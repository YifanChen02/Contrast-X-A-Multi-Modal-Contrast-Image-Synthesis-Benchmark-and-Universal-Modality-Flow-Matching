"""Fine-tune the pretrained MAISI autoencoder as a per-modality slot encoder on Contrast-X.

Every modality is equal and encoded on its own by the same weights, so the modality axis is
folded into the batch: a (B, M, ...) volume is reshaped to (B*M, 1, ...), run through MAISI, and
folded back. The autoencoder therefore never sees more than one modality at a time and its slot
field is homogeneous by construction.

The loss follows MAISI's own recipe (generation/maisi tutorial), which matters most for the KL:
MAISI's encoder returns z_sigma, a STANDARD DEVIATION, and its KL is

    0.5 * sum_over_latent(mu^2 + sigma^2 - log(sigma^2) - 1)   averaged over the batch

-- a per-sample sum, not the mean-over-elements form the old slot trainer used with logvar.
Getting this wrong rescales the KL by ~(latent size) and either collapses the posterior or lets
it drift, so it is reproduced exactly. Weights are MAISI's defaults: kl 1e-7, perceptual 0.3,
adversarial 0.1 after a warm-up.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator

from contrastx_dataloader import COHORTS
from contrastx_dataloader.prep_dataset import PREP_ROOT, PrepSet
from autoencoder.maisi_slot import MaisiSlotAE


def kl_maisi(z_mu, z_sigma):
    """MAISI's KL, verbatim: sigma is a std, summed per sample, averaged over the batch."""
    eps = 1e-10
    kl = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
                         dim=list(range(1, z_mu.dim())))
    return kl.mean()


def psnr_ssim(pred, target):
    from monai.metrics import SSIMMetric
    lo = target.min(); rng = (target.max() - lo).clamp(min=1e-6)
    t = (target - lo) / rng
    p = ((pred - lo) / rng).clamp(0, 1)
    ps = float(-10 * torch.log10(F.mse_loss(p, t).clamp(min=1e-12)))
    ss = float(SSIMMetric(spatial_dims=3, data_range=1.0)(p, t).mean())
    return ps, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(COHORTS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prep-root", default=PREP_ROOT)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-weight", type=float, default=1e-7)
    ap.add_argument("--lat-l2", type=float, default=0.0,
                    help="penalty on latent values beyond lat-thresh*std -- targets the "
                         "heavy TAIL only (|max| 120, kurt 48), not the signal bulk")
    ap.add_argument("--lat-thresh", type=float, default=6.0,
                    help="tail threshold in units of latent std")
    ap.add_argument("--perc-weight", type=float, default=0.3)
    ap.add_argument("--adv-weight", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=5, help="epochs before the discriminator")
    ap.add_argument("--recon-loss", default="l1", choices=["l1", "l2", "l1+l2"],
                    help="PSNR is MSE-defined but L1 optimises the median, so a run judged on "
                         "PSNR should carry an L2 term. l1+l2 uses 0.5 of each.")
    ap.add_argument("--z-noise", type=float, default=0.0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast for the forward/backward. The AE has been training in "
                         "fp32 while the flow already used bf16; VRAM sits at 16%% and the GPU at "
                         "99%%, so the bottleneck is arithmetic, not memory -- precision buys "
                         "throughput where a larger batch only amortises launch overhead. bf16 "
                         "rather than fp16 because it keeps fp32's exponent range and needs no "
                         "loss scaling.")
    ap.add_argument("--init-from", default="",
                    help="continue from an existing slot-AE checkpoint instead of the MAISI brain "
                         "weights. Changing only how best.pt is SELECTED does not invalidate the "
                         "weights, so a rerun for that reason should not throw away the epochs "
                         "already spent; restarting discards the optimiser state as well as the weights.")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--val-window", type=int, default=128,
                    help="depth of the validation volumes. It used to inherit --patch (64), but "
                         "every downstream use -- latent extraction, the flow, every image-space "
                         "eval -- runs at 128, and the two read 3-5 dB apart. Selecting best.pt "
                         "on a 64-slab is selecting on a distribution the checkpoint is never "
                         "used on.")
    ap.add_argument("--val-cap", type=int, default=0,
                    help="cases used for validation; 0 = the whole split. The old hard-coded 16 "
                         "left about 1 dB of run-to-run noise on CT (five identical 1-epoch runs "
                         "), which can be larger than the effect being compared, so "
                         "best.pt was partly being selected on noise.")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    lg = logging.getLogger("mslot"); lg.setLevel(logging.INFO); lg.handlers.clear()
    for h in (logging.FileHandler(os.path.join(a.out_dir, "log.txt")), logging.StreamHandler()):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")); lg.addHandler(h)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    keys = COHORTS[a.cohort][1]; M = len(keys)

    ae = MaisiSlotAE(latent_channels=4,
                     ckpt=None if a.no_pretrained else None if a.no_pretrained else
                     os.environ.get("FLOWMI_MAISI_CKPT", "weights/autoencoder_MAISI_Brain.pt")).to(dev)
    lg.info(f"MAISI slot AE: pretrained {ae.loaded:.0%} loaded, {sum(p.numel() for p in ae.parameters())/1e6:.1f}M params")
    if a.init_from:
        _ck = torch.load(a.init_from, map_location=dev, weights_only=False)
        ae.load_state_dict(_ck["model"])
        lg.info(f"continuing from {a.init_from} (its ep{_ck.get('epoch')}, "
                f"val_psnr {_ck.get('val_psnr')})")

    disc = PatchDiscriminator(spatial_dims=3, num_layers_d=3, channels=32,
                              in_channels=1, out_channels=1).to(dev)
    adv = PatchAdversarialLoss(criterion="least_squares")
    perc = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                          is_fake_3d=True, fake_3d_ratio=0.2).eval().to(dev)
    for q in perc.parameters():
        q.requires_grad_(False)

    optg = torch.optim.Adam(ae.parameters(), lr=a.lr, eps=1e-6)
    optd = torch.optim.Adam(disc.parameters(), lr=a.lr, eps=1e-6)

    tr = PrepSet(a.prep_root, a.cohort, "train", z_window=a.patch, train=True)
    va = PrepSet(a.prep_root, a.cohort, "val", z_window=a.val_window, train=False,
                 cap=a.val_cap)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch_size, shuffle=True,
                                     num_workers=8, drop_last=True)
    lg.info(f"cohort {a.cohort}: {len(tr)} train / {len(va)} val, M={M}, "
            f"val window {a.val_window} (downstream uses 128)")

    def crop64(x):
        """MAISI trains at 64^3; take a random in-plane + z window so a slot fits one patch."""
        _, _, H, W, D = x.shape
        p = a.patch
        hs = np.random.randint(0, max(H - p, 0) + 1); ws = np.random.randint(0, max(W - p, 0) + 1)
        ds = np.random.randint(0, max(D - p, 0) + 1)
        return x[..., hs:hs+p, ws:ws+p, ds:ds+p]

    csvp = os.path.join(a.out_dir, "metrics.csv")
    if not os.path.isfile(csvp):
        with open(csvp, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "recon", "psnr", "ssim"] +
                                   [f"{k}_psnr" for k in keys] + ["gate_absmax", "gate_kurt"])

    best_psnr, best_ep = -float("inf"), -1

    for ep in range(a.epochs):
        ae.train(); disc.train(); run = 0.0; n = 0; t0 = time.time()
        for b in dl:
            x = crop64(b["x"].to(dev)).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                out = ae(x, sample=True, z_noise=a.z_noise)
            rec = out["img"].float()
            sigma = torch.exp(0.5 * out["logvar"].float())
            if a.recon_loss == "l1":
                l1 = F.l1_loss(rec, x)
            elif a.recon_loss == "l2":
                l1 = F.mse_loss(rec, x)
            else:
                l1 = 0.5 * F.l1_loss(rec, x) + 0.5 * F.mse_loss(rec, x)
            klv = kl_maisi(out["mu"].float().reshape(-1, *out["mu"].shape[2:]),
                           sigma.reshape(-1, *sigma.shape[2:]))
            pl = 0.0
            rf = rec.reshape(-1, 1, *rec.shape[2:]); xf = x.reshape(-1, 1, *x.shape[2:])
            pl = perc(rf, xf)
            g = l1 + a.kl_weight * klv + a.perc_weight * pl
            if a.lat_l2 > 0:
                mu = out["mu"]
                thr = a.lat_thresh * mu.detach().std()
                excess = (mu.abs() - thr).clamp(min=0.0)
                g = g + a.lat_l2 * excess.pow(2).mean()
            if ep >= a.warmup:
                logits = disc(rf.contiguous())[-1]
                g = g + a.adv_weight * adv(logits, target_is_real=True, for_discriminator=False)
            optg.zero_grad(set_to_none=True); g.backward(); optg.step()

            if ep >= a.warmup:
                optd.zero_grad(set_to_none=True)
                lf = adv(disc(rf.contiguous().detach())[-1], target_is_real=False, for_discriminator=True)
                lr_ = adv(disc(xf.contiguous())[-1], target_is_real=True, for_discriminator=True)
                (0.5 * (lf + lr_)).backward(); optd.step()
            run += float(l1); n += 1
        msg = f"ep{ep} recon {run/max(n,1):.4f} | {(time.time()-t0)/60:.1f} min"

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            ae.eval()
            pm = {k: [] for k in keys}; ss_all = []; mus = []
            with torch.no_grad():
                for i in range(len(va)):
                    x = va[i]["x"].unsqueeze(0).to(dev).float()
                    z0 = min(int(va[i]["z0"]), x.shape[-1])
                    x = x[..., :((x.shape[-1] // 4) * 4)]      # MAISI needs /4 depth
                    out = ae(x, sample=False)
                    rec = out["img"]
                    mus.append(out["mu"].flatten().float().cpu())
                    for c, k in enumerate(keys):
                        ps, ss = psnr_ssim(rec[:, c:c+1, ..., :z0], x[:, c:c+1, ..., :z0])
                        pm[k].append(ps); ss_all.append(ss)
            mp = np.mean([v for vs in pm.values() for v in vs]); ms = np.mean(ss_all)
            # Latent gate, on the same units the FM will see (one global scale to std 1).
            # Without this the tail penalty costs reconstruction with nothing to show for it.
            mu = torch.cat(mus); mu = mu / mu.std().clamp(min=1e-8)
            g_max = float(mu.abs().max())
            g_kurt = float(((mu - mu.mean()) ** 4).mean() / (mu.var() ** 2))
            lg.info(msg + "   VAL psnr %.3f ssim %.4f  | " % (mp, ms) +
                    "  ".join(f"{k}={np.mean(pm[k]):.2f}" for k in keys) +
                    f"  || GATE |max| {g_max:.2f} kurt {g_kurt:.2f} "
                    f"{'PASS' if (g_max < 12 and g_kurt < 12) else 'FAIL'}")
            with open(csvp, "a", newline="") as f:
                csv.writer(f).writerow([ep, f"{run/max(n,1):.4f}", f"{mp:.4f}", f"{ms:.4f}"] +
                                       [f"{np.mean(pm[k]):.4f}" for k in keys] +
                                       [f"{g_max:.3f}", f"{g_kurt:.3f}"])
            ck = {"model": ae.state_dict(), "epoch": ep, "cohort": a.cohort, "z_ch": 4,
                  "keys": keys, "args": vars(a), "val_psnr": float(mp), "val_ssim": float(ms),
                  "gate_absmax": g_max, "gate_kurt": g_kurt}
            torch.save(ck, os.path.join(a.out_dir, "last.pt"))
            if mp > best_psnr:                       # best.pt was overwritten unconditionally
                best_psnr, best_ep = mp, ep
                torch.save(ck, os.path.join(a.out_dir, "best.pt"))
                lg.info(f"  new best  psnr {mp:.3f} @ep{ep}  -> best.pt")
        else:
            lg.info(msg)
    lg.info(f"BEST val psnr {best_psnr:.3f} @ep{best_ep}  ({a.out_dir}/best.pt)")
    lg.info("MSLOT_OK")


if __name__ == "__main__":
    main()
