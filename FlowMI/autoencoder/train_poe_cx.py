"""PoE-AE on the Contrast-X cohorts (breast_dce, ct).

autoencoder/stage_a_poe.py is wired to brats_dataloader and cannot see these cohorts, which
reach the disk through contrastx_dataloader.PrepSet. This is the same trainer loop as
train_maisi_slot.py -- same split handling, same best-by-val-PSNR selection, same latent gate,
same checkpoint keys -- with PoEAE in place of MaisiSlotAE.

One real difference: PoE exists to make every missing pattern meaningful, so training must SHOW
it missing patterns. Each step draws a random observed subset (never empty) and the loss scores
BOTH the reconstruction of what was observed and the completion of what was not. Reporting them
separately matters -- on the slot AE the two are the same number, here they are not, and the gap
is the thing PoE is supposed to close.
"""
from __future__ import annotations

import argparse, csv, logging, os, time

import numpy as np
import torch
import torch.nn.functional as F

from contrastx_dataloader import COHORTS
from contrastx_dataloader.prep_dataset import PREP_ROOT, PrepSet
from .poeae import PoEAE, PoECfg


def psnr_ssim(pred, target):
    from monai.metrics import SSIMMetric
    lo = target.min(); rng = (target.max() - lo).clamp(min=1e-6)
    t = (target - lo) / rng
    p = ((pred - lo) / rng).clamp(0, 1)
    ps = float(-10 * torch.log10(F.mse_loss(p, t).clamp(min=1e-12)))
    ss = float(SSIMMetric(spatial_dims=3, data_range=1.0)(p, t).mean())
    return ps, ss


def kl_std(mu, logvar):
    """KL(q||N(0,I)) per element, averaged. logvar is already clamped inside PoEAE."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(COHORTS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prep-root", default=PREP_ROOT)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--shared-ch", type=int, default=4)
    ap.add_argument("--private-ch", type=int, default=2,
                    help="0 gives a shared-only PoE; the private route is what an FM fills")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--n-res", type=int, default=2)
    ap.add_argument("--dec-n-res", type=int, default=2)
    ap.add_argument("--kl-shared", type=float, default=1e-6)
    ap.add_argument("--kl-private", type=float, default=1e-6)
    ap.add_argument("--perc-weight", type=float, default=0.1)
    ap.add_argument("--recon-loss", default="l1+l2", choices=["l1", "l2", "l1+l2"])
    ap.add_argument("--w-missing", type=float, default=1.0,
                    help="weight on the completion term relative to the reconstruction term")
    ap.add_argument("--p-drop", type=float, default=0.5,
                    help="probability a given modality is hidden; the observed set is resampled "
                         "until it is non-empty, so with M=2 this is close to a coin flip on "
                         "which single phase is seen")
    ap.add_argument("--drop-warmup", type=int, default=5,
                    help="epochs of full-observation training before dropping starts; a PoE "
                         "trained on holes from step 0 has no reconstruction to fall back on")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--val-cap", type=int, default=0)
    ap.add_argument("--val-window", type=int, default=128)
    ap.add_argument("--init-from", default="")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    lg = logging.getLogger("poecx"); lg.setLevel(logging.INFO); lg.handlers.clear()
    for h in (logging.FileHandler(os.path.join(a.out_dir, "log.txt")), logging.StreamHandler()):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")); lg.addHandler(h)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    keys = COHORTS[a.cohort][1]; M = len(keys)

    cfg = PoECfg(in_mod=M, base=a.base, n_res=a.n_res, dec_n_res=a.dec_n_res,
                 shared_ch=a.shared_ch, private_ch=a.private_ch)
    ae = PoEAE(cfg).to(dev)
    lg.info(f"PoE-AE shared={cfg.shared_ch} private={cfg.private_ch}  "
            f"{sum(p.numel() for p in ae.parameters())/1e6:.2f}M params")
    if a.init_from:
        ck = torch.load(a.init_from, map_location=dev, weights_only=False)
        ae.load_state_dict(ck["model"])
        lg.info(f"continuing from {a.init_from} (its ep{ck.get('epoch')}, "
                f"val_psnr {ck.get('val_psnr')})")

    from monai.losses import PerceptualLoss
    perc = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                          is_fake_3d=True, fake_3d_ratio=0.2).eval().to(dev)
    for q in perc.parameters():
        q.requires_grad_(False)

    opt = torch.optim.Adam(ae.parameters(), lr=a.lr, eps=1e-6)

    tr = PrepSet(a.prep_root, a.cohort, "train", z_window=a.patch, train=True)
    va = PrepSet(a.prep_root, a.cohort, "val", z_window=a.val_window, train=False, cap=a.val_cap)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch_size, shuffle=True,
                                     num_workers=8, drop_last=True)
    lg.info(f"cohort {a.cohort}: {len(tr)} train / {len(va)} val, M={M} ({', '.join(keys)})")

    def crop(x):
        _, _, H, W, D = x.shape; p = a.patch
        hs = np.random.randint(0, max(H - p, 0) + 1); ws = np.random.randint(0, max(W - p, 0) + 1)
        ds = np.random.randint(0, max(D - p, 0) + 1)
        return x[..., hs:hs+p, ws:ws+p, ds:ds+p]

    def draw_mask(B):
        """A random non-empty observed subset per sample."""
        m = (torch.rand(B, M) > a.p_drop).float()
        empty = m.sum(1) == 0
        if empty.any():                       # force one observed modality rather than resample
            j = torch.randint(0, M, (int(empty.sum()),))
            m[empty, j] = 1.0
        return m.to(dev)

    def recon_term(rec, x):
        if a.recon_loss == "l1":
            return F.l1_loss(rec, x)
        if a.recon_loss == "l2":
            return F.mse_loss(rec, x)
        return 0.5 * F.l1_loss(rec, x) + 0.5 * F.mse_loss(rec, x)

    csvp = os.path.join(a.out_dir, "metrics.csv")
    if not os.path.isfile(csvp):
        with open(csvp, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "recon_psnr", "recon_ssim",
                                    "compl_psnr", "compl_ssim"] +
                                   [f"{k}_psnr" for k in keys] + ["gate_absmax", "gate_kurt"])

    best_psnr, best_ep = -float("inf"), -1

    for ep in range(a.epochs):
        ae.train(); run = 0.0; n = 0; t0 = time.time()
        for b in dl:
            x = crop(b["x"].to(dev)).float()
            B = x.shape[0]
            mask = x.new_ones(B, M) if ep < a.drop_warmup else draw_mask(B)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                out = ae(x, mask=mask, sample=True)
            rec = out["img"].float()

            # score the observed and the missing halves separately -- averaging them would let a
            # good reconstruction hide a failed completion, which is the whole question here
            w = mask.view(B, M, 1, 1, 1)
            n_obs = w.sum().clamp(min=1.0); n_mis = (1 - w).sum().clamp(min=1.0)
            l_obs = (recon_term(rec * w, x * w) * w.numel() / n_obs)
            l_mis = (recon_term(rec * (1 - w), x * (1 - w)) * w.numel() / n_mis)
            loss = l_obs + a.w_missing * l_mis

            loss = loss + a.kl_shared * kl_std(out["a_mu"].float(), out["a_logvar"].float())
            if out["priv_mu"] is not None:
                loss = loss + a.kl_private * kl_std(out["priv_mu"].float(),
                                                    out["priv_logvar"].float())
            rf = rec.reshape(-1, 1, *rec.shape[2:]); xf = x.reshape(-1, 1, *x.shape[2:])
            loss = loss + a.perc_weight * perc(rf, xf)

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += float(loss); n += 1
        msg = f"ep{ep} loss {run/max(n,1):.4f} | {(time.time()-t0)/60:.1f} min"

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            ae.eval()
            rp, rs, cp, cs = [], [], [], []
            per = {k: [] for k in keys}; lat = []
            with torch.no_grad():
                for i in range(len(va)):
                    x = va[i]["x"].unsqueeze(0).to(dev).float()
                    z0 = min(int(va[i]["z0"]), x.shape[-1])
                    x = x[..., :((x.shape[-1] // 4) * 4)]
                    full = ae(x, mask=x.new_ones(1, M), sample=False)
                    lat.append(full["a_mu"].flatten().float().cpu())
                    for c, k in enumerate(keys):
                        ps, ss = psnr_ssim(full["img"][:, c:c+1, ..., :z0], x[:, c:c+1, ..., :z0])
                        rp.append(ps); rs.append(ss); per[k].append(ps)
                    # completion: hide one modality at a time and score only that one
                    for c in range(M):
                        m1 = x.new_ones(1, M); m1[0, c] = 0.0
                        o1 = ae(x, mask=m1, sample=False)
                        ps, ss = psnr_ssim(o1["img"][:, c:c+1, ..., :z0], x[:, c:c+1, ..., :z0])
                        cp.append(ps); cs.append(ss)
            mrp, mrs = float(np.mean(rp)), float(np.mean(rs))
            mcp, mcs = float(np.mean(cp)), float(np.mean(cs))
            mu = torch.cat(lat); mu = mu / mu.std().clamp(min=1e-8)
            g_max = float(mu.abs().max())
            g_kurt = float(((mu - mu.mean()) ** 4).mean() / (mu.var() ** 2))
            lg.info(msg + f"   RECON psnr {mrp:.3f} ssim {mrs:.4f}"
                          f" | COMPL psnr {mcp:.3f} ssim {mcs:.4f}  | " +
                    "  ".join(f"{k}={np.mean(per[k]):.2f}" for k in keys) +
                    f"  || GATE |max| {g_max:.2f} kurt {g_kurt:.2f} "
                    f"{'PASS' if (g_max < 12 and g_kurt < 12) else 'FAIL'}")
            with open(csvp, "a", newline="") as f:
                csv.writer(f).writerow([ep, f"{run/max(n,1):.4f}", f"{mrp:.4f}", f"{mrs:.4f}",
                                        f"{mcp:.4f}", f"{mcs:.4f}"] +
                                       [f"{np.mean(per[k]):.4f}" for k in keys] +
                                       [f"{g_max:.3f}", f"{g_kurt:.3f}"])
            ck = {"model": ae.state_dict(), "epoch": ep, "cohort": a.cohort,
                  "keys": keys, "args": vars(a), "cfg": vars(cfg),
                  "val_psnr": mrp, "val_ssim": mrs,
                  "compl_psnr": mcp, "compl_ssim": mcs,
                  "gate_absmax": g_max, "gate_kurt": g_kurt}
            torch.save(ck, os.path.join(a.out_dir, "last.pt"))
            if mrp > best_psnr:
                best_psnr, best_ep = mrp, ep
                torch.save(ck, os.path.join(a.out_dir, "best.pt"))
                lg.info(f"  new best  recon psnr {mrp:.3f} @ep{ep}  -> best.pt")
        else:
            lg.info(msg)
    lg.info(f"BEST recon psnr {best_psnr:.3f} @ep{best_ep}  ({a.out_dir}/best.pt)")
    lg.info("POECX_OK")


if __name__ == "__main__":
    main()
