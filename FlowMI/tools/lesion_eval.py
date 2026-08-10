"""Is the enhancing lesion lost in the AE or in the FM?

Whole-volume PSNR cannot see a 1%-of-voxels region, so a model can look fine and still erase
every lesion. This scores the same prediction twice -- over the whole volume and over an
enhancement ROI -- and puts three things on the same axis:

    copy      : emit the observed pre-contrast phase unchanged (the "no enhancement" answer)
    ae_oracle : encode the TRUE target and decode it (the ceiling any FM can reach)
    fm        : the flow's prediction

If ae_oracle is already near `copy` on the ROI, the lesion never survives the autoencoder and no
loss reweighting in the FM can bring it back. If ae_oracle is well above fm, the information is
in the latent and the FM is the thing to fix.

ROI = the top `--roi-pct` voxels of the enhancement map (target - observed pre-contrast), which
needs no segmentation labels and is exactly what is clinically salient.
"""
import argparse, json, os, sys
import os as _os
RESULTS = _os.environ.get("FLOWMI_RESULTS", "results")
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrastx_dataloader import COHORTS
from contrastx_dataloader.prep_dataset import PREP_ROOT, PrepSet
from autoencoder.maisi_slot import MaisiSlotAE
from flow.slot_fm import SlotFlow, sample, tailclip_fwd, tailclip_inv, flow_start

# (name, observed pattern, predicted modality, index of the pre-contrast phase to diff against)
TASKS = {"breast_dce": [("DCE1_to_DCE2", (1, 0, 0), "dce2", 0),
                        ("DCE1and3_to_DCE2", (1, 0, 1), "dce2", 0)],
         "ct": [("CT_to_CTC", (1, 0), "ctc", 0)]}

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", required=True)
ap.add_argument("--fm-ckpt", required=True)
ap.add_argument("--roi-pct", type=float, default=1.0, help="top %% of the enhancement map")
ap.add_argument("--cases", type=int, default=24)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--force-noise", type=float, default=-1.0,
                help="override the checkpoint's start noise at inference. 0 makes the flow "
                     "deterministic, so its output IS the model's conditional mean rather than a "
                     "sample -- which separates sampling error from a genuinely worse mapping.")
ap.add_argument("--dev-q", type=float, default=0.0,
                help="keep the flow's departure from the base only where it exceeds this "
                     "quantile of |z_fm - base|, and fall back to the base elsewhere. The ROI is "
                     "1%% of voxels while the whole-volume damage is spread over the other 99%%, "
                     "so if the damage is diffuse and the gain is concentrated they separate. "
                     "0 disables (keep everything).")
ap.add_argument("--dev-soft", type=float, default=0.0,
                help="softness of that gate in quantile units; 0 = hard threshold")
ap.add_argument("--sample-k", type=int, default=1,
                help="latent samples averaged before decoding. A stochastic model draws one "
                     "sample per call, whose per-voxel PSNR is necessarily worse than the "
                     "conditional mean -- while validation R2 already scores the K-sample mean. "
                     "Scoring k=1 against a deterministic copy baseline compares different "
                     "quantities.")
ap.add_argument("--cfg-adaptive", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                help="train tells underfit from a generalisation gap; the "
                     "hard-coded test-only default hid that distinction")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
keys = COHORTS[a.cohort][1]; M = len(keys)
fb = torch.load(a.fm_ckpt, map_location=dev, weights_only=False)
Z = fb["Z"]; scale = fb["scale"]; asinh = fb.get("asinh", 0.0); tclip = fb.get("tailclip", 0.0)
net = SlotFlow(M, Z, gate=fb.get("gate_head", False)).to(dev).eval(); net.load_state_dict(fb["ema"])
ae = MaisiSlotAE(latent_channels=Z, ckpt=None).to(dev).eval()
ae.load_state_dict(torch.load(fb["ae_ckpt"], map_location=dev, weights_only=False)["model"])
ns = float(fb.get("noise_start", 0.0)); bm = fb.get("base", "zero")
if a.force_noise >= 0: ns = a.force_noise
um = fb.get("umap"); rl = fb.get("rel")
print(f"FM ep{fb['epoch']} tailclip {tclip} noise {ns} base {bm} cfg {a.cfg}"
      f"{' ADAPTIVE' if a.cfg_adaptive else ''} k={a.sample_k} devq={a.dev_q} | AE {fb['ae_ckpt']}")


def fwd(z):
    z = z * scale
    if tclip > 0: z = tailclip_fwd(z, tclip)
    return asinh * torch.asinh(z / asinh) if asinh > 0 else z


def inv(z):
    if asinh > 0: z = asinh * torch.sinh(z / asinh)
    if tclip > 0: z = tailclip_inv(z, tclip)
    return z / scale


def psnr_masked(pred, tg, mask):
    """PSNR restricted to `mask`, normalised by the WHOLE-volume target range so the ROI
    number stays on the same scale as the whole-volume one."""
    lo = tg.min(); rng = (tg.max() - lo).clamp(min=1e-6)
    t = (tg - lo) / rng; p = ((pred - lo) / rng).clamp(0, 1)
    se = (((p - t) ** 2) * mask).sum() / mask.sum().clamp(min=1)
    return float(-10 * torch.log10(se.clamp(min=1e-12)))


ds = PrepSet(PREP_ROOT, a.cohort, a.split, z_window=128, train=False)
n = min(a.cases, len(ds))
acc = {}
for i in range(n):
    it = ds[i]; x = it["x"]
    x = x[..., :(x.shape[-1] // 32 * 32)]
    z0 = min(int(it["z0"]), x.shape[-1])
    xs = x.unsqueeze(0).to(dev).float()
    with torch.no_grad():
        mu, _ = ae.encode(xs)
        rec_all = ae.decode(mu)                       # AE oracle: true latent, decoded
    for tname, pat, k, pre_idx in TASKS[a.cohort]:
        m = keys.index(k)
        tg = xs[:, m:m+1, ..., :z0]
        obs = xs[:, pre_idx:pre_idx+1, ..., :z0]      # the "copy" answer
        enh = (tg - obs)[0, 0]
        thr = torch.quantile(enh.flatten().float()[::7], 1 - a.roi_pct / 100.0)
        roi = (enh > thr).float()[None, None]
        with torch.no_grad():
            zf = fwd(mu)
            mask = torch.tensor([pat], dtype=torch.float32, device=dev)
            gs = []
            for _ in range(max(a.sample_k, 1)):
                st, cd, bfv, _ = flow_start(zf, mask, noise=ns, base=bm, rel=rl, umap=um)
                gs.append(sample(net, st, mask, M, Z, steps=a.steps, cfg=a.cfg,
                                 cfg_adaptive=a.cfg_adaptive, cond=cd, base=bfv))
            g = torch.stack(gs).mean(0)
            if a.dev_q > 0:
                delta = g - bfv                     # NOT `dev` -- that name is the torch device
                mag = delta.abs()
                thr = mag.flatten()[::7].quantile(a.dev_q)
                if a.dev_soft > 0:
                    w = torch.sigmoid((mag - thr) / (a.dev_soft * thr.clamp(min=1e-6)))
                else:
                    w = (mag >= thr).to(delta.dtype)
                g = bfv + delta * w
            g = g.reshape(1, M, Z, *mu.shape[3:])
            fm = ae.decode(inv(g))[:, m:m+1, ..., :z0]
        row = acc.setdefault(tname, {c: [] for c in
                                     ("copy_roi", "ae_roi", "fm_roi",
                                      "copy_all", "ae_all", "fm_all")})
        # COPY pays no AE tax -- it is the real image. The flow's output is decoded, so it
        # carries the round-trip loss that copy never pays. copy_ae is the same "emit the
        # observed phase" answer put through encode+decode, i.e. the bar the flow actually has
        # to clear once both sides are charged the same tax.
        row.setdefault("copy_ae_roi", []).append(
            psnr_masked(rec_all[:, pre_idx:pre_idx+1, ..., :z0], tg, roi))
        row.setdefault("copy_ae_all", []).append(
            psnr_masked(rec_all[:, pre_idx:pre_idx+1, ..., :z0], tg,
                        torch.ones_like(roi).expand_as(tg)))
        row["copy_roi"].append(psnr_masked(obs, tg, roi))
        row["ae_roi"].append(psnr_masked(rec_all[:, m:m+1, ..., :z0], tg, roi))
        row["fm_roi"].append(psnr_masked(fm, tg, roi))
        one = torch.ones_like(roi).expand_as(tg)
        row["copy_all"].append(psnr_masked(obs, tg, one))
        row["ae_all"].append(psnr_masked(rec_all[:, m:m+1, ..., :z0], tg, one))
        row["fm_all"].append(psnr_masked(fm, tg, one))

print(f"\nROI = top {a.roi_pct}% of the enhancement map, {n} cases\n")
print(f"{'task':>18} | {'copy':>7} {'copyAE':>7} {'ae_orc':>7} {'fm':>7}  (ROI)"
      f" | {'copy':>7} {'copyAE':>7} {'ae_orc':>7} {'fm':>7}  (whole)")
out = {}
for t, r in acc.items():
    v = {c: float(np.mean(r[c])) for c in r}
    out[t] = v
    print(f"{t:>18} | {v['copy_roi']:7.2f} {v['copy_ae_roi']:7.2f} {v['ae_roi']:7.2f} "
          f"{v['fm_roi']:7.2f} | {v['copy_all']:7.2f} {v['copy_ae_all']:7.2f} "
          f"{v['ae_all']:7.2f} {v['fm_all']:7.2f}")
    print(f"{'':>18}   vs the AE-taxed copy: whole {v['fm_all']-v['copy_ae_all']:+.2f} dB, "
          f"ROI {v['fm_roi']-v['copy_ae_roi']:+.2f} dB")
    head = v["ae_roi"] - v["copy_roi"]
    got = v["fm_roi"] - v["copy_roi"]
    print(f"{'':>18}   ROI headroom above copy: {head:+.2f} dB;  FM captured "
          f"{got:+.2f} dB ({100*got/head if abs(head) > 1e-6 else float('nan'):.0f}%)")
jp = f"{RESULTS}/lesion_{a.cohort}_{a.tag or 'std'}.json"
json.dump(out, open(jp, "w"), indent=1)
print(f"\nwrote {jp}\nLESION_EVAL_OK")
