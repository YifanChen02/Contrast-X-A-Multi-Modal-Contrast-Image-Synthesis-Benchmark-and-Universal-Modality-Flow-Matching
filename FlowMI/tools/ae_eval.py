"""Score AE checkpoints on a FULL split, with a spread that says whether a gap is readable.

The trainer validates on `cap=16` cases, but CT has 64 val cases and huge per-case variance
across cases. Repeat runs of an identical config spread noticeably on a capped
set, so single-run gaps under ~1 dB have been uninterpretable. This re-scores any set of
checkpoints on every case of a split and prints the standard error, so a comparison can state
whether the difference clears the noise.

    python ae_eval.py --cohort ct --split val runs/mslot_ct/best.pt runs/mslot_ct3/best.pt
"""
import argparse, os, sys
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrastx_dataloader import COHORTS
from contrastx_dataloader.prep_dataset import PREP_ROOT, PrepSet
from autoencoder.maisi_slot import MaisiSlotAE

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--cohort", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--cap", type=int, default=0, help="0 = every case in the split")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
keys = COHORTS[a.cohort][1]
ds = PrepSet(PREP_ROOT, a.cohort, a.split, z_window=128, train=False, cap=a.cap)
print(f"{a.cohort} / {a.split}: {len(ds)} cases (trainer uses cap=16)\n")


def psnr_ssim(pred, target):
    from monai.metrics import SSIMMetric
    lo = target.min(); rng = (target.max() - lo).clamp(min=1e-6)
    t = (target - lo) / rng; p = ((pred - lo) / rng).clamp(0, 1)
    return (float(-10 * torch.log10(F.mse_loss(p, t).clamp(min=1e-12))),
            float(SSIMMetric(spatial_dims=3, data_range=1.0)(p, t).mean()))


print(f"{'checkpoint':>28} {'ep':>4} {'PSNR':>7} {'+-SE':>6} {'SSIM':>7} {'+-SE':>6}   per-modality")
rows = {}
for cp in a.ckpts:
    ck = torch.load(cp, map_location="cpu", weights_only=False)
    ae = MaisiSlotAE(latent_channels=ck.get("z_ch", 4), ckpt=None).to(dev).eval()
    ae.load_state_dict(ck["model"])
    for q in ae.parameters():
        q.requires_grad_(False)
    per = {k: [] for k in keys}; ss = []; case_mean = []
    for i in range(len(ds)):
        x = ds[i]["x"].unsqueeze(0).to(dev).float()
        z0 = min(int(ds[i]["z0"]), x.shape[-1])
        x = x[..., :((x.shape[-1] // 4) * 4)]
        with torch.no_grad():
            rec = ae(x, sample=False)["img"]
        cm = []
        for c, k in enumerate(keys):
            p, s = psnr_ssim(rec[:, c:c+1, ..., :z0], x[:, c:c+1, ..., :z0])
            per[k].append(p); ss.append(s); cm.append(p)
        case_mean.append(np.mean(cm))
    cm = np.array(case_mean)
    se = cm.std(ddof=1) / np.sqrt(len(cm))           # SE over CASES, the unit of replication
    ssa = np.array(ss); sse = ssa.std(ddof=1) / np.sqrt(len(ssa))
    tag = "/".join(cp.rstrip("/").split("/")[-2:])
    print(f"{tag:>28} {ck.get('epoch',-1):>4} {cm.mean():7.3f} {se:6.3f} "
          f"{ssa.mean()*100:7.2f} {sse*100:6.2f}   " +
          "  ".join(f"{k}={np.mean(per[k]):.2f}" for k in keys))
    rows[tag] = (cm, ssa.mean())
    del ae
    if dev == "cuda":
        torch.cuda.empty_cache()

# Paired comparison: the same cases go through every model, so pair them instead of
# comparing two independent means -- it removes the case-difficulty variance entirely.
names = list(rows)
if len(names) > 1:
    print(f"\npaired differences (same cases, so this is the readable comparison):")
    base = names[0]
    for n in names[1:]:
        d = rows[n][0] - rows[base][0]
        dse = d.std(ddof=1) / np.sqrt(len(d))
        sig = "significant" if abs(d.mean()) > 2 * dse else "NOT resolvable"
        print(f"  {n} - {base}: {d.mean():+.3f} dB +- {dse:.3f} (2SE={2*dse:.3f})  -> {sig}")
print("AE_EVAL_OK")
