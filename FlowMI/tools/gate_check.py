"""Latent gate for any SlotAE checkpoint: did the tail penalty tame the heavy tail, and what
did it cost?

The flow works best when the scaled latent has |max| < 6 and kurtosis ~3 after the one global scale to
std 1. The unregularized breast_dce AE sat at |max| 120, kurt 48 -- the failure the spec calls
the hidden killer. Pass several checkpoints to see the tail and the reconstruction move together.

    python gate_check.py --cohort breast_dce runs/mslot_dce/best.pt runs/mslot_dce4/best.pt
"""
import argparse, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrastx_dataloader.prep_dataset import PREP_ROOT, PrepSet
from autoencoder.maisi_slot import MaisiSlotAE

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--cohort", default="breast_dce")
ap.add_argument("--split", default="train")
ap.add_argument("--stride", type=int, default=4, help="take every Nth case")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ds = PrepSet(PREP_ROOT, a.cohort, a.split, z_window=128, train=False)
idxs = list(range(0, len(ds), a.stride))
print(f"cohort {a.cohort} / {a.split}: {len(idxs)} of {len(ds)} cases\n")
print(f"{'checkpoint':>26} {'ep':>4} {'std':>6} {'|max|':>8} {'kurt':>7} {'p99.9':>7} "
      f"{'gate':>5}   per-channel std")

for cp in a.ckpts:
    ck = torch.load(cp, map_location="cpu", weights_only=False)
    ae = MaisiSlotAE(latent_channels=ck.get("z_ch", 4), ckpt=None).to(dev).eval()
    ae.load_state_dict(ck["model"])
    for q in ae.parameters():
        q.requires_grad_(False)
    Z = []
    for i in idxs:
        x = ds[i]["x"]
        if x.shape[-1] > 128:
            x = x[..., :128]
        with torch.no_grad():
            mu, _ = ae.encode(x.unsqueeze(0).to(dev))
        Z.append(mu[0].half().cpu())
    Z = torch.stack(Z, 0).float()
    zs = Z / Z.std().clamp(min=1e-8)                  # ONE global scale, as the FM does
    k = float(((zs - zs.mean()) ** 4).mean() / (zs.var() ** 2))
    amax = float(zs.abs().max())
    # torch.quantile caps out around 16M elements; stride down deterministically instead.
    fl = zs.abs().flatten().float()
    p999 = float(fl[:: max(1, fl.numel() // 4_000_000)].quantile(0.999))
    pc = [round(float(zs[:, :, c].std()), 3) for c in range(zs.shape[2])]
    ok = "PASS" if (amax < 12 and k < 12) else "FAIL"
    tag = "/".join(cp.rstrip("/").split("/")[-2:])
    print(f"{tag:>26} {ck.get('epoch', -1):>4} {float(zs.std()):6.3f} {amax:8.2f} "
          f"{k:7.2f} {p999:7.2f} {ok:>5}   {pc}")
    del ae
    torch.cuda.empty_cache() if dev == "cuda" else None
