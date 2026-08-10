"""Cache SlotAE latents for a Contrast-X cohort.

The BraTS extractor reproduces that loader's 9:1 split; here the split is already on disk, so
this only has to read prep128, encode, and record which split each case came from.

Two things are computed alongside the latents because the flow model needs them and they must be
the SAME numbers everywhere:

  the GLOBAL scale -- one scalar over the whole training tensor, so that scaled latents have unit
      standard deviation. Not per-channel: rescaling channels independently destroys their
      relative amplitudes, and the velocity field is defined jointly across channels, so a
      channel that is quiet for a reason would be amplified into competing with one that carries
      signal.
  the distribution gate -- standard deviation, extreme value and kurtosis of the scaled latents.
      Heavy tails are what break a latent flow: MSE is dominated by the extremes, which in
      medical images are the high-signal regions. If the gate fails the fix belongs in the
      autoencoder (a light latent L2, or soft clipping), not in the flow.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from contrastx_dataloader import COHORTS
from contrastx_dataloader.prep_dataset import PREP_ROOT, PrepSet
from .slotae import SlotAE, SlotCfg
from .maisi_slot import MaisiSlotAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-ckpt", required=True)
    ap.add_argument("--cohort", required=True, choices=list(COHORTS))
    ap.add_argument("--prep-root", default=PREP_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ae-kind", default="slot", choices=["slot", "maisi"])
    ap.add_argument("--z-window", type=int, default=128,
                    help="centre slab encoded per case, so every latent has the same depth")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ae_ckpt, map_location=dev, weights_only=False)
    if a.ae_kind == "maisi":
        ae = MaisiSlotAE(latent_channels=4, ckpt=None).to(dev).eval()
        ae.load_state_dict(ck["model"])
    else:
        cfg = SlotCfg(**ck['cfg']) if isinstance(ck.get('cfg'), dict) else SlotCfg()
        ae = SlotAE(cfg).to(dev).eval()
        ae.load_state_dict(ck['model'])
    for q in ae.parameters():
        q.requires_grad_(False)
    print("frozen %s AE %s epoch %s" % (a.ae_kind, a.ae_ckpt, ck.get("epoch")))

    Z, ids, sps = [], [], []
    t0 = time.time()
    for split in ("train", "val", "test"):
        ds = PrepSet(a.prep_root, a.cohort, split, train=False)
        for i in range(len(ds)):
            it = ds[i]
            x = it["x"]
            z0 = min(int(it["z0"]), x.shape[-1])
            # a fixed centre slab: the flow needs one shape, and the centre is where the
            # anatomy is once the volumes are centred
            if z0 > a.z_window:
                s = (z0 - a.z_window) // 2
                x = x[..., s:s + a.z_window]
            elif x.shape[-1] > a.z_window:
                x = x[..., :a.z_window]
            if x.shape[-1] < a.z_window:
                pad = a.z_window - x.shape[-1]
                x = torch.nn.functional.pad(x, [pad // 2, pad - pad // 2])
            with torch.no_grad():
                mu, _ = ae.encode(x.unsqueeze(0).to(dev))
            Z.append(mu[0].half().cpu())
            ids.append(it["case"]); sps.append(split)
            if len(Z) % 100 == 0:
                el = time.time() - t0
                print(f"  {len(Z)}  {el/60:.1f} min", flush=True)

    Z = torch.stack(Z, 0)
    tr = torch.tensor([s == "train" for s in sps])
    zt = Z[tr].float()
    scale = 1.0 / float(zt.std())                      # ONE scalar, all channels together
    zs = zt * scale
    k = float(((zs - zs.mean()) ** 4).mean() / (zs.var() ** 2))
    print(f"\n=== gate on the scaled TRAIN latents (scale {scale:.5f})")
    print(f"  std       {float(zs.std()):.4f}   (want ~1.0)")
    print(f"  |max|     {float(zs.abs().max()):.3f}    (want <6; >10 means outliers)")
    print(f"  kurtosis  {k:.2f}     (want ~3; >10 means heavy tails)")
    per_slot = [float(zt[:, m].std() * scale) for m in range(Z.shape[1])]
    print(f"  per-slot std after the global scale: "
          f"{np.array2string(np.array(per_slot), precision=3)}")
    per_ch = [float(zt[:, :, c].std() * scale) for c in range(Z.shape[2])]
    print(f"  per-channel std (NOT normalised away, on purpose): "
          f"{np.array2string(np.array(per_ch), precision=3)}")
    ok = float(zs.std()) > 0.5 and float(zs.abs().max()) < 12 and k < 12
    print(f"  VERDICT: {'usable for a latent flow' if ok else 'FAILS -- fix the AE, not the FM'}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"latents": Z, "ids": ids, "split": sps, "cohort": a.cohort,
                "ae_ckpt": a.ae_ckpt, "cfg": ck.get("cfg", {}), "scale": scale,
                "gate": {"std": float(zs.std()), "absmax": float(zs.abs().max()),
                         "kurtosis": k, "pass": bool(ok)},
                "keys": COHORTS[a.cohort][1]}, a.out)
    print(f"\nwrote {a.out}   latents {tuple(Z.shape)}  "
          f"{Z.numel()*Z.element_size()/1e6:.0f} MB")
    print(f"  train {sps.count('train')}  val {sps.count('val')}  test {sps.count('test')}")
    print("LATENTS_OK")


if __name__ == "__main__":
    main()
