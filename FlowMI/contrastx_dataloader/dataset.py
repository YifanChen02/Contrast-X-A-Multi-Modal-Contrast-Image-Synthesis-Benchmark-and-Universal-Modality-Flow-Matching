"""Contrast-X loader — two cohorts, one task.

Both cohorts are the same problem: the anatomy is fixed and only the CONTRAST STATE varies.

    breast DCE   <case>/<case>_dce{1,2,3}.nii                      M = 3
    CT           <case>/<study>/<case>_<date>_{CT,CTC}.nii          M = 2

Two things differ from the BraTS path and both are deliberate.

1. THE SPLIT IS THE RELEASE'S. BraTS had no official split so worldmodel_loader takes a 9:1
   per-cohort slice; Contrast-X ships train/val/test directories and those are used verbatim,
   so numbers stay comparable with anything else trained on this dataset.

2. NORMALISATION IS PER CASE, NOT PER MODALITY. This is the important one. The BraTS pipeline
   z-scores every modality independently (NormalizeIntensityd channel_wise=True), which is
   right there because t1/t2/flair are different physical quantities with no common scale.
   Here it would destroy the task: dce1/dce2/dce3 are THE SAME sequence at three times, and
   the only thing separating them is how much contrast agent has arrived. Normalising each
   phase to its own mean and variance cancels precisely that difference -- the model would be
   asked to predict an enhancement that the preprocessing had already removed. So:
     * CT  uses a fixed Hounsfield window. HU is quantitative and absolute, so one window
           serves every case and every study, and iodine enhancement survives as a real
           intensity increase.
     * DCE derives ONE affine from the pre-contrast phase and applies it unchanged to all
           three phases, so enhancement survives as a relative increase over baseline.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai import transforms as mt
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate

# cohort -> (directory glob, modality key order, isotropic spacing in mm)
#
# The spacing is NOT a free parameter, it is measured. Sampling 12 headers per cohort gave
#   CT  median FOV 400 x 400 x 218 mm, slice thickness ranging 1.25-5.0 mm
#   DCE median FOV 190 x 190 x 134 mm
# so the whole anatomy lands inside a 128^3 box at 3.5 mm for CT (the widest case seen was
# 449 mm, and 449/128 = 3.5) and at 1.5 mm for breast DCE (190/1.5 = 127).
#
# Getting this wrong is what broke the first CT run. Resampling body CT to 1 mm and then
# taking a 128^3 crop covers 128 mm of a 400 mm torso -- under 4% of the volume -- so almost
# every crop missed the patient and training collapsed.
COHORTS: Dict[str, Tuple[str, List[str], float]] = {
    "breast_dce": ("Breast_dce_train_val_test_nii_mask", ["dce1", "dce2", "dce3"], 1.5),
    "ct": ("*_CT_train_val_test", ["ct", "ctc"], 3.5),
}

# Global modality ids, shared across every dataset this project touches, so one model can be
# handed any mixture later without renumbering. BraTS occupies 0-3.
MODALITY_ID = {"dce1": 4, "dce2": 5, "dce3": 6, "ct": 7, "ctc": 8}

# NO fixed window. The first version mapped Hounsfield [-1000, 1000] onto [-1, 1] on the
# reasoning that HU is absolute, so one window would serve every study. That is true of CT in
# general and false of THESE files: the packager already applied a window, and not the same
# one everywhere. Measured over 40 sampled studies the stored ranges are
#     (-135, 215) x17    (-50, 150) x14    (-140, 260) x6    (-1150, 350) x3
# which is why a [-1000, 1000] window left the data occupying only [-0.14, +0.26] with a
# standard deviation of 0.088, an order of magnitude below the DCE cohort, and the model
# collapsed.
#
# What IS consistent, and what the task depends on, is that CT and CTC of the same study
# always share a window -- 40 out of 40 sampled. So the enhancement is real rather than a
# windowing artefact, and only the cross-study scale varies. A per-case affine removes exactly
# that, and it is the same rule the DCE cohort already uses.
#
# Residual limitation, not fixable by normalisation: a study clipped to (-50, 150) contains no
# air or bone at all, while one stored at (-1150, 350) does. Rescaling cannot put back content
# the packager discarded, so the cohort stays somewhat heterogeneous in what it depicts.


def _case_dicts_breast(root: str, split: str) -> List[dict]:
    out = []
    for cdir in sorted(glob.glob(os.path.join(root, split, "*"))):
        if not os.path.isdir(cdir):
            continue
        case = os.path.basename(cdir)
        paths = {m: os.path.join(cdir, f"{case}_{m}.nii") for m in ("dce1", "dce2", "dce3")}
        if not all(os.path.isfile(p) for p in paths.values()):
            continue
        paths.update(case=case, split=split, cohort="breast_dce")
        out.append(paths)
    return out


def _case_dicts_ct(root: str, split: str) -> List[dict]:
    """One entry per STUDY, not per patient: several cohorts are longitudinal and each study
    is an independent CT/CTC pair."""
    out = []
    cohort = os.path.basename(root).replace("_train_val_test", "")
    for cdir in sorted(glob.glob(os.path.join(root, split, "*"))):
        if not os.path.isdir(cdir):
            continue
        for sdir in sorted(glob.glob(os.path.join(cdir, "*"))):
            if not os.path.isdir(sdir):
                continue
            ct = glob.glob(os.path.join(sdir, "*_CT.nii"))
            ctc = glob.glob(os.path.join(sdir, "*_CTC.nii"))
            if len(ct) != 1 or len(ctc) != 1:
                continue
            out.append({
                "ct": ct[0], "ctc": ctc[0],
                "case": os.path.basename(cdir), "split": split, "cohort": cohort,
            })
    return out


def build_contrastx_dicts(extracted_root: str, cohort: str,
                          splits: Sequence[str] = ("train", "val", "test")
                          ) -> Dict[str, List[dict]]:
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; expected one of {sorted(COHORTS)}")
    pattern, _, _ = COHORTS[cohort]
    roots = sorted(glob.glob(os.path.join(extracted_root, pattern)))
    if not roots:
        raise FileNotFoundError(f"no directory matching {pattern!r} under {extracted_root}")

    out: Dict[str, List[dict]] = {s: [] for s in splits}
    for root in roots:
        for s in splits:
            if not os.path.isdir(os.path.join(root, s)):
                continue
            out[s] += (_case_dicts_breast(root, s) if cohort == "breast_dce"
                       else _case_dicts_ct(root, s))
    for s in splits:
        n_src = len({d["cohort"] for d in out[s]})
        print(f"[contrastx] {cohort} {s}: {len(out[s])} item(s) from {n_src} source cohort(s)")
    return out


class SharedNormalised(mt.MapTransform):
    """One intensity affine per CASE, applied to every modality.

    See the module docstring: a per-modality affine would cancel the enhancement, which is the
    entire signal here. `ref` names the modality the statistics come from -- the pre-contrast
    phase, so enhancement reads as a positive excursion above baseline.
    """

    def __init__(self, keys, ref: str):
        super().__init__(keys)
        self.ref = ref

    def __call__(self, data):
        d = dict(data)
        r = d[self.ref]
        arr = r.numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        fg = arr[arr > arr.min()]                       # drop the air/background plateau
        if fg.size == 0:
            fg = arr.reshape(-1)
        # Percentile-based, not mean/std: several CT studies arrive hard-clipped, so a large
        # spike sits exactly on the window edge and would drag a plain mean and inflate a
        # plain standard deviation by an amount that varies with which window was used.
        lo, mid, hi = np.percentile(fg, [5.0, 50.0, 95.0])
        mu = float(mid)
        sd = float(hi - lo) / 3.29                      # 5-95 spread of a normal, in sigmas
        if sd < 1e-6:
            sd = 1.0
        for k in self.keys:
            d[k] = (d[k] - mu) / sd
        return d


def _transforms(cohort: str, patch, spacing, train: bool):
    _, keys, iso = COHORTS[cohort]
    t = [
        mt.LoadImaged(keys=keys),
        mt.EnsureChannelFirstd(keys=keys),
        mt.Orientationd(keys=keys, axcodes="RAS"),
        # `spacing` is ignored in favour of the cohort's measured isotropic value
        mt.Spacingd(keys=keys, pixdim=(iso, iso, iso), mode="bilinear"),
    ]
    # One rule for both cohorts: normalise by the PRE-CONTRAST volume of this case and apply
    # that same affine to every phase, so enhancement survives as an excursion above baseline.
    # keys[0] is dce1 for breast and ct for the CT cohorts -- the pre-contrast acquisition in
    # each.
    t.append(SharedNormalised(keys=keys, ref=keys[0]))
    # NO CROPPING. The volume is resampled to the cohort's isotropic spacing above, which is
    # chosen so the whole anatomy already fits, and then centred in the box: short axes get
    # padded, the rare over-long one is trimmed at its ends. Proportions are exact -- this is
    # pure physical resampling, not an anisotropic squeeze onto a cube -- and every sample now
    # shows the model the entire organ instead of a 128 mm window onto it.
    t.append(mt.ResizeWithPadOrCropd(keys=keys, spatial_size=patch))
    if train:
        t.append(mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=0))
    # (B, M, H, W, D) -- the trainer and PoEAE both read M straight off this axis
    t.append(mt.ConcatItemsd(keys=keys, name="target", dim=0))
    t.append(mt.DeleteItemsd(keys=keys))
    t.append(mt.EnsureTyped(keys=["target"]))
    return mt.Compose(t)


def build_contrastx_dataloaders(
    extracted_root: str,
    cohort: str,
    patch_size=(128, 128, 128),
    spacing=(1.0, 1.0, 1.0),
    batch_size: int = 1,
    num_workers: int = 4,
    cache_rate: float = 0.0,
    max_train_cases: int = 0,
    val_cases: int = 12,
    seed: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], List[int]]:
    import random

    d = build_contrastx_dicts(extracted_root, cohort)
    rng = random.Random(seed)
    train_files, val_files = list(d["train"]), list(d["val"])
    rng.shuffle(train_files)
    if max_train_cases and max_train_cases < len(train_files):
        train_files = train_files[:max_train_cases]
        print(f"[contrastx] train capped to {len(train_files)}")
    val_files = val_files[:val_cases]
    print(f"[contrastx] train={len(train_files)}  val={len(val_files)}")

    def _ds(files, tf):
        if cache_rate > 0:
            return CacheDataset(files, tf, cache_rate=cache_rate, num_workers=num_workers)
        return Dataset(files, tf)

    train_loader = DataLoader(
        _ds(train_files, _transforms(cohort, patch_size, spacing, True)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=list_data_collate, drop_last=True,
    )
    val_loader = None
    if val_files:
        val_loader = DataLoader(
            _ds(val_files, _transforms(cohort, patch_size, spacing, False)),
            batch_size=1, shuffle=False, num_workers=num_workers,
            collate_fn=list_data_collate,
        )
    mod_ids = [MODALITY_ID[k] for k in COHORTS[cohort][1]]
    return train_loader, val_loader, mod_ids
