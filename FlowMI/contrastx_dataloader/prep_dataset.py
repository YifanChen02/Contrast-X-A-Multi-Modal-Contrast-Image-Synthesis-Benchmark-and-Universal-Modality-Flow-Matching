"""Read the preprocessed 128x128xZ volumes.

The offline pass fixed the voxel size per case and left z at whatever the acquisition covered,
so depth varies from 128 (the padded floor) to about 460. Training takes a random 128-slice
window, which both fixes the tensor shape for batching and acts as augmentation along the one
axis where the field of view genuinely differs between scans. Evaluation keeps the whole volume
and lets the caller slide over it, so no anatomy is dropped from the metric.

Nothing here resamples or normalises: that was all done once, offline, and the manifest records
how to undo it.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from . import COHORTS

PREP_ROOT = os.environ.get("FLOWMI_PREP_ROOT", "data/prep128")


def build_prep_dicts(root: str, cohort: str,
                     splits: Sequence[str] = ("train", "val", "test")) -> Dict[str, List[dict]]:
    keys = COHORTS[cohort][1]
    out: Dict[str, List[dict]] = {s: [] for s in splits}
    for s in splits:
        for cdir in sorted(glob.glob(os.path.join(root, cohort, s, "*"))):
            if not os.path.isdir(cdir):
                continue
            case = os.path.basename(cdir)
            paths = {k: os.path.join(cdir, f"{case}_{k}.nii.gz") for k in keys}
            if not all(os.path.isfile(p) for p in paths.values()):
                continue
            paths.update(case=case, split=s, cohort=cohort)
            out[s].append(paths)
        print(f"[prep128] {cohort} {s}: {len(out[s])} case(s)")
    return out


def load_manifest(root: str, cohort: str) -> dict:
    p = os.path.join(root, f"manifest_{cohort}.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


class PrepSet(Dataset):
    """One item = one case as [M, 128, 128, Z].

    `z_window` crops a random slab at train time and is ignored otherwise, so a validation item
    and a training item of the same case differ only in what a random draw would have changed.
    """

    def __init__(self, root, cohort, split, z_window=128, train=False, cap=0, seed=0,
                 pad_multiple=16):
        self.items = build_prep_dicts(root, cohort, splits=[split])[split]
        if cap:
            self.items = self.items[:cap]
        self.keys = COHORTS[cohort][1]
        self.M = len(self.keys)
        self.z_window = z_window
        self.train = train
        self.pad_multiple = pad_multiple
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        vols = [np.asanyarray(nib.load(it[k]).dataobj).astype(np.float32) for k in self.keys]
        x = torch.from_numpy(np.stack(vols, 0))                 # [M, 128, 128, Z]
        z0 = z = x.shape[-1]
        if self.train and z > self.z_window:
            s = int(self.rng.integers(0, z - self.z_window + 1))
            x = x[..., s:s + self.z_window]
            z0 = self.z_window
        elif z < self.z_window:                                  # only the padded floor cases
            pad = self.z_window - z
            b = pad // 2
            x = torch.nn.functional.pad(x, [b, pad - b])
            z0 = self.z_window
        m = self.pad_multiple
        if x.shape[-1] % m:                                      # so the encoder can halve twice
            extra = m - x.shape[-1] % m
            x = torch.nn.functional.pad(x, [0, extra])           # trailing, so z0 crops it back
        if self.train:
            if self.rng.random() < 0.5:
                x = torch.flip(x, dims=[1])                      # left-right, as before
        return {"x": x, "case": it["case"], "z0": z0}
