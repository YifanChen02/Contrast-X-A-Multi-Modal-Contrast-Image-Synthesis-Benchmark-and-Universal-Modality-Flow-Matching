"""Latent dataset for the conditional flow.

ARBITRARY MISSINGNESS IS SAMPLED, NOT STORED. Each item holds the per-modality experts, and
the PoE shared code for any subset is a precision-weighted sum of them:

    mu_S = sum_{m in S} tau_m mu_m / (sum_{m in S} tau_m + 1),   tau_m = exp(-logvar_m)

so a fresh mask can be drawn every step and mu_S recomputed on the spot. Dumping a fixed list
of subsets instead would have capped M=3 at whichever few were written out, and 2^M - 2 grows
fast.

WHY THE TARGET DOES NOT DEPEND ON THE MASK. `encode_expert(x_m)` never sees the mask, so the
private code of a modality is the same whether it was encoded alone or alongside others. The
flow's target is therefore fixed per case; only the conditioning varies with the subset.

SCALING. Raw latents run to |a| ~ 60 and |r| ~ 145, far outside the unit-ish range a flow's
noise schedule assumes. Statistics are computed once over the training split and cached next
to the data; both codes are divided by their own global standard deviation. This is the same
move as Stable Diffusion's 0.18215, done per cohort because these two differ by ~4x.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def poe_shared(expert_mu, expert_logvar, mask, eps: float = 1e-6):
    """(B,M,S,...) experts + (B,M) mask -> (B,S,...) shared posterior mean.

    The +1 is the unit prior precision, exactly as in PoEAE.poe, so a code built here is the
    same object the decoder was trained against."""
    w = mask.view(*mask.shape, *([1] * (expert_mu.dim() - 2)))
    prec = torch.exp(-expert_logvar) * w
    return (expert_mu * prec).sum(1) / (prec.sum(1) + 1.0).clamp(min=eps)


def poe_full(expert_mu, expert_logvar):
    """The shared code aggregated over EVERY modality. Unreachable at inference -- it needs the
    absent modalities -- which is exactly why it is a target worth predicting."""
    ones = expert_mu.new_ones(expert_mu.shape[:2])
    return poe_shared(expert_mu, expert_logvar, ones)


def modality_corr(root, cohort, limit=80):
    """Mean within-case correlation between each pair of modalities' private codes.

    Cached beside the latents. Within-case rather than pooled: a pooled correlation would be
    dominated by between-case variance and would overstate how useful one modality is as a
    reference for another in the SAME case, which is the only thing that matters here.
    """
    import itertools
    p = os.path.join(root, cohort, "modality_corr.json")
    if os.path.isfile(p):
        with open(p) as f:
            return np.array(json.load(f))
    fs = sorted(glob.glob(os.path.join(root, cohort, "train", "*.npz")))[:limit]
    if not fs:
        raise FileNotFoundError(f"no training latents under {root}/{cohort}/train")
    pr = np.stack([np.load(f, allow_pickle=True)["priv"].astype(np.float32) for f in fs])
    N, M = pr.shape[0], pr.shape[1]
    flat = pr.reshape(N, M, -1)
    C = np.eye(M)
    for i, j in itertools.combinations(range(M), 2):
        c = float(np.mean([np.corrcoef(flat[k, i], flat[k, j])[0, 1] for k in range(N)]))
        C[i, j] = C[j, i] = c
    with open(p, "w") as f:
        json.dump(C.tolist(), f)
    return C


def _stats_path(root, cohort):
    return os.path.join(root, cohort, "scale_stats.json")


def compute_stats(root, cohort, limit: int = 200):
    """Global std of each code, over the training split. Cached; deleting the json recomputes."""
    p = _stats_path(root, cohort)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    fs = sorted(glob.glob(os.path.join(root, cohort, "train", "*.npz")))[:limit]
    if not fs:
        raise FileNotFoundError(f"no training latents under {root}/{cohort}/train")
    a, r = [], []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        a.append(d["expert_mu"].astype(np.float32).ravel())
        r.append(d["priv"].astype(np.float32).ravel())
    st = {"expert_std": float(np.concatenate(a).std()),
          "priv_std": float(np.concatenate(r).std()),
          "n_files": len(fs)}
    with open(p, "w") as f:
        json.dump(st, f, indent=1)
    return st


class LatentSubsets(Dataset):
    """One item = one case plus a freshly drawn incomplete mask.

    `fixed_mask` pins the subset, which is what evaluation wants: every pattern scored
    separately rather than averaged over a random draw.
    """

    def __init__(self, root, cohort, split, stats=None, fixed_mask=None, seed=0,
                 sample_target=False, with_image=False, data_root=None):
        self.files = sorted(glob.glob(os.path.join(root, cohort, split, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no latents under {root}/{cohort}/{split}")
        self.stats = stats or compute_stats(root, cohort)
        self.fixed_mask = fixed_mask
        # SAMPLE THE TARGET instead of returning the posterior mean.
        #
        # Learning a conditional DISTRIBUTION needs several observations per condition, and the
        # stored latents give exactly ONE target per (case, mask) pair. With a single point per
        # condition the likelihood is maximised by a delta on it, and whatever generalisation
        # survives is the local mean -- which is the collapse that was measured, and it has
        # nothing to do with the AE's KL.
        #
        # Drawing r ~ N(priv_mu, exp(priv_logvar)) afresh each visit is what turns one target
        # into many, and it is only meaningful once that posterior has real width. It did not:
        # the private KL was absent entirely and sigma sat on the clamp at 0.0498 against a mu
        # spread of ~10. So this switch and the new KL term only pay off together.
        self.sample_target = sample_target
        # loading the volume alongside the latent is what makes an image-space loss possible at
        # all; the latents alone cannot be decoded back to a target
        self.with_image = with_image
        self.data_root = data_root
        self._imap = None
        if with_image:
            from contrastx_dataloader import COHORTS
            from contrastx_dataloader.dataset import build_contrastx_dicts, _transforms
            iso = COHORTS[cohort][2]
            self._tf = _transforms(cohort, (128, 128, 128), (iso, iso, iso), train=False)
            items = build_contrastx_dicts(data_root, cohort, splits=[split])[split]
            self._imap = {it["case"]: it for it in items}
        self.rng = np.random.default_rng(seed)
        d = np.load(self.files[0], allow_pickle=True)
        self.M = d["priv"].shape[0]
        self.P = d["priv"].shape[1]
        self.S = d["expert_mu"].shape[1]
        # Read the GLOBAL modality ids off the file rather than assuming position==id. That
        # assumption is exactly what wired two earlier evaluations to untrained embedding rows.
        if "mod_ids" in d.files:
            self.mod_ids = [int(x) for x in d["mod_ids"]]
        else:
            raise KeyError(f"{self.files[0]} has no mod_ids -- re-run extract_latents")

    def __len__(self):
        return len(self.files)

    def _draw_mask(self):
        # uniform over the 2^M - 2 proper subsets: at least one present so the shared code has
        # evidence, at least one absent so there is something for the flow to generate
        while True:
            m = self.rng.integers(0, 2, self.M)
            if 0 < m.sum() < self.M:
                return m.astype(np.float32)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        es = self.stats["expert_std"]
        ps = self.stats["priv_std"]
        emu = torch.from_numpy(d["expert_mu"].astype(np.float32)) / es
        elv = torch.from_numpy(d["expert_logvar"].astype(np.float32))
        priv = torch.from_numpy(d["priv"].astype(np.float32)) / ps
        if self.sample_target:
            plv = torch.from_numpy(d["priv_logvar"].astype(np.float32))
            # the posterior sigma is in RAW units, so it scales the same way priv does
            priv = priv + torch.randn_like(priv) * torch.exp(0.5 * plv) / ps
        mask = torch.from_numpy(np.asarray(self.fixed_mask, dtype=np.float32)
                                if self.fixed_mask is not None else self._draw_mask())
        # logvar is NOT rescaled: it only ever enters as a precision WEIGHT in the PoE sum,
        # where a common factor cancels between numerator and denominator.
        out = {"expert_mu": emu, "expert_logvar": elv, "priv": priv,
               "mask": mask, "case": str(d["case"])}
        if self.with_image:
            it = self._imap.get(str(d["case"]))
            if it is not None:
                try:
                    out["image"] = torch.as_tensor(self._tf(it)["target"]).float()
                except Exception:
                    pass
        return out
