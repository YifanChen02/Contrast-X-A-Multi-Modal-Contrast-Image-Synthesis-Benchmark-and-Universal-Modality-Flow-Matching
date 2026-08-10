"""PoE-AE — a shared tissue posterior aggregated over whatever modalities are present,
plus a private code per modality.

The design question was: what latent makes all 2^M - 1 missing patterns meaningful AND
related, without a mask branch and without asking the flow model to invent everything?

The answer is to stop treating the latent as a point and treat it as a POSTERIOR over one
shared variable. Assume every sequence is an observation of the same tissue state a:

    q(a | x_S) proportional to  p(a) * prod_{m in S} q_m(a | x_m)

Gaussian experts give a closed form -- precision-weighted addition:

    Lambda_S = Lambda_0 + sum_{m in S} Lambda_m
    mu_S     = Lambda_S^-1 ( Lambda_0 mu_0 + sum_{m in S} Lambda_m mu_m )

Everything the design needed falls out of that formula rather than being engineered:

  * one model for every subset -- the sum IS the rule, no mask embedding, no branch
  * all subsets are related, because they are posteriors over the SAME a
  * order-invariant: the aggregation is a SUM, so the order the experts are visited in
    cannot matter. (It is deliberately NOT swap-invariant across modality identities --
    t1 and t2 are different observations and must stay distinguishable.)
  * M is limited by the modality VOCABULARY, not by this dataset's channel count, so one
    model spans CT (M=1) and a DCE series (M=6) by giving each sequence a global id
  * uncertainty is automatically right: fewer modalities -> smaller Lambda_S -> wider
    posterior. That is a mathematical consequence, not a design choice
  * compact: a single latent, not one per modality

WHY THE PRIVATE CODE IS NOT OPTIONAL. A shared-only PoE (plain MVAE) has to push every
modality-specific detail through a, and the measurements here say the modalities share
only about a third of their content, so it cannot fit and every modality comes out blurry.
That is the usual reason PoE disappoints on cross-modal generation. Each modality
therefore also gets a private r_m: used directly when m is observed (which is where
reconstruction PSNR actually comes from), and drawn from the prior when it is not -- and
generating that r_m, conditioned on a, is precisely the job left for a flow model.

IDENTIFIABILITY. Nothing above stops the model from dumping everything into r_m and
leaving a useless, which is exactly the "it only learned to handle missingness" failure.
The fix is already in the training loop: modalities must go MISSING during training,
because when m is absent r_m is unavailable and only a can produce x_m. The masking
curriculum is not a training trick here, it is the identifiability mechanism for this
factorisation -- which is also why removing it in favour of "pure reconstruction" would
break the model rather than simplify it.
"""
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PoECfg:
    in_mod: int = 4              # modalities in THIS dataset
    vocab: int = 32              # size of the global modality vocabulary. The embedding is
                                 # indexed by a modality ID, not by position, so one model can
                                 # span datasets: t1/t1ce/t2/flair/CT/DCE-t1..tN each get an id
                                 # and M is limited by the vocabulary, not by in_mod.
    base: int = 32
    ch_mult: Sequence[int] = field(default_factory=lambda: (1, 2, 4))   # factor-4
    n_res: int = 2
    dec_n_res: int = 2
    shared_ch: int = 4           # a : the tissue posterior
    private_ch: int = 2          # r_m: 0 disables, giving a plain shared-only PoE
    norm: str = "group"
    groups: int = 8
    logvar_clamp: float = 6.0    # bounds the expert precision; an overconfident expert
                                 # otherwise dominates the product, the classic PoE failure

    @property
    def channels(self):
        return [self.base * m for m in self.ch_mult]

    @property
    def n_levels(self):
        return len(self.ch_mult) - 1


def _norm(name, c, groups):
    return nn.GroupNorm(min(groups, c), c) if name == "group" else nn.Identity()


class ResBlock(nn.Module):
    def __init__(self, c, cfg):
        super().__init__()
        self.n1, self.n2 = _norm(cfg.norm, c, cfg.groups), _norm(cfg.norm, c, cfg.groups)
        self.c1 = nn.Conv3d(c, c, 3, 1, 1)
        self.c2 = nn.Conv3d(c, c, 3, 1, 1)
        self.a = nn.SiLU()

    def forward(self, x):
        h = self.c1(self.a(self.n1(x)))
        h = self.c2(self.a(self.n2(h)))
        return x + h


class PoEAE(nn.Module):
    def __init__(self, cfg: PoECfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.channels

        # ---- one expert, shared weights, 1 channel in: this is what makes M free ----
        self.stem = nn.Conv3d(1, C[0], 3, 1, 1)
        self.enc_res, self.down = nn.ModuleList(), nn.ModuleList()
        for l in range(cfg.n_levels):
            self.enc_res.append(nn.Sequential(*[ResBlock(C[l], cfg) for _ in range(cfg.n_res)]))
            self.down.append(nn.Conv3d(C[l], C[l + 1], 3, 2, 1))
        self.enc_mid = nn.Sequential(*[ResBlock(C[-1], cfg) for _ in range(cfg.n_res)])
        # modality identity enters the ENCODER, so one shared expert can still specialise
        self.mod_emb_enc = nn.Parameter(torch.randn(cfg.vocab, C[-1]) * 0.02)

        self.to_mu = nn.Conv3d(C[-1], cfg.shared_ch, 1)
        self.to_logvar = nn.Conv3d(C[-1], cfg.shared_ch, 1)
        if cfg.private_ch:
            self.to_priv_mu = nn.Conv3d(C[-1], cfg.private_ch, 1)
            self.to_priv_logvar = nn.Conv3d(C[-1], cfg.private_ch, 1)

        # ---- decoder: (a, r_m, modality id) -> x_m ----
        zin = cfg.shared_ch + cfg.private_ch
        self.from_z = nn.Conv3d(zin, C[-1], 1)
        self.mod_emb_dec = nn.Parameter(torch.randn(cfg.vocab, C[-1]) * 0.02)
        self.dec_mid = nn.Sequential(*[ResBlock(C[-1], cfg) for _ in range(cfg.dec_n_res)])
        self.up, self.dec_res = nn.ModuleList(), nn.ModuleList()
        for l in reversed(range(cfg.n_levels)):
            self.up.append(nn.Conv3d(C[l + 1], C[l], 3, 1, 1))
            self.dec_res.append(nn.Sequential(*[ResBlock(C[l], cfg) for _ in range(cfg.dec_n_res)]))
        self.out_norm = _norm(cfg.norm, C[0], cfg.groups)
        self.out_act = nn.SiLU()
        self.out_head = nn.Conv3d(C[0], 1, 3, 1, 1)

    # ------------------------------------------------------------------ #
    def encode_expert(self, x, m_idx):
        """x (N,1,...) -> this modality's opinion about a, plus its private code."""
        h = self.stem(x)
        for l in range(self.cfg.n_levels):
            h = self.enc_res[l](h)
            h = self.down[l](h)
        h = self.enc_mid(h) + self.mod_emb_enc[m_idx].view(1, -1, 1, 1, 1)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-self.cfg.logvar_clamp, self.cfg.logvar_clamp)
        if self.cfg.private_ch:
            pmu = self.to_priv_mu(h)
            plv = self.to_priv_logvar(h).clamp(-self.cfg.logvar_clamp, self.cfg.logvar_clamp)
        else:
            pmu = plv = None
        return mu, logvar, pmu, plv

    def poe(self, mus, logvars, mask):
        """Precision-weighted product of the PRESENT experts, plus the N(0,1) prior.

        mus/logvars: (B, M, C, ...)   mask: (B, M) with 1 = observed.
        The prior term keeps Lambda_S non-singular when a modality is alone, and makes
        the empty set fall back to the prior instead of dividing by zero."""
        w = mask.view(*mask.shape, 1, 1, 1, 1)
        prec = torch.exp(-logvars) * w                      # absent experts contribute 0
        prec_sum = prec.sum(1) + 1.0                        # +1 = prior precision
        mu_sum = (mus * prec).sum(1)                        # prior mean is 0
        mu_S = mu_sum / prec_sum
        logvar_S = -torch.log(prec_sum)
        return mu_S, logvar_S

    @staticmethod
    def _rsample(mu, logvar, sample=True):
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_from(self, a, r, m_idx):
        """Decode ONE modality from an explicit (shared, private) pair.

        Kept separate from forward() so a downstream completer can supply its own predicted
        private code without re-running the encoder."""
        z = torch.cat([a, r], 1) if self.cfg.private_ch else a
        h = self.from_z(z) + self.mod_emb_dec[m_idx].view(1, -1, 1, 1, 1)
        h = self.dec_mid(h)
        for j in range(self.cfg.n_levels):
            h = F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False)
            h = self.up[j](h)
            h = self.dec_res[j](h)
        return self.out_head(self.out_act(self.out_norm(h)))

    def forward(self, image, mask=None, sample=True, mod_ids=None, priv_mask=None):
        """image (B,M,H,W,D); mask (B,M) 1=observed. Reconstructs ALL M modalities.

        mod_ids gives each of the M inputs its GLOBAL modality id; defaults to 0..M-1, which
        is the single-dataset case. Passing explicit ids is what lets one model take CT
        (M=1) or a DCE series (M=6) with the same weights."""
        B, M = image.shape[:2]
        if mask is None:
            mask = image.new_ones(B, M)
        if mod_ids is None:
            mod_ids = list(range(M))
        # priv_mask lets the private route be withheld independently of the input
        # mask, so the shared code can be forced to carry an OBSERVED modality too
        if priv_mask is None:
            priv_mask = mask
        mus, lvs, pmus, plvs = [], [], [], []
        for m in range(M):
            mu, lv, pmu, plv = self.encode_expert(image[:, m:m + 1], mod_ids[m])
            mus.append(mu); lvs.append(lv)
            if self.cfg.private_ch:
                pmus.append(pmu); plvs.append(plv)
        mus = torch.stack(mus, 1); lvs = torch.stack(lvs, 1)

        mu_S, logvar_S = self.poe(mus, lvs, mask)           # the shared posterior
        a = self._rsample(mu_S, logvar_S, sample and self.training)

        outs = []
        for m in range(M):
            if self.cfg.private_ch:
                # observed -> use this modality's own private code;
                # absent   -> draw it from the prior, which is exactly the slot an FM fills
                pm = self._rsample(pmus[m], plvs[m], sample and self.training)
                obs = priv_mask[:, m].view(B, 1, 1, 1, 1)
                r = obs * pm + (1 - obs) * torch.randn_like(pm)
                z = torch.cat([a, r], 1)
            else:
                z = a
            h = self.from_z(z) + self.mod_emb_dec[mod_ids[m]].view(1, -1, 1, 1, 1)
            h = self.dec_mid(h)
            for j in range(self.cfg.n_levels):
                h = F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False)
                h = self.up[j](h)
                h = self.dec_res[j](h)
            outs.append(self.out_head(self.out_act(self.out_norm(h))))
        img = torch.cat(outs, 1)

        return {"img": img, "a_mu": mu_S, "a_logvar": logvar_S,
                "expert_mu": mus, "expert_logvar": lvs,
                "priv_mu": (torch.stack(pmus, 1) if self.cfg.private_ch else None),
                "priv_logvar": (torch.stack(plvs, 1) if self.cfg.private_ch else None)}


def poe_cfg(shared_ch: int = 4, private_ch: int = 2) -> PoECfg:
    return PoECfg(shared_ch=shared_ch, private_ch=private_ch)
