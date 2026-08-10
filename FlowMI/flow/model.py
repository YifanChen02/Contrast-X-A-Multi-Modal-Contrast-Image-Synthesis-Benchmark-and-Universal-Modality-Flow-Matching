"""Conditional 3D flow-matching network over PoE private codes.

WHAT IS BEING FLOWED, and what is not. The shared code is never generated: PoE aggregation
hands it over for free from whatever is observed. What is missing is the private code of each
absent modality, so the flow transports noise to

    p( r_m for m absent | mu_S , { r_m for m present } , S )

All M private codes are carried through the network together, but the OBSERVED ones are pinned
to their true values at every t rather than being noised. They are conditioning, not targets;
the velocity loss is taken on absent channels only. This is the inpainting form of conditional
flow matching, and it is what lets ONE model serve every subset instead of one model per
pattern -- the mask is an input, not a different network.

Conditioning enters three ways:
  * mu_S, recomputed from the experts under the sampled mask, concatenated as channels
  * the mask itself, broadcast to the spatial grid, so the network knows WHICH channels it is
    being asked to invent rather than having to infer it from the noise statistics
  * t, through a sinusoidal embedding that FiLM-modulates every residual block
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float().view(-1, 1) * freqs.view(1, -1)
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class Block(nn.Module):
    """Residual block with FiLM from the time embedding."""

    def __init__(self, cin, cout, tdim, groups=8):
        super().__init__()
        self.n1 = nn.GroupNorm(min(groups, cin), cin)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(tdim, cout * 2)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)          # start as identity: the block adds nothing
        nn.init.zeros_(self.c2.bias)            # until it has learnt to

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(temb)).chunk(2, 1)
        h = self.n2(h) * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        return self.skip(x) + self.c2(F.silu(h))


class FlowUNet(nn.Module):
    def __init__(self, m_mod, priv_ch, shared_ch, base=96, mults=(1, 2, 4), tdim=256,
                 n_res=2, rich_cond=False, fix_shared=False, translate=False,
                 hybrid=False):
        super().__init__()
        self.M, self.P, self.S = m_mod, priv_ch, shared_ch
        self.rich_cond = rich_cond
        # fix_shared adds S output channels for a correction to the SHARED code. Everything so
        # decoded from a_S, the aggregate over observed modalities only, including the
        # `oracle` row -- so a ceiling computed that way is understated. With the TRUE private
        # code throughout, a_full decodes better than a_S by a margin that no completion method
        # here reaches, not even the deterministic control.
        self.fix_shared = fix_shared
        # mu_S is a PRECISION-WEIGHTED SUM of the experts and is not invertible, so handing the
        # network only the aggregate discards the individual expert means. It also never saw
        # any posterior uncertainty, which varies a lot with how many modalities are observed.
        # rich_cond adds both: the per-modality expert means (masked, so absent ones read zero)
        # and the shared posterior's log-variance.
        # TRANSLATE MODE. The flow transports the INPUT latent to the COMPLETE latent instead
        # of transporting noise to the missing private codes:
        #     x_0 = [a_S , priv with the absent entries zeroed]      what is actually available
        #     x_1 = [a_full , priv_true]                             what is wanted
        # Three things fall out that the noise->data formulation had to be told separately.
        # The transport starts from an informative point rather than pure noise, so there is far
        # less distance to cover. The shared-code correction a_full - a_S is part of the
        # transport, so the headroom that was unreachable before is now inside the objective
        # rather than outside it. And the velocity target IS the residual, without needing a
        # separate deterministic model subtracted off first.
        self.translate = translate
        # HYBRID. The leak test settled how these two halves must differ.
        #
        # For an ABSENT modality the source x0 holds zeros, so x_t = t*x1 and x1 = x_t/t is
        # recoverable by algebra: a bridge there leaks its own answer, which is what the
        # leak test showed (the model followed an impostor x1 far more closely than its own). There is no information in x0 to bridge FROM, so noise is the only
        # honest source and standard flow matching is the right tool.
        #
        # The SHARED code is the opposite case: a_S is genuinely informative about a_full, and
        # the map between them carries no ambiguity worth sampling. That is a regression, and
        # giving it a flow only adds 50 integration steps and a leak. So it gets a plain
        # deterministic head predicting a_full - a_S, which is where that headroom lives.
        self.hybrid = hybrid
        nflow = m_mod * priv_ch
        if translate:
            nflow = shared_ch + m_mod * priv_ch
        extra = (m_mod * shared_ch + shared_ch) if rich_cond else 0
        cond = m_mod * priv_ch + shared_ch if translate else 0   # x_0 kept as conditioning
        # fix_shared rides the shared-code delta along as S EXTRA channels of x_t, so the input
        # width has to grow with it. Adding them to the output alone -- which is what the first
        # version did -- leaves the input conv expecting fewer channels than it is handed, and
        # the failure surfaces only when that mode is actually run.
        cin = nflow + shared_ch + m_mod + extra + cond
        self.nflow = nflow
        cout = nflow if translate else (m_mod * priv_ch
                                        + (shared_ch if (fix_shared or hybrid) else 0))
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.tdim = tdim
        self.inp = nn.Conv3d(cin, base, 3, padding=1)

        chs = [base * m for m in mults]
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        prev = base
        for c in chs:
            self.down.append(nn.ModuleList([Block(prev if i == 0 else c, c, tdim)
                                            for i in range(n_res)]))
            self.downsample.append(nn.Conv3d(c, c, 3, stride=2, padding=1))
            prev = c
        self.mid1 = Block(prev, prev, tdim)
        self.mid2 = Block(prev, prev, tdim)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for c in reversed(chs):
            self.upsample.append(nn.ConvTranspose3d(prev, c, 4, stride=2, padding=1))
            self.up.append(nn.ModuleList([Block(c * 2 if i == 0 else c, c, tdim)
                                          for i in range(n_res)]))
            prev = c
        self.outn = nn.GroupNorm(8, prev)
        self.outc = nn.Conv3d(prev, cout, 3, padding=1)
        nn.init.zeros_(self.outc.weight)        # predict zero velocity at init, so the first
        nn.init.zeros_(self.outc.bias)          # steps cannot throw the trajectory anywhere

    def forward(self, x_t, mu_s, mask, t, expert_mu=None, logvar_s=None, x0=None):
        B = x_t.shape[0]
        mk = mask.view(B, self.M, 1, 1, 1).expand(B, self.M, *x_t.shape[2:])
        parts = [x_t, mu_s, mk]
        if self.translate:
            # the starting point stays available as conditioning at every t, so the network
            # never has to reconstruct it from x_t alone
            parts.append(x0 if x0 is not None else torch.zeros_like(x_t))
        if self.rich_cond:
            if expert_mu is None:
                expert_mu = x_t.new_zeros(B, self.M, self.S, *x_t.shape[2:])
            if logvar_s is None:
                logvar_s = x_t.new_zeros(B, self.S, *x_t.shape[2:])
            em = expert_mu * mask.view(B, self.M, 1, 1, 1, 1)
            parts += [em.reshape(B, self.M * self.S, *x_t.shape[2:]), logvar_s]
        h = self.inp(torch.cat(parts, 1))
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        skips = []
        for blocks, ds in zip(self.down, self.downsample):
            for b in blocks:
                h = b(h, temb)
            skips.append(h)
            h = ds(h)
        h = self.mid2(self.mid1(h, temb), temb)
        for us, blocks, sk in zip(self.upsample, self.up, reversed(skips)):
            h = us(h)
            h = torch.cat([h, sk], 1)
            for b in blocks:
                h = b(h, temb)
        return self.outc(F.silu(self.outn(h)))


def flow_batch(priv, mask, pred_x=False, t_dist="uniform", t_mean=0.0,
               t_std=1.0):
    """Build one conditional-flow training pair.

    x_t is the straight-line interpolation from noise to the true private codes, but only on
    ABSENT channels. Observed channels are handed over clean at every t: they are context the
    model may read, never something it must invent. The target velocity for a linear path is
    simply r - noise, and the loss weight below restricts it to the absent channels.
    """
    B, M, P = priv.shape[:3]
    sp = priv.shape[3:]
    if t_dist == "logitnormal":
        t = torch.sigmoid(t_mean + t_std * torch.randn(B, device=priv.device))
    else:
        t = torch.rand(B, device=priv.device)
    noise = torch.randn(priv.shape, device=priv.device)
    tt = t.view(B, 1, 1, *([1] * len(sp)))
    x_t = (1 - tt) * noise + tt * priv
    obs = mask.view(B, M, 1, *([1] * len(sp)))
    x_t = obs * priv + (1 - obs) * x_t                     # observed stay clean
    # v-prediction regresses priv - noise, which contains the noise draw itself; x-prediction
    # regresses priv, which does not. On a conditional this sharp the difference is the whole
    # ball game.
    target = priv if pred_x else (priv - noise)
    w = (1 - obs).expand_as(priv)                           # loss on absent channels only
    return (x_t.reshape(B, M * P, *sp), t,
            target.reshape(B, M * P, *sp), w.reshape(B, M * P, *sp))


@torch.no_grad()
def sample(net, mu_s, mask, priv_obs, steps=50, expert_mu=None, logvar_s=None,
           cfg=1.0, pred_x=False):
    """Euler integration from t=0 to t=1, observed channels re-pinned after every step so the
    conditioning cannot drift as the trajectory advances.

    `cfg` is classifier-free guidance and it is the knob this model previously lacked: at 1.0
    the behaviour is unchanged, above 1.0 the velocity is extrapolated away from the
    unconditional branch, trading diversity for fidelity to the observation; below 1.0 it moves
    the other way. Since the measured problem here is collapse onto the conditional mean
    (diversity ratio 0.037), the interesting direction is BELOW 1, and having the dial at all
    means the trade can be chosen at inference instead of being baked in.
    """
    B, M, P = priv_obs.shape[:3]
    sp = priv_obs.shape[3:]
    obs = mask.view(B, M, 1, *([1] * len(sp)))
    x = torch.randn(priv_obs.shape, device=priv_obs.device)
    x = obs * priv_obs + (1 - obs) * x
    dt = 1.0 / steps
    npriv = M * P
    zmu = torch.zeros_like(mu_s)
    for i in range(steps):
        t = torch.full((B,), i * dt, device=x.device)
        xf = x.reshape(B, npriv, *sp)
        out = net(xf, mu_s, mask, t, expert_mu, logvar_s)
        # a hybrid net emits the shared-code correction as trailing channels; only the private
        # ones are transported by the ODE
        if pred_x:
            # x-PREDICTION as a residual from x_t: the head emits x1_hat - x_t, so its zero
            # init is the identity rather than "x1 = 0". The velocity is then out/(1-t), which
            # is the same ODE, only reached from a starting point the model has actually seen.
            x1h = xf.reshape(B, M, P, *sp) + out[:, :npriv].reshape(B, M, P, *sp)
            v = (x1h - x) / max(1.0 - float(t[0]), 1e-3)
        else:
            v = out[:, :npriv].reshape(B, M, P, *sp)
        if cfg != 1.0:
            ou = net(xf, zmu, mask, t,
                     None if expert_mu is None else torch.zeros_like(expert_mu),
                     None if logvar_s is None else torch.zeros_like(logvar_s)
                     )[:, :npriv].reshape(B, M, P, *sp)
            vu = ((ou - x) / max(1.0 - float(t[0]), 1e-3)) if pred_x else ou
            v = vu + cfg * (v - vu)
        x = x + dt * v
        x = obs * priv_obs + (1 - obs) * x
    if out.shape[1] > npriv:
        return x, out[:, npriv:]        # (private codes, shared-code correction)
    return x


@torch.no_grad()
def translate_sample(net, x0, mu_s, mask, steps=50, expert_mu=None, logvar_s=None,
                     noise=0.0, cfg=1.0, pred_x=False):
    """Integrate from the INPUT latent to the complete one.

    `noise` perturbs the starting point. At 0 the map is deterministic -- same input, same
    output -- so any diversity has to be injected here; this is the knob the user's "one channel
    condition, one channel noise" framing asks for, and it is separate from cfg.
    """
    x = x0 + noise * torch.randn_like(x0) if noise > 0 else x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        out = net(x, mu_s, mask, t, expert_mu, logvar_s, x0)
        # x-prediction emits x1 - x_t, so the velocity along the straight path is out/(1-t)
        v = out / max(1.0 - i * dt, 1e-3) if pred_x else out
        if cfg != 1.0:
            ou = net(x, torch.zeros_like(mu_s), mask, t, None, None, x0)
            vu = ou / max(1.0 - i * dt, 1e-3) if pred_x else ou
            v = vu + cfg * (v - vu)
        x = x + dt * v
    return x
