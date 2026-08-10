"""SlotAE — the slot-latent 3D VAE from Claude_BrainWorldModel_AE_Design.md (2026-07-21).

The architectural law: **the encoder is strictly unimodal.** Each modality is
encoded on its own by *shared* weights, so the latent field is a stack of
co-registered, homogeneous slots

    Z = [z_t1, z_t1ce, z_t2, z_flair],   z_m in R^{z_ch x 32 x 32 x 32}

and slot m depends on modality m ALONE. Nothing here fuses, masks, or completes:
completion / translation is latent inpainting on this field, and it belongs to
the downstream flow-matching model. That keeps the present slots invariant to
whatever else is missing, which is what makes them a stable FM condition.

Consequences that fall out of the law, and why they matter:

  * pure conv, no content-moving spatial attention -> the encoder stays
    spatially equivariant, so z_t1[:,i,j,k] and z_t1ce[:,i,j,k] describe the
    SAME physical location (BraTS is co-registered). That voxel-to-voxel
    correspondence is what turns the FM's job into a field of *local*
    conditional generations rather than global synthesis.
  * shared weights across modalities -> slots share channel semantics, so the
    FM sees one homogeneous field instead of four private "languages".
  * no modality embedding in the AE. Modality identity is the FM's business
    (a modality-position embedding per slot); baking it in here would make the
    slots non-homogeneous for no gain.

So the model below is, deliberately, just a clean single-channel 3D VAE. It is
applied to each modality independently and never sees a modality mask.

A fused autoencoder squeezes every modality through one bottleneck and pays for
segmentation, absent-modality completion and mixup on top. The load, rather than
the latent capacity, is what limits it. SlotAE spends its whole capacity on
reconstruction and leaves completion to the flow.
"""
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SlotCfg:
    base: int = 32
    ch_mult: Sequence[int] = field(default_factory=lambda: (1, 2, 4))  # 2 downs -> factor-4
    n_res: int = 4            # encoder res-blocks per level
    dec_n_res: int = 6        # heavier decoder (#1 asymmetric; the one toggle Wave R liked)
    z_ch: int = 4             # channels PER SLOT (MAISI-scale at factor-4)
    vae: bool = True          # light-KL VAE, not deterministic
    norm: str = "group"
    groups: int = 8
    act: str = "silu"
    n_mod: int = 0            # >0 turns on per-block modality conditioning
    emb_dim: int = 128
    z_scale: float = 1.0      # ONE global scalar; see the note on per-channel scaling

    @property
    def channels(self):
        return [self.base * m for m in self.ch_mult]

    @property
    def n_levels(self):
        return len(self.ch_mult) - 1

    @property
    def factor(self):
        return 2 ** self.n_levels


def _act(name):
    return {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[name]()


def _norm(name, c, groups):
    if name == "group":
        return nn.GroupNorm(min(groups, c), c)
    if name == "instance":
        return nn.InstanceNorm3d(c, affine=True)
    return nn.Identity()


class ResBlock(nn.Module):
    """Optionally FiLM-conditioned on the modality, at every block rather than at the entry."""

    def __init__(self, c, cfg: SlotCfg, emb_dim: int = 0):
        super().__init__()
        self.n1, self.n2 = _norm(cfg.norm, c, cfg.groups), _norm(cfg.norm, c, cfg.groups)
        self.c1 = nn.Conv3d(c, c, 3, 1, 1)
        self.c2 = nn.Conv3d(c, c, 3, 1, 1)
        self.a = _act(cfg.act)
        self.emb = nn.Linear(emb_dim, 2 * c) if emb_dim else None
        if self.emb is not None:
            # start as the identity, so adding conditioning cannot destabilise a warm start
            nn.init.zeros_(self.emb.weight)
            nn.init.zeros_(self.emb.bias)

    def forward(self, x, e=None):
        h = self.c1(self.a(self.n1(x)))
        h = self.n2(h)
        if self.emb is not None and e is not None:
            scale, shift = self.emb(e).chunk(2, dim=1)
            h = h * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        h = self.c2(self.a(h))
        return x + h


class ResStack(nn.Module):
    """nn.Sequential cannot pass the conditioning through, so the stack is explicit."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, e=None):
        for b in self.blocks:
            x = b(x, e)
        return x


class SlotAE(nn.Module):
    """Single-channel 3D VAE, applied per modality with shared weights."""

    def __init__(self, cfg: SlotCfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.channels

        # ---- encoder: 1 channel in, pure conv, no attention ----
        self.stem = nn.Conv3d(1, C[0], 3, 1, 1)
        E = cfg.emb_dim if cfg.n_mod > 0 else 0
        self.mod_emb = nn.Embedding(cfg.n_mod, E) if E else None
        self.enc_res = nn.ModuleList()
        self.down = nn.ModuleList()
        for l in range(cfg.n_levels):
            self.enc_res.append(ResStack([ResBlock(C[l], cfg, E) for _ in range(cfg.n_res)]))
            self.down.append(nn.Conv3d(C[l], C[l + 1], 3, 2, 1))
        self.enc_mid = ResStack([ResBlock(C[-1], cfg, E) for _ in range(cfg.n_res)])

        # ---- latent: the ONLY encoder->decoder path ----
        self.to_z = nn.Conv3d(C[-1], cfg.z_ch, 1)
        self.to_logvar = nn.Conv3d(C[-1], cfg.z_ch, 1) if cfg.vae else None
        self.from_z = nn.Conv3d(cfg.z_ch, C[-1], 1)

        # ---- decoder: heavier (dec_n_res), 1 channel out ----
        self.dec_mid = ResStack([ResBlock(C[-1], cfg, E) for _ in range(cfg.dec_n_res)])
        self.dec_res = nn.ModuleList()
        self.up = nn.ModuleList()
        for l in reversed(range(cfg.n_levels)):
            self.up.append(nn.Conv3d(C[l + 1], C[l], 3, 1, 1))
            self.dec_res.append(ResStack([ResBlock(C[l], cfg, E) for _ in range(cfg.dec_n_res)]))
        self.out_norm = _norm(cfg.norm, C[0], cfg.groups)
        self.out_act = _act(cfg.act)
        self.out_head = nn.Conv3d(C[0], 1, 3, 1, 1)

    # ------------------------------------------------------------------ #
    # single-slot primitives: x is (N, 1, H, W, D)
    # ------------------------------------------------------------------ #
    def _emb(self, mod_id, n, device):
        if self.mod_emb is None or mod_id is None:
            return None
        if not torch.is_tensor(mod_id):
            mod_id = torch.full((n,), int(mod_id), dtype=torch.long, device=device)
        return self.mod_emb(mod_id.to(device))

    def encode_one(self, x, mod_id=None):
        e = self._emb(mod_id, x.shape[0], x.device)
        h = self.stem(x)
        for l in range(self.cfg.n_levels):
            h = self.enc_res[l](h, e)
            h = self.down[l](h)
        h = self.enc_mid(h, e)
        mu = self.to_z(h)
        logvar = (self.to_logvar(h).clamp(-30.0, 20.0) if self.to_logvar is not None
                  else torch.zeros_like(mu))
        return mu, logvar

    def decode_one(self, z, mod_id=None):
        e = self._emb(mod_id, z.shape[0], z.device)
        h = self.from_z(z)
        h = self.dec_mid(h, e)
        for j in range(self.cfg.n_levels):
            h = F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False)
            h = self.up[j](h)
            h = self.dec_res[j](h, e)
        return self.out_head(self.out_act(self.out_norm(h)))

    # ------------------------------------------------------------------ #
    # multi-slot: image is (B, M, H, W, D) -> fold M into the batch so the
    # SAME weights see every modality. This is what makes the slots homogeneous.
    # ------------------------------------------------------------------ #
    def _ids(self, B, M, device):
        """Folding M into the batch means the ids must be tiled the same way."""
        if self.mod_emb is None:
            return None
        return torch.arange(M, device=device).repeat(B)

    def encode(self, image):
        B, M = image.shape[:2]
        x = image.reshape(B * M, 1, *image.shape[2:])
        mu, logvar = self.encode_one(x, self._ids(B, M, image.device))
        z_shape = mu.shape[1:]
        return (mu.reshape(B, M, *z_shape), logvar.reshape(B, M, *z_shape))

    def decode(self, z):
        B, M = z.shape[:2]
        x = self.decode_one(z.reshape(B * M, *z.shape[2:]), self._ids(B, M, z.device))
        return x.reshape(B, M, *x.shape[2:])

    def forward(self, image, sample=True, z_noise: float = 0.0):
        """image (B, M, H, W, D) -> dict. No mask argument: by design.

        ``z_noise`` adds extra Gaussian noise to the latent before decoding.
        The point is NOT regularisation of the encoder but robustness of the
        DECODER: at inference the FM's samples will not land exactly on the AE
        manifold, so the decoder needs a basin around real latents rather than a
        knife edge. (Design doc, "light-KL earns its keep three times", #3.)
        """
        mu, logvar = self.encode(image)
        if self.cfg.vae and sample and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        if z_noise > 0 and self.training:
            z = z + torch.randn_like(z) * z_noise
        img = self.decode(z)
        return {"img": img, "mu": mu, "logvar": logvar, "z": z}


def slot_cfg(z_ch: int = 4, vae: bool = True, n_mod: int = 0) -> SlotCfg:
    """The converged spec: factor-4, light-KL, heavy decoder, pure conv."""
    return SlotCfg(base=32, ch_mult=(1, 2, 4), n_res=4, dec_n_res=6,
                   z_ch=z_ch, vae=vae, n_mod=n_mod)
