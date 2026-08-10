import os
"""MAISI's autoencoder as the slot encoder, with its pretrained weights.

The architecture is already what the slot design asks for: one channel in, one channel out, a
4-channel latent at factor-4. Folding the modality axis into the batch means every modality is
encoded by the same weights, one at a time, so a 4-modality volume becomes a 4-slot latent field
and nothing about the model depends on how many modalities there are.

    image (B, M, H, W, D)  ->  latent (B, M, 4, H/4, W/4, D/4)  ->  image (B, M, H, W, D)

Weights come from autoencoder_MAISI_Brain.pt. The other file in that directory,
autoencoder_MAISI_MR.pt, holds a `unet_state_dict` -- it is a diffusion UNet, not an
autoencoder, and cannot be used here.

The config below is not guessed: it is read off the checkpoint's own tensor shapes
(64 -> 128 -> 256 -> 4, two downsamples), so the load is exact rather than approximate. The
loaded fraction is asserted, because a rename or a shape drift that silently leaves most of the
network at its random init is indistinguishable from a successful warm start in the logs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

MAISI_BRAIN = os.environ.get("FLOWMI_MAISI_CKPT", "weights/autoencoder_MAISI_Brain.pt")


def build_maisi_ae(latent_channels: int = 4):
    """Exactly the shape the checkpoint was trained with."""
    return AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        num_channels=(64, 128, 256),
        num_res_blocks=(2, 2, 2),
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=False,
        num_splits=1,
        dim_split=1,
    )


class MaisiSlotAE(nn.Module):
    """Per-modality MAISI autoencoder; the modality axis lives in the batch."""

    def __init__(self, latent_channels: int = 4, ckpt: str | None = MAISI_BRAIN,
                 min_load: float = 0.95):
        super().__init__()
        self.z_ch = latent_channels
        self.net = build_maisi_ae(latent_channels)
        self.loaded = 0.0
        if ckpt:
            self.loaded = self.load_pretrained(ckpt, min_load)

    def load_pretrained(self, path: str, min_load: float = 0.95) -> float:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        cur = self.net.state_dict()
        ok = {k: v for k, v in sd.items()
              if k in cur and tuple(v.shape) == tuple(cur[k].shape)}
        frac = len(ok) / max(len(cur), 1)
        self.net.load_state_dict(ok, strict=False)
        if frac < min_load:
            missing = sorted(set(cur) - set(ok))
            raise RuntimeError(
                f"MAISI warm start transferred only {frac:.0%} ({len(ok)}/{len(cur)}); "
                f"the config does not match the checkpoint. Missing e.g. {missing[:5]}")
        return frac

    # ---------------------------------------------------------------- slots
    def encode(self, image):
        """(B, M, H, W, D) -> mu, logvar each (B, M, z_ch, H/4, W/4, D/4)."""
        B, M = image.shape[:2]
        x = image.reshape(B * M, 1, *image.shape[2:])
        mu, sigma = self.net.encode(x)
        z = mu.reshape(B, M, *mu.shape[1:])
        lv = (2.0 * torch.log(sigma.clamp(min=1e-6))).reshape(B, M, *mu.shape[1:])
        return z, lv

    def decode(self, z):
        """(B, M, z_ch, h, w, d) -> (B, M, H, W, D)."""
        B, M = z.shape[:2]
        x = self.net.decode(z.reshape(B * M, *z.shape[2:]))
        return x.reshape(B, M, *x.shape[2:])

    def forward(self, image, sample=False, z_noise: float = 0.0):
        mu, logvar = self.encode(image)
        z = mu
        if sample and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        if z_noise > 0 and self.training:
            # the decoder needs a basin around real latents: at inference the flow's samples do
            # not land exactly on the autoencoder's manifold
            z = z + torch.randn_like(z) * z_noise
        return {"img": self.decode(z), "mu": mu, "logvar": logvar, "z": z}
