"""BraTS autoencoder: MAISI conv backbone + cross-modal attention.

Encodes any subset of the 4 modalities into one shared latent and decodes all 4
modalities (completion) + 4 segmentation regions (WT, TC, ET, BG).
"""

from .config import AEConfig, get_config, PRESETS, MAISI_DEBUG, MAISI_SMALL, MAISI_LARGE
from .model import BratsAutoencoder, reparameterize
from .losses import ae_loss, dice_bce_loss, kl_loss, dice_score

__all__ = [
    "AEConfig",
    "get_config",
    "PRESETS",
    "MAISI_DEBUG",
    "MAISI_SMALL",
    "MAISI_LARGE",
    "BratsAutoencoder",
    "reparameterize",
    "ae_loss",
    "dice_bce_loss",
    "kl_loss",
    "dice_score",
]
