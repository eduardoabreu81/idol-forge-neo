"""The packed latent of a generation: the object [B, 8192, 16] and the input camera [B, 1, 5] in one tensor.

Forge's sampler works on one tensor, in the [B, 1, 1, N] form the MiniMax H3 extension uses: it reads the latent's
dims 2 and 3 as an (H, W) area, so it must be 4D. The conditioning travels the same way: the DINOv3 tokens
[B, 4101, 1280] and the FLUX.2 VAE tokens [B, 4101, 128] are one [B, 4101, 1408] context.
"""

import torch

from ..contracts import CAMERA_CHANNELS, LATENT_CHANNELS, Q_TOKENS

LATENT_SIZE = Q_TOKENS * LATENT_CHANNELS
PACKED_SIZE = LATENT_SIZE + CAMERA_CHANNELS
PACKED_SHAPE = (1, 1, PACKED_SIZE)
DINOV3_CHANNELS = 1280
FLUX2_CHANNELS = 128
CONTEXT_TOKENS = 4101

# Forge's RGB live preview needs a picture: the latent tokens laid out as a 64 x 128 map of 16 channels
PREVIEW_HEIGHT = 64
PREVIEW_WIDTH = 128


def pack(latent: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    flat = torch.cat([latent.reshape(latent.shape[0], -1), camera.reshape(camera.shape[0], -1)], dim=-1)
    return flat.reshape(flat.shape[0], 1, 1, -1)


def unpack(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = packed.shape[0]
    flat = packed.reshape(b, -1)
    return flat[:, :LATENT_SIZE].reshape(b, Q_TOKENS, LATENT_CHANNELS), flat[:, LATENT_SIZE:].reshape(b, 1, CAMERA_CHANNELS)


def pack_context(feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
    return torch.cat([feature1, feature2], dim=-1)


def unpack_context(context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return context[..., :DINOV3_CHANNELS], context[..., DINOV3_CHANNELS:]


def preview_map(packed: torch.Tensor) -> torch.Tensor:
    """The latent tokens as [1, 16, 64, 128]; anything that is not a packed latent comes back unchanged."""
    if packed.shape[-1] != PACKED_SIZE:
        return packed
    latent, _ = unpack(packed[:1])
    return latent.movedim(-1, 1).reshape(1, LATENT_CHANNELS, PREVIEW_HEIGHT, PREVIEW_WIDTH)
