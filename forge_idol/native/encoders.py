"""The two image encoders, as idol-sdk conditioning.py reads them.

feature1 is DINOv3's token sequence (class, registers, patches) with a final non-affine layer norm. feature2 is the FLUX.2
VAE latent as tokens, behind 5 empty tokens that line it up with DINOv3's class and register tokens.
"""

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# facebook/dinov3-vith16plus-pretrain-lvd1689m
DINOV3_CONFIG = {"hidden_size": 1280, "intermediate_size": 5120, "num_hidden_layers": 32, "num_attention_heads": 20,
                 "hidden_act": "silu", "patch_size": 16, "num_channels": 3, "num_register_tokens": 4,
                 "use_gated_mlp": True, "layer_norm_eps": 1e-5, "layerscale_value": 1.0, "rope_theta": 100.0,
                 "pos_embed_rescale": 2.0, "query_bias": True, "key_bias": False, "value_bias": True, "proj_bias": True,
                 "mlp_bias": True, "attention_dropout": 0.0, "drop_path_rate": 0.0}


def dinov3_model():
    """The architecture without weights; build it inside the operations that should hold them."""
    from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel

    config = DINOv3ViTConfig(**DINOV3_CONFIG)
    config._attn_implementation = "sdpa"
    return DINOv3ViTModel(config)


def dinov3_tokens(model, images: torch.Tensor) -> torch.Tensor:
    """images [B, 3, H, W] in [0, 1] -> [B, 5 + patches, 1280]."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    images = (images.float() - mean) / std
    dtype = next(model.parameters()).dtype
    tokens = model(pixel_values=images.to(dtype)).last_hidden_state.float()
    return F.layer_norm(tokens, tokens.shape[-1:])


def flux2_tokens(latents: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """The VAE's raw latent [B, 32, H, W] -> [B, 5 + (H/2)(W/2), 128]: 2 x 2 patches, normalized by the VAE's batch norm
    statistics, flattened.

    Forge's own AutoencoderKLFlux2.encode does not fit here: it calls F.batch_norm in its default training mode, which
    normalizes by the statistics of the latent itself instead of the stored ones."""
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
    mean = running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + eps)
    tokens = ((latents - mean) / std).float().flatten(2).transpose(1, 2).contiguous()
    padding = torch.zeros(batch, 5, tokens.shape[2], dtype=tokens.dtype, device=tokens.device)
    return torch.cat([padding, tokens], dim=1)
