"""Idol registration for huggingface_guess."""

import os

import torch
from huggingface_guess.latent import LatentFormat
from huggingface_guess.model_list import BASE, ModelType

from ..contracts import LATENT_CHANNELS, SHIFT
from . import detect as detection
from .streams import preview_map

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "Idol")


class IdolLatent(LatentFormat):
    # the object latent is not a picture: the RGB live preview only shows it settling, in grey
    def __init__(self):
        self.latent_channels = LATENT_CHANNELS
        self.scale_factor = 1.0
        self.latent_rgb_factors = [[0.25, 0.25, 0.25]] * LATENT_CHANNELS
        self.latent_rgb_factors_bias = [0.0, 0.0, 0.0]

    def latent_rgb_factors_reshape(self, sample):
        return preview_map(sample)


class Idol(BASE):
    # absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part
    huggingface_repo = CONFIG_DIR

    unet_config = {"image_model": "idol"}

    # discrete flow, timestep = sigma * 1000 as the denoiser expects
    sampling_settings = {"shift": SHIFT, "multiplier": 1000}

    # Forge's estimate for [B, 1, 1, N] in fp32: about 1.8 GB per batch item for the 12,294 tokens
    memory_usage_factor = 0.17
    supported_inference_dtypes = [torch.float16, torch.float32]

    unet_extra_config = {}
    latent_format = IdolLatent

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        return {"dinov3": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLOW


def detect(state_dict: dict, key_prefix: str) -> dict | None:
    return detection.detect(state_dict, key_prefix)
