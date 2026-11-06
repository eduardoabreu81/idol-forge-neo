# Adapted from https://github.com/speridlabs/idol-sdk, which adapted
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py
# Copyright 2025 The Wan Team and The HuggingFace Team. Licensed under the Apache License, Version 2.0.
"""The denoiser: a rectified-flow transformer over the object latent (8192 tokens) and the input camera (1 token),
conditioned on the image tokens. Joint attention over [latent, image, camera] with learned rotary positions.

Forge calls forward with the packed latent and context (streams.py) and treats the model as any DiT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import streams
from .layers import (
    MLP,
    PcdAbsolutePositionEmbedder,
    RePo3DRotaryEmbedding,
    SinusoidalEmbedder,
    UnifiedTransformerBlock,
)

BLOCK_DTYPE = torch.float16


def sobol_positions(length: int) -> torch.Tensor:
    """The latent tokens' positions: a fixed scrambled Sobol sequence, stored with the SDK's weights and left out of
    the ComfyUI repack, whose converter checks that they are this sequence."""
    return torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(length)[None].float()


class IdolDenoiser(nn.Module):
    def __init__(self, q_token_length: int = 8192, in_channels: int = 16, cam_channels: int = 5,
                 model_channels: int = 1024, cond_channels: int = 1280, cond2_channels: int = 128,
                 num_blocks: int = 24, num_refiner_blocks: int = 2, num_heads: int = 16, num_head_channels: int = 64,
                 mlp_ratio: float = 4, **kwargs):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.block_dtype = BLOCK_DTYPE
        # a plain attribute, not a buffer: it is computed, and Forge loads nothing into it
        self.positions = sobol_positions(q_token_length)
        # turbo.TurboLoras while Turbo is on
        self.turbo = None

        def blocks(count, modulation=True):
            return nn.ModuleList([UnifiedTransformerBlock(model_channels, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                          modulation=modulation) for _ in range(count)])

        def rope_layers(count):
            return nn.ModuleList([RePo3DRotaryEmbedding(model_channels, num_heads=num_heads,
                                                        head_dim=num_head_channels) for _ in range(count)])

        self.t_embedder = SinusoidalEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = nn.Linear(cond2_channels, model_channels)

        self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)
        self.noise_repo_layers = rope_layers(num_refiner_blocks)
        self.context_repo_layers = rope_layers(num_refiner_blocks)
        self.repo_layers = rope_layers(num_blocks)

        # each stream is refined on its own, then all tokens go through the joint blocks
        self.noise_refiner = blocks(num_refiner_blocks)
        self.context_refiner = blocks(num_refiner_blocks, modulation=False)
        self.cam_refiner = MLP(cam_channels, model_channels, model_channels, mlp_layer_num=num_refiner_blocks)
        self.blocks = blocks(num_blocks)

        self.shift_table = nn.Parameter(torch.zeros(1, 2, model_channels))
        self.out_layer = nn.Linear(model_channels, in_channels)
        self.cam_out_layer = nn.Linear(model_channels, cam_channels)

    def forward(self, x, timestep, context=None, control=None, transformer_options={}, **kwargs):
        """Forge's call: x is the packed latent [B, 1, 1, N], timestep [B] in [0, 1000], context [B, L, 1408]."""
        z, cam = streams.unpack(x.float())
        feature1, feature2 = streams.unpack_context(context.float())
        if self.turbo is not None:
            self.turbo.select(timestep)
        latent, camera = self.forward_streams(z, cam, timestep, feature1, feature2)
        return streams.pack(latent, camera)

    def forward_streams(self, z, cam, t, feature1, feature2):
        """z [B, 8192, 16], cam [B, 1, 5], t [B] in [0, 1000], feature1 [B, L, 1280] (DINOv3), feature2 [B, L, 128]
        (FLUX.2 VAE) -> the velocity of z and of cam, in the SDK's mix of fp16 blocks and fp32 ends."""
        h_x = self.input_layer(z).type(self.block_dtype)
        h_cond = self.cond_embedder(feature1) + self.cond_embedder2(feature2)
        h_cond = h_cond.type(self.block_dtype)

        t_emb = self.t_embedder(t.float())
        t_mod = self.adaLN_modulation(t_emb).type(self.block_dtype)

        h_x = h_x + self.pos_embedder(self.positions.to(z.device)).type(self.block_dtype)
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = block(h_x, mod=t_mod, rotary_emb=repo(h_x))
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h_cond = block(h_cond, mod=None, rotary_emb=repo(h_cond))
        h_cam = self.cam_refiner(cam).type(self.block_dtype)

        h = torch.cat([h_x, h_cond, h_cam], dim=1)
        for repo, block in zip(self.repo_layers, self.blocks):
            h = block(h, mod=t_mod, rotary_emb=repo(h))

        h_x = h[:, :z.shape[1]].type(z.dtype)
        h_x = F.layer_norm(h_x, h_x.shape[-1:])
        h_cam = h[:, -cam.shape[1]:].type(cam.dtype)
        h_cam = F.layer_norm(h_cam, h_cam.shape[-1:])
        shift, scale = (self.shift_table.to(t_emb.device, t_emb.dtype) + t_emb.unsqueeze(1)).chunk(2, dim=1)
        h_x = h_x * (1 + scale) + shift
        h_cam = h_cam * (1 + scale) + shift
        return self.out_layer(h_x), self.cam_out_layer(h_cam)
