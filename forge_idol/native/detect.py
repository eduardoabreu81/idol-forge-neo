"""Recognize an Idol denoiser and read its configuration from tensor shapes. The stock TripoSplat denoiser has the same
layout, so it is recognized too."""

import re

IDENTITY_KEYS = ("cam_out_layer.weight", "cam_refiner.mlp.0.weight", "noise_refiner.0.attn.qkv.weight",
                 "context_refiner.0.norm1.weight")


def count_blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    found = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    return max(found) + 1 if found else 0


def is_idol(state_dict, key_prefix=""):
    return all(f"{key_prefix}{k}" in state_dict for k in IDENTITY_KEYS)


def detect(state_dict, key_prefix=""):
    """The IdolDenoiser keyword arguments for this checkpoint, or None when it is not an Idol denoiser."""
    if not is_idol(state_dict, key_prefix):
        return None

    def shape(key):
        return tuple(state_dict[f"{key_prefix}{key}"].shape)

    keys = list(state_dict.keys())
    return {
        "image_model": "idol",
        "model_channels": shape("input_layer.weight")[0],
        "in_channels": shape("input_layer.weight")[1],
        "cam_channels": shape("cam_out_layer.weight")[0],
        "cond_channels": shape("cond_embedder.weight")[1],
        "cond2_channels": shape("cond_embedder2.weight")[1],
        "num_blocks": count_blocks(keys, f"{key_prefix}blocks."),
        "num_refiner_blocks": count_blocks(keys, f"{key_prefix}noise_refiner."),
        "num_heads": shape("blocks.0.attn.q_norm.gamma")[0],
        "num_head_channels": shape("blocks.0.attn.q_norm.gamma")[1],
    }
