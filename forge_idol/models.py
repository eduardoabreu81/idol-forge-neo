"""Read only safetensors headers; never deserialize a checkpoint to discover it."""

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .contracts import IdolError

MAX_HEADER_BYTES = 32 * 1024 * 1024

DENOISER_KEYS = ("cam_out_layer.weight", "noise_refiner.0.attn.qkv.weight")
DECODERS_KEYS = ("octree.in_proj.weight", "gs.in_proj.weight")
OCTREE_KEYS = ("l_embedder.mlp.0.weight", "out_proj.weight", "blocks.0.cross_attn.to_q.weight")
GS_KEYS = ("points_offset_perturbation", "out_proj.weight")
FLUX2_KEYS = ("bn.running_mean", "encoder.conv_in.weight")
DINOV3_KEYS = ("embeddings.cls_token", "layer.0.attention.q_proj.weight")


def role_of(keys) -> str | None:
    """Which part of Idol a file holds, from its tensor names."""
    if all(k in keys for k in DENOISER_KEYS):
        return "dit"
    if all(k in keys for k in DECODERS_KEYS):
        return "decoders"
    if all(k in keys for k in GS_KEYS):
        return "gs"
    if all(k in keys for k in OCTREE_KEYS):
        return "octree"
    if all(k in keys for k in FLUX2_KEYS):
        return "flux2"
    if all(k in keys for k in DINOV3_KEYS):
        return "dinov3"
    return None


@lru_cache(maxsize=64)
def _read_keys(path, size, mtime):
    try:
        with open(path, "rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise IdolError("Invalid safetensors header: missing length.")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
                raise IdolError("Invalid or oversized safetensors header.")
            header = json.loads(stream.read(length))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IdolError(f"Cannot read safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise IdolError("Invalid safetensors header object.")
    header.pop("__metadata__", None)
    return frozenset(header)


def inspect_model(path) -> str | None:
    """The role of a .safetensors file, None for anything else (other models, other formats)."""
    path = Path(path)
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        return None
    stat = path.stat()
    return role_of(_read_keys(str(path), stat.st_size, stat.st_mtime_ns))


@dataclass(frozen=True)
class Components:
    dit: str
    dinov3: str
    flux2: str
    decoders: tuple


def resolve_components(dit_path, module_paths) -> Components:
    """The files of one generation; says what is missing before Forge starts loading anything."""
    found = {}
    for path in module_paths:
        role = inspect_model(path)
        if role is not None:
            found.setdefault(role, []).append(str(path))
    missing = []
    if "dinov3" not in found:
        missing.append("the DINOv3 image encoder (dino_v3_vit_h_bf16)")
    if "flux2" not in found:
        missing.append("the FLUX.2 VAE (flux2-vae)")
    if "decoders" not in found and not ("octree" in found and "gs" in found):
        missing.append("the Idol decoder (idol_decoder_fp16)")
    if missing:
        raise IdolError(f"Idol needs {' and '.join(missing)}: select them under VAE / Text Encoder.")
    decoders = found.get("decoders") or (found["octree"] + found["gs"])
    return Components(str(dit_path), found["dinov3"][0], found["flux2"][0], tuple(decoders))
