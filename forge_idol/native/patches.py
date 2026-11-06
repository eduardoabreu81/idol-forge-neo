"""Runtime hooks into Forge Neo's loader and sampler, all held by this extension: no file of Forge changes.

Every hook hands anything that is not Idol to the original Forge Neo function untouched.
"""

import contextlib
import functools
import logging
import os

import torch
import torch.nn as nn
from backend import loader, memory_management
from backend.nn import krea
from backend.operations import using_forge_operations
from backend.state_dict import load_state_dict
from huggingface_guess import detection, model_list
from transformers.modeling_utils import no_init_weights

from ..models import role_of
from . import encoders, model
from .decoder import IdolDecoders
from .denoiser import IdolDenoiser
from .engine import IdolEngine

logger = logging.getLogger("forge_idol")

MISSING_VAE = "Idol needs the FLUX.2 VAE and its own decoder: select both under VAE / Text Encoder"
MISSING_TE = "Idol needs the DINOv3 image encoder: select dino_v3_vit_h_bf16 under VAE / Text Encoder"

_applied = False


@contextlib.contextmanager
def _swapped(module, name: str, value):
    # the loader imports these classes inside the function, so a temporary swap routes one call to our class
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def cast_on_the_fly(module: nn.Module) -> None:
    """Every Forge layer casts its weights to the type of its input: the file can be all fp16 while the blocks run in fp16
    and the embeddings in fp32, as in the SDK. Casting to the type already held costs nothing."""
    for layer in module.modules():
        if hasattr(layer, "parameters_manual_cast"):
            layer.parameters_manual_cast = True


def flux2_keys(asd: dict) -> dict:
    # the same renaming Forge's replace_state_dict gives the standalone FLUX.2 VAE file
    renames = (("decoder.post_quant_conv.", "post_quant_conv."), ("encoder.quant_conv.", "quant_conv."))
    result = {}
    for key, value in asd.items():
        for old, new in renames:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        result[key] = value
    return result


def _replace(sd: dict, prefix: str, asd: dict) -> None:
    for k in [k for k in sd if k.startswith(prefix)]:
        del sd[k]
    for k, v in asd.items():
        sd[prefix + k] = v


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = model.detect(state_dict, key_prefix)
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_replace_state_dict() -> None:
    original = loader.replace_state_dict

    def replace_state_dict(sd: dict, asd: dict, guess, path):
        if not isinstance(guess, model.Idol):
            return original(sd, asd, guess, path)
        role = role_of(asd)
        if role == "dinov3":
            _replace(sd, f"{guess.text_encoder_key_prefix[0]}dinov3.", asd)
        elif role == "flux2":
            _replace(sd, f"{guess.vae_key_prefix[0]}flux2.", flux2_keys(asd))
        elif role == "decoders":
            _replace(sd, f"{guess.vae_key_prefix[0]}decoders.", asd)
        elif role in ("octree", "gs"):
            _replace(sd, f"{guess.vae_key_prefix[0]}decoders.{role}.", asd)
        else:
            return original(sd, asd, guess, path)
        return sd

    loader.replace_state_dict = replace_state_dict


class AutoencoderIdol(nn.Module):
    """The FLUX.2 VAE that encodes the picture and the decoders that make the object, as one Forge component; the state
    dict uses the "flux2." and "decoders." prefixes."""

    def __init__(self, flux2, decoders):
        super().__init__()
        self.flux2 = flux2
        self.decoders = decoders


def _load_vae(state_dict) -> AutoencoderIdol:
    if not isinstance(state_dict, dict):
        raise ValueError(MISSING_VAE)
    flux2 = {k[len("flux2."):]: v for k, v in state_dict.items() if k.startswith("flux2.")}
    decoders = {k[len("decoders."):]: v for k, v in state_dict.items() if k.startswith("decoders.")}
    if len(flux2) <= 16 or len(decoders) <= 16:
        raise ValueError(MISSING_VAE)

    from backend.nn.vae import AutoencoderKLFlux2

    config = AutoencoderKLFlux2.load_config(os.path.join(loader.HF, "black-forest-labs", "FLUX.2-klein-9B", "vae"))
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype(), extra_dtype="vae"):
            flux2_model = AutoencoderKLFlux2.from_config(config)
    load_state_dict(flux2_model, flux2, ignore_start="loss.")

    dtype = next(iter(decoders.values())).dtype
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=dtype, manual_cast_enabled=True):
            decoders_model = IdolDecoders()
    load_state_dict(decoders_model, decoders, log_name="Idol decoder")
    decoders_model.eval().requires_grad_(False)
    return AutoencoderIdol(flux2_model, decoders_model)


def _load_dinov3(state_dict) -> nn.Module:
    if not isinstance(state_dict, dict) or len(state_dict) <= 16:
        raise ValueError(MISSING_TE)
    device = memory_management.get_torch_device()
    dtype = torch.bfloat16 if memory_management.should_use_bf16(device) else torch.float32
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=dtype, manual_cast_enabled=True):
            dinov3 = encoders.dinov3_model()
    load_state_dict(dinov3, state_dict, log_name="DINOv3")
    dinov3.to(dtype)
    return dinov3.eval().requires_grad_(False)


def _hook_components() -> None:
    original = loader.load_huggingface_component

    # wraps leaves __wrapped__ behind, which inspect.getsource follows: the compat checks of the other extensions
    # that read Forge's loader source still reach it through this hook
    @functools.wraps(original)
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.Idol):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "AutoencoderIdol":
            return _load_vae(state_dict)
        if cls_name == "DINOv3ViTModel":
            return _load_dinov3(state_dict)
        if cls_name == "IdolDenoiser":
            # the SDK's weights store the positions the denoiser computes
            state_dict.pop("pos_pe", None)
            # the Krea 2 branch is a generic single-stream DiT load: dtype and device handling
            with _swapped(krea, "SingleStreamDiT", IdolDenoiser):
                dit = original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            cast_on_the_fly(dit)
            # Forge casts the latent and context to this type on the way in: they stay in fp32, as the SDK has them,
            # and the denoiser moves to fp16 for its blocks
            dit.computation_dtype = torch.float32
            return dit

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _hook_first_stage() -> None:
    # the object is not an image: TAESD has nothing to say about it, so Settings > VAE never reroutes Idol
    from modules import processing, sd_samplers_common, sd_vae_taesd

    original_decoder = sd_vae_taesd.decoder_model

    def decoder_model():
        from modules import shared
        return None if getattr(shared.sd_model, "is_idol", False) else original_decoder()

    sd_vae_taesd.decoder_model = decoder_model

    original_encoder = sd_vae_taesd.encoder_model

    def encoder_model():
        from modules import shared
        return None if getattr(shared.sd_model, "is_idol", False) else original_encoder()

    sd_vae_taesd.encoder_model = encoder_model

    original_decode = sd_samplers_common.decode_first_stage

    def decode_first_stage(model, x):
        if not getattr(model, "is_idol", False):
            return original_decode(model, x)
        return sd_samplers_common.samples_to_images_tensor(x, 0, model)

    sd_samplers_common.decode_first_stage = decode_first_stage
    # processing imported the function by name
    processing.decode_first_stage = decode_first_stage


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.Idol not in model_list.models:
        model_list.models.append(model.Idol)
    if IdolEngine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, IdolEngine)
    _hook_replace_state_dict()
    _hook_components()
    _hook_first_stage()

    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works without the preset: the panel sets Euler, Simple, 50 steps and CFG 4
        logger.warning(f"[Idol] could not add the idol UI preset: {e}")

    _applied = True
    print(f"[Idol] native backend enabled (torch {torch.__version__})")
