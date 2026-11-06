"""Minimal stand-ins for the Forge Neo pieces the extension touches, for CPU tests without Forge.

Only the names the extension uses exist; anything else raises AttributeError, so a test notices when the extension starts
depending on more of Forge.
"""

import contextlib
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


class CastLinear(nn.Linear):
    """Forge's Linear with manual cast: the weights take the type of the input."""

    parameters_manual_cast = True

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight.to(x.dtype), bias)


@contextlib.contextmanager
def cast_layers():
    original = nn.Linear
    nn.Linear = CastLinear
    try:
        yield
    finally:
        nn.Linear = original


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class FakePatcher:
    def __init__(self, model=None, load_device=None, offload_device=None):
        self.model = model
        self.load_device = load_device or torch.device("cpu")


class FakeClip:
    def __init__(self, model_dict, tokenizer_dict):
        self.cond_stage_model = types.SimpleNamespace(**model_dict)
        self.patcher = FakePatcher()


class FakeVAE:
    def __init__(self, model, is_flux2):
        self.first_stage_model = model
        self.patcher = FakePatcher()
        self.device = torch.device("cpu")
        self.vae_dtype = torch.float32
        self.memory_used_encode = lambda shape, dtype: 0


class FakeUnetPatcher:
    def __init__(self, model):
        self.model = types.SimpleNamespace(diffusion_model=model)

    @classmethod
    def from_model(cls, model, diffusers_scheduler, k_predictor, config):
        return cls(model)


class FakeForgeObjects:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet, self.clip, self.vae, self.clipvision = unet, clip, vae, clipvision

    def shallow_copy(self):
        return FakeForgeObjects(self.unet, self.clip, self.vae, self.clipvision)


class FakeEngineBase:
    def __init__(self, estimated_config, huggingface_components):
        self.model_config = estimated_config

    def _get_predictor(self):
        return None


class FakeAutoencoder:
    @staticmethod
    def encode(model, x):
        return torch.zeros(x.shape[0], 32, x.shape[2] // 8, x.shape[3] // 8)


def _forget_forge_modules():
    """The modules that import Forge, so the next import binds to the stand-ins (or to nothing, after restore)."""
    package = sys.modules.get("forge_idol.native")
    for name in ("engine", "patches", "model", "presets"):
        sys.modules.pop(f"forge_idol.native.{name}", None)
        if package is not None and hasattr(package, name):
            delattr(package, name)


def install():
    """Everything forge_idol.native.engine and patches import; returns what was replaced, for restore()."""
    memory = module("backend.memory_management", load_models_gpu=lambda *a, **k: None, vae_device=lambda: torch.device("cpu"),
                    vae_offload_device=lambda: torch.device("cpu"), get_torch_device=lambda: torch.device("cpu"),
                    should_use_bf16=lambda device: False, cpu=torch.device("cpu"), vae_dtype=lambda: torch.float32)
    loader = module("backend.loader", HF="", possible_models=(), load_huggingface_component=lambda *a: "forge",
                    replace_state_dict=lambda sd, asd, guess, path: "forge")
    krea = module("backend.nn.krea", SingleStreamDiT=object)
    vae_nn = module("backend.nn.vae", IntegratedAutoencoderKL=FakeAutoencoder, AutoencoderKLFlux2=object)
    base = module("backend.diffusion_engine.base", ForgeDiffusionEngine=FakeEngineBase, ForgeObjects=FakeForgeObjects)
    patcher_base = module("backend.patcher.base", ModelPatcher=FakePatcher)
    patcher_clip = module("backend.patcher.clip", CLIP=FakeClip)
    patcher_unet = module("backend.patcher.unet", UnetPatcher=FakeUnetPatcher)
    patcher_vae = module("backend.patcher.vae", VAE=FakeVAE)
    operations = module("backend.operations", using_forge_operations=lambda **k: contextlib.nullcontext())
    state_dict = module("backend.state_dict", load_state_dict=lambda *a, **k: None)
    latent = module("huggingface_guess.latent", LatentFormat=object)

    class BASE:
        pass

    model_list = module("huggingface_guess.model_list", BASE=BASE, models=[],
                        ModelType=types.SimpleNamespace(FLOW="flow"))
    detection = module("huggingface_guess.detection", detect_unet_config=lambda sd, prefix: "forge")
    stubs = {
        "backend": module("backend", memory_management=memory, loader=loader),
        "backend.memory_management": memory,
        "backend.loader": loader,
        "backend.nn": module("backend.nn", krea=krea),
        "backend.nn.krea": krea,
        "backend.nn.vae": vae_nn,
        "backend.diffusion_engine": module("backend.diffusion_engine", base=base),
        "backend.diffusion_engine.base": base,
        "backend.patcher": module("backend.patcher"),
        "backend.patcher.base": patcher_base,
        "backend.patcher.clip": patcher_clip,
        "backend.patcher.unet": patcher_unet,
        "backend.patcher.vae": patcher_vae,
        "backend.operations": operations,
        "backend.state_dict": state_dict,
        "huggingface_guess": module("huggingface_guess", detection=detection, model_list=model_list),
        "huggingface_guess.latent": latent,
        "huggingface_guess.model_list": model_list,
        "huggingface_guess.detection": detection,
        "transformers": sys.modules.get("transformers") or module("transformers"),
        "transformers.modeling_utils": module("transformers.modeling_utils", no_init_weights=contextlib.nullcontext),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    _forget_forge_modules()
    return previous


def restore(previous):
    for name, value in previous.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value
    _forget_forge_modules()
