"""Idol diffusion engine for Forge Neo: Forge samples the object latent like any other, with the image in place of a prompt.

The script hands over a request (the original picture, its input type, the number of Gaussians); the conditioning step
prepares and encodes the picture, and the decoding step turns the latent into Gaussians for the script to write, giving
Forge the prepared picture for its gallery.
"""

import contextlib

import numpy as np
import torch
from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.nn.vae import IntegratedAutoencoderKL
from backend.patcher.base import ModelPatcher
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from .. import assets
from ..camera import decode_camera
from ..contracts import (
    GAUSSIANS_PER_ANCHOR,
    IdolError,
    crop_margin,
    raise_pending_error,
)
from ..preprocess import prepare_image
from . import encoders, streams, turbo
from .model import Idol

DINOV3_MEMORY = 2 * 1024 ** 3
BIREFNET_MEMORY = 2 * 1024 ** 3
DECODER_CHANNELS = 1024


def decoder_memory(num_gaussians: int) -> int:
    """Working memory for one decode, from ComfyUI's estimate; the anchors' self attention grows with their number."""
    anchors = num_gaussians // GAUSSIANS_PER_ANCHOR
    return 3 * (streams.LATENT_SIZE // 16 * 4 + anchors * 10) * DECODER_CHANNELS * 2


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.amax()) and torch.isfinite(tensor.amin()))


@contextlib.contextmanager
def seeded(seed: int, device: torch.device):
    """The VAE's sampling and the octree's random placement follow the generation's seed and leave Forge's RNG as it was."""
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed & 0xffffffffffffffff)
        yield


class IdolEngine(ForgeDiffusionEngine):
    matched_guesses = [Idol]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        pair = huggingface_components["vae"]
        clip = CLIP(model_dict={"dinov3": huggingface_components["text_encoder"]}, tokenizer_dict={})
        vae = VAE(model=pair.flux2, is_flux2=True)
        self.decoders = pair.decoders
        self.decoder_patcher = ModelPatcher(self.decoders, load_device=memory_management.vae_device(),
                                            offload_device=memory_management.vae_offload_device())

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._get_predictor(), config=estimated_config)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_idol = True
        # Forge's Shift slider (the "idol" UI preset) sets the flow shift
        self.use_shift = True

        self.request = None
        self.background_remover = None  # (model, patcher), loaded the first time a picture has no alpha
        self.prepared = None            # the 1024 x 1024 RGB picture the encoders saw
        self.tokens = None              # the conditioning [L, 1408] of the current picture
        self.gaussian = None            # the decoded object, on the CPU, for the script to write
        self.camera = None              # the predicted input camera

    def set_request(self, request, turbo_loras=None) -> None:
        """Called by the script for every generation, before Forge encodes the prompts."""
        self.request = request
        self.prepared = self.tokens = self.gaussian = self.camera = None
        turbo.set_turbo(self.forge_objects.unet.model.diffusion_model, turbo_loras)

    def release_generation(self) -> None:
        self.request = None
        self.prepared = self.tokens = self.gaussian = self.camera = None

    def _remove_background(self, image):
        """BiRefNet, only for pictures without a real alpha channel. A plain module loaded whole: its BatchNorms have no
        Forge operation to stream them from system RAM."""
        if self.background_remover is None:
            from safetensors.torch import load_file

            from .birefnet import BiRefNet

            model = BiRefNet().eval().requires_grad_(False)
            model.load_state_dict(load_file(assets.ensure(assets.BIREFNET)), strict=True)
            device = memory_management.get_torch_device()
            if device.type == "cuda":
                model.half()
            patcher = ModelPatcher(model, load_device=device, offload_device=memory_management.vae_offload_device())
            self.background_remover = (model, patcher)
        model, patcher = self.background_remover
        memory_management.load_models_gpu([patcher], memory_required=BIREFNET_MEMORY, force_full_load=True)
        return model.remove_background(image)

    def _flux2_latent(self, pixels: torch.Tensor) -> torch.Tensor:
        vae = self.forge_objects.vae
        memory_management.load_models_gpu([vae.patcher], memory_required=vae.memory_used_encode(pixels.shape, vae.vae_dtype))
        model = vae.first_stage_model
        # the parent's encode: the encoder and a sample of its distribution, without Flux2's own normalization
        latents = IntegratedAutoencoderKL.encode(model, (pixels * 2 - 1).to(vae.device, vae.vae_dtype))
        return encoders.flux2_tokens(latents, model.bn.running_mean, model.bn.running_var, model.bn_eps)

    def _encode_picture(self) -> torch.Tensor:
        if self.tokens is not None:
            return self.tokens
        request = self.request
        if request is None:
            raise IdolError("Idol makes an object from a picture: use the img2img tab with an input image.")
        self.prepared = prepare_image(request.image, self._remove_background, crop_margin(request.mode))
        pixels = torch.from_numpy(np.array(self.prepared)).permute(2, 0, 1).float().div(255.0)[None]

        clip = self.forge_objects.clip
        memory_management.load_models_gpu([clip.patcher], memory_required=DINOV3_MEMORY)
        feature1 = encoders.dinov3_tokens(clip.cond_stage_model.dinov3, pixels.to(clip.patcher.load_device)).cpu()
        vae_device = self.forge_objects.vae.device
        with seeded(request.seed, vae_device):
            feature2 = self._flux2_latent(pixels).cpu()
        self.tokens = streams.pack_context(feature1, feature2)[0]
        return self.tokens

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        """The picture stands for the prompt and zeros for the negative prompt, whatever they say; Forge's cache by prompt
        is cleared for every generation by the script."""
        raise_pending_error()
        tokens = self._encode_picture()
        if getattr(prompt, "is_negative_prompt", False):
            tokens = torch.zeros_like(tokens)
        return [tokens for _ in prompt]

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        return 0, 75

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        # Forge's img2img init hands over its flattened copy of the input; the script already took the original, and the
        # placeholder latent is replaced by the packed one before sampling
        raise_pending_error()
        return torch.zeros((1, 1, 1, 1), device=x.device)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        request = self.request
        if request is None or x.shape[-1] != streams.PACKED_SIZE:
            raise RuntimeError("[Idol] the latent does not belong to the current Idol generation")
        if not finite(x[:1]):
            raise IdolError("Idol sampling produced NaN or Inf values: the object would come out empty. Try another "
                            "sampler or attention option.")
        latent, camera = streams.unpack(x[:1].float())
        self.camera = decode_camera(camera[0, 0].tolist())

        memory_management.load_models_gpu([self.decoder_patcher], memory_required=decoder_memory(request.num_gaussians))
        device = self.decoder_patcher.load_device
        with seeded(request.seed, device):
            gaussian = self.decoders.decode(latent.to(device), request.num_gaussians)
        if not finite(gaussian.xyz):
            raise IdolError("The Idol decoder produced NaN or Inf values: the object would come out empty.")
        for name in ("xyz", "features_dc", "scaling", "rotation", "opacity"):
            setattr(gaussian, name, getattr(gaussian, name).float().cpu())
        self.gaussian = gaussian

        picture = torch.from_numpy(np.array(self.prepared)).permute(2, 0, 1).float().div(127.5).sub(1.0)
        return picture[None]
