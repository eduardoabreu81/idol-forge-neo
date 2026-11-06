"""Checks that the installed Forge Neo has every piece the Idol backend builds on, before anything is patched."""

import importlib
import importlib.util
import inspect

# what the backend imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.memory_management": ("load_models_gpu", "vae_device", "vae_offload_device", "should_use_bf16", "cpu"),
    "backend.operations": ("using_forge_operations", "weights_manual_cast"),
    "backend.state_dict": ("load_state_dict",),
    "backend.loader": ("load_huggingface_component", "replace_state_dict", "possible_models", "HF"),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.nn.vae": ("AutoencoderKLFlux2", "IntegratedAutoencoderKL"),
    "backend.patcher.base": ("ModelPatcher",),
    "backend.patcher.clip": ("CLIP",),
    "backend.patcher.unet": ("UnetPatcher",),
    "backend.patcher.vae": ("VAE",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine", "ForgeObjects"),
    "modules.processing": ("decode_first_stage",),
    "modules.sd_samplers_common": ("decode_first_stage", "samples_to_images_tensor"),
    "modules.sd_vae_taesd": ("decoder_model", "encoder_model", "download_model"),
    "modules.paths_internal": ("models_path",),
    "huggingface_guess.detection": ("detect_unet_config",),
    "huggingface_guess.model_list": ("models", "BASE", "ModelType"),
    "huggingface_guess.latent": ("LatentFormat",),
    "transformers.models.dinov3_vit": ("DINOv3ViTConfig", "DINOv3ViTModel"),
    "scipy.ndimage": ("binary_closing", "binary_fill_holes", "gaussian_filter", "map_coordinates"),
    "scipy.sparse.csgraph": ("connected_components",),
}

# the denoiser goes through this branch of the loader
LOADER_BRANCHES = ("Krea2Transformer2DModel",)


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        from backend import loader

        # the whole module, not load_huggingface_component: an extension that loaded first has already put its own
        # wrapper there, and a wrapper's source holds none of Forge's branches
        source = inspect.getsource(loader)
        missing = [b for b in LOADER_BRANCHES if f'"{b}"' not in source]
        if missing:
            problems.append(f"backend.loader has no {', '.join(missing)} branch")

    return problems


def report(problems: list[str]) -> str:
    lines = [
        "[Idol] This Forge Neo is too old for the extension, so nothing was changed and Idol is disabled.",
        "[Idol] Update Forge Neo (git pull on the neo branch) and restart. Details:",
    ]
    lines += [f"[Idol]   - {p}" for p in problems]
    message = "\n".join(lines)
    print(message)
    return message
