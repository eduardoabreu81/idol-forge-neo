"""Idol inside Forge's own img2img flow, through script callbacks.

Forge loads the model (native/patches.py registers it), samples and decodes as usual. The callbacks here take the input
picture for the engine, give the sampler the packed noise, and write the PLY.
"""

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from PIL import Image

from .contracts import (
    DEFAULT_FORMATS,
    DEFAULT_GAUSSIANS,
    DEFAULT_MODE,
    FORMATS,
    IMAGE_SIZE,
    PLACEHOLDER_PROMPT,
    SAMPLER,
    SCHEDULER,
    SHIFT,
    TURBO_CFG,
    GenerationRequest,
    IdolError,
    align_gaussians,
    crop_margin,
    set_pending_error,
)
from .mesh import DEFAULT_RESOLUTION, DEFAULT_SMOOTH
from .models import inspect_model, resolve_components

# StableDiffusionProcessing.distilled_cfg_scale when a request does not set it
API_DEFAULT_DISTILLED_CFG = 3.5

_turbo = None
# the viewer's file; the PLY rotations start from (in the temporary folder when .ply is not ticked); the name of the last
# object's files, without a suffix
_latest = None
_latest_source = None
_latest_base = None


def checkpoint_info(value):
    # the same lookup Forge applies to a checkpoint override: alias, then title substring, with or without the hash
    from modules import sd_models
    return sd_models.get_closet_checkpoint_match(value)


def select_idol(p):
    """The selected checkpoint when it is an Idol denoiser; read from its header, before Forge loads it."""
    from modules import shared
    overrides = getattr(p, "override_settings", {})
    info = checkpoint_info(overrides.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint))
    if info is None:
        return None
    return info if inspect_model(info.filename) == "dit" else None


def module_paths(values):
    from modules_forge import main_entry
    result = []
    for value in values or []:
        # Forge Neo keys its list by file name; with "Show filenames without folder" off the dropdown sends the path
        # under models/VAE or models/text_encoder, which Forge itself reduces to the file name (modules_change)
        path = main_entry.module_list.get(value) or main_entry.module_list.get(Path(value).name, value)
        if not Path(path).is_file():
            raise IdolError(f"Selected Idol component is unavailable: {Path(value).name}. Refresh the model list.")
        result.append(str(Path(path).resolve()))
    return result


def has_input_image(p) -> bool:
    images = getattr(p, "init_images", None)
    return bool(images) and images[0] is not None


def validate_processing(p):
    if getattr(p, "n_iter", 1) != 1:
        raise IdolError("Idol currently supports Batch Count = 1.")
    if getattr(p, "enable_hr", False) or getattr(p, "txt2img_upscale", False):
        raise IdolError("Disable Hires. fix for Idol.")
    if getattr(p, "image_mask", None) is not None:
        raise IdolError("Idol has no inpainting. Use the regular img2img image input.")
    if (getattr(p, "subseed_strength", 0) > 0 or getattr(p, "seed_resize_from_w", -1) > 0
            or getattr(p, "seed_resize_from_h", -1) > 0):
        raise IdolError("Disable variation seed and seed resize for Idol.")
    if not has_input_image(p):
        # Forge reads the image's bytes before it first calls the model: with a stand-in, the message below is what
        # the user sees instead of Forge's AttributeError
        p.init_images = [Image.new("RGB", (64, 64))]
        raise IdolError("Add an input image for Idol in img2img: the object is made from it.")


def chosen_formats(formats) -> tuple:
    """What a request asks to save, in the panel's order; an API call that says nothing gets the default ones."""
    if formats is None:
        return DEFAULT_FORMATS
    unknown = [name for name in formats if name not in FORMATS]
    if unknown:
        raise IdolError(f"Idol cannot save {', '.join(unknown)}: choose from {', '.join(FORMATS)}.")
    if not formats:
        raise IdolError("Tick at least one file type under Save as.")
    return tuple(name for name in FORMATS if name in formats)


def before_process(p, mode=DEFAULT_MODE, num_gaussians=DEFAULT_GAUSSIANS, use_turbo=False, formats=None,
                   mesh_resolution=DEFAULT_RESOLUTION, mesh_smooth=DEFAULT_SMOOTH):
    """Before Forge loads the model: check the request and settle what Idol fixes."""
    p.idol_request = None
    set_pending_error(None)
    try:
        _before_process(p, mode, num_gaussians, use_turbo, formats, mesh_resolution, mesh_smooth)
    except IdolError as error:
        set_pending_error(error)
        raise


def script_before_process(p, mode=DEFAULT_MODE, num_gaussians=DEFAULT_GAUSSIANS, use_turbo=False, formats=None,
                          mesh_resolution=DEFAULT_RESOLUTION, mesh_smooth=DEFAULT_SMOOTH, *ignored):
    """before_process as Forge's script runner calls it: it logs any exception with its traceback, so a rejected request
    stays quiet here; the error is pending and stops the generation when the model is first called, with its message."""
    try:
        before_process(p, mode, num_gaussians, use_turbo, formats, mesh_resolution, mesh_smooth)
    except IdolError:
        pass


def _before_process(p, mode, num_gaussians, use_turbo, formats=None, mesh_resolution=DEFAULT_RESOLUTION,
                    mesh_smooth=DEFAULT_SMOOTH):
    info = select_idol(p)
    if info is None:
        return
    from modules import shared
    # first of all: a rejected request still goes through the other extensions before it stops at the model
    if not str(getattr(p, "prompt", "") or "").strip():
        p.prompt = PLACEHOLDER_PROMPT
    validate_processing(p)
    crop_margin(mode)
    formats = chosen_formats(formats)
    overrides = getattr(p, "override_settings", {})
    resolve_components(info.filename, module_paths(overrides.get("forge_additional_modules",
                                                                  shared.opts.forge_additional_modules)))
    # the picture as it came, before Forge flattens its alpha
    image = p.init_images[0].copy()
    p.idol_request = GenerationRequest(image=image, mode=mode, num_gaussians=align_gaussians(num_gaussians),
                                       turbo=bool(use_turbo), formats=formats, mesh_resolution=int(mesh_resolution),
                                       mesh_smooth=int(mesh_smooth))
    p.batch_size = 1
    # Forge's own copy only feeds a placeholder latent: the encoders get their 1024 x 1024 crop
    p.width = p.height = IMAGE_SIZE
    p.resize_mode = 0
    p.denoising_strength = 1.0
    if use_turbo:
        # the turbo's guidance is folded into its LoRAs, which were made for this integrator
        p.cfg_scale = TURBO_CFG
        p.sampler_name = SAMPLER
        p.scheduler = SCHEDULER
    print(f"[Idol] {request_text(p.idol_request, p.steps, p.cfg_scale)}")


def request_text(request, steps, cfg) -> str:
    turbo = ", Turbo" if request.turbo else ""
    return f"{request.num_gaussians} Gaussians from a {request.mode} image, {steps} steps, CFG {cfg:g}{turbo}"


def turbo_loras():
    """Both phases, read once and kept on the CPU."""
    global _turbo
    if _turbo is None:
        from safetensors.torch import load_file

        from . import assets
        from .native import turbo

        _turbo = turbo.TurboLoras(turbo.lora_pairs(load_file(assets.ensure(assets.TURBO_SLOW))),
                                  turbo.lora_pairs(load_file(assets.ensure(assets.TURBO_FAST))))
    return _turbo


def process(p):
    request = getattr(p, "idol_request", None)
    if request is None:
        return
    if not getattr(p.sd_model, "is_idol", False):
        raise IdolError("Forge did not load the Idol model. Check the VAE / Text Encoder selection and the console.")
    from modules import shared

    from .native.presets import PRESET

    # Forge caches the conditioning by prompt, which knows nothing of the picture
    p.clear_prompt_cache()
    p.sd_model.set_request(request, turbo_loras() if request.turbo else None)

    if getattr(shared.opts, "forge_preset", None) != PRESET:
        # outside the idol preset the slider is another model's Distilled CFG; keep Idol's own shift (and infotext)
        p.distilled_cfg_scale = SHIFT
    elif getattr(p, "is_api", False) and p.distilled_cfg_scale == API_DEFAULT_DISTILLED_CFG:
        # an API request that leaves distilled_cfg_scale out gets Forge's Flux default; use the preset's Shift
        p.distilled_cfg_scale = getattr(shared.opts, f"{PRESET}_i2i_dcfg", SHIFT)
    p.extra_generation_params.update({"Idol Input": request.mode, "Idol Gaussians": request.num_gaussians})
    if request.turbo:
        p.extra_generation_params["Idol Turbo"] = True


def process_batch(p, seeds):
    """Right before Forge encodes the conditioning: the seed the sampler will use and the infotext will show. It can
    differ from p.all_seeds at script.process, where other extensions may still replace the seeds."""
    request = getattr(p, "idol_request", None)
    if request is not None:
        request.seed = int(seeds[0])


def before_sampling(p, noise):
    import torch
    from modules import rng

    from .native.streams import PACKED_SHAPE

    if getattr(p, "idol_request", None) is None:
        return
    # Forge made p.rng for an image latent; samplers that add noise on the way draw from it too, so it has to give
    # the packed shape
    p.rng = rng.ImageRNG(PACKED_SHAPE, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
    if getattr(p, "init_latent", None) is not None:
        # img2img samples from init_latent at full denoise; the packed start is pure noise
        p.init_latent = torch.zeros_like(p.modified_noise)


def postprocess(p, processed):
    """After Forge's own outputs: write the object next to them."""
    request = getattr(p, "idol_request", None)
    if request is None:
        return
    engine = p.sd_model
    from modules import shared
    try:
        if shared.state.interrupted or shared.state.skipped:
            processed.comments += "Idol object not written: the generation was interrupted\n"
            return
        if getattr(engine, "gaussian", None) is None:
            return
        _write_object(p, processed, request, engine)
    finally:
        engine.release_generation()


def _write_object(p, processed, request, engine):
    global _latest, _latest_source, _latest_base
    from modules import shared

    directory = Path(p.outpath_samples or shared.opts.outdir_samples or "outputs/idol").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"idol-{request.seed}-{uuid.uuid4().hex[:12]}"
    elements = engine.gaussian.to_ply_elements()
    source, saved, note = save_files(elements, base, request.formats, request.mesh_resolution, request.mesh_smooth)
    infotext = processed.infotexts[0] if processed.infotexts else processed.info
    sidecar = dict(seed=request.seed, mode=request.mode, num_gaussians=request.num_gaussians, turbo=request.turbo, files=saved,
                   steps=p.steps, sampler=p.sampler_name, scheduler=p.scheduler, cfg=p.cfg_scale, camera=engine.camera,
                   infotext=infotext)
    base.with_name(base.name + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    _latest_source = source
    _latest = viewer_file(source, base, saved)
    _latest_base = base
    names = " and ".join(saved)
    processed.comments += f"Idol object saved to {base}: {names}\n{note}"
    print(f"[Idol] object saved to {base}: {names}")
    if note:
        print(f"[Idol] {note.strip()}")


def viewer_file(source, base, saved) -> str:
    """What the panel's viewer opens: the mesh when it is all that was asked for, the object otherwise."""
    return str(Path(base).with_name(Path(base).name + ".glb")) if saved == [".glb"] else source


def save_files(elements, base, formats, mesh_resolution=DEFAULT_RESOLUTION, mesh_smooth=DEFAULT_SMOOTH) -> tuple:
    """Writes the chosen files. Returns a PLY to rotate from, in the temporary folder when .ply is not one of them, the
    suffixes saved, and a line for the user when the mesh could not be made, which does not cost the other files."""
    from .gaussian import write_ply
    from .mesh import glb_bytes, mesh_from_elements
    from .spz import spz_bytes

    base = Path(base)
    if ".ply" in formats:
        source = base.with_name(base.name + ".ply")
    else:
        folder = Path(tempfile.gettempdir()) / "idol-forge-neo"
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / (base.name + ".ply")
    write_ply(elements, source)
    if ".spz" in formats:
        base.with_name(base.name + ".spz").write_bytes(spz_bytes(elements))
    saved, note = list(formats), ""
    if ".glb" in formats:
        try:
            mesh = mesh_from_elements(elements, int(mesh_resolution), int(mesh_smooth))
            base.with_name(base.name + ".glb").write_bytes(glb_bytes(mesh))
        except IdolError as error:
            saved.remove(".glb")
            note = f"{error}\n"
    return str(source), saved, note


def rotate_latest(x, y, z, formats, mesh_resolution=DEFAULT_RESOLUTION, mesh_smooth=DEFAULT_SMOOTH) -> tuple:
    """Turn the last object, always from how it came out, and save the turned files next to it; the original stays.
    Returns the viewer's PLY and a line for the panel."""
    global _latest
    from .gaussian import read_ply
    from .rotation import rotate_elements

    formats = chosen_formats(formats)
    if _latest_source is None:
        raise IdolError("Generate an object first: the rotation applies to the last one.")
    if not Path(_latest_source).is_file():
        raise IdolError(f"The last object is no longer there: {_latest_source}.")
    base = Path(_latest_base)
    base = base.with_name(f"{base.name}-rotated-x{x:g}-y{y:g}-z{z:g}")
    source, saved, note = save_files(rotate_elements(read_ply(_latest_source), x, y, z), base, formats, mesh_resolution, mesh_smooth)
    _latest = viewer_file(source, base, saved)
    return _latest, f"Saved {base.name}: {' and '.join(saved)}. {note}".strip()


def viewer_path(path, cwd=None, temp=None):
    """A path Gradio will serve: it refuses files outside the working and temporary folders, and Forge's output folder
    can be anywhere."""
    path = Path(path).resolve()
    roots = [Path(cwd or Path.cwd()).resolve(), Path(temp or tempfile.gettempdir()).resolve()]
    if any(root == path or root in path.parents for root in roots):
        return str(path)
    copy = roots[1] / "idol-forge-neo" / path.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, copy)
    return str(copy)


def latest_object():
    """The object of the last generation, for the panel's viewer; None until there is one."""
    return viewer_path(_latest) if _latest else None

