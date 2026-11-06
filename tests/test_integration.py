"""The hooks and the engine against stand-ins for Forge."""

import sys
import types

import pytest
import torch
import torch.nn as nn
from forge_stubs import install, restore
from PIL import Image

from forge_idol import integration
from forge_idol.contracts import (
    PLACEHOLDER_PROMPT,
    GenerationRequest,
    IdolError,
    set_pending_error,
)
from forge_idol.native import encoders, streams


@pytest.fixture
def forge():
    previous = install()
    yield
    restore(previous)
    set_pending_error(None)


def flux2_model():
    return types.SimpleNamespace(bn=types.SimpleNamespace(running_mean=torch.zeros(128), running_var=torch.ones(128)),
                                 bn_eps=1e-4)


class FakeDinov3:
    def __call__(self, pixel_values):
        return types.SimpleNamespace(last_hidden_state=torch.randn(1, 4101, 1280))

    def parameters(self):
        return iter([torch.zeros(1)])


def fake_engine():
    from forge_idol.native import engine, model

    config = model.Idol()
    decoders = nn.Linear(1, 1)
    pair = types.SimpleNamespace(flux2=flux2_model(), decoders=decoders)
    return engine.IdolEngine(config, {"vae": pair, "text_encoder": FakeDinov3(), "transformer": nn.Linear(1, 1)})


class NegativePrompts(list):
    is_negative_prompt = True


def test_patches_leave_other_models_to_forge(forge):
    from backend import loader
    from huggingface_guess import detection, model_list

    from forge_idol.native import patches

    patches._applied = False
    patches._hook_detection()
    patches._hook_replace_state_dict()
    patches._hook_components()
    assert detection.detect_unet_config({}, "") == "forge"
    assert loader.replace_state_dict({}, {}, object(), "file") == "forge"
    assert loader.load_huggingface_component(object(), "vae", "diffusers", "AutoencoderKL", "", {}) == "forge"
    assert model_list.BASE is not None


def test_files_land_where_the_engine_reads_them(forge):
    from backend import loader

    from forge_idol.native import model, patches

    patches._hook_replace_state_dict()
    guess = model.Idol()
    sd = {}
    flux2 = {"bn.running_mean": 1, "encoder.conv_in.weight": 2, "decoder.post_quant_conv.weight": 3, "encoder.quant_conv.bias": 4}
    loader.replace_state_dict(sd, flux2, guess, "flux2")
    loader.replace_state_dict(sd, {"embeddings.cls_token": 5, "layer.0.attention.q_proj.weight": 6}, guess, "dinov3")
    loader.replace_state_dict(sd, {"octree.in_proj.weight": 7, "gs.in_proj.weight": 8}, guess, "decoders")
    assert sd == {"vae.flux2.bn.running_mean": 1, "vae.flux2.encoder.conv_in.weight": 2, "vae.flux2.post_quant_conv.weight": 3,
                  "vae.flux2.quant_conv.bias": 4, "text_encoders.dinov3.embeddings.cls_token": 5,
                  "text_encoders.dinov3.layer.0.attention.q_proj.weight": 6, "vae.decoders.octree.in_proj.weight": 7,
                  "vae.decoders.gs.in_proj.weight": 8}


def test_split_decoders_get_the_repack_prefixes(forge):
    from backend import loader

    from forge_idol.native import model, patches

    patches._hook_replace_state_dict()
    guess = model.Idol()
    sd = {}
    loader.replace_state_dict(sd, {"l_embedder.mlp.0.weight": 1, "out_proj.weight": 2, "blocks.0.cross_attn.to_q.weight": 3}, guess, "o")
    loader.replace_state_dict(sd, {"points_offset_perturbation": 4, "out_proj.weight": 5}, guess, "g")
    assert sd["vae.decoders.octree.out_proj.weight"] == 2 and sd["vae.decoders.gs.out_proj.weight"] == 5


def test_every_layer_casts_on_the_fly(forge):
    from forge_idol.native import patches

    class Layer(nn.Module):
        parameters_manual_cast = False

    model = nn.Sequential(Layer(), nn.Sequential(Layer()))
    patches.cast_on_the_fly(model)
    assert all(layer.parameters_manual_cast for layer in model.modules() if isinstance(layer, Layer))


def test_flux2_tokens_use_the_stored_statistics():
    latents = torch.full((1, 32, 128, 128), 3.0)
    tokens = encoders.flux2_tokens(latents, torch.full((128,), 1.0), torch.full((128,), 4.0), eps=0.0)
    assert tokens.shape == (1, 4101, 128)
    assert torch.equal(tokens[:, :5], torch.zeros(1, 5, 128))
    # (3 - 1) / 2, whatever the latent's own mean and variance are
    assert torch.allclose(tokens[:, 5:], torch.ones(1, 4096, 128))


def test_flux2_patches_follow_forges_layout():
    # rearrange "c (i pi) (j pj) -> (c pi pj) i j": channels of one patch go together, row first
    latents = torch.arange(32 * 4, dtype=torch.float32).reshape(1, 32, 4)[:, :, None]
    latents = latents.reshape(1, 32, 2, 2)
    tokens = encoders.flux2_tokens(latents, torch.zeros(128), torch.ones(128), eps=0.0)
    assert tokens[0, 5].tolist() == latents[0].permute(1, 2, 0).reshape(-1, 32).reshape(4, 32).permute(1, 0).flatten().tolist()


def test_negative_prompt_gets_zeros_and_the_picture_is_encoded_once(forge):
    engine = fake_engine()
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (20, 20, 40, 40))
    engine.set_request = lambda request, loras=None: setattr(engine, "request", request)
    engine.set_request(GenerationRequest(image=image, mode="synthetic", num_gaussians=32768, turbo=False, seed=3))
    positive = engine.get_learned_conditioning(["anything"])
    assert len(positive) == 1 and positive[0].shape == (4101, 1408)
    assert positive[0][0, 0] != 0
    first = engine.tokens
    negative = engine.get_learned_conditioning(NegativePrompts([""]))
    assert engine.tokens is first
    assert torch.equal(negative[0], torch.zeros(4101, 1408))
    assert engine.prepared.size == (1024, 1024)


def test_stale_error_does_not_answer_for_the_next_generation(forge):
    engine = fake_engine()
    set_pending_error(IdolError("Add an input image"))
    with pytest.raises(IdolError, match="input image"):
        engine.encode_first_stage(torch.zeros(1, 3, 8, 8))
    # the next click, in txt2img, where the script does not run
    with pytest.raises(IdolError, match="use the img2img tab"):
        engine.get_learned_conditioning(["a chair"])


def request_for_prompt(prompt, monkeypatch):
    shared = types.SimpleNamespace(opts=types.SimpleNamespace(sd_model_checkpoint="idol", forge_additional_modules=[]))
    monkeypatch.setitem(sys.modules, "modules", types.SimpleNamespace(shared=shared))
    monkeypatch.setitem(sys.modules, "modules.shared", shared)
    monkeypatch.setattr(integration, "select_idol", lambda p: types.SimpleNamespace(filename="idol.safetensors"))
    monkeypatch.setattr(integration, "module_paths", lambda values: [])
    monkeypatch.setattr(integration, "resolve_components", lambda dit, modules: None)
    p = types.SimpleNamespace(n_iter=1, init_images=[Image.new("RGB", (8, 8))], subseed_strength=0, seed_resize_from_w=-1,
                              seed_resize_from_h=-1, prompt=prompt, steps=5, cfg_scale=1.0)
    integration.before_process(p)
    return p


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_empty_prompt_gets_a_placeholder_other_extensions_can_read(prompt, monkeypatch):
    # sd-dynamic-prompts raises StopIteration on an empty prompt
    assert request_for_prompt(prompt, monkeypatch).prompt == PLACEHOLDER_PROMPT


def test_a_rejected_request_gets_the_placeholder_too(monkeypatch):
    shared = types.SimpleNamespace(opts=types.SimpleNamespace(sd_model_checkpoint="idol", forge_additional_modules=[]))
    monkeypatch.setitem(sys.modules, "modules", types.SimpleNamespace(shared=shared))
    monkeypatch.setitem(sys.modules, "modules.shared", shared)
    monkeypatch.setattr(integration, "select_idol", lambda p: types.SimpleNamespace(filename="idol.safetensors"))
    p = types.SimpleNamespace(n_iter=1, init_images=[None], subseed_strength=0, seed_resize_from_w=-1, seed_resize_from_h=-1,
                              prompt="", steps=5, cfg_scale=1.0)
    with pytest.raises(IdolError, match="input image"):
        integration.before_process(p)
    assert p.prompt == PLACEHOLDER_PROMPT


def test_a_written_prompt_stays(monkeypatch):
    assert request_for_prompt("a red chair", monkeypatch).prompt == "a red chair"


def test_txt2img_is_told_to_use_img2img(forge):
    engine = fake_engine()
    with pytest.raises(IdolError, match="img2img"):
        engine.get_learned_conditioning(["a chair"])


def test_pending_error_stops_conditioning(forge):
    engine = fake_engine()
    set_pending_error(IdolError("select the encoder"))
    with pytest.raises(IdolError, match="select the encoder"):
        engine.get_learned_conditioning([""])


def test_placeholder_latent_and_foreign_latents(forge):
    engine = fake_engine()
    assert engine.encode_first_stage(torch.zeros(1, 3, 8, 8)).shape == (1, 1, 1, 1)
    with pytest.raises(RuntimeError, match="does not belong"):
        engine.decode_first_stage(torch.zeros(1, 4, 8, 8))


def test_nan_latent_is_a_clear_error(forge):
    engine = fake_engine()
    engine.request = GenerationRequest(image=None, mode="real", num_gaussians=32768, turbo=False)
    x = torch.full((1, 1, 1, streams.PACKED_SIZE), float("nan"))
    with pytest.raises(IdolError, match="NaN"):
        engine.decode_first_stage(x)


def test_validation_names_what_to_change():
    def request(**settings):
        values = dict(n_iter=1, init_images=[object()], subseed_strength=0, seed_resize_from_w=-1, seed_resize_from_h=-1)
        return types.SimpleNamespace(**{**values, **settings})

    integration.validate_processing(request())
    for settings, message in ((dict(n_iter=2), "Batch Count"), (dict(init_images=[]), "input image"),
                              (dict(image_mask=object()), "inpainting"), (dict(subseed_strength=0.5), "variation")):
        with pytest.raises(IdolError, match=message):
            integration.validate_processing(request(**settings))


def test_missing_image_leaves_a_stand_in_so_the_message_reaches_the_user():
    # the img2img tab sends [None] with no image, and Forge reads the bytes of what is there before calling the model
    p = types.SimpleNamespace(n_iter=1, init_images=[None], subseed_strength=0, seed_resize_from_w=-1, seed_resize_from_h=-1)
    with pytest.raises(IdolError, match="input image"):
        integration.validate_processing(p)
    assert p.init_images[0].tobytes()


def test_the_seed_comes_from_the_batch_forge_is_about_to_run():
    request = GenerationRequest(image=None, mode="real", num_gaussians=32768, turbo=False, seed=1)
    integration.process_batch(types.SimpleNamespace(idol_request=request), [711730114])
    assert request.seed == 711730114
    integration.process_batch(types.SimpleNamespace(), [5])  # not an Idol generation


def test_viewer_path_copies_only_what_gradio_would_refuse(tmp_path):
    cwd, temp, elsewhere = tmp_path / "forge", tmp_path / "temp", tmp_path / "out"
    for folder in (cwd, temp, elsewhere):
        folder.mkdir()
    inside = cwd / "a.ply"
    inside.write_bytes(b"x")
    outside = elsewhere / "b.ply"
    outside.write_bytes(b"y")
    assert integration.viewer_path(inside, cwd, temp) == str(inside.resolve())
    copied = integration.viewer_path(outside, cwd, temp)
    assert str(temp.resolve()) in copied and open(copied, "rb").read() == b"y"

