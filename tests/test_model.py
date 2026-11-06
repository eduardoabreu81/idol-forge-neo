"""The denoiser, decoders, turbo and PLY on a small random model."""

import numpy as np
import torch
from forge_stubs import cast_layers
from PIL import Image

from forge_idol.camera import decode_camera
from forge_idol.contracts import IdolError
from forge_idol.gaussian import PLY_FIELDS
from forge_idol.native import streams, turbo
from forge_idol.native.decoder import IdolDecoders, systematic_sample
from forge_idol.native.denoiser import IdolDenoiser, sobol_positions
from forge_idol.preprocess import has_real_alpha, prepare_image

TINY = dict(model_channels=64, num_blocks=2, num_refiner_blocks=2, num_heads=4, num_head_channels=16)


def tiny_denoiser():
    torch.manual_seed(0)
    with cast_layers():
        model = IdolDenoiser(**TINY)
    return model.eval()


def tiny_inputs(batch=1):
    torch.manual_seed(1)
    x = torch.randn(batch, 1, 1, streams.PACKED_SIZE)
    context = torch.randn(batch, 16, streams.DINOV3_CHANNELS + streams.FLUX2_CHANNELS)
    return x, torch.full((batch,), 500.0), context


def test_streams_round_trip():
    latent, camera = torch.randn(2, 8192, 16), torch.randn(2, 1, 5)
    packed = streams.pack(latent, camera)
    assert packed.shape == (2, 1, 1, streams.PACKED_SIZE)
    back = streams.unpack(packed)
    assert torch.equal(back[0], latent) and torch.equal(back[1], camera)


def test_context_round_trip():
    feature1, feature2 = torch.randn(1, 4101, 1280), torch.randn(1, 4101, 128)
    back = streams.unpack_context(streams.pack_context(feature1, feature2))
    assert torch.equal(back[0], feature1) and torch.equal(back[1], feature2)


def test_preview_map_only_touches_packed_latents():
    assert streams.preview_map(torch.zeros(1, 1, 1, streams.PACKED_SIZE)).shape == (1, 16, 64, 128)
    other = torch.zeros(1, 4, 8, 8)
    assert streams.preview_map(other) is other


def test_forge_call_returns_the_packed_velocity():
    with torch.no_grad():
        out = tiny_denoiser()(*tiny_inputs())
    assert out.shape == (1, 1, 1, streams.PACKED_SIZE)
    assert torch.isfinite(out).all()


def test_batched_cfg_equals_two_passes():
    model = tiny_denoiser()
    x, t, context = tiny_inputs(2)
    x = x[:1].repeat(2, 1, 1, 1)
    with torch.no_grad():
        both = model(x, t, context)
        first = model(x[:1], t[:1], context[:1])
        second = model(x[1:], t[1:], context[1:])
    assert torch.allclose(both[:1], first, atol=2e-2)
    assert torch.allclose(both[1:], second, atol=2e-2)


def test_positions_are_the_release_sequence():
    # idol-sdk's stored pos_pe is this sequence: ComfyUI-Idol's converter checks it with torch.equal
    positions = sobol_positions(8192)
    assert positions.shape == (1, 8192, 3)
    assert torch.equal(positions, torch.quasirandom.SobolEngine(3, scramble=True, seed=123).draw(8192)[None])


def lora_files(model):
    def one(seed):
        torch.manual_seed(seed)
        files = {}
        for name, layer in model.named_modules():
            if name.endswith(("attn.qkv", "attn.out", "mlp.mlp.0", "mlp.mlp.2")):
                files[f"{name}.lora_down.weight"] = torch.randn(4, layer.in_features) * 0.01
                files[f"{name}.lora_up.weight"] = torch.randn(layer.out_features, 4) * 0.01
        return files

    return turbo.lora_pairs(one(2)), turbo.lora_pairs(one(3))


def test_turbo_follows_the_phase_and_comes_off_cleanly():
    model = tiny_denoiser()
    x, _, context = tiny_inputs()
    high, low = torch.full((1,), 900.0), torch.full((1,), 100.0)
    slow, fast = lora_files(model)
    with torch.no_grad():
        plain = model(x, high, context)
        loras = turbo.TurboLoras(slow, fast)
        turbo.set_turbo(model, loras)
        first = model(x, high, context)
        assert loras.active == "slow"
        model(x, low, context)
        assert loras.active == "fast"
        turbo.set_turbo(model, turbo.TurboLoras(fast, fast))
        fast_at_high = model(x, high, context)
        turbo.set_turbo(model, None)
        again = model(x, high, context)
    assert not torch.allclose(first, plain, atol=1e-4)
    assert not torch.allclose(first, fast_at_high, atol=1e-4)  # the phase comes from the time, and the files differ
    assert torch.equal(again, plain)
    assert model.turbo is None


def test_systematic_sample_keeps_the_count():
    probs = torch.softmax(torch.randn(1, 5, 8), dim=-1)
    counts = torch.tensor([[3, 0, 7, 1, 100]])
    out = systematic_sample(probs, counts)
    assert torch.equal(out.sum(-1), counts)


def tiny_decoders():
    torch.manual_seed(4)
    with cast_layers():
        model = IdolDecoders()
    return model.eval()


def test_decoders_make_the_asked_number_of_gaussians(tmp_path):
    latent = torch.randn(1, 64, 16)
    with torch.no_grad():
        gaussian = tiny_decoders().decode(latent, 32 * 8)
    assert gaussian.xyz.shape == (256, 3)
    assert gaussian.features_dc.shape == (256, 1, 3)
    assert gaussian.rotation.shape == (256, 4)
    assert torch.allclose(gaussian.rotation.norm(dim=-1), torch.ones(256), atol=1e-4)

    path = tmp_path / "object.ply"
    gaussian.save_ply(path)
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    assert "format binary_little_endian 1.0" in header and "element vertex 256" in header
    assert [line.split()[-1] for line in header.splitlines() if line.startswith("property")] == list(PLY_FIELDS)
    rows = np.frombuffer(data[end:], dtype=[(name, "<f4") for name in PLY_FIELDS])
    assert len(rows) == 256
    assert np.isfinite(np.stack([rows[name] for name in PLY_FIELDS])).all()
    # z-up object written y-up: (x, y, z) -> (x, -z, y)
    assert np.allclose(rows["y"], -gaussian.xyz[:, 2].numpy()) and np.allclose(rows["z"], gaussian.xyz[:, 1].numpy())


def test_camera_of_a_perspective_and_an_orthographic_view():
    camera = decode_camera([0.0, 1.0, 0.0, 0.5, 0.5])
    assert camera["radius"] == 2.0 and abs(camera["yaw"]) < 1e-9 and abs(camera["pitch"]) < 1e-9
    assert decode_camera([0.0, 0.0, 0.0, 0.0, 1.0])["radius"] == 1000.0


def square_with_alpha():
    image = Image.new("RGBA", (200, 100), (255, 0, 0, 0))
    image.paste((0, 255, 0, 255), (80, 30, 120, 70))
    return image


def test_real_alpha_is_used_without_the_remover():
    def remover(image):
        raise AssertionError("the picture has its own alpha")

    assert has_real_alpha(square_with_alpha())
    prepared = prepare_image(square_with_alpha(), remover, 1.2)
    assert prepared.mode == "RGB" and prepared.size == (1024, 1024)
    center, corner = np.array(prepared)[512, 512], np.array(prepared)[0, 0]
    assert center[1] > 200 and center[0] < 50 and corner.max() == 0


def test_opaque_pictures_go_to_the_remover():
    calls = []

    def remover(image):
        calls.append(image.size)
        result = image.copy()
        alpha = Image.new("L", image.size, 0)
        alpha.paste(255, (40, 40, image.size[0] - 40, image.size[1] - 40))
        result.putalpha(alpha)
        return result

    prepared = prepare_image(Image.new("RGB", (100, 100), (9, 9, 9)), remover, 1.7, size=256)
    assert calls == [(256, 256)]
    assert prepared.size == (256, 256)


def test_empty_alpha_is_a_clear_error():
    def remover(image):
        image.putalpha(0)
        return image

    try:
        prepare_image(Image.new("RGB", (64, 64)), remover, 1.2, size=64)
    except IdolError as error:
        assert "no object" in str(error)
    else:
        raise AssertionError("an empty alpha must stop the generation")
