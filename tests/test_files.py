"""The real release files, by their headers: tests/fixtures/idol_headers.json.gz holds tensor names, types and shapes."""

import gzip
import json
import struct
from pathlib import Path

import pytest
import torch
from forge_stubs import cast_layers

from forge_idol.contracts import IdolError
from forge_idol.models import inspect_model, resolve_components, role_of
from forge_idol.native import detect
from forge_idol.native.decoder import IdolDecoders
from forge_idol.native.denoiser import IdolDenoiser

FIXTURE = Path(__file__).parent / "fixtures" / "idol_headers.json.gz"
HEADERS = json.loads(gzip.decompress(FIXTURE.read_bytes()))
DENOISER = "Comfy-Org/Idol/diffusion_models/idol_fp16.safetensors"
DECODERS = "Comfy-Org/Idol/vae/idol_decoder_fp16.safetensors"
DINOV3 = "Comfy-Org/Idol/clip_vision/dino_v3_vit_h_bf16.safetensors"
FLUX2 = "Comfy-Org/Idol/vae/flux2-vae.safetensors"
SDK_DENOISER = "speridlabs/idol/denoiser.safetensors"
SDK_OCTREE = "speridlabs/idol/decoder.safetensors"
SDK_GS = "speridlabs/idol/decoder_gs.safetensors"


class Shaped:
    def __init__(self, shape):
        self.shape = shape


def shapes_of(module):
    return {k: tuple(v.shape) for k, v in module.state_dict().items()}


def expected(name):
    return {k: tuple(v[1]) for k, v in HEADERS[name].items()}


@pytest.mark.parametrize("name, role", [(DENOISER, "dit"), (SDK_DENOISER, "dit"), (DECODERS, "decoders"), (DINOV3, "dinov3"),
                                        (FLUX2, "flux2"), (SDK_OCTREE, "octree"), (SDK_GS, "gs")])
def test_each_file_has_its_role(name, role):
    assert role_of(HEADERS[name]) == role


def test_denoiser_layout_matches_the_repack():
    with cast_layers(), torch.device("meta"):
        model = IdolDenoiser(**{k: v for k, v in detected_config().items() if k != "image_model"})
    assert shapes_of(model) == expected(DENOISER)


def test_sdk_denoiser_differs_only_by_its_stored_positions():
    assert set(expected(SDK_DENOISER)) - set(expected(DENOISER)) == {"pos_pe"}
    assert expected(SDK_DENOISER)["pos_pe"] == (1, 8192, 3)


def test_decoders_layout_matches_the_repack():
    with cast_layers(), torch.device("meta"):
        model = IdolDecoders()
    assert shapes_of(model) == expected(DECODERS)


def test_sdk_decoders_are_the_repack_without_prefixes():
    assert {f"octree.{k}" for k in expected(SDK_OCTREE)} | {f"gs.{k}" for k in expected(SDK_GS)} == set(expected(DECODERS))


def detected_config():
    state = {k: Shaped(v) for k, v in expected(DENOISER).items()}
    return detect.detect(state)


def test_detection_reads_the_sdk_config():
    assert detected_config() == {"image_model": "idol", "model_channels": 1024, "in_channels": 16, "cam_channels": 5,
                                 "cond_channels": 1280, "cond2_channels": 128, "num_blocks": 24, "num_refiner_blocks": 2,
                                 "num_heads": 16, "num_head_channels": 64}


def test_detection_leaves_other_models_alone():
    assert detect.detect({k: Shaped(v) for k, v in expected(DECODERS).items()}) is None


def write_header(path, keys):
    header = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    return path


def test_inspect_reads_only_the_header(tmp_path):
    path = write_header(tmp_path / "idol.safetensors", HEADERS[DENOISER])
    assert inspect_model(path) == "dit"
    assert inspect_model(tmp_path / "missing.safetensors") is None
    assert inspect_model(write_header(tmp_path / "other.safetensors", ["some.weight"])) is None


def test_components_name_what_is_missing(tmp_path):
    dit = write_header(tmp_path / "dit.safetensors", HEADERS[DENOISER])
    dinov3 = write_header(tmp_path / "dinov3.safetensors", HEADERS[DINOV3])
    flux2 = write_header(tmp_path / "flux2.safetensors", HEADERS[FLUX2])
    decoders = write_header(tmp_path / "decoders.safetensors", HEADERS[DECODERS])
    with pytest.raises(IdolError, match="DINOv3 image encoder.*Idol decoder"):
        resolve_components(dit, [flux2])
    components = resolve_components(dit, [dinov3, flux2, decoders])
    assert components.dinov3 == str(dinov3) and components.flux2 == str(flux2) and components.decoders == (str(decoders),)


def test_split_sdk_decoders_count_as_the_decoder(tmp_path):
    dit = write_header(tmp_path / "dit.safetensors", HEADERS[DENOISER])
    dinov3 = write_header(tmp_path / "dinov3.safetensors", HEADERS[DINOV3])
    flux2 = write_header(tmp_path / "flux2.safetensors", HEADERS[FLUX2])
    octree = write_header(tmp_path / "decoder.safetensors", HEADERS[SDK_OCTREE])
    with pytest.raises(IdolError, match="Idol decoder"):
        resolve_components(dit, [dinov3, flux2, octree])
    gs = write_header(tmp_path / "decoder_gs.safetensors", HEADERS[SDK_GS])
    assert resolve_components(dit, [dinov3, flux2, octree, gs]).decoders == (str(octree), str(gs))
