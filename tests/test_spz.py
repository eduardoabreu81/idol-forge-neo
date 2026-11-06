"""The .spz against a decoder written from the format's own loader (load-spz.cc), independent of the writer."""

import gzip
import math
import struct

import numpy as np
import torch

from forge_idol.gaussian import Gaussian
from forge_idol.spz import COLOR_SCALE, spz_bytes


def decode(data):
    raw = gzip.decompress(data)
    magic, version, count, sh_degree, fractional_bits, flags, reserved = struct.unpack_from("<IIIBBBB", raw)
    assert (magic, version, sh_degree, flags, reserved) == (0x5053474E, 3, 0, 0, 0)
    offset = 16

    def take(size):
        nonlocal offset
        block = np.frombuffer(raw, np.uint8, count=size, offset=offset)
        offset += size
        return block

    b = take(count * 9).reshape(count, 3, 3).astype(np.int64)
    fixed = (b[..., 2] << 16) | (b[..., 1] << 8) | b[..., 0]
    positions = np.where(fixed & 0x800000, fixed - 0x1000000, fixed) / (1 << fractional_bits)
    alphas = take(count) / 255.0
    colors = (take(count * 3).reshape(count, 3) / 255.0 - 0.5) / COLOR_SCALE
    scales = take(count * 3).reshape(count, 3) / 16.0 - 10.0
    packed = take(count * 4).reshape(count, 4).astype(np.int64)
    assert offset == len(raw)

    quaternions = np.zeros((count, 4))
    for row in range(count):
        word = int(packed[row, 0] | packed[row, 1] << 8 | packed[row, 2] << 16 | packed[row, 3] << 24)
        largest = word >> 30
        total = 0.0
        for component in (3, 2, 1, 0):
            if component == largest:
                continue
            value = (word & 0x1FF) / 511.0 * math.sqrt(0.5)
            if word >> 9 & 1:
                value = -value
            word >>= 10
            quaternions[row, component] = value
            total += value * value
        quaternions[row, largest] = math.sqrt(max(0.0, 1.0 - total))
    return positions, alphas, colors, scales, quaternions


def sample_object(count=400):
    torch.manual_seed(5)
    return Gaussian(raw_xyz=torch.rand(count, 3), features_dc=torch.randn(count, 1, 3) * 0.8,
                    raw_scaling=torch.randn(count, 3) * 0.5, raw_rotation=torch.randn(count, 4),
                    raw_opacity=torch.randn(count, 1), scaling_bias=0.004, opacity_bias=0.1, minimum_kernel_size=0.0009)


def test_every_attribute_survives_within_its_quantization():
    elements = sample_object().to_ply_elements()
    positions, alphas, colors, scales, _ = decode(spz_bytes(elements))

    # y and z flip between the PLY and SPZ's y-up frame
    expected = np.stack([elements["x"], -elements["y"], -elements["z"]], axis=1)
    assert np.abs(positions - expected).max() <= 0.5 / 4096 + 1e-6
    assert np.abs(alphas - 1 / (1 + np.exp(-elements["opacity"].astype(np.float64)))).max() <= 0.5 / 255 + 1e-6
    wanted = np.stack([elements[f"f_dc_{i}"] for i in range(3)], axis=1)
    inside = np.abs(wanted).max(axis=1) < 3.0
    assert inside.any()
    assert np.abs(colors - wanted)[inside].max() <= 0.5 / (255 * COLOR_SCALE) + 1e-6
    wanted = np.stack([elements[f"scale_{i}"] for i in range(3)], axis=1)
    inside = (wanted > -10).all(axis=1) & (wanted < 5.9).all(axis=1)
    assert inside.any()
    assert np.abs(scales - wanted)[inside].max() <= 0.5 / 16 + 1e-6


def test_rotations_are_the_same_rotation_in_the_flipped_frame():
    elements = sample_object().to_ply_elements()
    *_, quaternions = decode(spz_bytes(elements))
    expected = np.stack([elements["rot_1"], -elements["rot_2"], -elements["rot_3"], elements["rot_0"]], axis=1).astype(np.float64)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    # q and -q are the same rotation
    agreement = np.abs((quaternions * expected).sum(axis=1))
    assert agreement.min() > 0.9995


def test_values_outside_the_range_saturate_instead_of_wrapping():
    elements = sample_object(4).to_ply_elements()
    elements = elements.copy()
    elements["f_dc_0"] = 1000.0
    elements["scale_0"] = 1000.0
    elements["opacity"] = 1000.0
    _, alphas, colors, scales, _ = decode(spz_bytes(elements))
    assert (alphas == 1.0).all()
    assert (colors[:, 0] == (1.0 - 0.5) / COLOR_SCALE).all()
    assert (scales[:, 0] == 255 / 16 - 10).all()


def test_save_spz_writes_the_file_a_good_deal_smaller_than_the_ply(tmp_path):
    gaussian = sample_object(2000)
    gaussian.save_ply(tmp_path / "object.ply")
    gaussian.save_spz(tmp_path / "object.spz")
    assert (tmp_path / "object.spz").read_bytes()[:2] == b"\x1f\x8b"
    assert (tmp_path / "object.spz").stat().st_size < (tmp_path / "object.ply").stat().st_size / 2
