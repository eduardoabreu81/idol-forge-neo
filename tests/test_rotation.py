"""Turning a finished object, and the files it makes."""

import numpy as np
import pytest
import torch

from forge_idol import integration
from forge_idol.contracts import DEFAULT_FORMATS, FORMATS, IdolError
from forge_idol.gaussian import Gaussian, read_ply, write_ply
from forge_idol.rotation import quaternion_matrix, rotate_elements, rotation_quaternion


def sample_elements(count=200):
    torch.manual_seed(9)
    gaussian = Gaussian(raw_xyz=torch.rand(count, 3), features_dc=torch.randn(count, 1, 3), raw_scaling=torch.randn(count, 3),
                        raw_rotation=torch.randn(count, 4), raw_opacity=torch.randn(count, 1), scaling_bias=0.004,
                        opacity_bias=0.1, minimum_kernel_size=0.0009)
    return gaussian.to_ply_elements()


def y_up(elements):
    # the PLY is y-down: what a viewer shows has Y and Z reversed
    return np.stack([elements["x"], -elements["y"], -elements["z"]], axis=1)


def covariance(elements):
    # each Gaussian's shape in space, which must turn with the object
    q = np.stack([elements[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scales = np.exp(np.stack([elements[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))
    matrices = np.stack([quaternion_matrix(row) for row in q])
    return (matrices * (scales ** 2)[:, None, :]) @ matrices.transpose(0, 2, 1)


@pytest.mark.parametrize("angles, axis_from, axis_to", [
    # y-up, right-handed: Y up turned 90 degrees about X points towards the viewer (+Z)
    ((90, 0, 0), (0, 1, 0), (0, 0, 1)),
    # Z towards the viewer turned 90 degrees about Y points right (+X)
    ((0, 90, 0), (0, 0, 1), (1, 0, 0)),
    # X right turned 90 degrees about Z points up (+Y)
    ((0, 0, 90), (1, 0, 0), (0, 1, 0)),
])
def test_angles_follow_the_axes_a_viewer_shows(angles, axis_from, axis_to):
    elements = sample_elements(1).copy()
    # y_up((x, y, z)) = axis_from
    elements["x"], elements["y"], elements["z"] = axis_from[0], -axis_from[1], -axis_from[2]
    turned = rotate_elements(elements, *angles)
    assert np.allclose(y_up(turned)[0], axis_to, atol=1e-6)


def test_orientations_turn_with_the_object():
    elements = sample_elements()
    turned = rotate_elements(elements, 30, -70, 110)
    matrix = quaternion_matrix(rotation_quaternion(30, -70, 110))
    assert np.allclose(covariance(turned), matrix @ covariance(elements) @ matrix.T, atol=1e-5)


def test_only_positions_and_orientations_change():
    elements = sample_elements()
    turned = rotate_elements(elements, 45, 45, 45)
    for name in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"):
        assert np.array_equal(turned[name], elements[name])
    assert not np.array_equal(turned["x"], elements["x"])


def test_no_turn_and_a_full_turn_change_nothing():
    elements = sample_elements()
    for angles in ((0, 0, 0), (360, 0, 0), (0, 360, 0), (0, 0, 360)):
        turned = rotate_elements(elements, *angles)
        assert np.allclose(y_up(turned), y_up(elements), atol=1e-5)
        assert np.allclose(covariance(turned), covariance(elements), atol=1e-5)


def test_ply_round_trip(tmp_path):
    elements = sample_elements()
    write_ply(elements, tmp_path / "object.ply")
    assert np.array_equal(read_ply(tmp_path / "object.ply"), elements)


@pytest.fixture
def last_object(tmp_path, monkeypatch):
    source = tmp_path / "idol-1-abc.ply"
    write_ply(sample_elements(), source)
    monkeypatch.setattr(integration, "_latest", str(source))
    monkeypatch.setattr(integration, "_latest_source", str(source))
    monkeypatch.setattr(integration, "_latest_base", tmp_path / "idol-1-abc")
    return tmp_path


def test_rotating_saves_new_files_and_keeps_the_original(last_object):
    before = (last_object / "idol-1-abc.ply").read_bytes()
    viewer, message = integration.rotate_latest(90, 0, -45, list(DEFAULT_FORMATS))
    assert (last_object / "idol-1-abc.ply").read_bytes() == before
    ply = last_object / "idol-1-abc-rotated-x90-y0-z-45.ply"
    assert viewer == str(ply) and ply.is_file() and ply.with_suffix(".spz").is_file()
    assert ply.stem in message
    assert integration.latest_object() == str(ply.resolve()) or ply.name in integration.latest_object()


def test_rotations_start_from_the_original_not_the_last_rotation(last_object):
    integration.rotate_latest(90, 0, 0, [".ply"])
    integration.rotate_latest(90, 0, 0, [".ply"])
    once = read_ply(last_object / "idol-1-abc-rotated-x90-y0-z0.ply")
    assert np.allclose(y_up(once), y_up(rotate_elements(sample_elements(), 90, 0, 0)), atol=1e-5)


def test_only_the_chosen_formats_are_saved(last_object):
    viewer, _ = integration.rotate_latest(0, 90, 0, [".spz"])
    assert (last_object / "idol-1-abc-rotated-x0-y90-z0.spz").is_file()
    assert not (last_object / "idol-1-abc-rotated-x0-y90-z0.ply").exists()
    # the viewer still needs a PLY: a temporary one, not in the output folder
    assert viewer.endswith(".ply") and "idol-forge-neo" in viewer


def test_rotating_needs_an_object_and_a_format(last_object, monkeypatch):
    with pytest.raises(IdolError, match="at least one"):
        integration.rotate_latest(0, 0, 0, [])
    monkeypatch.setattr(integration, "_latest_source", None)
    with pytest.raises(IdolError, match="Generate an object first"):
        integration.rotate_latest(0, 0, 0, [".ply"])


def test_formats_are_checked():
    assert integration.chosen_formats([".spz"]) == (".spz",)
    assert integration.chosen_formats(None) == DEFAULT_FORMATS
    assert integration.chosen_formats(list(reversed(FORMATS))) == FORMATS
    with pytest.raises(IdolError, match="at least one"):
        integration.chosen_formats([])
    with pytest.raises(IdolError, match=".obj"):
        integration.chosen_formats([".obj"])
