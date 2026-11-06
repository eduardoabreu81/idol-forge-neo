"""The mesh and its .glb, on simple objects made of Gaussians."""

import json
import struct

import numpy as np
import pytest
import torch

from forge_idol.contracts import IdolError
from forge_idol.gaussian import Gaussian
from forge_idol.mesh import (
    Mesh,
    clean_components,
    glb_bytes,
    mesh_from_elements,
    otsu_level,
    surface_nets,
    taubin_smooth,
    vertex_normals,
)


def surfels(positions, normals, colour=(0.8, 0.2, 0.1), thin=0.0009, wide=0.012, opacity=4.0):
    """Flat Gaussians on a surface, as the model makes them: rows of a PLY, y-down."""
    count = len(positions)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(np.tile(z, (count, 1)), normals)
    w = 1.0 + normals @ z
    quaternions = np.concatenate([w[:, None], axis], 1)
    flip = w < 1e-6
    quaternions[flip] = [0.0, 1.0, 0.0, 0.0]
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    elements = np.zeros(count, dtype=[(name, "<f4") for name in (
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3")])
    elements["x"], elements["y"], elements["z"] = positions.T
    for i in range(3):
        elements[f"f_dc_{i}"] = (colour[i] - 0.5) / 0.28209479
    elements["opacity"] = opacity
    elements["scale_0"], elements["scale_1"], elements["scale_2"] = np.log(wide), np.log(wide), np.log(thin)
    for i in range(4):
        elements[f"rot_{i}"] = quaternions[:, i]
    return elements


def sphere(count=20000, radius=0.35, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return surfels(normals * radius, normals, **kwargs)


def plate(side=0.6, thickness=0.05, step=0.006):
    """A flat box lined up with the grid, its faces made of thin Gaussians: they can fall between the grid's points."""
    u = np.arange(-side / 2, side / 2, step)
    a, b = (grid.ravel() for grid in np.meshgrid(u, u))
    faces = []
    for sign in (-1.0, 1.0):
        faces.append(surfels(np.stack([a, b, np.full_like(a, sign * thickness / 2)], 1), np.tile([0.0, 0.0, sign], (len(a), 1)), wide=0.006))
    v = np.arange(-thickness / 2, thickness / 2, step)
    c, d = (grid.ravel() for grid in np.meshgrid(u, v))
    for sign in (-1.0, 1.0):
        faces.append(surfels(np.stack([np.full_like(c, sign * side / 2), c, d], 1), np.tile([sign, 0.0, 0.0], (len(c), 1)), wide=0.006))
        faces.append(surfels(np.stack([c, np.full_like(c, sign * side / 2), d], 1), np.tile([0.0, sign, 0.0], (len(c), 1)), wide=0.006))
    return np.concatenate(faces)


def signed_volume(mesh):
    a, b, c = (mesh.vertices[mesh.faces[:, i]].astype(np.float64) for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def edge_uses(faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0), 1)
    _, counts = np.unique(edges[:, 0] * (int(faces.max()) + 1) + edges[:, 1], return_counts=True)
    return counts


def test_a_shell_of_flat_gaussians_becomes_one_solid_sphere():
    mesh = mesh_from_elements(sphere(), 96)
    expected = 4 / 3 * np.pi * 0.35 ** 3
    # the hollow inside is filled: the outer surface alone, a little fatter than the shell, wound outwards
    assert expected * 0.95 < signed_volume(mesh) < expected * 1.35
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert radii.min() > 0.33 and radii.max() < 0.42
    # about the faces of one surface: not two walls
    assert len(mesh.faces) < 4 * np.pi * (0.38 / (0.72 / 96)) ** 2 * 2.5
    assert (edge_uses(mesh.faces) == 2).all()


def test_flat_faces_lined_up_with_the_grid_do_not_vanish():
    mesh = mesh_from_elements(plate(), 96)
    # front and back are there, with the inside closed
    assert signed_volume(mesh) > 0.6 * 0.6 * 0.05 * 0.7
    z = mesh.vertices[:, 2]
    assert z.max() > 0.02 and z.min() < -0.02
    assert (edge_uses(mesh.faces) == 2).mean() > 0.99


def test_vertices_get_the_colours_of_the_gaussians_in_linear_light():
    mesh = mesh_from_elements(sphere(colour=(0.8, 0.2, 0.1)), 64)
    linear = [((value + 0.055) / 1.055) ** 2.4 if value > 0.04045 else value / 12.92 for value in (0.8, 0.2, 0.1)]
    assert np.allclose(mesh.colors.mean(0), linear, atol=0.01)


def test_the_mesh_is_y_up_and_the_object_y_down():
    # a sphere with a small second piece at the PLY's +y: down there, up in the mesh
    elements = np.concatenate([sphere(), sphere(4000, radius=0.1, seed=2)])
    elements["y"][-4000:] = elements["y"][-4000:] + 0.45
    mesh = mesh_from_elements(elements, 96)
    assert mesh.vertices[:, 1].min() < -0.5
    assert mesh.vertices[:, 1].max() < 0.42


def test_floaters_go_away():
    elements = np.concatenate([sphere(), sphere(30, radius=0.005, seed=3)])
    elements["x"][-30:] += 0.48
    elements["y"][-30:] += 0.48
    mesh = mesh_from_elements(elements, 96)
    assert np.linalg.norm(mesh.vertices, axis=1).max() < 0.42


def test_nothing_solid_is_a_clear_error():
    with pytest.raises(IdolError, match="transparent"):
        mesh_from_elements(sphere(100, opacity=-20.0), 64)


def test_smoothing_keeps_the_volume():
    mesh = mesh_from_elements(sphere(), 96)
    smoothed = Mesh(taubin_smooth(mesh.vertices, mesh.faces, 20), mesh.faces, mesh.colors)
    assert abs(signed_volume(smoothed) - signed_volume(mesh)) / signed_volume(mesh) < 0.03
    assert not np.array_equal(smoothed.vertices, mesh.vertices)


def test_otsu_splits_two_groups():
    values = np.concatenate([np.full(1000, 1.0), np.full(1000, 9.0)])
    assert 1.0 < otsu_level(values) < 9.0


def test_surface_nets_of_a_box_is_closed():
    volume = torch.zeros(12, 12, 12)
    volume[3:9, 3:9, 3:9] = 1.0
    vertices, faces = surface_nets(volume, 0.5, 1.0, torch.zeros(3))
    assert (edge_uses(faces) == 2).all()
    vertices, faces = clean_components(vertices, faces, 1)
    assert len(faces) > 0


def parse_glb(data):
    magic, version, total = struct.unpack_from("<III", data)
    assert (magic, version, total) == (0x46546C67, 2, len(data))
    length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A and length % 4 == 0
    gltf = json.loads(data[20:20 + length])
    binary_length, binary_kind = struct.unpack_from("<II", data, 20 + length)
    assert binary_kind == 0x004E4942 and binary_length % 4 == 0
    return gltf, data[28 + length:28 + length + binary_length]


def test_the_glb_holds_the_mesh_exactly():
    mesh = mesh_from_elements(sphere(), 64)
    gltf, binary = parse_glb(glb_bytes(mesh))
    primitive = gltf["meshes"][0]["primitives"][0]

    def read(accessor_index, dtype, width):
        accessor = gltf["accessors"][accessor_index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        assert view["byteOffset"] % 4 == 0 and view["byteOffset"] + view["byteLength"] <= len(binary)
        return np.frombuffer(binary, dtype, count=accessor["count"] * width, offset=view["byteOffset"]).reshape(-1, width)

    assert np.array_equal(read(primitive["attributes"]["POSITION"], "<f4", 3), mesh.vertices)
    assert np.array_equal(read(primitive["attributes"]["COLOR_0"], "<f4", 3), mesh.colors)
    assert np.array_equal(read(primitive["indices"], "<u4", 1).reshape(-1, 3), mesh.faces)
    position = gltf["accessors"][primitive["attributes"]["POSITION"]]
    assert np.allclose(position["min"], mesh.vertices.min(0)) and np.allclose(position["max"], mesh.vertices.max(0))
    assert "KHR_materials_unlit" in gltf["extensionsUsed"]


def test_normals_point_out_of_a_sphere():
    mesh = mesh_from_elements(sphere(), 64)
    normals = vertex_normals(mesh.vertices, mesh.faces)
    outward = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
    assert (normals * outward).sum(1).mean() > 0.97


def test_gaussian_elements_go_in_as_they_come():
    torch.manual_seed(1)
    gaussian = Gaussian(raw_xyz=torch.rand(50, 3), features_dc=torch.randn(50, 1, 3), raw_scaling=torch.randn(50, 3),
                        raw_rotation=torch.randn(50, 4), raw_opacity=torch.randn(50, 1), scaling_bias=0.004, opacity_bias=0.1,
                        minimum_kernel_size=0.0009)
    elements = gaussian.to_ply_elements()
    try:
        mesh_from_elements(elements, 64)
    except IdolError:
        pass  # a handful of scattered Gaussians may hold no surface; what must not happen is any other failure
