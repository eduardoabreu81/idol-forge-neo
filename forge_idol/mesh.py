"""A coloured triangle mesh from the finished object, written as a binary glTF (.glb).

The Gaussians are splatted into a density grid and a colour grid, the hollow inside is filled, Surface Nets extracts the
surface at a level Otsu's method picks, detached floaters go away, Taubin smoothing is optional, and each vertex samples
its colour. A port of ComfyUI's SplatToMesh (comfy_extras/nodes_gaussian_splat.py, GPL-3.0), without its progress reporting
and GPU shortcuts, and with solid_field between the density and the surface.

The surface is a sampled shell, good for blocking out and for reference; it needs retopology for production. The object's
PLY rows go in, y-down; the mesh comes out y-up, as glTF defines.
"""

import json
import struct
from dataclasses import dataclass

import numpy as np
import torch

from .contracts import IdolError

SH_C0 = 0.28209479177387814

DEFAULT_RESOLUTION = 256
MIN_RESOLUTION = 64
MAX_RESOLUTION = 512
DEFAULT_SMOOTH = 0
MAX_SMOOTH = 60
# a Gaussian's half width in voxels, level over Otsu's, the smallest piece kept in vertices, the faintest Gaussian used, colour
# sharpening
KERNEL = 5
LEVEL_BIAS = 0.07
MIN_COMPONENT = 500
MIN_OPACITY = 0.02
COLOR_SHARPEN = 2.0
CHUNK = 4096
# in voxels: the blur of the density, the gaps closed, and the blurs of the surface and of the colours
DENSITY_BLUR = 1.0
CLOSING = 2
SURFACE_BLUR = 1.0
COLOUR_BLUR = 1.0
FIELD_MARGIN = 4
MIN_THICKNESS = 0.6


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float32, y-up
    faces: np.ndarray     # (F, 3) int64, counter-clockwise from outside
    colors: np.ndarray    # (V, 3) float32, linear


def _inverse_covariance(scales: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Each Gaussian's inverse covariance (N, 3, 3); the scales get a floor, and quaternions are (w, x, y, z)."""
    w, x, y, z = quaternions.unbind(-1)
    rotation = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    inverse_variance = 1.0 / scales.clamp_min(1e-8).square()
    return (rotation * inverse_variance[:, None, :]) @ rotation.transpose(1, 2)


def _splat_density(xyz, opacity, scales, quaternions, rgb, resolution, device):
    """Each Gaussian as its oriented disk, out to 3 sigma, into a density grid and a colour grid. Colour weighs by
    w^COLOR_SHARPEN, which leans each voxel towards its dominant Gaussian. -> density, colour, normaliser, origin, voxel"""
    pad = 4.0 * scales.median()
    low = xyz.amin(0) - pad
    high = xyz.amax(0) + pad
    voxel = ((high - low).max() / resolution).clamp_min(1e-8)
    dx, dy, dz = (torch.ceil((high - low) / voxel).long() + 1).tolist()

    # most Gaussians are flat and far thinner than a voxel: at their real thickness they fall between the grid's sample
    # points, and a flat face lined up with the grid disappears altogether
    scales = scales.clamp_min(MIN_THICKNESS * voxel)
    inverse = _inverse_covariance(scales, quaternions)
    widths = torch.ceil(3.0 * scales.amax(-1) / voxel).long().clamp(1, KERNEL)
    density = torch.zeros(dx * dy * dz, device=device)
    colour = torch.zeros(dx * dy * dz, 3, device=device, dtype=torch.float16)
    normaliser = torch.zeros(dx * dy * dz, device=device, dtype=torch.float16)
    for width in range(1, KERNEL + 1):
        chosen = (widths == width).nonzero(as_tuple=True)[0]
        if chosen.numel() == 0:
            continue
        steps = torch.arange(-width, width + 1, device=device, dtype=torch.float32)
        offsets = torch.stack(torch.meshgrid(steps, steps, steps, indexing="ij"), -1).reshape(-1, 3)
        for start in range(0, chosen.numel(), CHUNK):
            batch = chosen[start:start + CHUNK]
            centres = xyz[batch]
            index = ((centres - low) / voxel).round()[:, None, :] + offsets[None]
            distance = (low + index * voxel) - centres[:, None, :]
            quadratic = torch.einsum("bmi,bij,bmj->bm", distance, inverse[batch], distance)
            weight = opacity[batch, None] * torch.exp(-0.5 * quadratic)
            weight = torch.where(quadratic < 9.0, weight, torch.zeros_like(weight))
            cells = index.long()
            flat = (cells[..., 0].clamp(0, dx - 1) * (dy * dz) + cells[..., 1].clamp(0, dy - 1) * dz
                    + cells[..., 2].clamp(0, dz - 1)).reshape(-1)
            density.index_add_(0, flat, weight.reshape(-1))
            sharpened = weight.pow(COLOR_SHARPEN)
            colour.index_add_(0, flat, (sharpened[..., None] * rgb[batch, None, :]).reshape(-1, 3).to(torch.float16))
            normaliser.index_add_(0, flat, sharpened.reshape(-1).to(torch.float16))
    return (density.reshape(dx, dy, dz), colour.reshape(dx, dy, dz, 3), normaliser.reshape(dx, dy, dz), low, float(voxel))


def otsu_level(values: np.ndarray, bins: int = 256) -> float:
    """The value that best splits inside from outside: the largest variance between the two classes."""
    histogram, edges = np.histogram(values, bins=bins)
    histogram = histogram.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) * 0.5
    below = np.cumsum(histogram)
    total = np.cumsum(histogram * centres)
    above = below[-1] - below
    mean_below = total / np.where(below > 0, below, 1.0)
    mean_above = (total[-1] - total) / np.where(above > 0, above, 1.0)
    between = below * above * (mean_below - mean_above) ** 2
    between[(below <= 0) | (above <= 0)] = -1.0
    return float(centres[int(np.argmax(between))])


def solid_field(volume: torch.Tensor) -> tuple:
    """A smooth inside/outside field, whose surface is at 0.5, and its margin in voxels: the solid can grow past the grid
    of a thin object, and a surface that met the grid's edge would stay open. The Gaussians sit on the surface and are about a voxel apart,
    fewer on the sides the picture did not show, so their density is a hollow, pitted shell: extracted directly, it gives
    about four times the faces, with an inner wall joined to the outer one and a lumpy back. A blur evens out the density,
    a low level and closing seal the gaps, filling makes it solid, and a second blur takes off the voxel steps."""
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

    density = gaussian_filter(volume.cpu().numpy(), sigma=DENSITY_BLUR)
    occupied = density[density > density.max() * 1e-3]
    if occupied.size == 0:
        raise IdolError("Idol could not make a mesh: the object has no solid surface.")
    margin = CLOSING + FIELD_MARGIN
    solid = np.pad(density >= otsu_level(occupied) * LEVEL_BIAS, margin)
    solid = binary_fill_holes(binary_closing(solid, iterations=CLOSING))
    field = gaussian_filter(solid.astype(np.float32), sigma=SURFACE_BLUR)
    return torch.from_numpy(field).to(volume.device), margin


def surface_nets(volume: torch.Tensor, level: float, voxel: float, origin: torch.Tensor):
    """One vertex per cell the surface crosses, at the mean of its edge crossings; a quad per crossed grid edge, wound
    counter-clockwise from the low density side. -> vertices (V, 3), faces (F, 3)"""
    empty = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    device = volume.device
    dx, dy, dz = volume.shape
    if min(dx, dy, dz) < 2:
        return empty
    inside = volume >= level
    corners = [inside[ox:ox + dx - 1, oy:oy + dy - 1, oz:oz + dz - 1]
               for ox, oy, oz in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))]
    any_inside = corners[0] | corners[1] | corners[2] | corners[3] | corners[4] | corners[5] | corners[6] | corners[7]
    all_inside = corners[0] & corners[1] & corners[2] & corners[3] & corners[4] & corners[5] & corners[6] & corners[7]
    active = any_inside & ~all_inside
    count = int(active.sum())
    if count == 0:
        return empty
    del any_inside, all_inside, corners

    cells = active.nonzero(as_tuple=False)
    offsets = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], device=device)
    edges = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]], device=device)
    start_corner, end_corner = edges[:, 0], edges[:, 1]
    start_offset, end_offset = offsets.float()[start_corner], offsets.float()[end_corner]
    local = []
    for start in range(0, count, 1 << 18):
        grid = cells[start:start + (1 << 18), None, :] + offsets[None]
        values = volume[grid[..., 0], grid[..., 1], grid[..., 2]]
        above = values >= level
        low, high = values[:, start_corner], values[:, end_corner]
        crossing = (above[:, start_corner] != above[:, end_corner])[..., None].float()
        difference = high - low
        t = torch.where(difference.abs() > 1e-12, (level - low) / difference, torch.full_like(difference, 0.5)).clamp(0, 1)
        points = torch.lerp(start_offset, end_offset, t[..., None])
        local.append((points * crossing).sum(1) / crossing.sum(1).clamp_min(1.0))
    vertices = origin.to(device) + (cells.float() + torch.cat(local, 0)) * voxel

    ids = torch.full((dx - 1, dy - 1, dz - 1), -1, dtype=torch.int32, device=device)
    ids[active] = torch.arange(count, dtype=torch.int32, device=device)
    faces = []

    def emit(crossed, low_inside, a, b, d, c):
        valid = crossed & (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        if not bool(valid.any()):
            return
        a, b, c, d, low_inside = a[valid], b[valid], c[valid], d[valid], low_inside[valid]
        second, fourth = torch.where(low_inside, b, c), torch.where(low_inside, c, b)
        faces.append(torch.stack([a, second, d], 1))
        faces.append(torch.stack([a, d, fourth], 1))

    low = inside[0:dx - 1, 1:dy - 1, 1:dz - 1]
    emit(low != inside[1:dx, 1:dy - 1, 1:dz - 1], low, ids[:, 0:dy - 2, 0:dz - 2], ids[:, 1:dy - 1, 0:dz - 2],
         ids[:, 1:dy - 1, 1:dz - 1], ids[:, 0:dy - 2, 1:dz - 1])
    low = inside[1:dx - 1, 0:dy - 1, 1:dz - 1]
    emit(low != inside[1:dx - 1, 1:dy, 1:dz - 1], low, ids[0:dx - 2, :, 0:dz - 2], ids[0:dx - 2, :, 1:dz - 1],
         ids[1:dx - 1, :, 1:dz - 1], ids[1:dx - 1, :, 0:dz - 2])
    low = inside[1:dx - 1, 1:dy - 1, 0:dz - 1]
    emit(low != inside[1:dx - 1, 1:dy - 1, 1:dz], low, ids[0:dx - 2, 0:dy - 2, :], ids[1:dx - 1, 0:dy - 2, :],
         ids[1:dx - 1, 1:dy - 1, :], ids[0:dx - 2, 1:dy - 1, :])
    if not faces:
        return empty
    return vertices.cpu().numpy().astype(np.float32), torch.cat(faces, 0).cpu().numpy().astype(np.int64)


def clean_components(vertices: np.ndarray, faces: np.ndarray, min_vertices: int = MIN_COMPONENT):
    """Drop floaters, and the inner shell: density around a surface of flat Gaussians gives a double wall, the inner one
    enclosed by the outer and wound the other way."""
    from scipy.ndimage import maximum, minimum
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    count = len(vertices)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    total, labels = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count)), directed=False)
    face_labels = labels[faces[:, 0]]
    keep = np.bincount(labels, minlength=total) >= min_vertices
    if keep.sum() > 1:
        face_counts = np.bincount(face_labels, minlength=total)
        largest = np.where(keep, face_counts, -1).argmax()
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        volume = np.bincount(face_labels, weights=np.einsum("ij,ij->i", a, np.cross(b, c)), minlength=total)
        index = np.arange(total)
        lows = np.stack([minimum(vertices[:, axis], labels, index) for axis in range(3)], 1)
        highs = np.stack([maximum(vertices[:, axis], labels, index) for axis in range(3)], 1)
        tolerance = 1e-4 * (highs[largest] - lows[largest]).max()
        enclosed = (lows >= lows[largest] - tolerance).all(1) & (highs <= highs[largest] + tolerance).all(1)
        keep &= ~(enclosed & (np.sign(volume) != np.sign(volume[largest])) & (index != largest))
    faces = faces[keep[face_labels]]
    if len(faces) == 0:
        return vertices[:0], faces
    used = np.unique(faces)
    remap = np.full(count, -1, np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def taubin_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Smooths the surface without the shrinkage of a plain Laplacian: the mu pass gives back what the lambda pass took."""
    if iterations <= 0 or len(vertices) == 0 or len(faces) == 0:
        return vertices
    from scipy.sparse import coo_matrix

    count = len(vertices)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    edges = np.concatenate([edges, edges[:, ::-1]], 0)
    adjacency = coo_matrix((np.ones(len(edges), np.float32), (edges[:, 0], edges[:, 1])), shape=(count, count)).tocsr()
    adjacency.data[:] = 1.0
    degree = np.clip(np.asarray(adjacency.sum(1)).ravel(), 1.0, None).astype(np.float32)[:, None]
    result = vertices.astype(np.float32)
    for _ in range(int(iterations)):
        for factor in (lam, mu):
            result = result + np.float32(factor) * ((adjacency @ result) / degree - result)
    return np.ascontiguousarray(result)


def _srgb_to_linear(colours: np.ndarray) -> np.ndarray:
    colours = np.clip(colours, 0, 1)
    return np.where(colours <= 0.04045, colours / 12.92, ((colours + 0.055) / 1.055) ** 2.4).astype(np.float32)


def mesh_from_elements(elements: np.ndarray, resolution: int = DEFAULT_RESOLUTION, smooth: int = DEFAULT_SMOOTH,
                       device: str = "cpu") -> Mesh:
    """The mesh of an object's PLY rows. Raises IdolError when there is no surface to extract."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    opacity = torch.sigmoid(torch.from_numpy(elements["opacity"].astype(np.float32)))
    keep = opacity >= MIN_OPACITY
    if not bool(keep.any()):
        raise IdolError("Idol could not make a mesh: every Gaussian of this object is almost transparent.")

    def column(*names):
        return torch.from_numpy(np.stack([elements[name] for name in names], 1).astype(np.float32))[keep].to(device)

    xyz = column("x", "y", "z")
    scales = column("scale_0", "scale_1", "scale_2").exp()
    quaternions = column("rot_0", "rot_1", "rot_2", "rot_3")
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rgb = (column("f_dc_0", "f_dc_1", "f_dc_2") * SH_C0 + 0.5).clamp(0, 1)
    opacity = opacity[keep].to(device)

    density, colour, normaliser, origin, voxel = _splat_density(xyz, opacity, scales, quaternions, rgb, resolution, device)
    field, margin = solid_field(density)
    vertices, faces = surface_nets(field, 0.5, voxel, origin - margin * voxel)
    del field
    del density
    if len(faces):
        vertices, faces = clean_components(vertices, faces)
    if len(faces) == 0:
        raise IdolError("Idol could not make a mesh: no surface came out of this object.")
    vertices = taubin_smooth(vertices, faces, smooth)

    # the numerator and normaliser are sampled apart and divided after: empty voxels at the edge do not pull colours to
    # black. A short blur of both lets a vertex that settled between Gaussians still reach them.
    origin_cpu = origin.cpu().numpy()
    coords = ((vertices - origin_cpu) / voxel).T
    colour_grid = colour.float().cpu().numpy()
    normaliser_grid = gaussian_filter(normaliser.float().cpu().numpy(), sigma=COLOUR_BLUR)
    sampled = [map_coordinates(gaussian_filter(colour_grid[..., channel], sigma=COLOUR_BLUR), coords, order=1, mode="nearest")
               for channel in range(3)]
    divisor = map_coordinates(normaliser_grid, coords, order=1, mode="nearest")
    colours = np.stack(sampled, -1) / np.clip(divisor, 1e-8, None)[:, None]

    # y-down to y-up: half a turn about X, a proper rotation, so the winding stays
    vertices = np.ascontiguousarray(vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32))
    return Mesh(vertices, faces, _srgb_to_linear(colours))


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    corners = vertices[faces]
    area_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]).astype(np.float64)
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], area_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.where(lengths > 0, lengths, 1.0)).astype(np.float32)


def glb_bytes(mesh: Mesh) -> bytes:
    """A binary glTF 2.0: positions, normals, vertex colours and indices, with an unlit material, so the colours show as they
    are in viewers that follow it and the normals are there for the ones that light it themselves."""
    positions = mesh.vertices.astype("<f4")
    normals = vertex_normals(positions, mesh.faces).astype("<f4")
    colours = mesh.colors.astype("<f4")
    indices = mesh.faces.astype("<u4").reshape(-1)
    arrays = [(positions, 34962), (normals, 34962), (colours, 34962), (indices, 34963)]
    binary = b""
    views = []
    for array, target in arrays:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": array.nbytes, "target": target})
        binary += array.tobytes()
    count = len(positions)
    gltf = {
        "asset": {"version": "2.0", "generator": "idol-forge-neo"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Idol object"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"name": "Vertex colours", "doubleSided": False,
                       "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1},
                       "extensions": {"KHR_materials_unlit": {}}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3",
             "min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": count, "type": "VEC3"},
            {"bufferView": 3, "componentType": 5125, "count": int(indices.size), "type": "SCALAR"},
        ],
    }
    header = json.dumps(gltf, separators=(",", ":")).encode("utf8")
    header += b" " * (-len(header) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(header) + 8 + len(binary)
    return b"".join([struct.pack("<III", 0x46546C67, 2, total), struct.pack("<II", len(header), 0x4E4F534A), header,
                     struct.pack("<II", len(binary), 0x004E4942), binary])
