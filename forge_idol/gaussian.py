"""The decoded object and its standard 3D Gaussian Splatting PLY (idol-sdk gaussian.py, written without plyfile)."""

import numpy as np
import torch
import torch.nn.functional as F

PLY_FIELDS = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class Gaussian:
    """Built from the Gaussian decoder's raw outputs; the activated values are what renderers and PLY files use.

    xyz [N, 3] positions, features_dc [N, 1, 3] colour (degree-0 spherical harmonics), scaling [N, 3], rotation [N, 4]
    unit quaternions (w, x, y, z), opacity [N, 1]. The object is z-up, inside [-0.5, 0.5]^3."""

    def __init__(self, raw_xyz, features_dc, raw_scaling, raw_rotation, raw_opacity,
                 scaling_bias: float, opacity_bias: float, minimum_kernel_size: float):
        device = raw_xyz.device
        aabb = torch.tensor([-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        scaling_offset = inverse_softplus(torch.tensor(scaling_bias)).to(device)
        opacity_offset = inverse_sigmoid(torch.tensor(opacity_bias)).to(device)
        rotation_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        self.xyz = raw_xyz * aabb[None, 3:] + aabb[None, :3]
        self.features_dc = features_dc
        scaling = F.softplus(raw_scaling + scaling_offset)
        self.scaling = torch.sqrt(torch.square(scaling) + minimum_kernel_size ** 2)
        self.rotation = F.normalize(raw_rotation + rotation_offset[None, :])
        self.opacity = torch.sigmoid(raw_opacity + opacity_offset)

    def to_ply_elements(self) -> np.ndarray:
        """One structured row per Gaussian; the z-up object is turned to the y-up convention viewers use."""
        xyz = self.xyz.detach().float().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.detach().float().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # clamp before the inverse activations: a saturated opacity or a zero scale would be written as +/-inf
        eps = 1e-6
        opacities = inverse_sigmoid(self.opacity.float().clamp(eps, 1 - eps)).detach().cpu().numpy()
        scale = torch.log(self.scaling.float().clamp_min(eps)).detach().cpu().numpy()
        rotation = self.rotation.detach().float().cpu().numpy()

        # z-up -> y-up: rotate +90 degrees about x, i.e. (x, y, z) -> (x, -z, y)
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
        rotation = quaternion_multiply(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), rotation)

        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1).astype(np.float32)
        elements = np.empty(attributes.shape[0], dtype=[(name, "<f4") for name in PLY_FIELDS])
        for column, name in enumerate(PLY_FIELDS):
            elements[name] = attributes[:, column]
        return elements

    def save_ply(self, path) -> None:
        write_ply(self.to_ply_elements(), path)

    def save_spz(self, path) -> None:
        from .spz import spz_bytes

        with open(path, "wb") as stream:
            stream.write(spz_bytes(self.to_ply_elements()))


def write_ply(elements: np.ndarray, path) -> None:
    properties = "".join(f"property float {name}\n" for name in PLY_FIELDS)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(elements)}\n{properties}end_header\n"
    with open(path, "wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(elements.tobytes())


def read_ply(path) -> np.ndarray:
    """The rows of a PLY this extension wrote."""
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    return np.frombuffer(data, dtype=[(name, "<f4") for name in PLY_FIELDS], offset=end)


def quaternion_multiply(a, b):
    """Hamilton product a * b of (w, x, y, z) quaternions; a is one quaternion, b is (N, 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)
