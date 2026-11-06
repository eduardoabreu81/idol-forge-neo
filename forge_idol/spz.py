"""The object as a .spz: Niantic's compressed Gaussian splat format, about a fifth of the PLY's size, for the web.

Version 3 of the format, the last gzip one: viewers read it widely, and it keeps rotations more accurately than version 2. A
16-byte header, then each attribute for all the points: positions as 24-bit fixed point, alphas, colours, scales and
rotations as "smallest three" quaternions (spz's load-spz.cc, packGaussians). SPZ stores y-up (right, up, back), while
the PLY is y-down (right, down, forward): y and z flip, in the positions and in the quaternions' vector part.
"""

import gzip
import struct

import numpy as np

MAGIC = 0x5053474E  # "NGSP"
VERSION = 3
FRACTIONAL_BITS = 12
COLOR_SCALE = 0.15
SQRT1_2 = 0.7071067811865476


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def pack_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Unit quaternions (N, 4) x, y, z, w -> (N, 4) bytes: the index of the largest component in the top 2 bits, then for
    each other component, in order, a sign bit and 9 bits of magnitude. -q is the same rotation, so the largest needs no
    sign and is rebuilt from the other three."""
    largest = np.abs(quaternions).argmax(axis=1)
    negate = quaternions[np.arange(len(quaternions)), largest] < 0
    packed = largest.astype(np.uint32)
    for component in range(4):
        values = quaternions[:, component]
        sign = ((values < 0) ^ negate).astype(np.uint32)
        magnitude = np.clip((511.0 * (np.abs(values) / SQRT1_2) + 0.5).astype(np.uint32), 0, 511)
        packed = np.where(largest != component, (packed << 10) | (sign << 9) | magnitude, packed)
    return packed.astype("<u4")[:, None].view(np.uint8).reshape(-1, 4)


def spz_bytes(elements: np.ndarray) -> bytes:
    """The PLY's structured rows (gaussian.to_ply_elements) as the bytes of a .spz file."""
    count = len(elements)
    positions = np.stack([elements["x"], -elements["y"], -elements["z"]], axis=1).astype(np.float64)
    fixed = np.clip(np.round(positions * (1 << FRACTIONAL_BITS)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
    position_bytes = np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], axis=-1).astype(np.uint8)

    alphas = _to_uint8(255.0 / (1.0 + np.exp(-elements["opacity"].astype(np.float64))))
    colors = np.stack([elements[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float64)
    colors = _to_uint8(colors * (COLOR_SCALE * 255.0) + 0.5 * 255.0)
    scales = np.stack([elements[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)
    scales = _to_uint8((scales + 10.0) * 16.0)

    # the PLY holds w, x, y, z
    quaternions = np.stack([elements["rot_1"], -elements["rot_2"], -elements["rot_3"], elements["rot_0"]], axis=1).astype(np.float64)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    header = struct.pack("<IIIBBBB", MAGIC, VERSION, count, 0, FRACTIONAL_BITS, 0, 0)
    body = [header, position_bytes.tobytes(), alphas.tobytes(), colors.tobytes(), scales.tobytes(),
            pack_quaternions(quaternions).tobytes()]
    return gzip.compress(b"".join(body), compresslevel=6)
