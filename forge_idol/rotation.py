"""Turning a finished object: no model involved, only its Gaussians' positions and orientations.

The angles are about the axes a viewer shows, y-up: X to the right, Y up, Z towards the viewer, applied in that order, in
degrees. The PLY's own frame is y-down (right, down, forward), the same axes with Y and Z reversed.
"""

import math

import numpy as np

from .gaussian import quaternion_multiply


def _half_angle(degrees: float) -> tuple[float, float]:
    half = math.radians(degrees) / 2
    return math.cos(half), math.sin(half)


def rotation_quaternion(x: float, y: float, z: float) -> np.ndarray:
    """The turn as a (w, x, y, z) quaternion in the PLY's frame."""
    cos, sin = _half_angle(x)
    turn = np.array([cos, sin, 0.0, 0.0])
    cos, sin = _half_angle(y)
    turn = quaternion_multiply(np.array([cos, 0.0, -sin, 0.0]), turn[None])[0]
    cos, sin = _half_angle(z)
    return quaternion_multiply(np.array([cos, 0.0, 0.0, -sin]), turn[None])[0]


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotate_elements(elements: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    """A new set of PLY rows with the object turned; colour, opacity and each Gaussian's own scales do not change."""
    turn = rotation_quaternion(x, y, z)
    result = elements.copy()
    positions = np.stack([elements["x"], elements["y"], elements["z"]], axis=1).astype(np.float64)
    positions = positions @ quaternion_matrix(turn).T
    for column, name in enumerate(("x", "y", "z")):
        result[name] = positions[:, column]
    quaternions = np.stack([elements[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    quaternions = quaternion_multiply(turn, quaternions)
    for column in range(4):
        result[f"rot_{column}"] = quaternions[:, column]
    return result
