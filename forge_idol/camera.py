"""The input camera the model predicts with the object (idol-sdk camera.py)."""

import math


def decode_camera(values) -> dict:
    """camera latent [5] as floats -> yaw and pitch (radians), radius and vertical field of view (degrees)."""
    dx, dy, dz, inverse_distance, half_width = (float(v) for v in values)
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = (dx / norm, dy / norm, dz / norm) if norm > 1e-6 else (0.0, 0.0, 1.0)
    if inverse_distance < 1e-3:  # orthographic
        radius = 1000.0
        fov = math.degrees(2 * math.atan(half_width / radius))
    else:
        radius = 1.0 / inverse_distance
        fov = math.degrees(2 * math.atan(half_width * inverse_distance))
    return {"yaw": math.atan2(dx, dy), "pitch": math.asin(max(-1.0, min(1.0, dz))), "radius": radius, "fov": fov}
