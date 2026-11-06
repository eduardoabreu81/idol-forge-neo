"""A turning GIF of a finished object, with the input picture beside it, for the wiki.

    python tools/turntable.py OBJECT.ply OUT.gif --picture INPUT.png [--mesh] [--frames 36] [--size 360]

Starts at the camera the model predicted, read from the .json next to the .ply. By default it draws the Gaussians themselves,
each with its size, orientation and opacity, back to front; with --mesh it extracts the mesh as the extension does and draws
its triangles, lit from the camera. Plain NumPy on the CPU, slow but with nothing to install: frames run in parallel.
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_idol.gaussian import read_ply  # noqa: E402
from forge_idol.mesh import mesh_from_elements, vertex_normals  # noqa: E402

BACKGROUND = 24 / 255.0
SPAN = 1.15

_scene = {}


def load_gaussians(path):
    rows = read_ply(path)
    w, x, y, z = (rows[f"rot_{i}"].astype(np.float64) for i in range(4))
    rotation = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)
    scale = np.exp(np.stack([rows[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))
    return {
        "xyz": np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64),
        "opacity": 1 / (1 + np.exp(-rows["opacity"].astype(np.float64))),
        "rgb": np.clip(0.5 + 0.28209479 * np.stack([rows[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float64), 0, 1),
        "covariance": (rotation * (scale ** 2)[:, None, :]) @ rotation.transpose(0, 2, 1),
    }


def camera_axes(yaw, pitch):
    """The view in the PLY's frame, which is (x, -z, y) of the z-up object the camera is given in."""
    direction = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    direction = np.array([direction[0], -direction[2], direction[1]])
    world_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(-direction, world_up)
    right /= np.linalg.norm(right)
    return direction, right, np.cross(right, -direction)


def draw_gaussians(scene, yaw, pitch, size):
    direction, right, up = camera_axes(yaw, pitch)
    axes = np.stack([right, up])
    centres = scene["xyz"] @ axes.T
    depth = scene["xyz"] @ direction
    px = (centres[:, 0] / SPAN + 0.5) * size
    py = (0.5 - centres[:, 1] / SPAN) * size
    covariance = axes @ scene["covariance"] @ axes.T * (size / SPAN) ** 2 + np.eye(2) * 0.15
    det = covariance[:, 0, 0] * covariance[:, 1, 1] - covariance[:, 0, 1] ** 2
    inv00, inv11, inv01 = covariance[:, 1, 1] / det, covariance[:, 0, 0] / det, -covariance[:, 0, 1] / det
    radius = np.minimum(np.ceil(3 * np.sqrt(np.maximum(covariance[:, 0, 0], covariance[:, 1, 1]))), 40).astype(int)
    image = np.full((size, size, 3), BACKGROUND)
    opacity, rgb = scene["opacity"], scene["rgb"]
    for i in np.argsort(depth):
        if opacity[i] < 0.02:
            continue
        cx, cy, r = px[i], py[i], radius[i]
        x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, size)
        y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5 - cx, np.arange(y0, y1) + 0.5 - cy)
        # the image's y runs down, so the off-diagonal flips
        d2 = inv00[i] * gx * gx + inv11[i] * gy * gy - 2 * inv01[i] * gx * gy
        alpha = (opacity[i] * np.exp(-0.5 * d2))[..., None]
        image[y0:y1, x0:x1] = image[y0:y1, x0:x1] * (1 - alpha) + rgb[i] * alpha
    return image


def draw_mesh(scene, yaw, pitch, size):
    # the mesh is y-up: (x, y, z) of the PLY's frame with y and z reversed
    direction, right, up = (axis * np.array([1.0, -1.0, -1.0]) for axis in camera_axes(yaw, pitch))
    vertices, faces = scene["vertices"], scene["faces"]
    px = (vertices @ right / SPAN + 0.5) * size
    py = (0.5 - vertices @ up / SPAN) * size
    depth = vertices @ direction
    shade = scene["srgb"] * (np.clip(scene["normals"] @ direction, 0, 1)[:, None] * 0.65 + 0.35)
    image = np.full((size, size, 3), BACKGROUND)
    zbuffer = np.full((size, size), -1e9)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    area = (px[b] - px[a]) * (py[c] - py[a]) - (px[c] - px[a]) * (py[b] - py[a])
    for t in np.nonzero(area < 0)[0]:
        ia, ib, ic = a[t], b[t], c[t]
        xs = (px[ia], px[ib], px[ic])
        ys = (py[ia], py[ib], py[ic])
        for yy in range(max(int(min(ys)), 0), min(int(max(ys)) + 2, size)):
            for xx in range(max(int(min(xs)), 0), min(int(max(xs)) + 2, size)):
                w0 = ((px[ib] - xx) * (py[ic] - yy) - (px[ic] - xx) * (py[ib] - yy)) / area[t]
                w1 = ((px[ic] - xx) * (py[ia] - yy) - (px[ia] - xx) * (py[ic] - yy)) / area[t]
                w2 = 1 - w0 - w1
                if w0 < -0.01 or w1 < -0.01 or w2 < -0.01:
                    continue
                z = w0 * depth[ia] + w1 * depth[ib] + w2 * depth[ic]
                if z > zbuffer[yy, xx]:
                    zbuffer[yy, xx] = z
                    image[yy, xx] = w0 * shade[ia] + w1 * shade[ib] + w2 * shade[ic]
    return image


def _start_worker(path, use_mesh):
    if use_mesh:
        data = np.load(path)
        _scene.update(vertices=data["vertices"], faces=data["faces"], srgb=data["srgb"], normals=data["normals"], mesh=True)
    else:
        _scene.update(load_gaussians(path), mesh=False)


def _frame(job):
    yaw, pitch, size = job
    draw = draw_mesh if _scene["mesh"] else draw_gaussians
    return (np.clip(draw(_scene, yaw, pitch, size), 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply")
    parser.add_argument("gif")
    parser.add_argument("--camera", help="the object's .json, when it is not next to the .ply")
    parser.add_argument("--picture", help="the input picture, shown on the left")
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--size", type=int, default=360)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    camera = json.loads(Path(args.camera or Path(args.ply).with_suffix(".json")).read_text(encoding="utf-8"))["camera"]
    source, temporary = args.ply, None
    if args.mesh:
        mesh = mesh_from_elements(read_ply(args.ply))
        temporary = str(Path(args.gif).with_suffix(".npz"))
        np.savez(temporary, vertices=mesh.vertices, faces=mesh.faces, normals=vertex_normals(mesh.vertices, mesh.faces),
                 srgb=np.clip(1.055 * np.clip(mesh.colors, 0, 1) ** (1 / 2.4) - 0.055, 0, 1))
        source = temporary

    jobs = [(camera["yaw"] + 2 * math.pi * i / args.frames, camera["pitch"], args.size) for i in range(args.frames)]
    with multiprocessing.Pool(args.workers, _start_worker, (source, args.mesh)) as pool:
        frames = pool.map(_frame, jobs)
    if temporary:
        os.remove(temporary)

    left = None
    if args.picture:
        left = Image.open(args.picture).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    images = []
    for frame in frames:
        image = Image.fromarray(frame)
        if left is not None:
            sheet = Image.new("RGB", (args.size * 2, args.size))
            sheet.paste(left, (0, 0))
            sheet.paste(image, (args.size, 0))
            image = sheet
        images.append(image.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    images[0].save(args.gif, save_all=True, append_images=images[1:], duration=80, loop=0, optimize=True, disposal=1)
    print(f"{args.gif}: {os.path.getsize(args.gif) / 2 ** 20:.1f} MB, {args.frames} frames")


if __name__ == "__main__":
    main()
