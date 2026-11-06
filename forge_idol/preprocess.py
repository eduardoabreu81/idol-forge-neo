"""The image the encoders see (idol-sdk preprocess.py)."""

import numpy as np
from PIL import Image, ImageFilter

from .contracts import IMAGE_SIZE, IdolError


def has_real_alpha(image: Image.Image) -> bool:
    return image.mode == "RGBA" and np.array(image.getchannel(3), dtype=np.int32).min() < 255


def prepare_image(image: Image.Image, remove_background, margin: float, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize so the short side is ``size``; if the image has no real alpha, ``remove_background`` predicts one; erode
    the alpha by 1 px so no background-coloured fringe survives; crop a square of side ``margin`` x the object's
    bounding box; resize to ``size``; composite on black. Returns the RGB image."""
    width, height = image.size
    scale = size / min(width, height)
    image = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.LANCZOS)
    if not has_real_alpha(image):
        image = remove_background(image.convert("RGB"))
    image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(3)))
    alpha = np.array(image.getchannel(3))
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        raise IdolError("Idol found no object in the input image: its alpha channel is empty.")
    left, top, right, bottom = xs.min(), ys.min(), xs.max(), ys.max()
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half = max(right - left, bottom - top) / 2 * margin
    image = image.crop([int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)])
    image = image.resize((size, size), Image.LANCZOS)
    composite = Image.new("RGB", (size, size), (0, 0, 0))
    composite.paste(image, mask=image.split()[3])
    return composite
