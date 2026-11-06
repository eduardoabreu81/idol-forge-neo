"""Generation contracts independent of Forge, CUDA and model libraries."""

from dataclasses import dataclass

# the object latent and the input camera the denoiser works on (idol-sdk config.json)
Q_TOKENS = 8192
LATENT_CHANNELS = 16
CAMERA_CHANNELS = 5
IMAGE_SIZE = 1024

# Euler + Simple reproduce the SDK's time grid; CFG 4 is the SDK's guidance 3, written (1 + w) * cond - w * uncond
SAMPLER = "Euler"
SCHEDULER = "Simple"
STEPS = 50
CFG = 4.0
SHIFT = 3.0
TURBO_STEPS = 5
TURBO_CFG = 1.0

# Idol ignores the prompt, but sd-dynamic-prompts raises StopIteration on an empty one, on every generation
PLACEHOLDER_PROMPT = "3D object"

MIN_GAUSSIANS = 32768
MAX_GAUSSIANS = 1048576
GAUSSIANS_PER_ANCHOR = 32
DEFAULT_GAUSSIANS = 262144

# square crop side = object bounding box x margin; renders and generated images fill more of the frame than photos
CROP_MARGIN = {"synthetic": 1.2, "real": 1.7}
DEFAULT_MODE = "synthetic"

# what a generation can save; the .json with its settings and camera always is
FORMATS = (".ply", ".spz", ".glb")
# the mesh takes time and is rarely wanted, so it is asked for
DEFAULT_FORMATS = (".ply", ".spz")


class IdolError(RuntimeError):
    """An actionable configuration or generation failure, already saying what to change.

    Forge's main thread prints the traceback of whatever stops a generation, which says nothing to the person reading
    the message. traceback.print_exc reads it from this attribute (Python 3.12 and later; before that the traceback
    still prints, which is harmless): the console gets the one line, and Python itself is not affected."""

    @property
    def __traceback__(self):
        return None

    @__traceback__.setter
    def __traceback__(self, value):
        pass


# Forge logs and swallows exceptions raised in script callbacks; a rejected request is kept here and raised again from
# the model, which Forge calls directly
_pending_error = None


def set_pending_error(error):
    global _pending_error
    _pending_error = error


def raise_pending_error():
    """Once only: the panel is not in txt2img, so nothing else would clear it before a generation there."""
    global _pending_error
    if _pending_error is not None:
        error, _pending_error = _pending_error, None
        raise error


def align_gaussians(value) -> int:
    """32,768 to 1,048,576, rounded down to a multiple of 32."""
    value = max(MIN_GAUSSIANS, min(MAX_GAUSSIANS, int(value)))
    return value // GAUSSIANS_PER_ANCHOR * GAUSSIANS_PER_ANCHOR


def crop_margin(mode: str) -> float:
    if mode not in CROP_MARGIN:
        names = " or ".join(CROP_MARGIN)
        raise IdolError(f"Unknown Idol input type {mode}: use {names}.")
    return CROP_MARGIN[mode]


@dataclass
class GenerationRequest:
    image: object  # the PIL image as the user gave it, alpha included: Forge flattens its own copy
    mode: str
    num_gaussians: int
    turbo: bool
    seed: int = 0
    formats: tuple = DEFAULT_FORMATS
    mesh_resolution: int = 256
    mesh_smooth: int = 0
