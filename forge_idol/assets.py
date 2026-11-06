"""The files Idol needs besides the ones chosen in the UI, kept in models/Idol and downloaded once, on first use."""

import os
from pathlib import Path

from .contracts import IdolError

BIREFNET = ("birefnet.safetensors", "https://huggingface.co/speridlabs/idol/resolve/main/birefnet.safetensors")
TURBO_SLOW = ("idol_turbo_slow_comfy.safetensors",
              "https://huggingface.co/speridlabs/idol-turbo/resolve/main/loras/idol_turbo_slow_comfy.safetensors")
TURBO_FAST = ("idol_turbo_fast_comfy.safetensors",
              "https://huggingface.co/speridlabs/idol-turbo/resolve/main/loras/idol_turbo_fast_comfy.safetensors")


def folder() -> Path:
    from modules import paths_internal
    return Path(paths_internal.models_path) / "Idol"


def ensure(asset) -> str:
    """The local path of an asset, downloading it with Forge's own downloader when it is not there yet."""
    from modules import sd_vae_taesd

    name, url = asset
    path = folder() / name
    if not path.is_file():
        os.makedirs(path.parent, exist_ok=True)
        print(f"[Idol] downloading {name} into {path.parent}")
        try:
            sd_vae_taesd.download_model(str(path), url)
        except Exception as e:
            raise IdolError(f"Could not download {name} ({e}). Download it from {url} and put it in {path.parent}.") from e
    return str(path)
