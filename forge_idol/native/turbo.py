"""Idol Turbo: speridlabs/idol-turbo's two phase LoRAs on the denoiser's Linears.

The turbo's guidance is folded into the LoRAs, and each covers one part of the schedule: "slow" the high-noise steps, "fast"
the low-noise ones (t >= 0.5 is slow, as in its turbo.py). Forge's LoRA system applies one set of weights for a whole
generation, so the extension adds them here: a forward hook on each Linear adds up(down(x)) of the active phase.
"""

import torch
import torch.nn.functional as F

PHASE_THRESHOLD = 0.5
DOWN = ".lora_down.weight"
UP = ".lora_up.weight"


def lora_pairs(state_dict: dict) -> dict:
    """The (down, up) of each layer in one file of the ComfyUI format, which is scaled for strength 1."""
    pairs = {}
    for key, down in state_dict.items():
        if key.endswith(DOWN):
            name = key[: -len(DOWN)].removeprefix("diffusion_model.")
            pairs[name] = (down.float(), state_dict[key[: -len(DOWN)] + UP].float())
    return pairs


class TurboLoras:
    """Holds both phases and hooks the layers while Turbo is on; detach() removes the hooks, so a generation without
    Turbo runs the plain denoiser."""

    def __init__(self, slow: dict, fast: dict):
        if set(slow) != set(fast) or not slow:
            raise ValueError("the slow and fast Idol Turbo LoRAs do not cover the same layers")
        self.phases = {"slow": slow, "fast": fast}
        self.active = None
        self.handles = []
        self.on_device = {}

    def attach(self, model) -> None:
        if self.handles:
            return
        for name in self.phases["slow"]:
            self.handles.append(model.get_submodule(name).register_forward_hook(self._hook(name)))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.on_device = {}
        self.active = None

    def select(self, timestep: torch.Tensor) -> None:
        self.active = "slow" if float(timestep.flatten()[0]) / 1000.0 >= PHASE_THRESHOLD else "fast"

    def _weights(self, name, device):
        key = (self.active, name)
        weights = self.on_device.get(key)
        if weights is None or weights[0].device != device:
            down, up = self.phases[self.active][name]
            weights = (down.to(device), up.to(device))
            self.on_device[key] = weights
        return weights

    def _hook(self, name):
        def add_lora(module, inputs, output):
            if self.active is None:
                return output
            x = inputs[0]
            down, up = self._weights(name, x.device)
            return output + F.linear(F.linear(x.float(), down), up).to(output.dtype)

        return add_lora


def set_turbo(model, turbo: "TurboLoras | None") -> None:
    """Turn the turbo on or off for the next generation."""
    current = getattr(model, "turbo", None)
    if current is not None and current is not turbo:
        current.detach()
    if turbo is not None:
        turbo.attach(model)
    model.turbo = turbo
