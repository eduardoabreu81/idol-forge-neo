"""Forge Neo discovers this always-on script when the extension is installed."""

import sys
import traceback
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import script_callbacks, scripts

from forge_idol import integration, ui
from forge_idol.native import compat

ENABLED = False

_problems = compat.check()

if _problems:
    compat.report(_problems)
else:
    try:
        from forge_idol.native import patches

        patches.apply()
        ENABLED = True
    except Exception:
        print("[Idol] failed to enable, Forge Neo is unchanged:")
        traceback.print_exc()


class Script(scripts.Script):
    sorting_priority = 10
    create_group = False

    def title(self):
        return "Idol"

    def show(self, is_img2img):
        # an object is made from a picture, so the panel lives in img2img alone
        return scripts.AlwaysVisible if ENABLED and is_img2img else False

    def ui(self, is_img2img):
        self.panel = ui.Panel()
        self.infotext_fields = [(self.panel.mode, "Idol Input"), (self.panel.gaussians, "Idol Gaussians"),
                                (self.panel.turbo, "Idol Turbo")]
        return self.panel.inputs

    def before_process(self, p, *args):
        integration.script_before_process(p, *args)

    def process(self, p, *args):
        integration.process(p)

    def process_batch(self, p, *args, **kwargs):
        integration.process_batch(p, kwargs["seeds"])

    def process_before_every_sampling(self, p, *args, **kwargs):
        integration.before_sampling(p, kwargs["noise"])

    def postprocess(self, p, processed, *args):
        integration.postprocess(p, processed)


script_callbacks.on_before_ui(ui.reset)
script_callbacks.on_after_component(ui.capture)
script_callbacks.on_ui_tabs(ui.bind_all)
script_callbacks.on_app_started(ui.check_bindings)
script_callbacks.on_script_unloaded(ui.reset)
