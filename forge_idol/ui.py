"""Small Idol accordion and native-control events for Forge's Gradio interface."""

import html
import logging

import gradio as gr

from .contracts import (
    CFG,
    DEFAULT_FORMATS,
    DEFAULT_GAUSSIANS,
    DEFAULT_MODE,
    FORMATS,
    GAUSSIANS_PER_ANCHOR,
    MAX_GAUSSIANS,
    MIN_GAUSSIANS,
    SAMPLER,
    SCHEDULER,
    STEPS,
    TURBO_CFG,
    TURBO_STEPS,
    IdolError,
)
from .integration import (
    checkpoint_info,
    latest_object,
    module_paths,
    rotate_latest,
    viewer_path,
)
from .mesh import (
    DEFAULT_RESOLUTION,
    DEFAULT_SMOOTH,
    MAX_RESOLUTION,
    MAX_SMOOTH,
    MIN_RESOLUTION,
)
from .models import inspect_model, resolve_components

COMPONENTS = {}
TAB = "img2img"
NATIVE_CONTROLS = ("steps", "sampling", "scheduler", "cfg_scale")
NATIVE_IDS = tuple(f"{TAB}_{name}" for name in NATIVE_CONTROLS)
INPUT_TYPES = [("Render or generated image", "synthetic"), ("Photo", "real")]
NEEDS_FILES = "Select the DINOv3 image encoder, the FLUX.2 VAE and the Idol decoder in VAE / Text Encoder."
READY = "All the files are selected: add a picture and press Generate."
PANELS = []
logger = logging.getLogger("forge_idol")


def _in_blocks():
    # Forge also creates components outside the UI build (on page load); events can only be bound inside it
    return gr.context.Context.root_block is not None


def capture(component, **kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id:
        COMPONENTS[elem_id] = component
        # whichever of the controls a panel needs comes last, in this Forge's build order, completes the binding
        if elem_id in ("setting_sd_model_checkpoint", "setting_sd_modules", f"html_log_{TAB}") or elem_id in NATIVE_IDS:
            bind_all(elem_id)


def preset_settings():
    """The sampler, schedule and CFG saved for the idol preset in Settings, which start from the SDK's; a CFG of 0 there
    keeps the current one (None), as Forge's presets do."""
    try:
        from modules import shared
        opts = shared.opts
    except (ImportError, AttributeError):
        opts = None
    sampler = getattr(opts, "idol_i2i_sampler", None) or SAMPLER
    scheduler = getattr(opts, "idol_i2i_scheduler", None) or SCHEDULER
    cfg = getattr(opts, "idol_i2i_cfg", None)
    cfg = CFG if cfg is None else cfg
    return sampler, scheduler, cfg if cfg > 0 else None


def turbo_settings(on):
    """What the Turbo checkbox sets, in NATIVE_CONTROLS order."""
    if on:
        return [gr.update(value=TURBO_STEPS), gr.update(value=SAMPLER), gr.update(value=SCHEDULER), gr.update(value=TURBO_CFG)]
    sampler, scheduler, cfg = preset_settings()
    return [gr.update(value=STEPS), gr.update(value=sampler), gr.update(value=scheduler),
            gr.update(value=cfg) if cfg is not None else gr.update()]


def viewer_update():
    path = latest_object()
    return gr.update(value=path) if path else gr.update()


def rotate_object(x, y, z, formats, mesh_resolution, mesh_smooth):
    try:
        path, message = rotate_latest(x, y, z, formats, mesh_resolution, mesh_smooth)
    except IdolError as error:
        return gr.update(), html.escape(str(error))
    return gr.update(value=viewer_path(path)), html.escape(message)


class Panel:
    def __init__(self):
        self.saved = gr.State({"active": False})
        with gr.Accordion("Idol", open=True, visible=False, elem_id=f"{TAB}_idol_panel") as self.accordion:
            gr.Markdown("The input image becomes a **3D object**, saved as a `.ply` next to your images. Use one object on a "
                        "plain background, or a PNG with transparency. Prompt, size and Denoising strength are not used.")
            self.mode = gr.Radio(choices=INPUT_TYPES, value=DEFAULT_MODE, label="Idol input",
                                 elem_id=f"{TAB}_idol_mode")
            self.gaussians = gr.Slider(minimum=MIN_GAUSSIANS, maximum=MAX_GAUSSIANS, step=GAUSSIANS_PER_ANCHOR,
                                       value=DEFAULT_GAUSSIANS, label="Idol Gaussians", elem_id=f"{TAB}_idol_gaussians")
            self.turbo = gr.Checkbox(value=False, label="Idol Turbo (5 steps, softer details)",
                                     info="Same checkpoint. Its two LoRAs download once into models/Idol and apply by "
                                          "themselves: add nothing to the prompt. Sets 5 steps and CFG 1.",
                                     elem_id=f"{TAB}_idol_turbo")
            with gr.Accordion("Save as", open=False, elem_id=f"{TAB}_idol_save_as"):
                self.formats = gr.CheckboxGroup(choices=list(FORMATS), value=list(DEFAULT_FORMATS), label="Idol files",
                                                info=".ply is the full object; .spz is about a fifth of the size, for the web; "
                                                     ".glb is a mesh for 3D programs, a draft that will want retopology.",
                                                elem_id=f"{TAB}_idol_formats")
                with gr.Row():
                    self.mesh_resolution = gr.Slider(minimum=MIN_RESOLUTION, maximum=MAX_RESOLUTION, step=16, value=DEFAULT_RESOLUTION,
                                                     label="Idol mesh resolution", info="Finer surface, more faces and more time.",
                                                     elem_id=f"{TAB}_idol_mesh_resolution")
                    self.mesh_smooth = gr.Slider(minimum=0, maximum=MAX_SMOOTH, step=1, value=DEFAULT_SMOOTH, label="Idol mesh smoothing",
                                                 info="Rounds the surface without shrinking it.", elem_id=f"{TAB}_idol_mesh_smooth")
                gr.Markdown("Turn the **last object** without generating it again, for example to stand it upright. The angles "
                            "follow the viewer: X to the right, Y up, Z towards you. The turned files are saved next to the "
                            "original, which stays as it was.")
                with gr.Row():
                    self.rotate_x = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Idol rotate X",
                                              elem_id=f"{TAB}_idol_rotate_x")
                    self.rotate_y = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Idol rotate Y",
                                              elem_id=f"{TAB}_idol_rotate_y")
                    self.rotate_z = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Idol rotate Z",
                                              elem_id=f"{TAB}_idol_rotate_z")
                self.rotate = gr.Button("Rotate and save", elem_id=f"{TAB}_idol_rotate")
                self.rotate_status = gr.Markdown("", elem_id=f"{TAB}_idol_rotate_status")
            self.status = gr.Markdown(NEEDS_FILES)
            self.viewer = gr.Model3D(label="Idol object", interactive=False, height=480, elem_id=f"{TAB}_idol_viewer")
        self.rotate.click(rotate_object, inputs=[self.rotate_x, self.rotate_y, self.rotate_z, self.formats, self.mesh_resolution,
                                                 self.mesh_smooth],
                          outputs=[self.viewer, self.rotate_status], queue=False, show_progress=False)
        self.bound = False
        self.attempts = []
        PANELS.append(self)

    @property
    def inputs(self):
        # the order Script.before_process takes
        return [self.mode, self.gaussians, self.turbo, self.formats, self.mesh_resolution, self.mesh_smooth]

    @property
    def needed(self):
        return [f"{TAB}_{name}" for name in NATIVE_CONTROLS] + ["setting_sd_model_checkpoint", "setting_sd_modules",
                                                               f"html_log_{TAB}"]

    def bind(self, trigger="?"):
        if self.bound:
            return
        if not _in_blocks():
            self.attempts.append(f"{trigger}: outside Blocks")
            return
        if any(name not in COMPONENTS for name in self.needed):
            self.attempts.append(f"{trigger}: waiting for {', '.join(n for n in self.needed if n not in COMPONENTS)}")
            return
        native = [COMPONENTS[f"{TAB}_{name}"] for name in NATIVE_CONTROLS]
        checkpoint = COMPONENTS["setting_sd_model_checkpoint"]
        modules = COMPONENTS["setting_sd_modules"]

        def update(value, module_values, saved, *values):
            info = checkpoint_info(value)
            status = NEEDS_FILES
            active = False
            if info:
                try:
                    active = inspect_model(info.filename) == "dit"
                    if active:
                        resolve_components(info.filename, module_paths(module_values))
                        status = READY
                except IdolError as exc:
                    status = html.escape(str(exc))
            saved = dict(saved or {"active": False})
            entering = active and not saved.get("active")
            leaving = not active and saved.get("active")
            updates = [gr.update() for _ in native]
            if entering:
                saved["native"] = list(values)
                sampler, scheduler, cfg = preset_settings()
                updates = [gr.update(value=STEPS), gr.update(value=sampler), gr.update(value=scheduler),
                           gr.update(value=cfg) if cfg is not None else gr.update()]
            elif leaving:
                updates = [gr.update(value=value) for value in saved.pop("native", values)]
            saved["active"] = active
            return [gr.update(visible=active), gr.update(value=status), saved] + updates

        inputs = [checkpoint, modules, self.saved] + native
        outputs = [self.accordion, self.status, self.saved] + native
        for event in (checkpoint.change, modules.change, gr.context.Context.root_block.load):
            event(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        self.turbo.change(turbo_settings, inputs=[self.turbo], outputs=native, queue=False, show_progress=False)
        # Forge's generate button belongs to Forge; its log line changes at the end of every generation
        COMPONENTS[f"html_log_{TAB}"].change(viewer_update, outputs=[self.viewer], queue=False, show_progress=False)
        self.bound = True
        logger.info("Idol native controls connected.")


def bind_all(trigger="ui_tabs"):
    for panel in PANELS:
        panel.bind(trigger)
    return []


def check_bindings(*_args):
    # Forge also builds panel instances it never renders; the tab is fine as long as one of its panels is bound
    if PANELS and not any(panel.bound for panel in PANELS):
        missing = sorted({name for panel in PANELS for name in panel.needed if name not in COMPONENTS})
        attempts = [attempt for panel in PANELS for attempt in panel.attempts]
        logger.error("Idol native controls were not found (missing: %s; attempts: %s); restart Forge after updating it.",
                     ", ".join(missing) or "none", "; ".join(attempts) or "none")


def reset():
    COMPONENTS.clear()
    PANELS.clear()
