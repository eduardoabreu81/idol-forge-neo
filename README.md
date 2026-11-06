# 🗿 Idol for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen)](https://github.com/eduardoabreu81/idol-forge-neo)
![Status](https://img.shields.io/badge/Status-Work_in_progress-orange)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Turn **one picture of an object into a 3D Gaussian splat** with **[Idol](https://huggingface.co/speridlabs/idol)** inside Forge Neo's img2img tab, using the checkpoint, sampler and Generate button you already know. One picture goes in; the whole object comes out, including the sides the picture never showed, as a `.ply`, a `.spz` or a `.glb` mesh.

<div align="center">

[![A cowboy figure next to the 3D object made from it, turning](https://raw.githubusercontent.com/wiki/eduardoabreu81/idol-forge-neo/media/cowboy-50-steps.gif)](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples)

*One picture in, one object out. Made by this extension, 50 steps. More in the [Examples](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples).*

</div>

> [!Important]
> **Work in progress.** Generation, Idol Turbo, the three file types and turning the finished object work in Forge Neo 2.31. So far it has been tested on one machine, with a handful of objects.
>
> On a Forge Neo that lacks something it needs, the extension stays disabled and says so in the console - Forge Neo itself keeps working as usual.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Examples](#-examples)
- [Features](#-features)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.1.0 - First Public Preview

- **One picture to a 3D object** inside Forge Neo's img2img tab, on Forge's own loader, samplers and memory management
- **Idol Turbo** - 5 steps instead of 50, with its LoRAs handled for you
- **`.ply`, `.spz` and `.glb`** - choose what to save under **Save as**
- **Turn the finished object** without generating it again
- **A 3D viewer** in the panel, and the background removed for you

---

## 🎥 Examples

Each one shows the input picture and the object turning. The sides the picture does not show are the model's guess.

| A 3D figure | A rendered character | Idol Turbo |
| :---: | :---: | :---: |
| [![A cowboy figure and its 3D object](https://raw.githubusercontent.com/wiki/eduardoabreu81/idol-forge-neo/media/cowboy-50-steps.gif)](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples#a-3d-figure) | [![An axolotl character and its 3D object](https://raw.githubusercontent.com/wiki/eduardoabreu81/idol-forge-neo/media/axolotl-50-steps.gif)](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples#a-rendered-character) | [![The same axolotl with Idol Turbo](https://raw.githubusercontent.com/wiki/eduardoabreu81/idol-forge-neo/media/axolotl-turbo.gif)](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples#idol-turbo) |
| 50 steps | 50 steps | 5 steps, softer |

| An anime drawing | The mesh, from a figure | The mesh, from a drawing |
| :---: | :---: | :---: |
| [![An original chibi drawing and its 3D object](https://raw.githubusercontent.com/wiki/eduardoabreu81/idol-forge-neo/media/chibi-50-steps.gif)](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples#an-anime-drawing) | [![The cowboy as a mesh](https://raw.githubusercontent.com/wiki/eduardoabreu81/idol-forge-neo/media/cowboy-mesh.gif)](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples#from-a-3d-figure) | [![The chibi as a mesh](https://raw.githubusercontent.com/wiki/eduardoabreu81/idol-forge-neo/media/chibi-mesh.gif)](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples#from-an-anime-drawing-the-weak-spot) |
| Works, with limits | A draft | The weak spot |

The settings and what to expect from each are in the **[Examples](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Examples)** page of the wiki.

---

## 🎯 Features

> [!Tip]
> Step-by-step guides and every setting explained are in the **[wiki](https://github.com/eduardoabreu81/idol-forge-neo/wiki)**.

### 🗿 One Picture to a 3D Object

- A render, a generated image or a photo of one object, with or without a background
- Up to **1,048,576 Gaussians**, including the sides the picture never showed
- The background is removed for you, unless your picture already has transparency
- The gallery shows the picture the model actually saw: background removed, cropped, 1024 × 1024
- The same seed gives the same object, byte for byte

### 📁 Three File Types

- **`.ply`** - the full object, the standard 3DGS format
- **`.spz`** - the same object, about a fifth of the size, for the web
- **`.glb`** - a coloured mesh extracted from the object, a draft for Blender and other 3D programs
- A `.json` next to them records the settings and the camera the model predicted

### 🔄 Turn the Finished Object

- **Rotate X, Y and Z** under **Save as**, following the axes the viewer shows
- No generation and no GPU: only the object's Gaussians are turned
- The turned files are saved next to the original, which stays as it was

### ⚡ Idol Turbo

- **5 steps instead of 50**, with no guidance, on the same checkpoint
- Its two LoRAs download by themselves and switch in the middle of the sampling: add nothing to the prompt
- A good shape in a fraction of the time, softer and with paler colours

### 🎛️ Familiar Forge Controls

- Runs inside img2img - no separate tab, no extra program
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- The **idol** UI preset sets the sampler, schedule, steps, CFG and Shift
- A small **Idol** panel holds the options and a **3D viewer** you can turn with the mouse

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Nothing to `pip install`: it uses what Forge Neo already has
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Reads each file's header first, and says what is missing before loading anything
- Other models are not affected

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/idol-forge-neo`
4. Click **Install** and restart the WebUI
5. Place the Idol files in the usual folders. None of these need a Hugging Face account:

| Part | File | Size | Folder |
| :--- | :--- | :---: | :--- |
| Idol checkpoint | [idol_fp16](https://huggingface.co/Comfy-Org/Idol/blob/main/diffusion_models/idol_fp16.safetensors) | 741 MB | `models/Stable-diffusion` |
| Image encoder | [dino_v3_vit_h_bf16](https://huggingface.co/Comfy-Org/Idol/blob/main/clip_vision/dino_v3_vit_h_bf16.safetensors) | 1.7 GB | `models/text_encoder` |
| FLUX.2 VAE | [flux2-vae](https://huggingface.co/Comfy-Org/Idol/blob/main/vae/flux2-vae.safetensors) | 336 MB | `models/VAE` |
| Idol decoder | [idol_decoder_fp16](https://huggingface.co/Comfy-Org/Idol/blob/main/vae/idol_decoder_fp16.safetensors) | 576 MB | `models/VAE` |

If you use FLUX.2 or Krea 2, you may already have `flux2-vae`.

6. Pick the **idol** UI preset, the `idol_fp16` checkpoint, and **all three** of the image encoder, the FLUX.2 VAE and the decoder under **VAE / Text Encoder**
7. Go to **img2img**: the **Idol** panel appears there

These download by themselves the first time they are needed, into `models/Idol`:

| File | Size | When |
| :--- | :---: | :--- |
| [birefnet](https://huggingface.co/speridlabs/idol/blob/main/birefnet.safetensors) | 444 MB | A picture without transparency |
| [idol_turbo_slow_comfy](https://huggingface.co/speridlabs/idol-turbo/blob/main/loras/idol_turbo_slow_comfy.safetensors) and [idol_turbo_fast_comfy](https://huggingface.co/speridlabs/idol-turbo/blob/main/loras/idol_turbo_fast_comfy.safetensors) | 59 MB each | **Idol Turbo** |

> [!Note]
> The release files of [speridlabs/idol](https://huggingface.co/speridlabs/idol) (`denoiser`, `decoder`, `decoder_gs`) are recognized too. More in the **[Models](https://github.com/eduardoabreu81/idol-forge-neo/wiki/Models)** page.

---

## ⚙️ Recommended Settings

| Setup | Sampler | Schedule type | Steps | CFG | Shift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regular** (the idol preset) | Euler | Simple | 50 | 4 | 3 |
| **Idol Turbo** | Euler | Simple | 5 (4 to 8) | 1 | 3 |

The panel sets these for you. CFG 4 is what the Idol SDK calls guidance 3. With **Idol Turbo**, CFG and the sampler are fixed, as its guidance is folded into its LoRAs.

**Input type:**

| Choice | For | The object fills |
| :--- | :--- | :---: |
| **Render or generated image** | Renders and generated images of one object | 1/1.2 of the frame |
| **Photo** | Photographs | 1/1.7 of the frame |

**Idol Gaussians** goes from 32,768 to 1,048,576. The default of 262,144 matches the octree's density; more makes the object denser and takes more memory, without new detail.

**Idol mesh resolution** goes from 64 to 512. The default of 256 makes about 370,000 faces and a file of about 11 MB, in 10 to 20 seconds on the CPU. **Idol mesh smoothing** rounds the surface without shrinking it.

Prompt, negative prompt, Width, Height, Resize mode and Denoising strength do nothing. An empty prompt is saved as `3D object`, because the sd-dynamic-prompts extension fails on an empty one.

---

## 💡 Tips

- Use **one object**, whole in the frame, on a plain background. Several objects, or a busy background, can confuse the background removal
- A **three-quarter view** shows the model more of the object than a straight front view
- **3D-looking pictures work best.** In a flat drawing, the lines and painted shadows become part of the object's colour. To generate a character for Idol, ask for a 3D-looking figure rather than line art
- If an object comes out strange, look at the **gallery**: it shows exactly what the model saw, and a bad cut-out shows there
- If the background removal fails, use a **PNG with transparency**: your own cut-out is used as it is
- For anime drawings, prefer the **`.ply`** or **`.spz`**: faces drawn with lines do not survive the step to a mesh
- Objects do not always face front. If yours arrives sideways, try **Idol rotate Y** at 90
- Do not add the Turbo LoRAs to the prompt or to `models/Lora`: the extension applies them by itself
- Keep the same seed when comparing settings
- `.spz` is y-up, as the format defines; `.ply` follows the usual 3DGS convention, y-down
- Keep the JSON file saved next to the object - it records the settings and the predicted camera

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[Idol](https://huggingface.co/speridlabs/idol)** and the **[Idol SDK](https://github.com/speridlabs/idol-sdk)** by [Speridlabs](https://speridlabs.com) - the model and its reference code
- **[Idol Turbo](https://huggingface.co/speridlabs/idol-turbo)** - the distilled LoRAs
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** - the base of the mesh extraction
- **[Comfy-Org](https://huggingface.co/Comfy-Org/Idol)** - the model files
- **[ComfyUI-Idol](https://github.com/PedroMarinhoDev/ComfyUI-Idol)** by PedroMarinhoDev - which showed that Idol runs on a stock sampler
- **[SPZ](https://github.com/nianticlabs/spz)** by Niantic Labs - the `.spz` format
- **[DINOv3](https://github.com/facebookresearch/dinov3)** by Meta and **[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)** - the image encoder and the background removal Idol uses
- **Example sources** - the axolotl and cowboy pictures come with the Idol SDK; the chibi is an original character made with Anima in Forge Neo

The parts adapted from other projects keep their licenses: see [NOTICE](NOTICE).

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/idol-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/idol-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
