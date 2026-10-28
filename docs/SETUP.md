# Setup (Windows, Linux, macOS)

Everything is free and runs locally. You need Python 3.12, Blender 5.2, ffmpeg, and a handful of
models that `tools/setup.py` downloads into the repo's `.tools/` folder (git-ignored).

## Windows 10 / 11

```powershell
winget install Python.Python.3.12 BlenderFoundation.Blender Gyan.FFmpeg Git.Git
# open a NEW terminal so PATH picks them up, then in the repo folder:
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tools\setup.py
.venv\Scripts\python tools\setup.py --check
```

Use `.venv\Scripts\python` for every command (or activate the venv; PowerShell may first need
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, which is your call). Forward slashes in paths
work too: `.venv\Scripts\python anchor.py episodes/ar01 --lang ar --relief assets/haj_relief/ref10 --frames 90`.

- No winget? Install Blender from blender.org or run `python tools/setup.py --blender` (portable zip
  into `.tools/`), and download an ffmpeg build (gyan.dev or BtbN), then set `FFMPEG` to its `ffmpeg.exe`.
- Blender in a non-standard place: set `BLENDER=C:\path\to\blender.exe`.
- Text is read and written as UTF-8 everywhere (Arabic transcripts). If you write your own scripts,
  set `PYTHONUTF8=1` as well.
- The Veo pool links frames instead of copying them: symlinks need admin rights on Windows, so it
  falls back to hard links (same drive) and then copies. Nothing to do.

## Linux (Ubuntu / Debian / Fedora / Arch)

```bash
sudo apt install python3.12 python3.12-venv ffmpeg git     # or dnf / pacman equivalents
python3.12 -m venv .venv && source .venv/bin/activate      # `uv venv -p 3.12` works too
pip install -r requirements.txt
python tools/setup.py --blender    # portable Blender 5.2.3 into .tools/ (skip if `blender` is on PATH)
python tools/setup.py --check
```

## macOS (Apple Silicon or Intel)

```bash
brew install python@3.12 ffmpeg git
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/setup.py
```

Install Blender 5.2 from blender.org (the .dmg) and set
`BLENDER=/Applications/Blender.app/Contents/MacOS/Blender`.

## What `tools/setup.py` fetches

| Item | Where | Licence |
|---|---|---|
| Depth Anything V2 Small (ONNX) | `.tools/models/depth-anything-v2-small/onnx/model.onnx` | Apache-2.0 |
| MediaPipe face / pose / hand landmarkers | `.tools/models/mediapipe/*.task` | Apache-2.0 |
| Piper voices `ar_JO-kareem-medium`, `en_US-lessac-medium` | `.tools/piper-voices/` | see each voice's MODEL_CARD |
| Rhubarb Lip Sync 1.14.0 (for your OS) | `.tools/Rhubarb-Lip-Sync-1.14.0-*/` | MIT |
| `--blender`: Blender 5.2.3 portable | `.tools/blender-5.2.3-*/` | GPL |
| `--rife`: RIFE ncnn-vulkan (optional) | `.tools/rife-ncnn-vulkan-*/` | MIT |
| `--stock`: two CC BY interview clips (optional) | `assets/capture/stock/` | CC BY 3.0 / 4.0 |

rembg downloads its own `isnet-general-use` model on first use (into `~/.u2net`).

## How tools are found

`tools/toolpaths.py` looks for each program in this order: environment variable, PATH, `.tools/`,
`~/.local/opt` (the original Linux install).

| Variable | Meaning |
|---|---|
| `BLENDER`, `RHUBARB`, `FFMPEG` | full path to that program |
| `HAJ_TOOLS` | another folder laid out like `.tools/` (models/, piper-voices/, ...) |
| `HAJ_MP_PYTHON` | a different Python that has MediaPipe (if your main one cannot install it) |
| `HAJ_SD_PYTHON` | a Python with torch + diffusers (optional local Stable Diffusion) |

## Graphics

Blender renders with Eevee, which needs a GPU with OpenGL 4.3 / Vulkan; any integrated GPU works
(it was built on an AMD Radeon 680M iGPU). On a GPU-less server, Eevee cannot start: render on a
machine with a GPU. Expect about 2-4 s per 1080p frame on a laptop.

## Optional extras

- **Veo gesture pool** (standing figure): `python tools/capture/veo_rebuild.py` (about a minute per
  clip, a few hundred MB in `assets/haj_salama/veo/lib*/` and `pool/`). Without it the standing figure
  uses its painted gesture poses.
- **EMAGE** speech-to-gesture (non-commercial models): clone `VolgaGerm/emage-onnx-export` into
  `.tools/emage/emage-onnx-export`, export its ONNX files there (its own venv, Python 3.10, CPU torch),
  then `<that venv's python> tools/capture/emage_trace.py voice.wav face_trace.json` with
  `soundfile` and `librosa` installed (requirements-optional.txt).
- **Local Stable Diffusion** inpainting (experimental): a separate venv with torch + diffusers,
  named by `HAJ_SD_PYTHON`; see `tools/gen/`.
- **New Veo clips**: see `assets/haj_salama/veo/PROMPTS.md`, then
  `python tools/capture/veo_add.py downloaded.mp4 NAME [--mirrored]`.
