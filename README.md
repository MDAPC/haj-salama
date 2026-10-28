# Haj Salama (الحاج سلامة): a code-driven news anchor

Give it a transcript (or a recorded voice, or a filmed speaker) and the painted Gulf caricature
**Haj Salama** presents it in Arabic: exact lip-sync, blinks, brows, head motion and presenter
gestures. Story images can show on the studio screen behind him, or he can stand alone on a green
screen for compositing. Everything runs locally on a CPU with free, open-source tools (Blender, Piper, Rhubarb,
MediaPipe, ONNX models). Works on **Windows, Linux and macOS**.

| | |
|---|---|
| **Seated in the studio** (`assets/haj_relief/ref10`) | the reference painting rebuilt as a relief mesh in Blender; 17 painted mouths, story cards on the screen, ticker and lower third |
| **Standing on green** (`assets/haj_relief/stand01`) | full figure for keying; arms play real in-between motion cut from Veo clips of him; six facial expressions |
| **Performance capture** | film yourself talking: your voice becomes his, your face and hands drive his |

## Quick start

```bash
python -m venv .venv          # Python 3.12 recommended (MediaPipe has no wheels for newer Pythons yet)
# Windows:  .venv\Scripts\activate        Linux / macOS:  source .venv/bin/activate
pip install -r requirements.txt
python tools/setup.py         # models, Piper voices, Rhubarb -> .tools/  (add --blender for a portable Blender)
python tools/setup.py --check # everything found?
```

Blender 5.2 and ffmpeg are the two system programs: `winget install BlenderFoundation.Blender Gyan.FFmpeg`
on Windows, your package manager elsewhere, or `python tools/setup.py --blender` for Blender.
Full details per OS: [docs/SETUP.md](docs/SETUP.md).

Render a 3-second check of each figure:

```bash
# seated, in the studio, Arabic speech synthesized from episodes/ar01/transcript.txt
python anchor.py episodes/ar01 --lang ar --relief assets/haj_relief/ref10 --frames 90

# standing on green (build the Veo gesture pool once, ~5 min; without it he uses painted poses)
python tools/capture/veo_rebuild.py
python anchor.py episodes/emage02 --lang ar --relief assets/haj_relief/stand01 --image assets/haj_salama/refs/stand01.png --frames 90
```

Output: `episodes/<name>/out/relief_<figure>.mp4`. Drop `--frames` for the whole episode
(about 2-4 s per frame on a laptop CPU/iGPU).

## Your own episode

Make a folder under `episodes/` with:

- `transcript.txt`: the Arabic text, **with tashkeel** for correct vowels (Piper reads it), or
- `voice.wav`: a recorded voice instead (Rhubarb estimates the mouth shapes), and optionally
- `screen.json`: `[{"t": 0, "image": "screen_default"}, {"t": 10.9, "image": "card_dust"}]` story cards
  for the studio screen (plates in `assets/haj_relief/ref10/plates/`, made by `tools/relief/prep_plate.py`),
- `expressions.json`: `[{"t": 2.0, "dur": 1.5, "name": "frown"}]` (standing figure: frown, stern,
  worried, skeptic, sad, closed).

Then `python anchor.py episodes/<name> --lang ar --relief assets/haj_relief/ref10`.

## From a filmed speaker

```bash
python tools/capture/from_video.py my_video.mp4 speaker01 --frames 240   # 8-second check
python tools/capture/from_video.py my_video.mp4 speaker01                # the whole clip
```

It trims and cleans the clip, takes the voice, traces face and body (MediaPipe), prints a quality
line (face / shoulders / wrists found), renders the standing figure and writes
`episodes/speaker01/out/side_by_side.mp4`. How to film: landscape, standing, head to below the waist,
hands in frame, front light, quiet room; send the original file (not a compressed chat video).

## Where things are

| Path | What |
|---|---|
| `anchor.py` | episode pipeline: speech -> mouth cues -> Blender render -> ffmpeg |
| `tools/relief/` | the relief rig: `prep_*.py` prepare a reference, `build_relief.py` animates it (runs inside Blender) |
| `tools/capture/` | performance capture (`trace_face.py`, `from_video.py`), Veo gesture library and pool, EMAGE |
| `tools/setup.py`, `tools/toolpaths.py` | downloads, checks, and where every external tool is found |
| `assets/haj_salama/` | references (`refs/`), Gemini paintings (`gen/`), Veo clips and prompts (`veo/`) |
| `assets/haj_relief/<figure>/` | prepared rigs: cut-out, depth, `rig.json` landmarks, aligned paintings (`var_*.png`) |
| `episodes/` | sample episodes (inputs only; renders go to `out/`, not committed) |
| `docs/` | [SETUP](docs/SETUP.md), [CAPTURE](docs/CAPTURE.md) (the full build log), [RESEARCH](docs/RESEARCH.md), [MOTION_RESEARCH](docs/MOTION_RESEARCH.md) |
| `AGENTS.md` | orientation for AI coding agents working on this repo |

Older approaches are still in the code: a layered 2D placeholder (`tools/build_scene.py`), a single-painting
mesh warp (`tools/build_warp.py`, `--layout layouts/*.json`) and a code-built 3D character
(`tools/haj3d/`, `--shot medium`). The relief rig replaced them for likeness.

## Licences and third-party material

See [THIRD_PARTY.md](THIRD_PARTY.md). In short: the pipeline uses only freely licensed tools and models;
Depth Anything V2 **Small** is Apache-2.0 (the Base/Large checkpoints are non-commercial, do not swap
them in); the optional EMAGE speech-to-gesture route is **non-commercial**. The character paintings
and Veo clips were generated with Google Gemini from the project's own reference images.
