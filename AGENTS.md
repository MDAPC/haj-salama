# AGENTS.md: working on the Haj Salama pipeline

Read this first, then `README.md` (what it does) and `docs/SETUP.md` (installing). `docs/CAPTURE.md`
is the build log: every pass, what was measured, what failed and why. Check it before re-trying an
idea; many obvious ones were tried already.

## What this is

A code-driven news-anchor video pipeline. Input: an episode folder (Arabic transcript, or a recorded
voice, or a filmed speaker). Output: an MP4 of the painted caricature **Haj Salama** presenting it.
Everything is local and free: Blender 5.2 (headless, Eevee), Piper TTS, Rhubarb Lip Sync, MediaPipe,
ONNX models. Windows, Linux and macOS.

## Set up and check (do this before changing anything)

```bash
python -m venv .venv            # Python 3.12
# Windows: .venv\Scripts\python ...   Linux/macOS: .venv/bin/python ...
python -m pip install -r requirements.txt
python tools/setup.py           # models, voices, Rhubarb -> .tools/   (--blender if no Blender 5.x)
python tools/setup.py --check   # must end with "All set."
```

Smoke test (about 2-5 minutes):

```bash
python anchor.py episodes/ar01 --lang ar --relief assets/haj_relief/ref10 --frames 60
# -> episodes/ar01/out/relief_ref10.mp4 and .../relief_ref10/frames/0001.png ...
```

`anchor.py` hides Blender's console. When a render fails, run the Blender step yourself to see the
traceback:

```bash
blender -b --factory-startup -P tools/relief/build_relief.py -- --ref assets/haj_relief/ref10 \
    --image assets/haj_salama/refs/ref10.png --episode episodes/ar01 --frames 3
```

(`build_relief.py` also takes `--from N` to re-render only frames N onward, and `--poses` for rig stills.)

## How a render flows

1. `anchor.py`: no `voice.wav` -> `tools/tts_align.py` (Piper, with phoneme timings: exact lip-sync);
   recorded `voice.wav` -> Rhubarb. Writes `mouth.json`, stamped with the voice's SHA-1, so a changed
   voice is noticed (file times do not survive a git clone).
2. Blender runs `tools/relief/build_relief.py`: the reference image as a relief mesh (cut-out +
   Depth Anything depth) with bones and shape keys, painted variants as twin layers, and `perform()`
   keyframing everything: mouth paintings per sound, blinks, gaze, brows, head (from a face trace if
   the episode has `face_trace.json`), breathing, shemagh cloth springs, gestures, expressions.
3. ffmpeg: frames + voice -> MP4 (`broadcast()` adds ticker/logo/strap unless the rig's `frame.json`
   says `"green": true`).

Key places in `build_relief.py`: `Relief` (mesh, armature, layers, `add_veo`, `set_veo`),
`gesture_targets` -> `calm_gestures` -> (`gesture_states` for paintings | `veo_track` for Veo takes),
`expression_track`, `perform`. Rig landmarks and switches live in `assets/haj_relief/<figure>/rig.json`
(keys documented by `_`-prefixed notes inside it).

## The two figures

- `ref10`: seated, studio plate with a green screen showing story cards (`episode/screen.json`).
- `stand01`: standing, plain chroma green (`frame.json` `"green": true`, no channel graphics). Body
  gestures play frames from Veo clips (`rig.json` `"veo_pool"`): build the pool once with
  `python tools/capture/veo_rebuild.py`; without it the rig falls back to its painted poses.
  Expressions (`expr_*` variants) are cued by `episode/expressions.json`.

## How to verify a change

There is no unit-test suite; renders are the test. Keep checks short (3-8 s, `--frames 90..240`):
the owner wants short checks, not long renders. Then look at frames, not just the exit code:

- contact sheets of a few frames (Pillow), crops of the area you changed;
- missing textures show as Blender magenta (R>150, B>150, G<110): scan every frame;
- pops: the biggest frame-to-frame pixel changes, then look at those frames;
- motion: e.g. the head-top row over time for leaning/bobbing.

Compare before/after on the same frames (`--from N --frames M` into a copy of the episode).

## What the owner wants (keep to it)

- **Likeness stays the reference's.** Reuse the reference pixels and paintings; code-built 3D was
  rejected as primitive.
- **Calm, settled presenter motion.** Exaggerated, fast, sudden or looping motion is the main
  complaint. Gestures are events with rests; nothing should rock, flicker or repeat in a rhythm.
- **Variety**: never the same gesture take twice in a row; random (seeded) lengths.
- **Free, local, open-source** tools for the pipeline. Gemini/Veo are used only by hand in a
  browser to make paintings and clips (see `assets/haj_salama/veo/PROMPTS.md`).
- **Real people's recordings stay private** (git-ignored `episodes/speaker*`, `episodes/capture*`).
  Never commit a filmed person's video, voice or renders made from them.

## Gotchas learned the hard way

- Blender 5: new shape keys start at value 1.0; set `kb.value = 0`.
- Blender image sequences clamp the scene frame to `frame_duration` *before* adding `frame_offset`:
  declare a huge duration or frames past the pool length render magenta.
- A hide mask that reaches only 0.99 leaves a visible ghost over flat green (sRGB lifts it): saturate.
- Python `set` order changes between runs: sort before `max()` or random choices (plans must be
  deterministic, or a partial re-render will not match).
- Veo takes must start and end at the rest pose: take edges come from the body picture
  (`veo_pool.deviation`), not wrist positions alone.
- Seated tuning does not carry to standing: pivot at the waist (`chest_y`, `hips_y`), halve sway
  (`body_sway`), keep only quick nods from a seated speaker's trace.
- Text files: always `encoding="utf-8"` (Windows defaults to a code page; transcripts are Arabic).
- Paths: use `pathlib` and `tools/toolpaths.py`; never hard-code `~/.local/...` or `C:\...`.
- Don't start a `pkill -f` pattern that also matches your own shell's command line.
- Depth Anything V2: Small only (Apache-2.0). Base/Large are non-commercial.
- EMAGE (optional gesture source) is non-commercial; keep it optional.

## Conventions

- Match the surrounding code: short functions, numpy, comments that explain *why*, no new
  dependencies unless they do substantial work.
- Long renders go in the background; never block on them.
- After a meaningful pass, add a dated section to `docs/CAPTURE.md` (what changed, what was measured).
- Generated intermediates (Veo libs/pool, renders) stay out of git; regenerate them with the tools.
