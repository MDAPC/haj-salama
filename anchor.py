"""Episode pipeline: transcript (+ optional voice) -> lip-synced anchor video.

    python anchor.py episodes/test01                                  # layered figure (figure.json)
    python anchor.py episodes/ar01 --lang ar --layout layouts/A_desk.json   # single painting (rig.json)
    python anchor.py episodes/ar01 --lang ar --shot medium            # 3D Haj Salama in the 3D studio

The episode folder holds transcript.txt and, optionally, voice.wav and screen.json
(story images for the studio screen, 3D only). Without voice.wav, Piper TTS reads the transcript.
Output: <episode>/out/anchor.mp4, <episode>/out/<layout>.mp4, or <episode>/out/3d_<shot>.mp4
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
import toolpaths  # noqa: E402  (finds Blender, Rhubarb, ffmpeg and the voices on any OS)

VOICES = {"en": toolpaths.voice("en_US-lessac-medium.onnx"),
          "ar": toolpaths.voice("ar_JO-kareem-medium.onnx")}
FPS = 30


def need(path, what):
    if not path or not Path(path).exists() and not shutil.which(str(path)):
        raise SystemExit(f"{what} not found. Run `python tools/setup.py` (or see docs/SETUP.md).")
    return path


def run(cmd, **kw):
    t = time.time()
    print("»", " ".join(str(c) for c in cmd[:4]), "…", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=toolpaths.utf8_env(), **kw)
    print(f"  {time.time() - t:.1f}s", flush=True)


def voice_id(voice):
    return hashlib.sha1(Path(voice).read_bytes()).hexdigest()


def stamp(mouth, voice):
    """Record which voice.wav the mouth cues belong to (file times do not survive a git clone)."""
    m = json.loads(mouth.read_text(encoding="utf-8"))
    m["voice_sha1"] = voice_id(voice)
    mouth.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")


def stale(mouth, voice):
    """The mouth cues were made from a different voice.wav (unstamped cues are trusted: delete
    mouth.json after replacing voice.wav by hand)."""
    sha = json.loads(mouth.read_text(encoding="utf-8")).get("voice_sha1")
    return sha is not None and sha != voice_id(voice)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode")
    p.add_argument("--lang", choices=VOICES, default="en")
    p.add_argument("--layout", help="layout JSON for a single-painting figure")
    p.add_argument("--shot", choices=("desk", "medium", "closeup"), help="render the 3D character")
    p.add_argument("--relief", help="animate a prepared reference image, e.g. assets/haj_relief/ref10")
    p.add_argument("--image", default=HERE / "assets/haj_salama/refs/ref10.png", help="the reference image")
    p.add_argument("--figure", default=HERE / "assets/anchor_placeholder")
    p.add_argument("--frames", type=int)
    a = p.parse_args()
    ep = Path(a.episode).resolve()

    voice = ep / "voice.wav"
    mouth = ep / "mouth.json"
    if not voice.exists():
        # synthesized speech comes with the synthesizer's own phoneme timings: exact lip-sync
        run([sys.executable, HERE / "tools/tts_align.py", need(VOICES[a.lang], f"Piper voice for {a.lang}"),
             ep / "transcript.txt", ep])
        stamp(mouth, voice)
    elif not mouth.exists() or stale(mouth, voice):
        # a recorded voice: estimate the mouth shapes from the audio
        # PocketSphinx only knows English; other languages use the language-independent recognizer
        rec = ["-d", ep / "transcript.txt"] if a.lang == "en" else ["--recognizer", "phonetic"]
        run([need(toolpaths.rhubarb(), "Rhubarb Lip Sync"), "-q", "-f", "json", "--extendedShapes", "GHX", *rec, "-o", mouth, voice])
        stamp(mouth, voice)

    BLENDER = need(toolpaths.blender(), "Blender 5.x")
    if a.relief:
        name = Path(a.relief).name
        frames = ep / "out" / f"relief_{name}" / "frames"
        cmd = [BLENDER, "-b", "--factory-startup", "-P", HERE / "tools/relief/build_relief.py", "--",
               "--ref", Path(a.relief).resolve(), "--image", Path(a.image).resolve(), "--episode", ep]
        video = ep / "out" / f"relief_{name}.mp4"
    elif a.shot:
        frames = ep / "out" / f"3d_{a.shot}" / "frames"
        cmd = [BLENDER, "-b", HERE / "assets/haj_salama_3d/haj.blend", "-P", HERE / "tools/haj3d/animate.py",
               "--", "--episode", ep, "--shot", a.shot]
        video = ep / "out" / f"3d_{a.shot}.mp4"
    elif a.layout:
        name = Path(a.layout).stem
        frames = ep / "out" / name / "frames"
        cmd = [BLENDER, "-b", "--factory-startup", "-P", HERE / "tools/build_warp.py", "--",
               "--layout", Path(a.layout).resolve(), "--episode", ep]
        video = ep / "out" / f"{name}.mp4"
    else:
        frames = ep / "out" / "frames"
        cmd = [BLENDER, "-b", "--factory-startup", "-P", HERE / "tools/build_scene.py", "--",
               "--figure", a.figure, "--episode", ep]
        video = ep / "out" / "anchor.mp4"
    shutil.rmtree(frames, ignore_errors=True)
    if a.frames:
        cmd += ["--frames", a.frames]
    run(cmd, stdout=subprocess.DEVNULL)

    green = a.relief and json.loads((Path(a.relief) / "frame.json").read_text(encoding="utf-8")).get("green")
    if (a.shot or a.relief) and not green:
        broadcast(frames, voice, video)
    else:
        run([toolpaths.ffmpeg(), "-y", "-v", "error", "-framerate", FPS, "-i", frames / "%04d.png",
             "-i", voice, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             "-c:a", "aac", "-b:a", "160k", "-shortest", video])
    print("wrote", video)


def broadcast(frames, voice, video):
    """Lay the channel graphics over the rendered studio: ticker running left-to-right under
    the 'breaking' tag, channel logo, and the name strap sliding in from the right."""
    S = HERE / "assets/studio"
    graph = ";".join([
        "[0:v][3:v]overlay=0:1010[a]",
        "[a][4:v]overlay=x='1730-overlay_w+110*t':y=1010[b]",
        "[b][5:v]overlay=1730:1015[c]",
        "[c][2:v]overlay=1460:40[d]",
        "[d][6:v]overlay=x='if(lt(t,0.8),W,if(lt(t,1.3),W-60-w+(1.3-t)/0.5*(w+80),W-60-w))':y=860[v]",
    ])
    inputs = ["-framerate", FPS, "-i", frames / "%04d.png", "-i", voice]
    for name in ("logo", "ticker_bar", "ticker_text", "ticker_tag", "lower_third"):
        inputs += ["-loop", "1", "-framerate", FPS, "-i", S / f"{name}.png"]
    n = len(list(Path(frames).glob("*.png")))  # looped overlays never end, so stop at the last frame
    run([toolpaths.ffmpeg(), "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "160k",
         "-frames:v", n, "-t", f"{n / FPS:.3f}", video])


if __name__ == "__main__":
    main()
