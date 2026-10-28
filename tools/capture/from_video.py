"""A filmed speaker -> Haj Salama performing it, in one command (Windows, Linux, macOS).

    python tools/capture/from_video.py VIDEO NAME [--start S] [--dur D] [--ref stand01] [--frames N]

Makes episodes/NAME/: source.mp4 (the clip, 30 fps, upright, trimmed), voice.wav (its sound),
face_trace.json (MediaPipe face + body), then renders the figure and puts the speaker and the
figure side by side in out/side_by_side.mp4. The speaker's voice is the figure's voice; lips,
blinks, brows and head follow the face; shoulders and hands choose the gestures.
--frames N renders only the first N frames (a quick check). MediaPipe runs in this Python, or in
the one named by the HAJ_MP_PYTHON environment variable.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import toolpaths  # noqa: E402


def run(cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, env=toolpaths.utf8_env(), **kw)


def qc(trace_path):
    """Is the recording usable? (the speaker can re-film if not)"""
    fr = json.loads(Path(trace_path).read_text(encoding="utf-8"))["frames"]
    n = max(len(fr), 1)
    face = sum(f.get("ok", False) for f in fr) / n
    body = [f["body"] for f in fr if f.get("body")]
    wr = lambda s: sum(b[f"wrist{s}"][2] > 0.5 for b in body) / n  # noqa: E731
    hands = sum(("openL" in b) or ("openR" in b) for b in body) / n
    print(f"QC: {n / 30:.1f} s; face {face:.0%}, shoulders {len(body) / n:.0%}, "
          f"left wrist {wr('L'):.0%}, right wrist {wr('R'):.0%}, a hand's fingers {hands:.0%}")
    if face < 0.9:
        print("QC: face lost too often: face the camera, keep hands away from the mouth, more light")
    if len(body) / n < 0.8:
        print("QC: shoulders not found: step back so the camera sees from the head to below the waist")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("name")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--dur", type=float)
    p.add_argument("--ref", default="stand01", help="rig folder under assets/haj_relief")
    p.add_argument("--frames", type=int, help="render only the first N frames")
    a = p.parse_args()
    E = ROOT / "episodes" / a.name
    E.mkdir(parents=True, exist_ok=True)
    ff = toolpaths.ffmpeg()
    trim = ["-ss", a.start] + (["-t", a.dur] if a.dur else [])
    # phones record upright/rotated, at variable frame rates: one clean 30 fps clip first
    run([ff, "-v", "error", "-y", *trim, "-i", a.video, "-vf", "fps=30,scale='min(1920,iw)':-2",
         "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", E / "source.mp4"])
    run([ff, "-v", "error", "-y", "-i", E / "source.mp4", "-vn", "-ac", "1", "-ar", "22050", E / "voice.wav"])
    if not (E / "transcript.txt").exists():
        (E / "transcript.txt").write_text("(recorded speech)\n", encoding="utf-8")
    (E / "mouth.json").unlink(missing_ok=True)
    run([toolpaths.mediapipe_python(), ROOT / "tools/capture/trace_face.py", E / "source.mp4", E / "face_trace.json",
         "--body", "--face", "largest"])
    qc(E / "face_trace.json")
    extra = ["--frames", a.frames] if a.frames else []
    run([sys.executable, ROOT / "anchor.py", E, "--lang", "ar", "--relief", ROOT / "assets/haj_relief" / a.ref,
         "--image", ROOT / "assets/haj_salama/refs" / f"{a.ref}.png", *extra])
    run([ff, "-v", "error", "-y", "-i", E / "source.mp4", "-i", E / "out" / f"relief_{a.ref}.mp4", "-filter_complex",
         "[0:v]scale=-2:1080,setsar=1[a];[1:v]setsar=1[b];[a][b]hstack=inputs=2,scale=-2:720[v]",
         "-map", "[v]", "-map", "1:a", "-shortest", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", E / "out" / "side_by_side.mp4"])
    print("wrote", E / "out" / "side_by_side.mp4")


if __name__ == "__main__":
    main()
