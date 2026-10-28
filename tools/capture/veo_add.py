"""Add a downloaded Veo clip of the standing figure to the gesture pool (Windows, Linux, macOS).

    python tools/capture/veo_add.py DOWNLOADED.mp4 NAME [--mirrored] [--about "what it shows"]

--mirrored: the clip was generated from the mirrored start frame (veo/start/stand01_mirror.jpg) to
get left-hand gestures; it is flipped back first. Writes assets/haj_salama/veo/NAME.mp4,
NAME_trace.json and lib_NAME/, records the clip in veo/clips.json, then rebuilds the pool.
MediaPipe runs in this Python, or in the one named by HAJ_MP_PYTHON.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import toolpaths  # noqa: E402
import veo_rebuild  # noqa: E402  (same folder: a script's own folder is on sys.path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("name")
    p.add_argument("--mirrored", action="store_true")
    p.add_argument("--about", default="")
    a = p.parse_args()
    V = veo_rebuild.V
    clip, trace = V / f"{a.name}.mp4", V / f"{a.name}_trace.json"
    if a.mirrored:
        subprocess.run([toolpaths.ffmpeg(), "-v", "error", "-y", "-i", a.src, "-vf", "hflip", "-c:a", "copy", clip],
                       check=True)
        shutil.move(a.src, V / f"{a.name}_mirrored_original.mp4")
    else:
        shutil.move(a.src, clip)
    subprocess.run([toolpaths.mediapipe_python(), ROOT / "tools/capture/trace_face.py", clip, trace, "--body"],
                   check=True, env=toolpaths.utf8_env())
    entry = {"lib": f"lib_{a.name}", "clip": clip.name, "trace": trace.name, "about": a.about}
    m = json.loads((V / "clips.json").read_text(encoding="utf-8"))
    m["clips"] = [c for c in m["clips"] if c["lib"] != entry["lib"]] + [entry]
    (V / "clips.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    veo_rebuild.build_lib(entry)
    veo_rebuild.build_pool()


if __name__ == "__main__":
    main()
