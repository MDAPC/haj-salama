"""Rebuild the Veo gesture libraries and pool from the committed clips (Windows, Linux, macOS).

    python tools/capture/veo_rebuild.py            # every clip in assets/haj_salama/veo/clips.json
    python tools/capture/veo_rebuild.py --pool     # only re-gather the pool from existing libraries

The libraries (cut-out body frames) and the pool are large and fully regenerable, so they are not
in the repository; a fresh clone runs this once (about a minute per clip) before rendering the
standing figure with Veo gestures. Traces are committed, so MediaPipe is not needed here.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "assets/haj_salama/veo"
REF, RIG = ROOT / "assets/haj_salama/refs/stand01.png", ROOT / "assets/haj_relief/stand01"


def manifest():
    return json.loads((V / "clips.json").read_text(encoding="utf-8"))["clips"]


def build_lib(c):
    subprocess.run([sys.executable, ROOT / "tools/capture/veo_library.py", V / c["clip"], V / c["trace"],
                    REF, RIG, V / c["lib"]], check=True)


def build_pool():
    libs = [V / c["lib"] for c in manifest() if (V / c["lib"] / "gestures.json").exists()]
    subprocess.run([sys.executable, ROOT / "tools/capture/veo_pool.py", V / "pool", RIG, *libs], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", action="store_true", help="only rebuild the pool")
    a = p.parse_args()
    if not a.pool:
        for c in manifest():
            print("library", c["lib"], "<-", c["clip"], flush=True)
            build_lib(c)
    build_pool()


if __name__ == "__main__":
    main()
