"""Painted mouth shapes for ref10 with local Stable Diffusion inpainting (optional, experimental).

    python tools/gen/batch_mouths.py

Rig pose -> reference space (from_pose.py) -> local inpainting at medium strength (inpaint.py).
Needs the episodes/ar01 rig pose stills (build_relief.py --poses) and a Python with torch +
diffusers, named by HAJ_SD_PYTHON (default: this Python). The Gemini paintings in
assets/haj_salama/gen/ replaced this route; it is kept for fully local runs.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "assets/haj_salama/gen_local"
POSES = ROOT / "episodes/ar01/out/relief_ref10"
PROMPT = {
    "D": "mouth wide open saying aa, jaw dropped, dark mouth inside, tongue low, gap teeth, grey mustache",
    "C": "mouth half open, dark mouth inside, a few teeth, grey mustache",
    "B": "lips slightly parted showing teeth, grey mustache",
    "E": "mouth open and rounded saying o, grey mustache",
    "F": "lips pursed and pushed forward in a small round o, grey mustache",
}


def main():
    sd = os.environ.get("HAJ_SD_PYTHON") or sys.executable
    for v, prompt in PROMPT.items():
        subprocess.run([sys.executable, ROOT / "tools/gen/from_pose.py", ROOT / "assets/haj_salama/refs/ref10.png",
                        ROOT / "assets/haj_relief/ref10/frame.json", POSES / f"pose_v_{v}.png", G / f"_init_{v}.png",
                        "--box", "620", "480", "880", "740"], check=True)
        subprocess.run([sd, ROOT / "tools/gen/inpaint.py", G / f"_init_{v}.png", G / f"mouth_{v}.png",
                        "--box", "620", "480", "880", "740", "--mask", "ellipse", "750", "590", "70", "42",
                        "--prompt", prompt, "--seeds", "3", "--strength", "0.55", "--steps", "30"], check=True)


if __name__ == "__main__":
    main()
