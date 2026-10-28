"""Get the free models and tools, then check that everything is in place (Windows, Linux, macOS).

    python tools/setup.py              # models, Piper voices, Rhubarb -> .tools/, then the check
    python tools/setup.py --check      # only the check
    python tools/setup.py --blender    # also a portable Blender 5.2 (Windows / Linux) into .tools/
    python tools/setup.py --rife       # also RIFE (optional: arm in-betweens, tools/relief/arm_tweens.py)
    python tools/setup.py --stock      # also the CC BY stock interview clips (assets/capture/stock/)

Python packages come from requirements.txt (pip install -r requirements.txt). ffmpeg and Blender
can also come from the system (winget / apt / brew); see docs/SETUP.md. Anything already found by
tools/toolpaths.py (env variable, PATH, .tools/, ~/.local/opt) is not downloaded again.
"""
import argparse
import importlib
import platform
import shutil
import stat
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import toolpaths  # noqa: E402

TOOLS = ROOT / ".tools"
OS = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
HF = "https://huggingface.co"
MP = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "depth-anything-v2-small/onnx/model.onnx": f"{HF}/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
    "mediapipe/face_landmarker.task": f"{MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "mediapipe/pose_landmarker_full.task": f"{MP}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "mediapipe/hand_landmarker.task": f"{MP}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
PV = f"{HF}/rhasspy/piper-voices/resolve/main"
VOICES = {
    "ar_JO-kareem-medium.onnx": f"{PV}/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
    "ar_JO-kareem-medium.onnx.json": f"{PV}/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json",
    "en_US-lessac-medium.onnx": f"{PV}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-lessac-medium.onnx.json": f"{PV}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
}
RHUBARB = ("https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/"
           "Rhubarb-Lip-Sync-1.14.0-{}.zip").format({"windows": "Windows", "macos": "macOS", "linux": "Linux"}[OS])
RIFE = ("https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/"
        "rife-ncnn-vulkan-20221029-{}.zip").format({"windows": "windows", "macos": "macos", "linux": "ubuntu"}[OS])
BLENDER = {"windows": "https://download.blender.org/release/Blender5.2/blender-5.2.3-windows-x64.zip",
           "linux": "https://download.blender.org/release/Blender5.2/blender-5.2.3-linux-x64.tar.xz"}
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"
STOCK = {  # motion-capture references, CC BY (see assets/capture/stock/SOURCE.md)
    "takouti.webm": COMMONS + "Wikimedia_Strategy_2030_-_Roles_%26_Responsibilities_-_Interview_Zeinab_Takouti_"
                              "(AR)_-_%D8%B2%D9%8A%D9%86%D8%A8_%D8%AA%D8%A7%D9%83%D9%88%D8%AA%D9%8A.webm",
    "wales.webm": COMMONS + "Interview_with_Jimmy_Wales_at_Wikimania_2025_Nairobi.webm",
}
PACKAGES = {"numpy": "numpy", "PIL": "pillow", "cv2": "opencv-python", "onnxruntime": "onnxruntime",
            "rembg": "rembg", "soundfile": "soundfile", "piper": "piper-tts", "mediapipe": "mediapipe"}


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  downloading {dest.name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "haj-salama-setup/1.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)


def unpack(archive, into):
    print(f"  unpacking {archive.name} ...", flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                out = z.extract(info, into)
                if info.external_attr >> 16 & 0o111:  # keep the executable bit (Linux / macOS)
                    Path(out).chmod(Path(out).stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into, filter="data")
    archive.unlink()


def get_models():
    for rel, url in MODELS.items():
        if not toolpaths.model(rel).exists():
            fetch(url, TOOLS / "models" / rel)
    for name, url in VOICES.items():
        if not toolpaths.voice(name).exists():
            fetch(url, TOOLS / "piper-voices" / name)


def get_archive(url, found, what):
    if found:
        return
    print(what)
    arc = TOOLS / url.rsplit("/", 1)[1]
    fetch(url, arc)
    unpack(arc, TOOLS)


def check():
    ok = True

    def row(name, good, hint=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok' if good else '--'}] {name}" + ("" if good else f"   -> {hint}"))

    print(f"\nCheck (Python {platform.python_version()}, {platform.system()}):")
    for mod, pkg in PACKAGES.items():
        try:
            importlib.import_module(mod)
            row(f"python package {pkg}", True)
        except Exception:  # noqa: BLE001  (a broken install raises more than ImportError)
            alt = mod == "mediapipe" and "HAJ_MP_PYTHON" in __import__("os").environ
            row(f"python package {pkg}" + (" (HAJ_MP_PYTHON is set)" if alt else ""), alt,
                "pip install -r requirements.txt")
    b = toolpaths.blender()
    row(f"Blender: {b}", b and Path(b).exists(), "install Blender 5.2 (winget install BlenderFoundation.Blender) "
                                                 "or python tools/setup.py --blender, or set BLENDER")
    f = shutil.which(toolpaths.ffmpeg()) or (Path(toolpaths.ffmpeg()).exists() and toolpaths.ffmpeg())
    row(f"ffmpeg: {f}", f, "winget install Gyan.FFmpeg / apt install ffmpeg / brew install ffmpeg, or set FFMPEG")
    r = toolpaths.rhubarb()
    row(f"Rhubarb Lip Sync: {r}", r and Path(r).exists(), "python tools/setup.py")
    for rel in MODELS:
        row(f"model {rel}", toolpaths.model(rel).exists(), "python tools/setup.py")
    for name in VOICES:
        row(f"Piper voice {name}", toolpaths.voice(name).exists(), "python tools/setup.py")
    pool = ROOT / "assets/haj_salama/veo/pool/pool.json"
    print(f"  [{'ok' if pool.exists() else '..'}] Veo gesture pool (standing figure)"
          + ("" if pool.exists() else "   -> optional: python tools/capture/veo_rebuild.py"))
    print("\nAll set." if ok else "\nSome pieces are missing (see the hints above).")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--blender", action="store_true")
    p.add_argument("--rife", action="store_true")
    p.add_argument("--stock", action="store_true")
    a = p.parse_args()
    if not a.check:
        print("Models and voices:")
        get_models()
        get_archive(RHUBARB, toolpaths.rhubarb(), "Rhubarb Lip Sync:")
        if a.blender:
            if OS == "macos":
                print("Blender on macOS: install it from blender.org (a .dmg), or set BLENDER.")
            else:
                get_archive(BLENDER[OS], toolpaths.blender(), "Blender 5.2:")
        if a.rife:
            get_archive(RIFE, (toolpaths.rife() / "rife-v4.6").exists(), "RIFE:")
        if a.stock:
            for name, url in STOCK.items():
                if not (ROOT / "assets/capture/stock" / name).exists():
                    fetch(url, ROOT / "assets/capture/stock" / name)
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
