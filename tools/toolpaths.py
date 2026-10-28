"""Where the external programs and models live, on Windows, Linux and macOS.

Each one is looked up in this order:
  1. its environment variable (BLENDER, RHUBARB, FFMPEG, HAJ_MP_PYTHON, HAJ_TOOLS for the folder),
  2. the program on PATH,
  3. the repo's own .tools/ folder (filled by `python tools/setup.py`),
  4. ~/.local/opt (the original Linux install).
Scripts import it after adding the tools/ folder to sys.path.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""


def homes():
    out = []
    if os.environ.get("HAJ_TOOLS"):
        out.append(Path(os.environ["HAJ_TOOLS"]))
    out += [ROOT / ".tools", Path.home() / ".local/opt"]
    return out


def _find(name, env, patterns, extra=()):
    if os.environ.get(env):
        return os.environ[env]
    found = shutil.which(name)
    if found:
        return found
    for h in list(homes()) + [Path(e) for e in extra]:
        for pat in patterns:
            hits = sorted(h.glob(pat)) if h.exists() else []
            if hits:
                return str(hits[-1])
    return None


def blender():
    win = [os.environ.get("ProgramFiles", r"C:\Program Files") + r"\Blender Foundation"] if os.name == "nt" else []
    return _find("blender", "BLENDER", [f"blender-*/blender{EXE}", f"Blender*/blender{EXE}"], win)


def rhubarb():
    return _find("rhubarb", "RHUBARB", [f"Rhubarb-Lip-Sync-*/rhubarb{EXE}"])


def ffmpeg():
    return _find("ffmpeg", "FFMPEG", [f"ffmpeg*/bin/ffmpeg{EXE}", f"ffmpeg*/ffmpeg{EXE}"]) or "ffmpeg"


def rife():
    """The RIFE ncnn-vulkan folder (optional: only tools/relief/arm_tweens.py uses it)."""
    for h in homes():
        hits = sorted(h.glob("rife-ncnn-vulkan*")) if h.exists() else []
        if hits:
            return hits[-1]
    return ROOT / ".tools" / "rife-ncnn-vulkan"


def _first(rel):
    for h in homes():
        if (h / rel).exists():
            return h / rel
    return ROOT / ".tools" / rel  # where tools/setup.py puts it


def model(rel):
    """A model file under models/, e.g. model("mediapipe/face_landmarker.task")."""
    return _first(Path("models") / rel)


def voice(name):
    """A Piper voice, e.g. voice("ar_JO-kareem-medium.onnx")."""
    return _first(Path("piper-voices") / name)


def emage():
    """The emage-onnx-export folder (optional, non-commercial; see docs/CAPTURE.md)."""
    return _first(Path("emage") / "emage-onnx-export")


def mediapipe_python():
    """The Python that has MediaPipe (the same one by default)."""
    return os.environ.get("HAJ_MP_PYTHON") or sys.executable


def utf8_env():
    """Environment for child processes: UTF-8 text everywhere (Windows defaults to a code page)."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env
