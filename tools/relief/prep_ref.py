"""Cut a character out of a reference image and estimate its depth (free, local, CPU).

    python prep_ref.py assets/haj_salama/refs/ref10.png assets/haj_relief/ref10

Writes  rgba.png     the reference with everything but the character made transparent
        depth.npy    relative depth per pixel (larger = nearer the camera), float32
        depth.png    the same, for looking at
Background removal: rembg (isnet-general-use). Depth: Depth Anything V2 Small (Apache-2.0).
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter
from rembg import new_session, remove

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toolpaths  # noqa: E402

DEPTH_MODEL = toolpaths.model("depth-anything-v2-small/onnx/model.onnx")


def cutout(img):
    """Salient-object matte, tightened with a green-screen key where the backdrop is green."""
    matte = remove(img, session=new_session("isnet-general-use"), only_mask=True)
    a = np.asarray(img.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = (g > r + 30) & (g > b + 30)
    m = np.asarray(matte).astype(float) / 255.0
    m[green] = 0
    m = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    return m


def depth(img, size=518):
    w, h = img.size
    s = size / max(w, h)
    tw, th = int(round(w * s / 14)) * 14, int(round(h * s / 14)) * 14
    x = np.asarray(img.convert("RGB").resize((tw, th), Image.BICUBIC)).astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    sess = ort.InferenceSession(str(DEPTH_MODEL), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    d = np.asarray(Image.fromarray(out.astype(np.float32)).resize((w, h), Image.BICUBIC))
    return d


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    m = cutout(img)
    rgba = img.copy()
    rgba.putalpha(m)
    rgba.save(out / "rgba.png")
    d = depth(img)
    np.save(out / "depth.npy", d.astype(np.float32))
    v = (d - d.min()) / (d.max() - d.min() + 1e-9)
    Image.fromarray((v * 255).astype(np.uint8)).save(out / "depth.png")
    print("ok", img.size, float(d.min()), float(d.max()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
