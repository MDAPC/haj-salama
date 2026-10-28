"""Closed-eyelid overlay for a reference image, from measured eye-opening ellipses.

    python lids.py REF.png RIG.json OUT.png

RIG.json "eyes": [{"center": [x, y], "axes": [a, b], "angle": deg}, ...] in image pixels.
"""
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LASH = (30, 16, 12, 255)
SS = 4


def lids(src, eyes):
    im = Image.open(src).convert("RGB")
    rgb = np.asarray(im).astype(float)
    over = Image.new("RGBA", (im.width * SS, im.height * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for e in eyes:
        (cx, cy), (a, b), ang = e["center"], e["axes"], math.radians(e.get("angle", 0))
        # lid skin: the colour of the lid just above the opening, darkening toward the lash line
        top = rgb[int(cy - b - 9):int(cy - b - 3), int(cx - a * 0.5):int(cx + a * 0.5)].reshape(-1, 3)
        top = top[top.max(1) - top.min(1) > 40]
        skin = np.median(top, 0) if len(top) else np.array([190, 110, 70])
        pts = []
        for t in np.linspace(0, 2 * math.pi, 72, endpoint=False):
            x, y = a * 1.06 * math.cos(t), b * 1.12 * math.sin(t)
            pts.append(((cx + x * math.cos(ang) - y * math.sin(ang)) * SS,
                        (cy + x * math.sin(ang) + y * math.cos(ang)) * SS))
        layer = Image.new("RGBA", over.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.polygon(pts, fill=tuple(int(v) for v in skin) + (255,))
        # soft shading band toward the bottom of the lid
        shade = Image.new("RGBA", over.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        for k in range(10):
            f = k / 10
            y = cy + b * (0.2 + 0.9 * f)
            sd.rectangle(((cx - a * 1.2) * SS, (y - b * 0.09) * SS, (cx + a * 1.2) * SS, (y + b * 0.09) * SS),
                         fill=(60, 20, 10, int(10 + 50 * f)))
        mask = Image.new("L", over.size, 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        layer.alpha_composite(Image.composite(shade, Image.new("RGBA", over.size, (0, 0, 0, 0)), mask))
        # feather the lid into the painting
        r, g, b_, al = layer.split()
        layer = Image.merge("RGBA", (r, g, b_, al.filter(ImageFilter.GaussianBlur(1.6 * SS))))
        over.alpha_composite(layer)
        # lid crease: a soft dark arc along the top of the closed lid
        crease = [p for p, t in zip(pts, np.linspace(0, 2 * math.pi, 72, endpoint=False)) if math.sin(t) < -0.35]
        crease.sort()
        d.line(crease, fill=(60, 26, 16, 150), width=int(1.6 * SS), joint="curve")
        # closed lash line sagging across the lower part, with a few lashes
        lash = []
        for t in np.linspace(-1, 1, 24):
            x = a * 0.98 * t
            y = b * (0.30 + 0.45 * (1 - t * t))
            lash.append(((cx + x * math.cos(ang) - y * math.sin(ang)) * SS,
                         (cy + x * math.sin(ang) + y * math.cos(ang)) * SS))
        d.line(lash, fill=LASH, width=int(3.0 * SS), joint="curve")
    over = over.resize(im.size, Image.LANCZOS)
    return over.filter(ImageFilter.GaussianBlur(0.4))


if __name__ == "__main__":
    src, rig, out = sys.argv[1:4]
    lids(src, json.load(open(rig, encoding="utf-8"))["eyes"]).save(out)
