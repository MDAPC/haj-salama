"""Paint a placeholder news-anchor figure as separate layers + figure.json.

This is the contract a real painted figure must follow: one PNG per layer
(any size, placed at `pos` on the canvas), named pivots, and the nine
Rhubarb mouth shapes A-H + X. Swap the PNGs for real paintings and keep
the manifest.

    python make_placeholder.py <out_dir>
"""
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

SS = 4  # supersampling for smooth edges
CANVAS = (1000, 1400)

SKIN = (222, 176, 140)
SKIN_SHADE = (190, 140, 110)
HAIR = (52, 36, 30)
SUIT = (38, 50, 78)
SHIRT = (236, 234, 228)
TIE = (150, 34, 40)
LIP = (178, 92, 90)
MOUTH_IN = (62, 20, 26)
TEETH = (242, 238, 228)
TONGUE = (196, 96, 104)
EYE_W = (245, 242, 236)
IRIS = (70, 48, 36)
LASH = (40, 26, 22)

rng = np.random.default_rng(7)


class Layer:
    """A supersampled RGBA drawing surface covering a canvas box."""

    def __init__(self, box):
        x0, y0, x1, y1 = box
        self.pos = (x0, y0)
        self.img = Image.new("RGBA", ((x1 - x0) * SS, (y1 - y0) * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def _p(self, pts):
        return [((x - self.pos[0]) * SS, (y - self.pos[1]) * SS) for x, y in pts]

    def ellipse(self, cx, cy, rx, ry, fill):
        self.d.ellipse(self._p([(cx - rx, cy - ry), (cx + rx, cy + ry)]), fill=fill)

    def poly(self, pts, fill):
        self.d.polygon(self._p(pts), fill=fill)

    def line(self, pts, fill, width):
        self.d.line(self._p(pts), fill=fill, width=int(width * SS), joint="curve")

    def rect(self, x0, y0, x1, y1, fill, r=0):
        self.d.rounded_rectangle(self._p([(x0, y0), (x1, y1)]), radius=r * SS, fill=fill)

    def save(self, path, paint=True):
        img = self.img.resize((self.img.width // SS, self.img.height // SS), Image.LANCZOS)
        if paint:
            img = painterly(img)
        img.save(path)


def painterly(img):
    """Fake brush texture: low-frequency tone drift + directional stroke grain."""
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    small = rng.normal(0, 1, (max(2, h // 40), max(2, w // 40))).astype(np.float32)
    drift = np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC))
    strokes = rng.normal(0, 1, (h, max(2, w // 6))).astype(np.float32)
    strokes = np.asarray(Image.fromarray(strokes).resize((w, h), Image.BILINEAR))
    grain = rng.normal(0, 1, (h, w)).astype(np.float32)
    k = 1 + 0.045 * drift + 0.03 * strokes + 0.018 * grain
    a[..., :3] = np.clip(a[..., :3] * k[..., None], 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def body():
    L = Layer((0, 700, 1000, 1400))
    # neck (lives on the body so the head can rotate over it)
    L.rect(440, 740, 560, 900, SKIN_SHADE, r=30)

    def up(pts, dy=-60):
        return [(x, y + dy) for x, y in pts]

    # jacket
    L.poly(up([(150, 1460), (110, 1080), (190, 960), (400, 905), (600, 905),
               (810, 960), (890, 1080), (850, 1460)]), SUIT)
    L.ellipse(205, 980, 100, 90, SUIT)
    L.ellipse(795, 980, 100, 90, SUIT)
    # shirt V and tie
    L.poly(up([(410, 905), (590, 905), (500, 1120)]), SHIRT)
    L.poly(up([(480, 925), (520, 925), (530, 960), (500, 975), (470, 960)]), TIE)
    L.poly(up([(472, 970), (528, 970), (550, 1120), (500, 1160), (450, 1120)]), TIE)
    # lapels
    L.poly(up([(400, 905), (455, 1000), (430, 1040), (500, 1220), (380, 1000)]), (30, 40, 64))
    L.poly(up([(600, 905), (545, 1000), (570, 1040), (500, 1220), (620, 1000)]), (30, 40, 64))
    return L


def head():
    L = Layer((300, 330, 700, 830))
    L.ellipse(335, 610, 26, 42, SKIN_SHADE)  # ears
    L.ellipse(665, 610, 26, 42, SKIN_SHADE)
    L.ellipse(500, 600, 165, 210, SKIN)
    L.ellipse(500, 700, 120, 95, (214, 166, 132))  # lower-face shade
    # hair with side part
    L.poly([(335, 560), (340, 470), (390, 400), (470, 372), (560, 378),
            (630, 420), (668, 500), (665, 570), (640, 500), (560, 460),
            (470, 470), (400, 500), (360, 560)], HAIR)
    # nose
    L.line([(505, 610), (515, 670), (495, 685)], SKIN_SHADE, 6)
    L.ellipse(500, 688, 22, 8, (205, 155, 122))
    return L


def eyes(closed):
    L = Layer((380, 570, 620, 630))
    for cx in (435, 565):
        if closed:
            L.ellipse(cx, 600, 36, 20, SKIN)
            L.line([(cx - 32, 600), (cx, 608), (cx + 32, 600)], LASH, 4)
        else:
            L.ellipse(cx, 600, 32, 18, EYE_W)
            L.ellipse(cx + 2, 601, 14, 14, IRIS)
            L.ellipse(cx + 2, 601, 6, 6, (15, 10, 10))
            L.ellipse(cx + 6, 596, 3, 3, (255, 255, 255))
            L.line([(cx - 34, 598), (cx - 10, 584), (cx + 14, 584), (cx + 34, 596)], LASH, 4)
    return L


def brows():
    L = Layer((380, 535, 620, 580))
    L.line([(400, 566), (430, 552), (468, 556)], HAIR, 10)
    L.line([(532, 556), (570, 552), (600, 566)], HAIR, 10)
    return L


MOUTH_BOX = (420, 695, 580, 785)
MC = (500, 735)


def mouth(shape):
    """Rhubarb shapes: A MBP, B clenched teeth, C open, D wide, E round,
    F pucker, G F/V, H L, X rest."""
    L = Layer(MOUTH_BOX)
    cx, cy = MC

    def opening(w, h, teeth_top=True, teeth_bot=False, tongue=False):
        L.ellipse(cx, cy, w + 7, h + 8, LIP)
        L.ellipse(cx, cy, w, h, MOUTH_IN)
        if tongue:
            L.ellipse(cx, cy + h * 0.7, w * 0.65, h * 0.45, TONGUE)
        if teeth_top:
            L.rect(cx - w * 0.75, cy - h, cx + w * 0.75, cy - h + min(12, h * 0.6), TEETH, r=3)
        if teeth_bot:
            L.rect(cx - w * 0.7, cy + h - min(9, h * 0.5), cx + w * 0.7, cy + h, TEETH, r=3)

    if shape == "X":
        L.line([(cx - 40, cy), (cx, cy + 3), (cx + 40, cy)], LIP, 9)
        L.line([(cx - 36, cy), (cx, cy + 2), (cx + 36, cy)], MOUTH_IN, 2)
    elif shape == "A":
        L.ellipse(cx, cy, 38, 9, LIP)
        L.line([(cx - 36, cy), (cx + 36, cy)], MOUTH_IN, 2)
    elif shape == "B":
        opening(44, 10, teeth_top=True, teeth_bot=True)
    elif shape == "C":
        opening(46, 20, tongue=True)
    elif shape == "D":
        opening(50, 32, tongue=True)
    elif shape == "E":
        opening(34, 22)
    elif shape == "F":
        L.ellipse(cx, cy, 24, 18, LIP)
        L.ellipse(cx, cy, 11, 9, MOUTH_IN)
    elif shape == "G":
        L.ellipse(cx, cy + 4, 46, 12, LIP)
        L.rect(cx - 34, cy - 10, cx + 34, cy + 2, TEETH, r=3)
    elif shape == "H":
        opening(44, 22, tongue=False)
        L.ellipse(cx, cy - 8, 20, 10, TONGUE)
    return L


def studio_bg(path):
    w, h = 1920, 1080
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    chans = [16 + 30 * (1 - y) + 10 * x, 30 + 40 * (1 - y), 62 + 60 * (1 - y)]
    base = np.stack([np.broadcast_to(c, (h, w)) for c in chans], -1)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    rnd = random.Random(3)
    for _ in range(60):
        r = rnd.uniform(20, 90)
        cx, cy = rnd.uniform(0, w), rnd.uniform(0, h * 0.7)
        c = rnd.choice([(90, 140, 220), (200, 210, 255), (70, 110, 200)])
        g.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c + (rnd.randint(20, 55),))
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    img.alpha_composite(glow)
    painterly(img).convert("RGB").save(path)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layers = []

    def add(name, layer, group, z, role=None):
        layer.save(out / f"{name}.png")
        e = {"name": name, "file": f"{name}.png", "pos": list(layer.pos), "group": group, "z": z}
        if role:
            e["role"] = role
        layers.append(e)

    add("body", body(), "body", 0)
    add("head", head(), "head", 1)
    add("eyes_open", eyes(False), "head", 2, "eyes_open")
    add("eyes_closed", eyes(True), "head", 2, "eyes_closed")
    add("brows", brows(), "brows", 3)
    mouths = {}
    for s in "ABCDEFGHX":
        m = mouth(s)
        m.save(out / f"mouth_{s}.png")
        mouths[s] = f"mouth_{s}.png"

    manifest = {
        "canvas": list(CANVAS),
        "groups": {
            "body": {"pivot": [500, 1400]},
            "head": {"pivot": [500, 800], "parent": "body"},
            "brows": {"pivot": [500, 556], "parent": "head"},
        },
        "layers": layers,
        "mouth": {"pos": [MOUTH_BOX[0], MOUTH_BOX[1]], "group": "head", "z": 3, "shapes": mouths},
    }
    (out / "figure.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    studio_bg(out.parent / "studio_bg.png")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets/anchor_placeholder")
