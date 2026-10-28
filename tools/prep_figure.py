"""Derive the extra paintings a single-image figure needs.

    python prep_figure.py assets/haj_salama

Writes  blink_overlay.png   transparent except closed eyelids over each eye
        figure_nosmoke.png  cigarette smoke masked out (desk shots hide the hand)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LASH = (24, 14, 12, 255)


def eye_mask(rgb, box):
    """Eye white plus the iris holes inside it, within box."""
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1].astype(int)
    white = (crop.min(-1) > 185) & (crop.max(-1) - crop.min(-1) < 45)
    m = Image.fromarray((white * 255).astype(np.uint8)).copy()  # fromarray may share a read-only buffer
    ImageDraw.floodfill(m, (0, 0), 128)  # everything reachable from outside
    a = np.asarray(m)
    filled = (a != 128)  # white + enclosed holes (iris, pupil)
    m = Image.fromarray((filled * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
    return np.asarray(m) > 0


def skin_tone(rgb, mask, box):
    """Median colour of the band just above the eye (the upper lid skin)."""
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(mask)
    top = ys.min()
    band = rgb[y0 + max(top - 22, 0): y0 + max(top - 10, 1), x0 + xs.min(): x0 + xs.max()]
    band = band.reshape(-1, 3)
    band = band[band.max(-1) - band.min(-1) > 30]  # skip black lines / white
    return tuple(int(v) for v in np.median(band, 0))


def blink_overlay(src, boxes):
    im = Image.open(src).convert("RGBA")
    rgb = np.asarray(im)[..., :3]
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for box in boxes:
        x0, y0 = box[:2]
        mask = eye_mask(rgb, box)
        skin = skin_tone(rgb, mask, box)
        ys, xs = np.nonzero(mask)
        lid = np.zeros(mask.shape + (4,), np.uint8)
        # lid skin, a touch darker toward the bottom like a lowered eyelid
        span = max(ys.max() - ys.min(), 1)
        shade = 1 - 0.12 * (np.arange(mask.shape[0]) - ys.min()).clip(0, span)[:, None] / span
        for c in range(3):
            lid[..., c] = np.clip(skin[c] * shade, 0, 255)
        lid[..., 3] = mask * 255
        over.alpha_composite(Image.fromarray(lid, "RGBA"), (x0, y0))
        # closed-lid line across the lower third, sagging like the painted lash lines
        left, right = xs.min(), xs.max()
        base = ys.min() + 0.62 * span
        pts = [(x0 + x, y0 + base + 0.22 * span * np.sin(np.pi * (x - left) / max(right - left, 1)))
               for x in np.linspace(left + 2, right - 2, 16)]
        d.line(pts, fill=LASH, width=5, joint="curve")
        for (px, py) in pts[3:-3:3]:  # a few lashes
            d.line([(px, py), (px - 3, py + 7)], fill=LASH, width=2)
    return over


def nosmoke(src, region):
    a = np.asarray(Image.open(src)).copy()
    x0, y0, x1, y1 = region
    a[y0:y1, x0:x1, 3] = 0
    return Image.fromarray(a)


def main(fig_dir):
    fig = Path(fig_dir)
    rig = json.loads((fig / "rig.json").read_text(encoding="utf-8"))
    blink_overlay(fig / "source.png", rig["eye_boxes"]).save(fig / "blink_overlay.png")
    nosmoke(fig / "source.png", rig["smoke_region"]).save(fig / "figure_nosmoke.png")
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
