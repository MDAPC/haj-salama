"""Separate the moving parts of the face from a reference image.

    python prep_face.py REF.png REF_DIR

Reads REF_DIR/rgba.png and REF_DIR/rig.json. Writes into REF_DIR:
  base.png      the cut-out character with the irises painted over by eye white
  iris.png      just the irises (moved for gaze, clipped by eye_mask.png)
  eye_mask.png  the visible eye openings
  mouth.png     the mouth interior: the painted opening (teeth removed) continued downward
                into a dark cavity with a tongue, revealed when the jaw drops
  teeth.png     the painted lower teeth, which ride on the jaw
All images share the reference's pixel grid.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def ellipse_mask(shape, e, grow=1.0):
    m = Image.new("L", (shape[1], shape[0]), 0)
    (cx, cy), (a, b), ang = e["center"], e["axes"], math.radians(e.get("angle", 0))
    pts = []
    for t in np.linspace(0, 2 * math.pi, 90, endpoint=False):
        x, y = a * grow * math.cos(t), b * grow * math.sin(t)
        pts.append((cx + x * math.cos(ang) - y * math.sin(ang), cy + x * math.sin(ang) + y * math.cos(ang)))
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return np.asarray(m) > 0


def mouth_polygon(M, drop=0):
    up = [tuple(p) for p in M["upper"]]
    lo = [(x, y + drop * (1 - abs(2 * (x - up[0][0]) / (up[-1][0] - up[0][0]) - 1) ** 2))
          for x, y in M["lower"]]
    return up + lo[::-1][1:-1]


def main(ref, ref_dir):
    d = Path(ref_dir)
    rig = json.loads((d / "rig.json").read_text(encoding="utf-8"))
    rgba = np.asarray(Image.open(d / "rgba.png")).copy()
    rgb = rgba[..., :3].copy()
    h, w = rgb.shape[:2]

    # ---- eyes: lift the irises out, paint eye white where they were
    base = rgb.copy()
    iris = np.zeros((h, w, 4), np.uint8)
    eye_mask = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for e in rig["eyes"]:
        inside = ellipse_mask((h, w), e, 0.94)
        eye_mask |= inside
        (ix, iy), r = e["iris"], e["iris_r"]
        dist = np.hypot(xx - ix, yy - iy)
        disc = dist < r + 0.5
        a = np.clip((r + 1.0 - dist) / 1.5, 0, 1)
        iris[..., :3] = np.where(disc[..., None], rgb, iris[..., :3])
        iris[..., 3] = np.maximum(iris[..., 3], (a * 255).astype(np.uint8))
        hole = (dist < r + 2.5) & inside
        # eye white continued in from its surroundings, then shaded under the upper lid
        base = cv2.inpaint(base, hole.astype(np.uint8) * 255, 6, cv2.INPAINT_TELEA)
        white = rgb[inside & ~(dist < r + 3)]
        white = white[white.min(1) > 150]
        if len(white):
            base[hole] = (0.5 * base[hole] + 0.5 * np.median(white, 0)).astype(np.uint8)
        (cx, cy), (ax, by) = e["center"], e["axes"]
        lid = np.clip(1 - (yy - (cy - by)) / (by * 1.1), 0, 1) * hole
        base = (base * (1 - 0.30 * lid[..., None])).astype(np.uint8)
        base = np.where(hole[..., None], cv2.GaussianBlur(base, (5, 5), 0), base)

    # ---- mouth: interior layer and the lower teeth
    M = rig["mouth"]
    drop = rig.get("jaw_drop", 24)
    open_now = Image.new("L", (w, h), 0)
    ImageDraw.Draw(open_now).polygon(mouth_polygon(M), fill=255)
    open_now = np.asarray(open_now) > 0
    # the interior sits behind the face and only shows through the opening, so make it generous:
    # the whole box the opening can sweep (corners and a full jaw drop included)
    xs = [p[0] for p in M["upper"]]
    ys_up = min(p[1] for p in M["upper"])
    ys_lo = max(p[1] for p in M["lower"])
    open_max = np.zeros((h, w), bool)
    open_max[ys_up - 8:ys_lo + drop + 16, min(xs) - 22:max(xs) + 23] = True
    face = cv2.erode((rgba[..., 3] > 128).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    open_max &= face  # never outside his silhouette (the corner is near the cheek edge in 3/4 view)

    teeth_m = open_now & (rgb.min(2) > 150)
    teeth_m = cv2.dilate(teeth_m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    teeth = np.zeros((h, w, 4), np.uint8)
    teeth[teeth_m, :3] = rgb[teeth_m]
    teeth[teeth_m, 3] = 255

    # cavity: dark warm gradient, a tongue low down, then the painted opening on top (minus teeth)
    ys = np.nonzero(open_max)[0]
    y0, y1 = ys.min(), ys.max()
    t = np.clip((yy - y0) / max(y1 - y0, 1), 0, 1)
    cav = np.stack([70 - 45 * t, 22 - 14 * t, 20 - 12 * t], -1)
    cx = np.mean([p[0] for p in M["upper"]])
    tongue = np.clip(1 - np.hypot((xx - cx) / 34.0, (yy - (M["lower"][3][1] + drop * 0.7)) / 12.0), 0, 1)
    tongue = np.clip(tongue * 3, 0, 1)[..., None]
    cav = cav * (1 - tongue) + np.array([150, 62, 58]) * tongue
    interior = cav.astype(np.uint8)
    painted = cv2.inpaint(rgb, teeth_m.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    interior[open_now] = painted[open_now]
    mouth = np.zeros((h, w, 4), np.uint8)
    mouth[..., :3] = interior
    mouth[..., 3] = (open_max * 255).astype(np.uint8)

    out = rgba.copy()
    out[..., :3] = base
    Image.fromarray(out).save(d / "base.png")
    Image.fromarray(iris).save(d / "iris.png")
    Image.fromarray((eye_mask * 255).astype(np.uint8)).save(d / "eye_mask.png")
    Image.fromarray(mouth).save(d / "mouth.png")
    Image.fromarray(teeth).save(d / "teeth.png")
    print("ok", "teeth px", int(teeth_m.sum()), "mouth rows", int(y0), int(y1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
