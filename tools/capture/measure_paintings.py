"""Measure each painted mouth in the same units as the face trace, so paintings and real speech
share one mouth space.

    python measure_paintings.py REF_DIR

The opening is segmented from pixels (dark cavity, teeth, tongue) inside the mouth area, then:
  open  = opening height / outer-eye-corner distance   (trace: inner-lip gap / same distance)
  width = mouth-corner distance / outer-eye-corner distance
  round = how much narrower and taller-than-wide the opening is than the reference's
Writes REF_DIR/mouth_space.json and REF_DIR/mouth_space.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def opening(img, box):
    x0, y0, x1, y1 = box
    c = img[y0:y1, x0:x1].astype(int)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(-1), c.min(-1)
    dark = mx < 95                                            # cavity and lip creases
    teeth = (mn > 175) & (mx - mn < 45)                       # off-white teeth
    tongue = (r > 150) & (r - g > 55) & (b > g - 10) & (mx - mn > 60) & (g < 120)
    m = (dark | teeth | tongue).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return 0.0, 0.0, None
    cx, cy = (x1 - x0) / 2, (y1 - y0) / 2.4
    # the opening: the biggest blob near the mouth centre
    best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA] / (1 + 0.02 * np.hypot(*(cen[i] - (cx, cy)))))
    ys, xs = np.nonzero(lab == best)
    # height as the median column height across the middle half (robust to stray creases)
    cols = [np.ptp(ys[xs == x]) + 1 for x in range(int(np.percentile(xs, 25)), int(np.percentile(xs, 75)) + 1)
            if np.any(xs == x)]
    return float(np.median(cols)), float(np.ptp(xs) + 1), lab == best


def main(ref_dir):
    d = Path(ref_dir)
    rig = json.loads((d / "rig.json").read_text(encoding="utf-8"))
    ref = np.asarray(Image.open(d / "base.png").convert("RGB"))
    eyes = rig["eyes"]
    face = np.hypot(*(np.array(eyes[1]["center"]) + [eyes[1]["axes"][0], 0]
                      - np.array(eyes[0]["center"]) + [eyes[0]["axes"][0], 0]))
    box = rig.get("mouth_box", [660, 540, 845, 690])
    items = {"base": ref}
    for k, v in rig["variants"].items():
        if v["region"].startswith("mouth"):
            items[k] = np.asarray(Image.open(d / f"var_{k}.png").convert("RGB"))
    # the closed mouths have no opening at all; their measure is 0 by definition
    closed = {"closed", "mb"}
    space, tiles = {}, []
    for k, im in items.items():
        h, w, mask = opening(im, box)
        if k in closed:
            h = 0.0
        space[k] = {"open": round(h / face, 4), "width": round(w / face, 4)}
        t = Image.fromarray(im[box[1]:box[3], box[0]:box[2]]).convert("RGB")
        if mask is not None and k not in closed:
            ov = np.asarray(t).copy()
            ov[mask] = (0.5 * ov[mask] + [0, 127, 127]).astype(np.uint8)
            t = Image.fromarray(ov)
        t = t.resize((t.width * 2, t.height * 2))
        ImageDraw.Draw(t).text((4, 4), f"{k} o{space[k]['open']:.2f} w{space[k]['width']:.2f}", fill=(255, 255, 0))
        tiles.append(t)
    ref_ratio = space["base"]["open"] / max(space["base"]["width"], 1e-3)
    for k, s in space.items():
        ratio = s["open"] / max(s["width"], 1e-3)
        s["round"] = round(float(np.clip((ratio - ref_ratio) * 2.5, 0, 1)), 3) if s["width"] else 0.0
    rig_round = {"oo": 1.0, "o": 0.6}  # pursed lips hide the opening's shape; trust the drawing's intent
    for k, v in rig_round.items():
        if k in space:
            space[k]["round"] = max(space[k]["round"], v)
    (d / "mouth_space.json").write_text(json.dumps(space, indent=1), encoding="utf-8")
    sheet = Image.new("RGB", (tiles[0].width * 4, tiles[0].height * ((len(tiles) + 3) // 4)))
    for i, t in enumerate(tiles):
        sheet.paste(t, (t.width * (i % 4), t.height * (i // 4)))
    sheet.save(d / "mouth_space.png")
    for k, s in space.items():
        print(f"{k:8s} open {s['open']:.3f} width {s['width']:.3f} round {s['round']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
