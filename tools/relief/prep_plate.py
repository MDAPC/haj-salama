"""Background plates from a reference shot: the studio with the character painted out,
widened to 16:9, and the green screen filled with each screen image.

    python prep_plate.py assets/haj_salama/refs/ref10.png assets/haj_relief/ref10 IMG [IMG...]

Needs rgba.png (from prep_ref.py) in the output folder. Writes plates/<image stem>.png and
frame.json (how reference pixels map onto the 1920x1080 frame).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

W, H = 1920, 1080


def green_key(rgb):
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return (g > r + 25) & (g > b + 25) & (g > 70)


def main(ref, out, screens):
    out = Path(out)
    rgb = np.asarray(Image.open(ref).convert("RGB"))
    alpha = np.asarray(Image.open(out / "rgba.png"))[..., 3]
    h, w = alpha.shape

    # paint the character out (hidden behind the relief anyway; only edges show when he turns)
    hole = cv2.dilate((alpha > 8).astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # where the hole sits on the green screen, fill with green so the key still finds the screen
    filled = cv2.inpaint(bgr, hole, 9, cv2.INPAINT_TELEA)

    # widen to 16:9: scale to the frame height and stretch the outermost columns
    # (neon bars become long bars, the screen and desk simply continue)
    s = H / h
    sw = int(round(w * s))
    scaled = cv2.resize(filled, (sw, H), interpolation=cv2.INTER_AREA)
    left = (W - sw) // 2
    right = W - sw - left
    wide = cv2.copyMakeBorder(scaled, 0, 0, left, right, cv2.BORDER_REPLICATE)
    plate = cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)

    key = green_key(plate).astype(np.uint8)
    key = cv2.morphologyEx(key, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    key = cv2.morphologyEx(key, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    ys, xs = np.nonzero(key)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    soft = np.asarray(Image.fromarray(key * 255).filter(ImageFilter.GaussianBlur(1.5))).astype(float)[..., None] / 255

    (out / "plates").mkdir(exist_ok=True)
    for img in screens:
        card = Image.open(img).convert("RGB")
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        k = max(bw / card.width, bh / card.height)  # cover the screen area
        card = card.resize((int(card.width * k) + 1, int(card.height * k) + 1), Image.LANCZOS)
        cx, cy = (card.width - bw) // 2, (card.height - bh) // 2
        layer = np.zeros_like(plate)
        layer[y0:y1 + 1, x0:x1 + 1] = np.asarray(card)[cy:cy + bh, cx:cx + bw]
        comp = (plate * (1 - soft) + layer * soft).astype(np.uint8)
        Image.fromarray(comp).save(out / "plates" / f"{Path(img).stem}.png")
    (out / "frame.json").write_text(json.dumps({"scale": s, "offset": [left, 0], "ref_size": [w, h],
                                                "screen_box": [int(x0), int(y0), int(x1), int(y1)]}), encoding="utf-8")
    print("ok", (x0, y0, x1, y1), "scale", round(s, 4), "offset", left)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
