"""Pick facial expressions out of a Veo close-up of the character and save them as paintings in
the reference's pixel space, ready for prep_variants.py.

    python veo_expressions.py CLIP.mp4 TRACE.json OUT_DIR --crop X0 Y0 X1 Y1

The close-up was generated from the reference cropped to (X0, Y0, X1, Y1) (same 16:9 shape), so
each video frame maps back onto that box. TRACE.json: trace_face.py on the clip (blendshapes).
For each expression the frame where it peaks is taken from the blendshapes:
  frown     brows pulled down          worried   inner brows up, not down
  skeptic   one brow up, the other not  sad       looking down with inner brows up
  closed    eyes shut (a sigh)          neutral   least of everything (for checking)
Writes OUT_DIR/expr_<name>.png (reference size, green outside the box) and OUT_DIR/picks.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H = 1920, 1080


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("trace")
    p.add_argument("out")
    p.add_argument("--crop", type=int, nargs=4, required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fr = json.loads(Path(a.trace).read_text(encoding="utf-8"))["frames"]
    n = len(fr)

    def bs(k):
        v = np.array([f.get("bs", {}).get(k, np.nan) for f in fr], float)
        ok = ~np.isnan(v)
        v = np.interp(np.arange(n), np.flatnonzero(ok), v[ok])
        return np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, "valid")  # steady peaks, not blips

    down = (bs("browDownLeft") + bs("browDownRight")) / 2
    inner = bs("browInnerUp")
    outer_l, outer_r = bs("browOuterUpLeft"), bs("browOuterUpRight")
    blink = (bs("eyeBlinkLeft") + bs("eyeBlinkRight")) / 2
    look_down = (bs("eyeLookDownLeft") + bs("eyeLookDownRight")) / 2
    open_eyes = blink < 0.35
    score = {
        "frown": down - 0.5 * inner,
        "worried": inner - down,
        "skeptic": np.abs(outer_l - outer_r) + np.abs(bs("browDownLeft") - bs("browDownRight")),
        "sad": look_down + 0.5 * inner,
        "closed": blink,
        "neutral": -(down + inner + np.abs(outer_l - outer_r) + blink + look_down),
    }
    picks, used = {}, []
    for name, s in score.items():
        s = s.copy()
        if name != "closed":
            s[~open_eyes] = -np.inf  # an expression with the eyes caught mid-blink is no use
        for u in used:  # different moments for different expressions
            s[max(0, u - 12):u + 12] = -np.inf
        f = int(np.argmax(s))
        used.append(f)
        picks[name] = {"frame": f, "t": round(f / 24, 2), "score": round(float(s[f]), 3)}
    cap = cv2.VideoCapture(a.clip)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x0, y0, x1, y1 = a.crop
    for name, pk in picks.items():
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:] = (0, 177, 64)
        canvas[y0:y1, x0:x1] = cv2.resize(frames[pk["frame"]], (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        Image.fromarray(canvas).save(out / f"expr_{name}.png")
    (out / "picks.json").write_text(json.dumps(picks, indent=1), encoding="utf-8")
    print(picks)


if __name__ == "__main__":
    main()
