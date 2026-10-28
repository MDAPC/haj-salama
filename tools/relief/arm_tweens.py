"""Cut-out in-betweens for the gesturing arm, like 2D cut-out animation.

    python arm_tweens.py REF.png REF_DIR [--preview OUT.png]

Optical flow and RIFE smear a hand that travels far between two paintings. Here the forearm + hand
of the higher pose is cut out and rotated about the elbow part of the way back toward the lower
pose's angle, over that pose's body with its own arm removed (painted green, which the variant
pipeline turns into "show the studio"). Writes gen/tween/<A>__<B>_<t>.png for each rig "tweens"
entry {"a": lower pose, "b": higher pose}, at t = 1/3 and 2/3, and registers them in rig.json.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toolpaths  # noqa: E402

GREEN = np.array([40, 170, 40], np.uint8)
TS = (0.33, 0.67)
HINGE = 70  # px around the elbow that does not swing


def load_full(ref_dir, rig, ref, name):
    """A pose as a full reference-space image (aligned patch over the reference)."""
    v = np.asarray(Image.open(ref_dir / f"var_{name}.png")).astype(np.float32)
    a = v[..., 3:] / 255
    return (ref * (1 - a) + v[..., :3] * a).astype(np.uint8)


def arm_mask(img, rest, box, elbow):
    """Pixels of this pose's raised forearm+hand: what differs from the resting pose inside the
    arm box, the component touching nearest the elbow, cleaned and slightly grown."""
    x0, y0, x1, y1 = box
    d = np.abs(img.astype(int) - rest.astype(int)).sum(2) > 70
    m = np.zeros(d.shape, np.uint8)
    m[y0:y1, x0:x1] = d[y0:y1, x0:x1]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m > 0
    keep = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))  # the arm only, no stray bits
    # only the forearm and hand swing: the part near the elbow stays where it is (a hinge)
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    keep &= np.hypot(xx - elbow[0], yy - elbow[1]) > HINGE
    keep &= (yy < elbow[1] - 15) & (xx < elbow[0] + 90)  # the raised forearm only, not sleeve on the desk
    return cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def angle(mask, elbow):
    ys, xs = np.nonzero(mask)
    return math.degrees(math.atan2(-(ys.mean() - elbow[1]), xs.mean() - elbow[0]))  # y up


def tween(a_img, b_img, b_mask, elbow, d_ang, t, desk_y):
    """b's arm rotated (1-t) of the way back to a's angle, over b's armless body."""
    base = b_img.copy()
    above = b_mask.copy()
    above[desk_y:] = False
    base[above] = GREEN  # b's own arm gone (the studio shows there)
    rot = -(1 - t) * d_ang  # cv2: positive = counter-clockwise on screen
    M = cv2.getRotationMatrix2D(tuple(map(float, elbow)), rot, 1.0)
    h, w = b_mask.shape
    cut = cv2.warpAffine(b_img, M, (w, h), flags=cv2.INTER_CUBIC)
    m = cv2.warpAffine(b_mask.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:h, 0:w]
    hinge = np.clip((np.hypot(xx - elbow[0], yy - elbow[1]) - HINGE) / 30, 0, 1)  # soft joint
    m = cv2.GaussianBlur(m * hinge, (0, 0), 1.2)[..., None]
    return (base * (1 - m) + cut * m).astype(np.uint8)


RIFE = toolpaths.rife()
RIFE_BELOW = 25  # degrees


def rife(a_img, b_img, t):
    """RIFE v4.6 (ncnn/Vulkan) in-between at time t."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        Image.fromarray(a_img).save(tmp / "a.png")
        Image.fromarray(b_img).save(tmp / "b.png")
        subprocess.run([str(RIFE / f"rife-ncnn-vulkan{toolpaths.EXE}"), "-0", str(tmp / "a.png"), "-1", str(tmp / "b.png"),
                        "-o", str(tmp / "o.png"), "-m", str(RIFE / "rife-v4.6"), "-s", str(t)],
                       check=True, capture_output=True)
        return np.asarray(Image.open(tmp / "o.png").convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("ref_dir")
    p.add_argument("--preview")
    a = p.parse_args()
    d = Path(a.ref_dir)
    rig_path = d / "rig.json"
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    ref = np.asarray(Image.open(a.ref).convert("RGB")).astype(np.float32)
    elbow = rig["gesture_arm"]["elbow"]
    box = rig.get("arm_box", [120, 330, 700, 905])
    desk_y = rig.get("desk_y", 860)
    rest_name = next(k for k, v in rig["variants"].items() if v.get("gesture") == "idle")
    rest = load_full(d, rig, ref, rest_name)
    out = (d / rig["variants"][rest_name]["file"]).parent / "tween"
    out.mkdir(exist_ok=True)
    rest_ang = rig.get("rest_arm_angle", -25.0)  # forearm lying on the desk, pointing down-right
    previews = []
    for tw in rig["tweens"]:
        A, B = tw["a"], tw["b"]
        a_img, b_img = load_full(d, rig, ref, A), load_full(d, rig, ref, B)
        b_mask = arm_mask(b_img, rest, box, elbow)
        a_ang = rest_ang if A == rest_name else angle(arm_mask(a_img, rest, box, elbow), elbow)
        b_ang = angle(b_mask, elbow)
        d_ang = b_ang - a_ang
        row = [a_img]
        for t in TS:
            if abs(d_ang) < RIFE_BELOW:  # small turn, the hand changes shape: RIFE interpolates it well
                img = rife(a_img, b_img, t)
            else:                         # big swing: rotate the cut-out forearm, never smears
                img = tween(a_img, b_img, b_mask, elbow, d_ang, t, desk_y)
            name = f"{A}__{B}_{int(t * 100)}"
            Image.fromarray(img).save(out / f"{name}.png")
            rel = Path(rig["variants"][rest_name]["file"]).parent / "tween" / f"{name}.png"
            rig["variants"][name] = {"file": str(rel),
                                     "region": "body", "feather": 4, "hide_green": True, "only_changed": True,
                                     "left_hand_from": "hand_open", "gesture": "tween", "tween": [A, B, t]}
            row.append(img)
        row.append(b_img)
        print(f"{A} -> {B}: arm {a_ang:.0f} deg -> {b_ang:.0f} deg")
        previews.append(np.concatenate([cv2.resize(r[380:1130, 150:1150], (400, 300)) for r in row], 1))
    rig_path.write_text(json.dumps(rig, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.preview:
        Image.fromarray(np.concatenate(previews, 0)).save(a.preview)


if __name__ == "__main__":
    main()
