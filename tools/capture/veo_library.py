"""Turn a Veo clip of the character (generated from the rig's own reference, on plain green) into
a library the rig can play: body frames lined up on the reference, where-the-arm-left masks, and
the gestures found in it.

    python veo_library.py CLIP.mp4 TRACE.json REF.png REF_DIR OUT_DIR

TRACE.json: tools/capture/trace_face.py CLIP.mp4 TRACE.json --body (same clip).
Writes OUT_DIR/body/NNNN.png (RGBA at reference size: the body below the head, keyed off the green,
moved so the torso sits where the reference's does, colour-matched), OUT_DIR/hide/NNNN.png (where
the reference's own body must vanish because Veo's arm moved away), OUT_DIR/gestures.json and
OUT_DIR/gestures.png (a strip per gesture).

Gestures come from the trace: each wrist's distance from where it rests at the start of the clip;
one hand away = "R" or "L" (the speaker's own right/left), both = "both", both wrists together in
front = "clasp". Each gesture is stored with its frame range (leaving rest .. back at rest).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "relief"))
from prep_variants import color_match, shape_mask  # noqa: E402

TORSO = (840, 440, 1080, 760)  # reference px: collar and chest, the part that defines "where he stands"
# Veo's head nods while the torso stays: cut a wider head than the rig's body mask does, so bits of
# Veo's shemagh never ride along on the body layer (the rig's own head is drawn instead)
VEO_HEAD = {"center": [965, 260], "radius": [205, 200], "sub": True}  # wide beside the head, narrow by the chin (raised hands)
SHOULDERS = 520  # reference row: Veo's shemagh can only be above this


def key(rgb):
    """Alpha from the plain green background, with the green spill taken out of the edges."""
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    excess = g - np.maximum(r, b)
    a = 1 - np.clip((excess - 35) / 45, 0, 1)
    f[..., 1] = np.minimum(g, np.maximum(r, b) + 10 * a)  # despill
    return f.astype(np.uint8), a


def align_torso(ref, img, torso_mask):
    """Translation that puts img's torso on the reference's (ECC on the collar/chest box)."""
    x0, y0, x1, y1 = TORSO
    a = cv2.GaussianBlur(cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_TRANSLATION,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                       torso_mask[y0:y1, x0:x1].astype(np.uint8), 5)
    except cv2.error:
        pass
    return warp


def labels(trace, n):
    fr = trace["frames"]

    def series(k, i):
        v = np.array([f["body"][k][i] if f.get("body") and f["body"][k][2] > 0.5 else np.nan for f in fr[:n]])
        ok = ~np.isnan(v)
        return np.interp(np.arange(n), np.flatnonzero(ok), v[ok]) if ok.any() else np.zeros(n)

    W = {s: np.stack([series(f"wrist{s}", 0), series(f"wrist{s}", 1)], 1) for s in "LR"}
    kern = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    kern /= kern.sum()
    for s in W:  # drop one-frame tracker jumps (median of 5), then smooth the jitter on a painted hand
        med = np.stack([np.median(np.lib.stride_tricks.sliding_window_view(np.pad(W[s][:, i], 2, mode="edge"), 5), 1)
                        for i in (0, 1)], 1)
        W[s] = np.stack([np.convolve(np.pad(med[:, i], 4, mode="edge"), kern, "valid") for i in (0, 1)], 1)
    # where each hand rests: the median of its lowest 30% of frames (clips need not start at rest)
    rest = {s: np.median(W[s][W[s][:, 1] <= np.percentile(W[s][:, 1], 30)], 0) for s in W}
    # a hand is gesturing when it has moved well away from rest, or a little with its palm in view
    # (the hand landmarker only finds a hand that shows its fingers; palms turned up barely move the wrist)
    seen = {s: np.array([bool(f.get("body") and f"open{s}" in f["body"]) for f in fr[:n]]) for s in "LR"}
    dist = {s: np.linalg.norm(W[s] - rest[s], axis=1) for s in W}
    away = {s: (dist[s] > 0.28) | (seen[s] & (dist[s] > 0.14)) for s in W}
    together = (np.abs(W["L"][:, 0]) < 0.5) & (np.abs(W["R"][:, 0]) < 0.5) & \
               (np.linalg.norm(W["L"] - W["R"], axis=1) < 0.75)
    point = {}  # index finger out, the others curled
    for s in "LR":
        fg = [f["body"].get(f"fingers{s}") if f.get("body") else None for f in fr[:n]]
        point[s] = np.array([bool(g) and g[0] > 1.5 and max(g[1:]) < 1.0 for g in fg])
    one = {s: away[s] & (dist[s] > 0.28) for s in W}  # a single raised hand: clearly away
    lab = np.where(together & away["L"] & away["R"], "clasp",
                   np.where(one["R"] & point["R"], "point_R",
                            np.where(one["L"] & point["L"], "point_L",
                                     np.where(away["L"] & away["R"], "both",
                                              np.where(away["R"], "R", np.where(away["L"], "L", "rest"))))))
    # majority filter: a label must hold a few frames
    out = lab.copy()
    for f in range(n):
        win = list(lab[max(0, f - 3):f + 4])
        out[f] = max(sorted(set(win)), key=win.count)
    return out


def segments(lab, fps):
    segs, f, n = [], 0, len(lab)
    while f < n:
        e = f
        while e < n and lab[e] == lab[f]:
            e += 1
        if lab[f] != "rest" and e - f >= int(0.35 * fps):
            segs.append({"kind": str(lab[f]), "start": f, "end": e})
        f = e
    return segs


def main(clip, trace_path, ref_path, ref_dir, out):
    out = Path(out)
    (out / "body").mkdir(parents=True, exist_ok=True)
    (out / "hide").mkdir(exist_ok=True)
    rig = json.loads((Path(ref_dir) / "rig.json").read_text(encoding="utf-8"))
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    ref_a = np.asarray(Image.open(Path(ref_dir) / "rgba.png"))[..., 3].astype(np.float32) / 255
    h, w = ref.shape[:2]
    body = cv2.GaussianBlur(shape_mask(ref.shape, rig["patch_masks"]["body"] + [VEO_HEAD]).astype(np.float32), (0, 0), 4)
    head_core = shape_mask(ref.shape, [{"center": [965, 270], "radius": [170, 190]}])
    torso_mask = np.zeros((h, w), bool)
    torso_mask[TORSO[1]:TORSO[3], TORSO[0]:TORSO[2]] = True
    ring = torso_mask & (ref_a > 0.5)
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    shifts, k = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_LANCZOS4)
        rgb, a = key(rgb)
        warp = align_torso(ref, rgb, torso_mask & (a > 0.5))
        M = (cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
        rgb = cv2.warpAffine(rgb, warp, (w, h), flags=M, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.warpAffine(a, warp, (w, h), flags=M, borderMode=cv2.BORDER_REPLICATE)  # the frame edge continues
        rgb = color_match(ref, rgb, ring)
        # Veo's shemagh, wherever its head nodded to, is not body: red above the shoulders that is
        # connected to the head area (skin is too close in colour; a raised hand stands apart from
        # the head, with background in between, so it is kept), plus its white checks
        r_, g_, b_ = (rgb[..., i].astype(int) for i in range(3))
        red = ((r_ > 140) & (g_ < 110) & (g_ < 0.62 * r_) & (a > 0.5)).astype(np.uint8)
        red[SHOULDERS:] = 0
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        nlab, lab = cv2.connectedComponents(red)
        touching = np.unique(lab[head_core & (lab > 0)])
        cloth = np.isin(lab, touching).astype(np.uint8)
        cloth = cv2.GaussianBlur(cv2.dilate(cloth, np.ones((13, 13), np.uint8)).astype(np.float32), (0, 0), 2)
        alpha = np.clip(a, 0, 1) * body * (1 - np.clip(cloth, 0, 1))
        # the reference's body must vanish where Veo shows background (the arm has moved away)
        hide = np.clip((1 - a) * body * (ref_a > 0.02), 0, 1)
        hide = cv2.GaussianBlur(cv2.dilate(hide, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        Image.fromarray(np.dstack([rgb, (alpha * 255).astype(np.uint8)])).save(out / "body" / f"{k + 1:04d}.png")
        Image.fromarray((np.clip(hide, 0, 1) * 255).astype(np.uint8)).save(out / "hide" / f"{k + 1:04d}.png")
        shifts.append([round(float(warp[0, 2]), 1), round(float(warp[1, 2]), 1)])
        k += 1
    lab = labels(trace, k)
    segs = segments(lab, fps)
    (out / "gestures.json").write_text(json.dumps({"clip": str(clip), "fps": fps, "frames": k,
                                                   "labels": lab.tolist(), "gestures": segs,
                                                   "torso_shift": shifts}, indent=1), encoding="utf-8")
    rows = []  # preview: per gesture, frames from leaving rest to back at rest
    for s in segs:
        a0, a1 = max(0, s["start"] - 6), min(k, s["end"] + 6)
        picks = np.linspace(a0, a1 - 1, 8).astype(int)
        row = Image.new("RGB", (8 * 200, 250), (0, 177, 64))
        for i, f in enumerate(picks):
            im = Image.open(out / "body" / f"{f + 1:04d}.png").crop((460, 180, 1460, 1080)).resize((200, 180))
            bg = Image.new("RGBA", im.size, (0, 177, 64, 255))
            bg.alpha_composite(im)
            row.paste(bg.convert("RGB"), (200 * i, 30))
            ImageDraw.Draw(row).text((200 * i + 4, 214), f"{f / fps:.2f}s", fill="white")
        ImageDraw.Draw(row).text((4, 6), f"{s['kind']}  {s['start'] / fps:.2f}-{s['end'] / fps:.2f}s", fill="white")
        rows.append(row)
    if rows:
        sheet = Image.new("RGB", (1600, 250 * len(rows)))
        for i, r in enumerate(rows):
            sheet.paste(r, (0, 250 * i))
        sheet.save(out / "gestures.png")
    print(k, "frames;", [(s["kind"], round(s["start"] / fps, 2), round(s["end"] / fps, 2)) for s in segs])
    print("torso shift range x", min(s[0] for s in shifts), max(s[0] for s in shifts),
          "y", min(s[1] for s in shifts), max(s[1] for s in shifts))


if __name__ == "__main__":
    main(*sys.argv[1:6])
