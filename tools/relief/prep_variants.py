"""Turn generated variants of a reference (e.g. Gemini edits) into patches that line up with it.

    python prep_variants.py REF.png REF_DIR

REF_DIR/rig.json "variants": {name: {"file": path, "region": "mouth" | "eyes"}} and "patch_masks".
For each variant: resize to the reference, refine the alignment with ECC (affine) on the face
around the region, match colour to the reference in a ring around the region, and cut the region
out with a feathered edge. Writes REF_DIR/var_<name>.png (RGBA, reference size) and
REF_DIR/variants_check.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

MATTE = None  # rembg session, made on first use


def shape_mask(shape, shapes):
    """Union of ellipses ({center, radius}) and rectangles ({rect: [x0, y0, x1, y1]});
    a shape with "sub": true is cut out of the union instead."""
    m = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(m)
    for s in sorted(shapes, key=lambda s: s.get("sub", False)):
        fill = 0 if s.get("sub") else 255
        if "rect" in s:
            d.rectangle(tuple(s["rect"]), fill=fill)
        else:
            (cx, cy), (rx, ry) = s["center"], s["radius"]
            d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill)
    return np.asarray(m) > 0


def align(ref, img, region, exclude):
    """Affine ECC on a face crop around the region, ignoring the edited area itself."""
    x0, y0, x1, y1 = region
    a = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    a, b = cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0)
    mask = (~exclude[y0:y1, x0:x1]).astype(np.uint8)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 5)
    except cv2.error as e:
        print("  ECC failed, using plain resize:", e)
        return img, np.eye(2, 3)
    # crop-local warp -> full-image warp
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float32)
    W = np.vstack([warp, [0, 0, 1]])
    full = (T @ W @ np.linalg.inv(T))[:2]
    out = cv2.warpAffine(img, full, (ref.shape[1], ref.shape[0]),
                         flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out, full


def color_match(ref, img, ring):
    out = img.astype(np.float32)
    for c in range(3):
        r, g = ref[..., c][ring].astype(np.float32), out[..., c][ring]
        gain = r.std() / max(g.std(), 1e-3)
        gain = np.clip(gain, 0.85, 1.15)
        out[..., c] = (out[..., c] - g.mean()) * gain + r.mean()
    return np.clip(out, 0, 255).astype(np.uint8)


def main(ref_path, ref_dir):
    d = Path(ref_dir)
    rig = json.loads((d / "rig.json").read_text(encoding="utf-8"))
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    h, w = ref.shape[:2]
    masks = rig["patch_masks"]
    face_box = rig["face_box"]
    checks = []
    aligned = {}
    order = sorted(rig["variants"], key=lambda k: 0 if any(v.get("left_hand_from") == k for v in rig["variants"].values()) else 1)
    for name in order:
        v = rig["variants"][name]
        img = np.asarray(Image.open(d / v["file"]).convert("RGB").resize((w, h), Image.LANCZOS))
        region = shape_mask(ref.shape, masks[v["region"]])
        excl = cv2.dilate(region.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        img, warp = align(ref, img, face_box, excl)
        if v.get("left_hand_from") in aligned:
            # one identical resting left hand in every pose, so it never jumps between paintings
            lh = cv2.GaussianBlur(shape_mask(ref.shape, masks["left_hand"]).astype(np.float32), (0, 0), 4)[..., None]
            img = (img * (1 - lh) + aligned[v["left_hand_from"]] * lh).astype(np.uint8)
        if "green_above" in rig:
            # Gemini tints the desk green where the clasped hands were: take the tint back out
            r_, g_, b_ = (img[rig["green_above"]:, :, i].astype(np.float32) for i in range(3))
            tint = np.clip((g_ - (r_ + b_) / 2) / 25, 0, 1)[..., None]
            grey = ((r_ + g_ + b_) / 3)[..., None]
            img[rig["green_above"]:] = (img[rig["green_above"]:] * (1 - tint) + grey * tint).astype(np.uint8)
        aligned[name] = img
        ring = cv2.dilate(region.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        ring &= ~(cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        img = color_match(ref, img, ring)
        if v.get("only_changed"):
            # big body poses: keep the reference's own background (Gemini repaints it slightly
            # differently) and take only the character plus whatever the pose actually changed
            ref_a = np.asarray(Image.open(d / "rgba.png"))[..., 3] > 128
            changed = np.abs(img.astype(int) - ref.astype(int)).sum(2) > 80
            changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            keep = cv2.dilate((ref_a | (changed > 0)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            region &= keep
        soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), v.get("feather", 6))
        # never paint the variant's own green background over the face (its outline can shift)
        r_, g_, b_ = (img[..., i].astype(int) for i in range(3))
        green = ((g_ > r_ + 20) & (g_ > b_ + 20)).astype(np.float32)
        if "green_above" in rig:  # below this line green is a reflection on the desk, not the screen
            green[rig["green_above"]:] = 0
        if v.get("matte"):
            # the variant's own background is not all green screen (Gemini paints studio floor below
            # it): cut the character out of the variant and treat everything else as background
            from rembg import new_session, remove
            global MATTE
            MATTE = MATTE or new_session("isnet-general-use")
            fg = np.asarray(remove(Image.fromarray(img), session=MATTE, only_mask=True)).astype(np.float32) / 255
            green = np.maximum(green, (fg < 0.5).astype(np.float32))
        green = cv2.GaussianBlur(cv2.dilate(green, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        if v.get("hide_green"):
            # a pose change (e.g. a raised arm): where the variant shows the green screen inside its
            # region, the reference's own body must vanish too, so the studio plate shows through
            hide = np.clip(green, 0, 1) * soft
            Image.fromarray((hide * 255).astype(np.uint8)).save(d / f"var_{name}_hide.png")
        soft *= 1 - np.clip(green, 0, 1)
        rgba = np.dstack([img, (soft * 255).astype(np.uint8)])
        Image.fromarray(rgba).save(d / f"var_{name}.png")
        comp = (ref * (1 - soft[..., None]) + img * soft[..., None]).astype(np.uint8)
        ys, xs = np.nonzero(region)
        pad = 40
        box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, w), min(ys.max() + pad, h))
        tile = Image.fromarray(comp).crop(box).resize((300, int(300 * (box[3] - box[1]) / (box[2] - box[0]))))
        checks.append((name, tile))
        print(f"{name}: shift {warp[0, 2]:+.1f},{warp[1, 2]:+.1f} px  scale {warp[0, 0]:.3f}")
    W_ = 300 * len(checks)
    H_ = max(t.height for _, t in checks) + 20
    sheet = Image.new("RGB", (W_, H_), (20, 20, 24))
    dr = ImageDraw.Draw(sheet)
    for i, (n, t) in enumerate(checks):
        sheet.paste(t, (300 * i, 20))
        dr.text((300 * i + 4, 4), n, fill=(255, 220, 90))
    sheet.save(d / "variants_check.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
