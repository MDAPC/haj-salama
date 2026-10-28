"""Gather Veo libraries (veo_library.py) into one numbered frame pool the rig can play.

    python veo_pool.py OUT_DIR REF_DIR LIB_DIR [LIB_DIR ...]

Writes OUT_DIR/body/NNNN.png and OUT_DIR/hide/NNNN.png (links to the library frames, numbered
through all clips), OUT_DIR/outside_union.png (every pixel any frame paints outside the reference's
silhouette: the surface a raised arm needs) and OUT_DIR/pool.json with the takes.

A take is one gesture from rest back to rest: a run of non-rest frames (short gaps bridged),
padded with a few rest frames each side. "rise" and "fall" are its natural going-up and coming-down
parts (source frames); "returns" is false when the clip ends before the hands are back at rest
(the rig then plays the rise backwards to get down). family: R, L, point_R, point_L, both, clasp.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PAD, GAP, MIN_LEN = 4, 3, 14  # source frames (24 fps)


def link(src, dst):
    """Symlink where allowed; Windows needs admin rights for symlinks, so fall back to a hard link
    (same drive), then a copy."""
    for make in (os.symlink, os.link, shutil.copyfile):
        try:
            make(src, dst)
            return
        except OSError:
            continue
    raise OSError(f"cannot link {src} -> {dst}")


def deviation(lib, ref_a):
    """Per frame: how far the body is from the reference's resting pose (pixels the arm moved
    away from, plus pixels painted outside the reference's silhouette), at quarter size. Cached."""
    cache = lib / "deviation.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    small = ref_a[::4, ::4]
    dev = []
    for f in sorted((lib / "hide").glob("*.png")):
        hide = np.asarray(Image.open(f).reduce(4), np.float32) / 255
        a = np.asarray(Image.open(lib / "body" / f.name).getchannel("A").reduce(4), np.float32) / 255
        dev.append(round(float(hide.sum() + (a * ~small).sum()), 1))
    cache.write_text(json.dumps(dev), encoding="utf-8")
    return dev


def takes(lab, dev=None):
    n = len(lab)
    act = np.array([x != "rest" for x in lab])
    runs, f = [], 0
    while f < n:
        if act[f]:
            e = f
            while e < n and act[e]:
                e += 1
            if runs and f - runs[-1][1] <= GAP:
                runs[-1][1] = e
            else:
                runs.append([f, e])
            f = e
        f += 1
    if dev is not None:  # widen each run to where the body really is back at rest
        dev = np.convolve(np.pad(np.asarray(dev, float), 2, mode="edge"), np.ones(5) / 5, "valid")
        lo, hi = np.percentile(dev, 10), np.percentile(dev, 90)
        near = dev <= lo + 0.12 * (hi - lo)
        for r in runs:
            while r[0] > 0 and not near[r[0] - 1]:
                r[0] -= 1
            while r[1] < n and not near[r[1]]:
                r[1] += 1
        merged = []
        for r in runs:  # widened runs that now touch are one gesture
            if merged and r[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], r[1])
            else:
                merged.append(r)
        runs = merged
    out = []
    for a, b in runs:
        if b - a < MIN_LEN:
            continue
        if dev is not None and dev[a:b].max() < lo + 0.3 * (hi - lo):
            continue  # the tracker jittered but the body never really moved
        kinds = [x for x in lab[a:b] if x != "rest"]
        fam = max(sorted(set(kinds)), key=kinds.count)
        s0 = max(0, a - PAD)
        s1 = min(n, b + PAD)
        span = s1 - s0
        rise = int(min(18, span / 3))
        out.append({"family": fam, "start": s0, "end": s1, "rise": rise, "fall": rise,
                    "returns": b + PAD <= n - 1, "kinds": sorted(set(kinds))})
    return out


def main(out, ref_dir, libs):
    out = Path(out)
    for sub in ("body", "hide"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*.png"):
            old.unlink()
    ref_a = np.asarray(Image.open(Path(ref_dir) / "rgba.png"))[..., 3] > 128
    union = None
    pool, k = [], 0
    for lib in libs:
        lib = Path(lib).resolve()
        g = json.loads((lib / "gestures.json").read_text(encoding="utf-8"))
        base = k
        for f in range(g["frames"]):
            k += 1
            for sub in ("body", "hide"):
                link(lib / sub / f"{f + 1:04d}.png", out / sub / f"{k:04d}.png")
            if f % 3 == 0:  # the arm surface: union of everything painted outside the silhouette
                a = np.asarray(Image.open(lib / "body" / f"{f + 1:04d}.png"))[..., 3]
                a = np.where(ref_a, 0, a)
                union = a if union is None else np.maximum(union, a)
        for t in takes(g["labels"], deviation(lib, ref_a)):
            t.update({"clip": lib.name, "start": base + t["start"] + 1, "end": base + t["end"] + 1})
            pool.append(t)  # start/end are pool frame numbers (1-based, end exclusive)
    rgba = np.zeros(union.shape + (4,), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = union
    Image.fromarray(rgba).save(out / "outside_union.png")
    (out / "pool.json").write_text(json.dumps({"frames": k, "fps": 24, "takes": pool}, indent=1), encoding="utf-8")
    for t in pool:
        print(f"{t['clip']} {t['family']:8s} {t['start']:4d}-{t['end']:4d} rise {t['rise']} returns {t['returns']} {t['kinds']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
