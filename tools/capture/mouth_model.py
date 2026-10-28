"""One mouth space for speech sounds, real faces and paintings.

A mouth is a point (open, spread, round): open 0..1, spread -1..1, round 0..1. Paintings are placed
in this space in rig.json "mouth_space". Each frame gets a target point, either

  * from speech sounds (phoneme cues): each sound class has a target, and the path between targets is
    shaped like a real mouth moves (critically damped, run forwards and backwards so it neither lags
    nor leads the sound; this is also coarticulation: neighbouring sounds pull on each other), or
  * from a traced video of a real speaker (tools/capture/trace_face.py), mapped by that speaker's own
    ranges, so any face drives the rig,

and then shows the nearest painting (with hysteresis and a minimum hold, so it never flickers),
plus a small continuous jaw warp for the gap between the target and that painting.
No module-level Blender imports: usable from Blender and from plain Python.
"""
import json
import math
from pathlib import Path

import numpy as np

# sound class (tts_align / Rhubarb letters) -> target (open, spread, round)
SOUND = {"A": (0.00, -0.1, 0.0),   # m b p: pressed
         "B": (0.22, 0.30, 0.0),   # t d s z n r: teeth together
         "I": (0.38, 0.60, 0.0),   # ee
         "C": (0.32, 0.00, 0.0),   # e, throat sounds, short a
         "D": (0.55, 0.10, 0.0),   # a
         "W": (0.72, 0.05, 0.1),   # long stressed aa
         "E": (0.42, -0.40, 0.6),  # o
         "F": (0.25, -0.80, 1.0),  # u w
         "G": (0.20, 0.10, 0.0),   # f v
         "H": (0.45, 0.10, 0.0),   # l
         "X": (0.00, 0.00, 0.0)}   # rest
WEIGHTS = np.array([1.0, 0.45, 0.8])   # how much each axis counts when picking the nearest painting
# paintings that only make sense for their own sound (a tongue or teeth-on-lip shown for any other
# sound looks wrong); everything else is chosen by distance alone
SPECIAL = {"mb": "A", "f": "G", "l": "H", "oo": "F"}
CALM, RISE = 1.0, 0.07   # tuned against a traced real speaker: a little calmer and slower than her


def sound_targets(cues, n, fps, calm=0.85):
    """Piecewise-constant targets from cues; `calm` scales how far the mouth opens."""
    T = np.zeros((n, 3))
    j = 0
    for f in range(n):
        t = f / fps
        while j + 1 < len(cues) and cues[j]["end"] <= t:
            j += 1
        c = cues[j]
        T[f] = SOUND.get(c["value"] if c["start"] <= t < c["end"] else "X", SOUND["B"])
    T[:, 0] *= calm
    T[:, 1] *= calm
    return T


def damp(x, fps, rise=0.11):
    """Zero-phase critically damped smoothing: a step reaches ~90% in about `rise` seconds,
    forwards then backwards (so the mouth starts moving slightly before each sound, as real
    speakers do, and nothing lags)."""
    w = 3.9 / rise  # 2nd-order critically damped: 90% at t ~ 3.9/w

    def run(sig):
        y = np.empty_like(sig)
        pos, vel = sig[0].copy(), np.zeros_like(sig[0])
        dt = 1.0 / fps
        sub = 4
        for i, target in enumerate(sig):
            for _ in range(sub):
                acc = w * w * (target - pos) - 2 * w * vel
                vel = vel + acc * dt / sub
                pos = pos + vel * dt / sub
            y[i] = pos
        return y

    fwd = run(np.asarray(x, float))
    return run(fwd[::-1])[::-1]


def trace_targets(trace, n, fps, calm=0.85, smooth_rise=0.06):
    """Targets from a traced real face, resampled to `fps`, using the speaker's own ranges."""
    F = trace["frames"]
    src_fps = trace["fps"]
    t_src = np.array([f["t"] for f in F])
    ok = np.array([f["ok"] for f in F])

    def series(fn):
        v = np.array([fn(f) if f["ok"] else np.nan for f in F], float)
        good = ~np.isnan(v)
        return np.interp(np.arange(len(v)), np.nonzero(good)[0], v[good]) if good.any() else np.zeros(len(v))

    op = series(lambda f: f["open"])
    wd = series(lambda f: f["width"])
    rd = series(lambda f: f["bs"]["mouthFunnel"] + f["bs"]["mouthPucker"])
    o_lo, o_hi = np.percentile(op[ok], 3), np.percentile(op[ok], 97)
    w_mid, w_span = np.median(wd[ok]), max(np.percentile(wd[ok], 95) - np.median(wd[ok]), 1e-3)
    r_hi = max(np.percentile(rd[ok], 97), 0.2)
    T_src = np.stack([np.clip((op - o_lo) / (o_hi - o_lo), 0, 1.2) * 0.68,  # calm: wide "aa" only at extremes
                      np.clip((wd - w_mid) / w_span, -1, 1),
                      np.clip(rd / r_hi, 0, 1)], -1)
    t_out = np.arange(n) / fps
    T = np.stack([np.interp(t_out, t_src - t_src[0], T_src[:, k]) for k in range(3)], -1)
    T[:, 0] *= calm / 0.85
    return damp(T, fps, smooth_rise)


def sound_per_frame(cues, n, fps):
    out, j = [], 0
    for f in range(n):
        t = f / fps
        while j + 1 < len(cues) and cues[j]["end"] <= t:
            j += 1
        c = cues[j]
        out.append(c["value"] if c["start"] <= t < c["end"] else "X")
    return out


def pick(T, space, hold=2, stick=0.06, sounds=None):
    """Nearest painting per frame, staying with the current one unless another is clearly
    closer (hysteresis), and holding each at least `hold` frames. Special paintings are only
    allowed during their own sound (or, with no sounds, mb and oo only, for a traced face)."""
    names = list(space)
    P = np.array([space[k] for k in names], float)
    seq, cur, since = [], None, 0
    for i, target in enumerate(T):
        d = np.sqrt((((P - target) * WEIGHTS) ** 2).sum(1))
        for k, name in enumerate(names):
            if name in SPECIAL:
                ok = (sounds[i] == SPECIAL[name]) if sounds is not None else name in ("mb", "oo")
                if not ok:
                    d[k] = np.inf
        best = int(np.argmin(d))
        if cur is not None and not np.isfinite(d[cur]):
            since = hold  # the current painting just became disallowed: switch now
        if cur is None or (since >= hold and d[best] < d[cur] - stick):
            cur, since = best, 0
        seq.append(names[cur])
        since += 1
    return seq


def residual_open(T, seq, space):
    """How much more (or less) open the target is than the painting shown: a small continuous
    jaw warp covers the gap between paintings, so motion stays smooth."""
    return np.array([T[i, 0] - space[s][0] for i, s in enumerate(seq)])


def load_trace(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan(cues=None, trace=None, n=0, fps=30, space=None, calm=CALM):
    """Everything the rig needs per frame: painting name, jaw residual, the raw target."""
    if trace is not None:
        T = trace_targets(trace, n, fps, calm)
        sounds = None
    else:
        T = damp(sound_targets(cues, n, fps, calm), fps, RISE)
        sounds = sound_per_frame(cues, n, fps)
    seq = pick(T, space, sounds=sounds)
    return {"target": T, "paint": seq, "residual": residual_open(T, seq, space)}


def speed_report(T, fps):
    """Median and 95th-percentile speed of the open axis (per second), to compare with a real speaker."""
    v = np.abs(np.diff(T[:, 0])) * fps
    return float(np.median(v)), float(np.percentile(v, 95))


if __name__ == "__main__":  # quick self-test: a closed -> a -> closed step
    cues = [{"start": 0, "end": 0.3, "value": "X"}, {"start": 0.3, "end": 0.6, "value": "D"},
            {"start": 0.6, "end": 1.0, "value": "A"}]
    T = damp(sound_targets(cues, 30, 30), 30)
    print(" ".join(f"{v:.2f}" for v in T[:, 0]))
    print(math.isfinite(T.sum()))
