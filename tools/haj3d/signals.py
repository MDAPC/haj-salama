"""Time signals shared by the animators: loudness envelope, Rhubarb mouth tracks,
phrases (speech between silences), stressed-syllable beats, smooth bumps."""
import math
import wave

import numpy as np

FPS = 30
# finer mouth classes from tts_align -> nearest Rhubarb shape
RHUBARB_FALLBACK = {"I": "B", "W": "D"}


def smooth(x, k):
    y, acc = np.empty(len(x)), float(x[0])
    for i, v in enumerate(x):
        acc += k * (v - acc)
        y[i] = acc
    return y


def envelope(wav_path, n):
    with wave.open(str(wav_path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    a = a.reshape(-1, ch).mean(1) / 32768.0
    hop = sr / FPS
    a = np.pad(a, (0, max(0, int(math.ceil((n + 1) * hop)) - len(a))))
    rms = np.array([np.sqrt(np.mean(a[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(n)])
    rms = np.clip(rms / max(np.percentile(rms, 95), 1e-6), 0, 1)
    return smooth(rms, 0.45), smooth(rms, 0.07)


def viseme_tracks(cues, n, table):
    k = len(next(iter(table.values())))
    tr = np.zeros((n, k))
    j = 0
    for f in range(n):
        t = f / FPS
        while j + 1 < len(cues) and cues[j]["end"] <= t:
            j += 1
        c = cues[j]
        v = c["value"] if c["start"] <= t < c["end"] else "X"
        tr[f] = table.get(v) or table[RHUBARB_FALLBACK.get(v, "B")]
    return np.stack([smooth(tr[:, j], 0.55) for j in range(k)], -1)


def phrases(cues, n, min_gap=0.22):
    """(start, end) frames of speech between Rhubarb silences."""
    talk = np.zeros(n, bool)
    for c in cues:
        if c["value"] != "X" or c["end"] - c["start"] < min_gap:
            talk[int(c["start"] * FPS):min(n, int(c["end"] * FPS) + 1)] = True
    out, f = [], 0
    while f < n:
        if talk[f]:
            s = f
            while f < n and talk[f]:
                f += 1
            if f - s > 8:
                out.append((s, f))
        f += 1
    return out


def bump(n, start, attack, hold, release):
    t = np.arange(n)
    up = np.clip((t - start) / max(attack, 1), 0, 1)
    down = 1 - np.clip((t - start - attack - hold) / max(release, 1), 0, 1)
    v = np.minimum(up, down)
    return v * v * (3 - 2 * v)


def beats(env_slow, n, min_gap=14):
    """Frames of stressed syllables: local peaks in the slow loudness."""
    peaks, last = [], -999
    for f in range(2, n - 2):
        if env_slow[f] > 0.55 and env_slow[f] >= env_slow[f - 2] and env_slow[f] >= env_slow[f + 2] \
                and f - last >= min_gap:
            peaks.append(f)
            last = f
    return peaks


