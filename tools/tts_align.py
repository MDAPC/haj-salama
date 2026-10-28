"""Speak a transcript with Piper and keep its exact phoneme timings as mouth cues.

    python tts_align.py MODEL.onnx transcript.txt OUT_DIR [--silence 0.35] [--lead 0.04]

Writes  voice.wav       the speech (one chunk per sentence, silence between)
        phonemes.json   [{phoneme, start, end}] straight from the synthesizer
        mouth.json      Rhubarb-style {"mouthCues": [{start, end, value}]} with shapes A-H, X
Because the timings come from the synthesizer itself, the lips match the audio exactly.
For a recorded voice, use Rhubarb instead (anchor.py does this automatically).
"""
import argparse
import json
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice

# espeak IPA (as Piper emits it, one codepoint at a time) -> mouth shape.
# Rhubarb's letters, plus two finer ones (rigs that don't know them fall back to B and D):
#   A closed lips (m b p)   B teeth together, lips parted (t d s z n r ...)   I spread "ee" (i y j)
#   C half open (e, throat sounds, short a)   D open (a)   W wide open (long stressed aa)
#   E rounded open (o)   F puckered (u w)   G lower lip on teeth (f v)   H tongue up (l)   X rest
SHAPES = {}
for chars, shape in (("mbp", "A"), ("fv", "G"), ("uwʊ", "F"), ("oɔ", "E"), ("aɑæɐ", "D"),
                     ("eɛəɪ", "C"), ("iyj", "I"), ("l", "H"),
                     ("tdnrɾszʃʒθðʔ", "B"), ("kgqxɣħʕhχʁ", "C")):
    for ch in chars:
        SHAPES[ch] = shape
MODIFIERS = set("ːˈˌˤ̪̩ʲʰ‿")      # length, stress, pharyngealisation etc. belong to the previous sound
PAUSES = set(",.;:!?،؛؟^$ \n")


def synthesize(model, text, silence):
    voice = PiperVoice.load(model, include_alignments=True)
    sr = voice.config.sample_rate
    audio, phon, t = [], [], 0.0
    for chunk in voice.synthesize(text, include_alignments=True):
        if not chunk.phoneme_alignments:
            raise SystemExit("this voice gave no alignments")
        for al in chunk.phoneme_alignments:
            dur = al.num_samples / sr
            phon.append({"phoneme": al.phoneme, "start": round(t, 4), "end": round(t + dur, 4)})
            t += dur
        a = chunk.audio_int16_array
        # alignments can be a few samples off the audio length; keep the audio as the clock
        t = sum(len(x) for x in audio) / sr + len(a) / sr
        audio.append(a)
        gap = np.zeros(int(silence * sr), np.int16)
        audio.append(gap)
        phon.append({"phoneme": " ", "start": round(t, 4), "end": round(t + silence, 4)})
        t += silence
    return np.concatenate(audio), sr, phon


def mouth_cues(phon, lead):
    """Merge modifiers into their sound, map to shapes, drop flickers, shift slightly early
    (lips move a little before the sound is heard)."""
    merged = []
    for p in phon:
        ph = p["phoneme"]
        if ph in MODIFIERS and merged:
            merged[-1]["end"] = p["end"]
            merged[-1]["long"] |= ph == "ː"
            continue
        shape = "X" if ph in PAUSES else SHAPES.get(ph, "B")
        merged.append({"start": p["start"], "end": p["end"], "value": shape, "long": False})
    for c in merged:  # how far "a" opens depends on how long it is held
        dur = c["end"] - c["start"]
        if c["value"] == "D":
            c["value"] = "W" if c["long"] and dur >= 0.11 else ("C" if dur < 0.07 else "D")
        del c["long"]
    # short pauses inside a phrase (word gaps) shouldn't snap the mouth to rest
    for i, c in enumerate(merged):
        if c["value"] == "X" and c["end"] - c["start"] < 0.09 and 0 < i < len(merged) - 1:
            c["value"] = merged[i - 1]["value"]
    cues = []
    for c in merged:
        if cues and cues[-1]["value"] == c["value"]:
            cues[-1]["end"] = c["end"]
        else:
            cues.append(dict(c))
    for c in cues:
        c["start"] = round(max(0.0, c["start"] - lead), 3)
        c["end"] = round(max(0.0, c["end"] - lead), 3)
    return {"mouthCues": cues}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("transcript")
    p.add_argument("out")
    p.add_argument("--silence", type=float, default=0.35)
    p.add_argument("--lead", type=float, default=0.04)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    audio, sr, phon = synthesize(a.model, Path(a.transcript).read_text(encoding="utf-8"), a.silence)
    with wave.open(str(out / "voice.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    (out / "phonemes.json").write_text(json.dumps(phon, ensure_ascii=False, indent=0), encoding="utf-8")
    cues = mouth_cues(phon, a.lead)
    (out / "mouth.json").write_text(json.dumps(cues, indent=1), encoding="utf-8")
    print(f"ok {len(audio) / sr:.2f}s, {len(phon)} phonemes, {len(cues['mouthCues'])} cues")


if __name__ == "__main__":
    main()
