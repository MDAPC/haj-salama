"""Perform an episode with the 3D Haj Salama and render transparent frames.

    blender -b assets/haj_salama_3d/haj.blend -P tools/haj3d/animate.py -- \
        --episode episodes/ar01 --shot desk [--frames N]

Reads   episode/voice.wav, episode/mouth.json (Rhubarb)
Writes  episode/out/3d_<shot>/frames/####.png (RGBA, character + desk only)
Everything below is a function of time: mouth shapes from Rhubarb, phrases from its silences,
emphasis from the loudness envelope, and seeded idle motion.
"""
import argparse
import json
import math
import random
import sys
import wave
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signals import beats, bump, envelope, phrases, smooth, viseme_tracks  # noqa: E402
from studio import SCREEN_LOC, SCREEN_SIZE, build_studio, image_panel  # noqa: E402

FPS = 30
SHOTS = {  # camera location, look-at target, focal length (framed like the reference sheet)
    "desk": ((0.45, -2.95, 1.45), (0.10, 0.0, 1.02), 35),
    "medium": ((0.30, -2.30, 1.25), (0.0, 0.0, 1.12), 45),
    "closeup": ((0.22, -1.70, 1.34), (0.0, 0.0, 1.26), 55),
}
# Rhubarb shape -> jaw degrees, wide, round, press, f/v
VISEMES = {"A": (0, 0, 0, 1.0, 0), "B": (4, 0.4, 0, 0.2, 0), "C": (10, 0.3, 0, 0, 0),
           "D": (17, 0.1, 0, 0, 0), "E": (9, 0, 0.5, 0, 0), "F": (3, 0, 0.9, 0.2, 0),
           "G": (3, 0.1, 0, 0, 1.0), "H": (11, 0.2, 0, 0, 0), "X": (1, 0, 0, 0.7, 0)}
LID_OPEN = {"R": -20, "L": -14}
LID_SHUT = 84


def args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True)
    p.add_argument("--shot", default="desk", choices=SHOTS)
    p.add_argument("--frames", type=int)
    p.add_argument("--no-render", action="store_true")
    return p.parse_args(argv)


# ------------------------------------------------------------------ scene

def setup_scene(shot, n):
    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = 1920, 1080
    s.render.fps = FPS
    s.render.film_transparent = False
    s.world.color = (0.01, 0.012, 0.018)
    s.eevee.taa_render_samples = 8
    s.frame_start, s.frame_end = 1, n
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    loc, target, lens = SHOTS[shot]
    cd = bpy.data.cameras.new("cam")
    cd.lens = lens
    cam = bpy.data.objects.new("cam", cd)
    s.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    s.camera = cam
    for ob in bpy.data.objects:
        if ob.name.startswith("col_"):
            ob.hide_render = True
    sh = bpy.data.objects["shemagh"]
    for m in sh.modifiers:
        if m.type == "CLOTH":
            m.point_cache.frame_start, m.point_cache.frame_end = 1, n
    return s


# ------------------------------------------------------------------ performance

def perform(ep, n):
    arm = bpy.data.objects["rig"]
    P = arm.pose.bones
    head = bpy.data.objects["head"]
    keys = head.data.shape_keys.key_blocks
    lids = {s: bpy.data.objects[f"lid.{s}"] for s in ("R", "L")}

    cues = json.loads((ep / "mouth.json").read_text(encoding="utf-8"))["mouthCues"]
    vis = viseme_tracks(cues, n, VISEMES)
    ef, es = envelope(ep / "voice.wav", n)
    ph = phrases(cues, n)
    bt = beats(es, n)
    rnd = random.Random(7)

    # gesture envelopes
    g_phrase = np.zeros(n)
    for k, (a, b) in enumerate(ph):
        g_phrase = np.maximum(g_phrase, bump(n, a - 6, 9, max(0, b - a - 12), 14))
    g_beat = np.zeros(n)
    for f in bt:
        g_beat += bump(n, f - 4, 4, 1, 7)
    g_beat = np.clip(g_beat, 0, 1)
    nod = np.zeros(n)
    for f in bt:
        nod += bump(n, f - 3, 3, 0, 8)
    open_hand = np.zeros(n)          # both palms out at the end of some phrases
    for a, b in ph[1::2]:
        open_hand = np.maximum(open_hand, bump(n, b - 12, 8, 6, 14))

    # blinks and gaze
    blink = np.zeros(n)
    t = rnd.uniform(0.8, 1.6)
    while t * FPS < n - 6:
        f0 = int(t * FPS)
        for i, v in enumerate((0.5, 1, 1, 0.6, 0.2)):
            blink[f0 + i] = max(blink[f0 + i], v)
        t += rnd.uniform(2.0, 4.5)
    sac_yaw, sac_pitch = np.zeros(n), np.zeros(n)
    f, y, p = 0, 0.0, 0.0
    while f < n:
        dur = rnd.randint(18, 50)
        y, p = rnd.uniform(-0.08, 0.08), rnd.uniform(-0.04, 0.05)
        sac_yaw[f:f + dur], sac_pitch[f:f + dur] = y, p
        f += dur
    sac_yaw, sac_pitch = smooth(sac_yaw, 0.6), smooth(sac_pitch, 0.6)

    phase = [rnd.uniform(0, 6.28) for _ in range(8)]

    def kr(bone, f, x=None, y=None, z=None):
        pb = P[bone]
        for i, v in enumerate((x, y, z)):
            if v is not None:
                pb.rotation_euler[i] = v
                pb.keyframe_insert("rotation_euler", index=i, frame=f)

    for f in range(n):
        t = f / FPS
        F = f + 1
        jaw, wide, rnd_, press, fv = vis[f]
        kr("jaw", F, x=math.radians(jaw))
        for name, v in (("wide", wide), ("round", rnd_), ("press", press), ("fv", fv)):
            keys[name].value = float(v)
            keys[name].keyframe_insert("value", frame=F)

        # head: slow wander, nods on stressed syllables, a tilt into each phrase
        turn = 0.10 * math.sin(2 * math.pi * 0.06 * t + phase[0]) + 0.05 * math.sin(2 * math.pi * 0.17 * t + phase[1])
        tilt = 0.05 * math.sin(2 * math.pi * 0.09 * t + phase[2]) + 0.04 * g_phrase[f]
        nodx = 0.07 * nod[f] - 0.03 * es[f] + 0.02 * math.sin(2 * math.pi * 0.11 * t + phase[3])
        kr("head", F, x=nodx, y=turn, z=tilt)
        kr("neck", F, x=0.4 * nodx, y=0.35 * turn, z=0.3 * tilt)
        # body: breathing, leaning in while talking
        breath = math.sin(2 * math.pi * t / 4.2)
        kr("chest", F, x=0.012 * breath + 0.025 * g_phrase[f])
        kr("spine", F, x=0.03 * g_phrase[f], y=0.25 * turn)
        for s, sign in (("R", 1), ("L", -1)):
            kr(f"shoulder.{s}", F, x=0.03 * open_hand[f] + 0.01 * breath)

        # eyes keep finding the camera: counter the head turn, plus small saccades
        for s in ("R", "L"):
            kr(f"eye.{s}", F, x=sac_pitch[f] - 0.5 * nodx, z=sac_yaw[f] - 0.8 * turn)
            v = blink[f]
            lids[s].rotation_euler.x = math.radians(LID_OPEN[s] + (LID_SHUT - LID_OPEN[s]) * v)
            lids[s].keyframe_insert("rotation_euler", index=0, frame=F)
        for s in ("R", "L"):
            b = P[f"brow.{s}"]
            b.location.y = 0.010 * min(1, max(0, (es[f] - 0.5) * 2.5)) + 0.006 * g_beat[f] + 0.004 * open_hand[f]
            b.keyframe_insert("location", index=1, frame=F)

        # hands rest clasped on the desk (the rig's rest pose). The right hand is the talker:
        # it lifts out of the clasp into each phrase, opens, and beats on stressed syllables.
        g, b_, oh = g_phrase[f], g_beat[f] * g_phrase[f], open_hand[f]
        kr("upperarm.R", F, x=-0.12 * g - 0.10 * oh)
        kr("forearm.R", F, x=0.50 * g + 0.15 * b_ + 0.20 * oh,
           y=-0.45 * g + 0.10 * math.sin(2 * math.pi * 0.4 * t) * g)
        kr("hand.R", F, x=-0.25 * g + 0.2 * b_)
        for fn in ("index", "middle", "ring", "pinky"):
            curl = -0.55 * max(g, oh) + 0.12 * b_
            kr(f"{fn}.1.R", F, x=curl)
            kr(f"{fn}.2.R", F, x=curl * 0.9)
        kr("thumb.1.R", F, x=-0.3 * g)
        # left hand stays on the desk: drums a finger while he talks, opens with the right
        tap = max(0.0, math.sin(2 * math.pi * 1.6 * t)) * g * (1 - oh) * 0.35
        kr("forearm.L", F, x=0.30 * oh, y=0.40 * oh)
        kr("hand.L", F, x=-0.2 * oh)
        for fn in ("index", "middle", "ring", "pinky"):
            uncurl = -0.5 * oh + (tap if fn == "index" else 0.0)
            kr(f"{fn}.1.L", F, x=uncurl)
            kr(f"{fn}.2.L", F, x=-0.4 * oh)


def screen_cues(studio, ep, n):
    """episode/screen.json: [{"t": seconds, "image": path}] -> cut the studio screen to each image."""
    path = ep / "screen.json"
    if not path.exists():
        return
    cues = json.loads(path.read_text(encoding="utf-8"))
    studio["screen"].hide_render = True
    panels = {}
    for k, c in enumerate(cues):
        if c["image"] not in panels:
            loc = (SCREEN_LOC[0], SCREEN_LOC[1] - 0.002 * (len(panels) + 1), SCREEN_LOC[2])
            panels[c["image"]], _ = image_panel(f"screen_cue{len(panels)}", ROOT / c["image"], SCREEN_SIZE, loc)
    for ob in panels.values():
        ob.hide_render = True
        ob.keyframe_insert("hide_render", frame=1)
    for c in cues:
        f = max(1, int(round(c["t"] * FPS)) + 1)
        for img, ob in panels.items():
            ob.hide_render = img != c["image"]
            ob.keyframe_insert("hide_render", frame=f)


def main():
    a = args()
    ep = (ROOT / a.episode).resolve() if not Path(a.episode).is_absolute() else Path(a.episode)
    with wave.open(str(ep / "voice.wav")) as w:
        n = int(math.ceil(w.getnframes() / w.getframerate() * FPS)) + FPS // 2
    if a.frames:
        n = min(n, a.frames)
    s = setup_scene(a.shot, n)
    studio = build_studio(ROOT)
    screen_cues(studio, ep, n)
    perform(ep, n)
    out = ep / "out" / f"3d_{a.shot}"
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    if not a.no_render:
        s.render.filepath = str(out / "frames") + "/"
        bpy.ops.render.render(animation=True)
    print(f"ANIM_DONE frames={n}")


main()
