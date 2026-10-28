"""Build and render a talking-anchor shot in Blender from a layered figure.

    blender -b -P build_scene.py -- --figure DIR --episode DIR [--frames N] [--no-render]

Reads   DIR/figure.json + layer PNGs   (see make_placeholder.py for the contract)
        episode/voice.wav, episode/mouth.json (Rhubarb), optional episode/../studio_bg.png
Writes  episode/out/scene.blend, episode/out/frames/####.png
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

FPS = 30
RES = (1920, 1080)
PX = 100.0  # pixels per Blender unit; ortho scale 19.2 => 1 px on canvas = 1 px on screen


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--figure", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--background")
    p.add_argument("--frames", type=int, help="limit frame count (for quick tests)")
    p.add_argument("--fig-origin", type=float, nargs=2, default=(460, -220),
                   help="screen px of the figure canvas top-left")
    p.add_argument("--no-render", action="store_true")
    return p.parse_args(argv)


def to_bu(sx, sy):
    return ((sx - RES[0] / 2) / PX, (RES[1] / 2 - sy) / PX)


# ---------------------------------------------------------------- scene setup

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            s.render.engine = eng
            break
        except TypeError:
            pass
    s.render.resolution_x, s.render.resolution_y = RES
    s.render.resolution_percentage = 100
    s.render.fps = FPS
    s.view_settings.view_transform = "Standard"  # keep painted colours exact
    s.view_settings.look = "None"
    s.eevee.taa_render_samples = 16
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    return s


def unlit_material(name, img_path):
    img = bpy.data.images.load(str(img_path))
    mat = bpy.data.materials.new(name)  # 5.x materials always have a node tree
    mat.surface_render_method = "BLENDED"
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "CLIP"
    emit = nt.nodes.new("ShaderNodeEmission")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(emit.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat, img.size[:]


def image_plane(name, img_path, parent=None, center_world=(0, 0), z=0.0):
    mat, (w, h) = unlit_material(name, img_path)
    hw, hh = w / PX / 2, h / PX / 2
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-hw, -hh, 0), (hw, -hh, 0), (hw, hh, 0), (-hw, hh, 0)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    for i, co in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[i].uv = co
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    px, py = (0, 0) if parent is None else parent["world"]
    ob.location = (center_world[0] - px, center_world[1] - py, z)
    if parent is not None:
        ob.parent = parent["obj"]
    return ob, (w, h)


def build_figure(fig_dir, origin):
    m = json.loads((fig_dir / "figure.json").read_text(encoding="utf-8"))
    ox, oy = origin

    def canvas_to_world(cx, cy):
        return to_bu(ox + cx, oy + cy)

    groups = {}

    def group(name):
        if name in groups:
            return groups[name]
        g = m["groups"][name]
        parent = group(g["parent"]) if g.get("parent") else None
        world = canvas_to_world(*g["pivot"])
        e = bpy.data.objects.new(f"grp_{name}", None)
        bpy.context.scene.collection.objects.link(e)
        if parent:
            e.parent = parent["obj"]
            e.location = (world[0] - parent["world"][0], world[1] - parent["world"][1], 0)
        else:
            e.location = (world[0], world[1], 0)
        groups[name] = {"obj": e, "world": world, "rest": tuple(e.location)}
        return groups[name]

    roles = {}
    for L in m["layers"]:
        w, h = _png_size(fig_dir / L["file"])
        cx, cy = canvas_to_world(L["pos"][0] + w / 2, L["pos"][1] + h / 2)
        ob, _ = image_plane(L["name"], fig_dir / L["file"], group(L["group"]), (cx, cy), L["z"] * 0.01)
        if "role" in L:
            roles[L["role"]] = ob

    mouths = {}
    mm = m["mouth"]
    for shape, f in mm["shapes"].items():
        w, h = _png_size(fig_dir / f)
        cx, cy = canvas_to_world(mm["pos"][0] + w / 2, mm["pos"][1] + h / 2)
        ob, _ = image_plane(f"mouth_{shape}", fig_dir / f, group(mm["group"]), (cx, cy), mm["z"] * 0.01)
        mouths[shape] = ob
    return groups, roles, mouths


def _png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def camera():
    cam = bpy.data.cameras.new("cam")
    cam.type = "ORTHO"
    cam.ortho_scale = RES[0] / PX
    ob = bpy.data.objects.new("cam", cam)
    ob.location = (0, 0, 50)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob


# ---------------------------------------------------------------- animation

def envelope(wav_path, n_frames):
    """Per-frame loudness 0..1 (fast) plus a slow version for emphasis."""
    with wave.open(str(wav_path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    a = a.reshape(-1, ch).mean(1) / 32768.0
    hop = sr / FPS
    a = np.pad(a, (0, max(0, int(math.ceil((n_frames + 1) * hop)) - len(a))))  # silent tail
    rms = np.array([np.sqrt(np.mean(a[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12)
                    for i in range(n_frames)])
    rms /= max(np.percentile(rms, 95), 1e-6)
    rms = np.clip(rms, 0, 1)

    def smooth(x, k):
        y, acc = np.empty_like(x), 0.0
        for i, v in enumerate(x):
            acc += k * (v - acc)
            y[i] = acc
        return y

    return smooth(rms, 0.45), smooth(rms, 0.08)


def mouth_track(cues, n_frames):
    shapes, j = [], 0
    for f in range(n_frames):
        t = f / FPS
        while j + 1 < len(cues) and cues[j]["end"] <= t:
            j += 1
        c = cues[j]
        v = c["value"] if c["start"] <= t < c["end"] else "X"
        shapes.append({"I": "B", "W": "D"}.get(v, v))
    return shapes


def animate(scene, groups, roles, mouths, cues, env_fast, env_slow, n):
    scene.frame_start, scene.frame_end = 1, n

    # mouths: one plane per shape, visibility switched only where the shape changes
    track = mouth_track(cues, n)
    prev = None
    for f, s in enumerate(track):
        if s == prev:
            continue
        for shape, ob in mouths.items():
            ob.hide_render = ob.hide_viewport = (shape != s)
            ob.keyframe_insert("hide_render", frame=f + 1)
            ob.keyframe_insert("hide_viewport", frame=f + 1)
        prev = s

    # blinks: every 2-5 s, closed for 4 frames
    rnd = random.Random(11)
    closed = set()
    t = rnd.uniform(1.0, 2.5)
    while t * FPS < n:
        start = int(t * FPS)
        closed.update(range(start, start + 4))
        t += rnd.uniform(2.0, 5.0)
    eo, ec = roles.get("eyes_open"), roles.get("eyes_closed")
    prev = None
    for f in range(n):
        c = f in closed
        if c == prev:
            continue
        for ob, hide in ((eo, c), (ec, not c)):
            if ob:
                ob.hide_render = ob.hide_viewport = hide
                ob.keyframe_insert("hide_render", frame=f + 1)
                ob.keyframe_insert("hide_viewport", frame=f + 1)
        prev = c

    # idle + speech-coupled motion
    body, head, brows = groups["body"], groups["head"], groups.get("brows")
    for f in range(n):
        t = f / FPS
        ef, es = float(env_fast[f]), float(env_slow[f])
        body["obj"].rotation_euler.z = 0.004 * math.sin(2 * math.pi * 0.07 * t)
        body["obj"].scale.y = 1 + 0.004 * math.sin(2 * math.pi * t / 4.2)
        body["obj"].keyframe_insert("rotation_euler", index=2, frame=f + 1)
        body["obj"].keyframe_insert("scale", index=1, frame=f + 1)

        head["obj"].rotation_euler.z = (0.014 * math.sin(2 * math.pi * 0.11 * t)
                                        + 0.007 * math.sin(2 * math.pi * 0.27 * t + 1.3)
                                        + 0.012 * es * math.sin(2 * math.pi * 0.45 * t))
        rx, ry, rz = head["rest"]
        head["obj"].location = (rx, ry - 0.04 * ef, rz)  # small nod on loud syllables
        head["obj"].keyframe_insert("rotation_euler", index=2, frame=f + 1)
        head["obj"].keyframe_insert("location", frame=f + 1)

        if brows:
            bx, by, bz = brows["rest"]
            lift = min(max((es - 0.55) * 3, 0), 1)
            brows["obj"].location = (bx, by + 0.06 * lift, bz)
            brows["obj"].keyframe_insert("location", frame=f + 1)


# ---------------------------------------------------------------- main

def main():
    a = args()
    fig, ep = Path(a.figure).resolve(), Path(a.episode).resolve()
    out = ep / "out"
    scene = reset()
    camera()

    bg = Path(a.background) if a.background else fig.parent / "studio_bg.png"
    if bg.exists():
        image_plane("background", bg, None, (0, 0), -1.0)

    groups, roles, mouths = build_figure(fig, a.fig_origin)

    with wave.open(str(ep / "voice.wav")) as w:
        dur = w.getnframes() / w.getframerate()
    n = int(math.ceil(dur * FPS)) + FPS // 2
    if a.frames:
        n = min(n, a.frames)
    cues = json.loads((ep / "mouth.json").read_text(encoding="utf-8"))["mouthCues"]
    env_fast, env_slow = envelope(ep / "voice.wav", n)
    animate(scene, groups, roles, mouths, cues, env_fast, env_slow, n)

    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    if not a.no_render:
        scene.render.filepath = str(out / "frames") + "/"
        bpy.ops.render.render(animation=True)
    print(f"ANCHOR_DONE frames={n}")


main()
