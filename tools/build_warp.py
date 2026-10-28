"""Build and render a talking shot from ONE flat painting, deformed by mesh shape keys.

    blender -b -P build_warp.py -- --layout layouts/A_desk.json --episode DIR [--frames N]
    blender -b -P build_warp.py -- --layout layouts/B_closeup.json --episode DIR --poses

The figure entry of the layout points at a folder with rig.json (landmarks in image px).
The painting becomes a dense grid mesh; every motion is a shape key computed here:
  mouth_close (negative = open wider than painted), mouth_narrow, blink, brows,
  head_tilt, head_nod, arm, breathe.
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

ROOT = Path(__file__).resolve().parent.parent
FPS = 30
RES = (1920, 1080)
PX = 100.0

# Rhubarb shape -> (mouth opening relative to the painting, narrowing)
VISEMES = {"A": (0.0, 0.0), "B": (0.28, 0.0), "C": (0.65, 0.0), "D": (1.05, 0.0),
           "E": (0.55, 0.55), "F": (0.22, 0.9), "G": (0.15, 0.1), "H": (0.7, 0.0), "X": (0.04, 0.0)}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--poses", action="store_true", help="render rig test stills instead")
    p.add_argument("--no-render", action="store_true")
    return p.parse_args(argv)


def to_bu(sx, sy):
    return ((sx - RES[0] / 2) / PX, (RES[1] / 2 - sy) / PX)


# ---------------------------------------------------------------- scene basics

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "BLENDER_EEVEE"
    s.render.resolution_x, s.render.resolution_y = RES
    s.render.fps = FPS
    s.view_settings.view_transform = "Standard"
    s.view_settings.look = "None"
    s.eevee.taa_render_samples = 8
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    cam = bpy.data.cameras.new("cam")
    cam.type = "ORTHO"
    cam.ortho_scale = RES[0] / PX
    ob = bpy.data.objects.new("cam", cam)
    ob.location = (0, 0, 50)
    s.collection.objects.link(ob)
    s.camera = ob
    return s


def unlit_material(name, img):
    mat = bpy.data.materials.new(name)
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
    return mat


def image_plane(name, path, pos, z):
    """Plane showing an image with its top-left at screen px `pos`."""
    img = bpy.data.images.load(str(path))
    w, h = img.size
    me = bpy.data.meshes.new(name)
    hw, hh = w / PX / 2, h / PX / 2
    me.from_pydata([(-hw, -hh, 0), (hw, -hh, 0), (hw, hh, 0), (-hw, hh, 0)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    for i, co in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[i].uv = co
    me.materials.append(unlit_material(name, img))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    cx, cy = to_bu(pos[0] + w / 2, pos[1] + h / 2)
    ob.location = (cx, cy, z)
    return ob, (w, h)


# ---------------------------------------------------------------- warp fields (image px, y down)

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def curve(pts, x):
    pts = np.asarray(pts, float)
    return np.interp(x, pts[:, 0], pts[:, 1])


def close_gap(X, Y, upper, lower, meet, reach_up, reach_down, eps=1.5):
    """dy that shuts the opening between two lid/lip curves.
    Inside the gap everything squeezes onto a line at `meet` (0 = upper edge,
    1 = lower edge); above and below, the push fades out over reach_up/down px."""
    x0, x1 = upper[0][0], upper[-1][0]
    inside_x = (X >= x0) & (X <= x1)
    u = curve(upper, X)
    l = curve(lower, X)
    gap = np.maximum(l - u, 0)
    m = u + meet * gap
    dy = np.zeros_like(X)
    safe = np.maximum(gap, 1e-6)
    a = inside_x & (Y >= u) & (Y <= l) & (gap > 0.5)
    dy[a] = (m[a] + ((Y[a] - u[a]) / safe[a] - meet) * eps) - Y[a]
    top_shift = m - meet * eps - u
    b = inside_x & (Y < u)
    dy[b] = top_shift[b] * (1 - smoothstep((u[b] - Y[b]) / reach_up))
    bot_shift = m + (1 - meet) * eps - l
    c = inside_x & (Y > l)
    dy[c] = bot_shift[c] * (1 - smoothstep((Y[c] - l[c]) / reach_down))
    # soften the vertical seams just past the corners
    edge = np.minimum(X - x0, x1 - X)
    soft = smoothstep(edge / 6.0)
    dy[~a] *= soft[~a]
    return dy


def ellipse_weight(X, Y, center, radius):
    r = np.sqrt(((X - center[0]) / radius[0]) ** 2 + ((Y - center[1]) / radius[1]) ** 2)
    return 1 - smoothstep(r)


def rotate_field(X, Y, pivot, deg, w):
    th = math.radians(deg)
    dx0, dy0 = X - pivot[0], Y - pivot[1]
    # y is down in image space, so a positive angle tips the head toward screen-right
    rx = dx0 * math.cos(th) - dy0 * math.sin(th)
    ry = dx0 * math.sin(th) + dy0 * math.cos(th)
    return (rx - dx0) * w, (ry - dy0) * w


def build_fields(rig, X, Y):
    f = {}
    M = rig["mouth"]
    f["mouth_close"] = (np.zeros_like(X),
                        close_gap(X, Y, M["upper"], M["lower"], M["meet"], M["reach_up"], M["reach_down"]))
    w = ellipse_weight(X, Y, M["center"], M["narrow_radius"])
    f["mouth_narrow"] = (-(X - M["center"][0]) * 0.32 * w, np.zeros_like(X))
    wb = sum(ellipse_weight(X, Y, b["center"], b["radius"]) for b in rig["brows"])
    f["brows"] = (np.zeros_like(X), -rig["brow_lift"] * np.clip(wb, 0, 1))
    H = rig["head"]
    wh = 1 - smoothstep((Y - H["full_above"]) / (H["zero_below"] - H["full_above"]))
    f["head_tilt"] = rotate_field(X, Y, H["pivot"], H["tilt_deg"], wh)
    f["head_nod"] = (np.zeros_like(X), H["nod_px"] * wh)
    A = rig["arm"]
    wx = smoothstep((A["x_zero"] - X) / (A["x_zero"] - A["x_full"]))
    y0, y1 = A["y_range"]
    wy = smoothstep((Y - (y0 - A["y_soft"])) / A["y_soft"]) * (1 - smoothstep((Y - y1) / A["y_soft"]))
    f["arm"] = rotate_field(X, Y, A["pivot"], -A["deg"], wx * wy)
    B = rig["breathe"]
    f["breathe"] = (np.zeros_like(X), -(B["base_y"] - Y) * B["amount"])
    return f


def figure_mesh(entry, z):
    fig_dir = ROOT / entry["figure"]
    rig = json.loads((fig_dir / "rig.json").read_text(encoding="utf-8"))
    img = bpy.data.images.load(str(fig_dir / entry["image"]))
    iw, ih = img.size
    s, (px, py) = entry["scale"], entry["pos"]
    bx0, by0, bx1, by1 = rig["bbox"]
    step = rig["grid"]
    xs = np.arange(bx0, bx1 + step, step, dtype=np.float64)
    ys = np.arange(by0, by1 + step, step, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    nx, ny = len(xs), len(ys)

    def to_world(Xi, Yi):
        return (np.stack([(px + Xi * s - RES[0] / 2) / PX, (RES[1] / 2 - (py + Yi * s)) / PX,
                          np.zeros_like(Xi)], -1)).reshape(-1, 3)

    base = to_world(X, Y)
    idx = np.arange(nx * ny).reshape(ny, nx)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)

    me = bpy.data.meshes.new("figure")
    me.from_pydata(base.tolist(), [], quads.tolist())
    uv = me.uv_layers.new()
    loop_v = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    uvs = np.stack([X.ravel() / iw, 1 - Y.ravel() / ih], -1)[loop_v]
    uv.data.foreach_set("uv", uvs.astype(np.float32).ravel())
    me.materials.append(unlit_material("figure", img))
    ob = bpy.data.objects.new("figure", me)
    ob.location.z = z
    bpy.context.scene.collection.objects.link(ob)

    ob.shape_key_add(name="Basis")
    for name, (dx, dy) in build_fields(rig, X, Y).items():
        kb = ob.shape_key_add(name=name, from_mix=False)
        kb.slider_min, kb.slider_max = -1.5, 1.5
        kb.data.foreach_set("co", to_world(X + dx, Y + dy).astype(np.float32).ravel())

    # closed-eye painting on a twin object sharing the mesh, so it rides every deformation
    lids = bpy.data.objects.new("figure_blink", me)
    lids.location.z = z + 0.005
    bpy.context.scene.collection.objects.link(lids)
    lids.material_slots[0].link = "OBJECT"
    lids.material_slots[0].material = unlit_material(
        "blink", bpy.data.images.load(str(fig_dir / "blink_overlay.png")))
    lids.hide_render = True
    return ob, lids


# ---------------------------------------------------------------- layout

def png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def build(layout):
    objs = {}
    for i, L in enumerate(layout):
        z = i * 0.01
        if "figure" in L:
            objs["figure"], objs["lids"] = figure_mesh(L, z)
            continue
        path = ROOT / L["file"]
        w, h = png_size(path)
        x, y = L["pos"]
        if L.get("align") == "right":
            x -= w
        ob, size = image_plane(Path(L["file"]).stem, path, (x, y), z)
        if L.get("role"):
            objs[L["role"]] = (ob, L, size)
    return objs


# ---------------------------------------------------------------- animation

def envelope(wav_path, n):
    with wave.open(str(wav_path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    a = a.reshape(-1, ch).mean(1) / 32768.0
    hop = sr / FPS
    a = np.pad(a, (0, max(0, int(math.ceil((n + 1) * hop)) - len(a))))
    rms = np.array([np.sqrt(np.mean(a[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(n)])
    rms = np.clip(rms / max(np.percentile(rms, 95), 1e-6), 0, 1)
    return smooth(rms, 0.45), smooth(rms, 0.06)


def smooth(x, k):
    y, acc = np.empty_like(x, dtype=float), float(x[0])
    for i, v in enumerate(x):
        acc += k * (v - acc)
        y[i] = acc
    return y


def mouth_targets(cues, n):
    opening, narrow, j = np.zeros(n), np.zeros(n), 0
    for f in range(n):
        t = f / FPS
        while j + 1 < len(cues) and cues[j]["end"] <= t:
            j += 1
        c = cues[j]
        v = c["value"] if c["start"] <= t < c["end"] else "X"
        o, w = VISEMES.get(v) or VISEMES[{"I": "B", "W": "D"}.get(v, "B")]
        opening[f], narrow[f] = o, w
    return smooth(opening, 0.6), smooth(narrow, 0.5)


def blink_track(n, seed=5):
    v = np.zeros(n)
    rnd = random.Random(seed)
    t = rnd.uniform(0.8, 2.0)
    while t * FPS < n - 6:
        f0 = int(t * FPS)
        for k, val in enumerate((0.55, 1.0, 1.0, 0.6, 0.2)):
            v[f0 + k] = val
        t += rnd.uniform(2.2, 5.0) if rnd.random() > 0.15 else 0.35  # occasional double blink
    return v


def key(kb, value, f):
    kb.value = float(value)
    kb.keyframe_insert("value", frame=f)


def animate(scene, objs, ep, n):
    scene.frame_start, scene.frame_end = 1, n
    fig = objs["figure"]
    kbs = fig.data.shape_keys.key_blocks
    cues = json.loads((ep / "mouth.json").read_text(encoding="utf-8"))["mouthCues"]
    opening, narrow = mouth_targets(cues, n)
    ef, es = envelope(ep / "voice.wav", n)
    blinks = blink_track(n) >= 0.5
    lids = objs["lids"]
    for f in range(n):
        if f == 0 or blinks[f] != blinks[f - 1]:
            lids.hide_render = not blinks[f]
            lids.keyframe_insert("hide_render", frame=f + 1)
    rnd = np.random.default_rng(3)
    ph = rnd.uniform(0, 6.28, 6)
    for f in range(n):
        t = f / FPS
        key(kbs["mouth_close"], 1 - opening[f], f + 1)
        key(kbs["mouth_narrow"], narrow[f], f + 1)
        key(kbs["brows"], np.clip((es[f] - 0.45) * 2.2, 0, 1) + 0.25 * max(0, math.sin(0.9 * t + ph[0])), f + 1)
        key(kbs["head_tilt"], 0.55 * math.sin(2 * math.pi * 0.09 * t + ph[1])
            + 0.3 * math.sin(2 * math.pi * 0.23 * t + ph[2]) + 0.35 * es[f] * math.sin(2 * math.pi * 0.4 * t), f + 1)
        key(kbs["head_nod"], 0.8 * ef[f] - 0.3, f + 1)
        key(kbs["arm"], es[f] * (0.6 + 0.6 * math.sin(2 * math.pi * 0.3 * t + ph[3])) - 0.2, f + 1)
        key(kbs["breathe"], math.sin(2 * math.pi * t / 4.0), f + 1)

    if "lower_third" in objs:  # slide in after a beat, from the side it is anchored to
        ob, L, size = objs["lower_third"]
        home = ob.location.x
        off = (size[0] + 80) / PX * (1 if L.get("align") == "right" else -1)
        ob.location.x = home + off
        ob.keyframe_insert("location", index=0, frame=1)
        ob.keyframe_insert("location", index=0, frame=int(0.8 * FPS))
        ob.location.x = home
        ob.keyframe_insert("location", index=0, frame=int(1.3 * FPS))
    if "ticker" in objs:  # Arabic tickers run left-to-right
        ob, L, size = objs["ticker"]
        start = ob.location.x
        for f in (1, n):
            ob.location.x = start + 110 * (f - 1) / FPS / PX
            ob.keyframe_insert("location", index=0, frame=f)
        set_linear(ob)


def set_linear(ob):
    """Constant-speed scroll: make the ticker's location curve linear."""
    ad = ob.animation_data
    if not ad or not ad.action:
        return
    try:
        bag = ad.action.layers[0].strips[0].channelbag(ad.action_slot)
        curves = bag.fcurves
    except (AttributeError, IndexError):
        curves = ad.action.fcurves
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


POSES = {
    "neutral": {},
    "closed": {"mouth_close": 1},
    "half": {"mouth_close": 0.5},
    "wide": {"mouth_close": -0.35},
    "round": {"mouth_close": 0.45, "mouth_narrow": 1},
    "blink": {"_lids": True, "head_tilt": 1.0},
    "brows": {"brows": 1},
    "tilt_l": {"head_tilt": 1.0, "head_nod": 1},
    "tilt_r": {"head_tilt": -1.0},
    "arm_up": {"arm": 1},
    "arm_down": {"arm": -1},
}


def main():
    a = args()
    layout = json.loads(Path(a.layout).read_text(encoding="utf-8"))
    ep = Path(a.episode).resolve()
    out = ep / "out" / Path(a.layout).stem
    scene = reset()
    objs = build(layout)
    out.mkdir(parents=True, exist_ok=True)

    if a.poses:
        kbs = objs["figure"].data.shape_keys.key_blocks
        for name, vals in POSES.items():
            for kb in kbs[1:]:
                kb.value = vals.get(kb.name, 0.0)
            objs["lids"].hide_render = not vals.get("_lids", False)
            scene.render.filepath = str(out / f"pose_{name}.png")
            bpy.ops.render.render(write_still=True)
        print("POSES_DONE")
        return

    with wave.open(str(ep / "voice.wav")) as w:
        n = int(math.ceil(w.getnframes() / w.getframerate() * FPS)) + FPS // 2
    if a.frames:
        n = min(n, a.frames)
    animate(scene, objs, ep, n)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    if not a.no_render:
        scene.render.filepath = str(out / "frames") + "/"
        bpy.ops.render.render(animation=True)
    print(f"ANCHOR_DONE frames={n}")


main()
