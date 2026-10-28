"""Animate a reference image as a 3D relief: every pixel is pushed out to its estimated depth
along its own camera ray, so at rest the render IS the reference; when the head turns or
nods the relief gives real parallax.

    blender -b -P tools/relief/build_relief.py -- --ref assets/haj_relief/ref10 \
        --image assets/haj_salama/refs/ref10.png --episode episodes/ar01 [--frames N] [--poses]

Inputs in --ref: rgba.png, depth.npy, rig.json, frame.json, blink.png, plates/*.png
Writes episode/out/relief_<ref>/frames/####.png (full frame: plate + character)
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools/haj3d"))
from signals import FPS, beats, bump, envelope, phrases, smooth, viseme_tracks  # noqa: E402
sys.path.insert(0, str(ROOT / "tools/capture"))
import mouth_model as mm  # noqa: E402

W, H = 1920, 1080
LENS = 50.0
F_PX = LENS / 36.0 * W
Z_FAR, Z_SPAN = 2.6, 0.55     # metres: the deepest part of the figure, and how far the nearest comes forward
BELOW = 14                    # px the mesh continues below a frame-cut figure, so body sway never shows the edge
GRID = 3                      # px between mesh vertices on the reference
# mouth shape -> (close, jaw, wide, round). The painting's mouth is a little open (~20 px):
# close squeezes the lips together over it, jaw drops the lower lip and chin to show the interior.
VISEMES = {"A": (1.00, 0.00, 0.10, 0.0),   # m b p: lips pressed
           "B": (0.55, 0.00, 0.45, 0.0),   # most consonants, i: teeth just apart, lips spread
           "C": (0.10, 0.25, 0.20, 0.0),   # e, throat sounds
           "D": (0.00, 0.85, 0.05, 0.0),   # long a
           "E": (0.20, 0.40, 0.00, 0.45),  # o
           "F": (0.55, 0.05, 0.00, 1.00),  # u w: pursed
           "G": (0.85, 0.00, 0.15, 0.0),   # f v: lower lip to the teeth
           "H": (0.15, 0.35, 0.15, 0.0),   # l
           "X": (0.75, 0.00, 0.00, 0.0)}   # rest: lips nearly together
VISEMES["I"] = VISEMES["B"]                # finer classes from tts_align, for the warp-only rig
VISEMES["W"] = VISEMES["D"]


def args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--poses", action="store_true")
    p.add_argument("--from", dest="from_frame", type=int, default=1, help="render from this frame on (redo a tail)")
    return p.parse_args(argv)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ image + depth

def load_rgba(path):
    img = bpy.data.images.load(str(path))
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return img, px.reshape(h, w, 4)[::-1]  # top row first


def dilate_depth(d, mask, steps=12):
    """Carry the figure's depth a few pixels past its silhouette so edges don't sink back."""
    out = np.where(mask, d, -1e9)
    for _ in range(steps):
        p = np.pad(out, 1, mode="edge")
        nb = np.max([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]], axis=0)
        out = np.where(out > -1e8, out, nb)
    return np.where(out > -1e8, out, d)


def box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def metric_depth(disp, mask):
    lo, hi = np.percentile(disp[mask], [3, 97])
    t = np.clip((disp - lo) / (hi - lo + 1e-9), -0.2, 1.2)
    return Z_FAR - t * Z_SPAN


# ------------------------------------------------------------------ warp fields (reference px)

def curve(pts, x):
    pts = np.asarray(pts, float)
    return np.interp(x, pts[:, 0], pts[:, 1])


def close_gap(X, Y, upper, lower, meet, reach_up, reach_down, eps=1.2):
    x0, x1 = upper[0][0], upper[-1][0]
    inside_x = (X >= x0) & (X <= x1)
    u, l = curve(upper, X), curve(lower, X)
    gap = np.maximum(l - u, 0)
    m = u + meet * gap
    dy = np.zeros_like(X)
    safe = np.maximum(gap, 1e-6)
    a = inside_x & (Y >= u) & (Y <= l) & (gap > 0.5)
    dy[a] = (m[a] + ((Y[a] - u[a]) / safe[a] - meet) * eps) - Y[a]
    b = inside_x & (Y < u)
    dy[b] = (m - meet * eps - u)[b] * (1 - smoothstep((u[b] - Y[b]) / reach_up))
    c = inside_x & (Y > l)
    dy[c] = (m + (1 - meet) * eps - l)[c] * (1 - smoothstep((Y[c] - l[c]) / reach_down))
    edge = np.minimum(X - x0, x1 - X)
    dy[~a] *= smoothstep(edge / 8.0)[~a]
    # the jaw: below the mouth the push spreads sideways a little past the corners
    return dy


def ellipse_w(X, Y, c, r):
    return 1 - smoothstep(np.sqrt(((X - c[0]) / r[0]) ** 2 + ((Y - c[1]) / r[1]) ** 2))


# ------------------------------------------------------------------ scene

def setup(n):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "BLENDER_EEVEE"
    s.render.resolution_x, s.render.resolution_y = W, H
    s.render.fps = FPS
    s.view_settings.view_transform = "Standard"
    s.view_settings.look = "None"
    s.eevee.taa_render_samples = 24  # fine painted lines shimmer under sub-pixel motion at 8
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    s.frame_start, s.frame_end = 1, n
    cd = bpy.data.cameras.new("cam")
    cd.lens, cd.sensor_width, cd.sensor_fit = LENS, 36.0, "HORIZONTAL"
    cam = bpy.data.objects.new("cam", cd)  # at the origin looking down -Z, +Y up
    s.collection.objects.link(cam)
    s.camera = cam
    return s


def unlit(name, img, blend=True, mask=None):
    """Self-lit picture material (the painting is already lit).
    blend: True = alpha-blended, "DITHERED" = cut-out drawn with the opaque pass, False = opaque.
    mask: optional image whose alpha clips this one at the *unshifted* UVs; the texture lookup
    then goes through a Mapping node whose Location animates (used to move the irises)."""
    m = bpy.data.materials.new(name)
    if blend:
        m.surface_render_method = "DITHERED" if blend == "DITHERED" else "BLENDED"
    nt = m.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Cubic"
    tex.extension = "CLIP"
    e = nt.nodes.new("ShaderNodeEmission")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], e.inputs["Color"])
    alpha = tex.outputs["Alpha"]
    if mask is not None:
        uv = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.name = "gaze"
        nt.links.new(uv.outputs["UV"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
        mt = nt.nodes.new("ShaderNodeTexImage")
        mt.image = mask
        mt.extension = "CLIP"
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(alpha, mul.inputs[0])
        nt.links.new(mt.outputs["Color"], mul.inputs[1])
        alpha = mul.outputs[0]
    # "fade" scales the alpha; animated to cross-fade painted variants
    fade = nt.nodes.new("ShaderNodeMath")
    fade.name = "fade"
    fade.operation = "MULTIPLY"
    fade.inputs[1].default_value = 1.0
    nt.links.new(alpha, fade.inputs[0])
    alpha = fade.outputs[0]
    nt.links.new(alpha, mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def plate_plane(name, path, z=-9.0):
    """Background plate filling the whole frame at distance -z."""
    img = bpy.data.images.load(str(path))
    hw, hh = W / 2 / F_PX * -z, H / 2 / F_PX * -z
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-hw, -hh, z), (hw, -hh, z), (hw, hh, z), (-hw, hh, z)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    for i, c in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[i].uv = c
    mat = unlit(name, img, blend=False)
    next(nd for nd in mat.node_tree.nodes if nd.type == "TEX_IMAGE").extension = "EXTEND"  # no dark rim at the frame edge
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


class Relief:
    def __init__(self, ref_dir, image_path):
        self.dir = Path(ref_dir)
        self.rig = json.loads((self.dir / "rig.json").read_text(encoding="utf-8"))
        fr = json.loads((self.dir / "frame.json").read_text(encoding="utf-8"))
        self.s, (self.ox, self.oy) = fr["scale"], fr["offset"]
        _, rgba = load_rgba(self.dir / "rgba.png")
        # the texture is base.png: the same cut-out with the irises lifted out (prep_face.py)
        self.img = bpy.data.images.load(str(self.dir / "base.png"))
        self.h, self.w = rgba.shape[:2]
        self.alpha = rgba[..., 3]
        self.rgb = rgba[..., :3]
        mask = self.alpha > 0.5
        disp = np.load(self.dir / "depth.npy")
        # smooth depth: a head turn should gently warp the painting, not tear it where depth
        # jumps (cheek in front of shemagh and studio); fine relief detail costs us more than it gives
        disp = box_blur(box_blur(dilate_depth(disp, mask, steps=20), 6), 6)
        self.Z = metric_depth(disp, mask)
        # painted mode: generated mouth/eye variants (prep_variants.py) are cross-faded on the
        # face instead of cutting the mouth open and bending the lips
        self.variants = [k for k in self.rig.get("variants", {}) if (self.dir / f"var_{k}.png").exists()]
        self.veo = None
        pool = self.rig.get("veo_pool")
        self.veo_pool = pool if pool and (self.dir / pool / "pool.json").exists() else None
        if pool and not self.veo_pool:
            print("NOTE: Veo gesture pool not built yet (python tools/capture/veo_rebuild.py); "
                  "using the painted gesture poses")
        self.painted = bool(self.variants)

    def project(self, u, v, z):
        """Reference pixel (u, v) at camera distance z -> world point (camera at origin)."""
        X = self.ox + self.s * u
        Y = self.oy + self.s * v
        return np.stack([(X - W / 2) / F_PX * z, -(Y - H / 2) / F_PX * z, -z], -1)

    def z_at(self, u, v):
        return float(self.Z[int(round(v)), int(round(u))])

    def point(self, u, v, dz=0.0):
        return Vector(self.project(np.array(u, float), np.array(v, float), self.z_at(u, v) + dz))

    def build(self):
        ys, xs = np.nonzero(self.alpha > 0.02)
        u0, u1 = xs.min() - GRID, xs.max() + GRID
        v0, v1 = ys.min() - GRID, ys.max() + GRID
        us = np.arange(max(0, u0), min(self.w - 1, u1) + 1, GRID, dtype=float)
        vs = np.arange(max(0, v0), min(self.h - 1, v1) + 1, GRID, dtype=float)
        if v1 >= self.h - 1:  # a figure cut by the frame (standing): the mesh runs past the bottom edge
            vs = np.append(vs[vs < self.h - 1], [self.h, self.h + BELOW])  # (its last pixel row stretched)
        U, V = np.meshgrid(us, vs)
        A = self.alpha[np.minimum(V, self.h - 1).astype(int), U.astype(int)]
        # cut the mouth opening out (the interior layer behind shows through it): grid points
        # inside the opening snap onto the nearer lip curve so the lip edges stay smooth, and
        # quads left spanning the opening are dropped
        M = self.rig["mouth"]
        x0, x1 = M["upper"][0][0], M["upper"][-1][0]
        up, lo = curve(M["upper"], U), curve(M["lower"], U)
        inside = (U > x0) & (U < x1) & (V > up) & (V < lo) & (not self.painted)
        to_up = inside & (V - up < lo - V)
        to_lo = inside & ~to_up
        V = np.where(to_up, up - 0.4, np.where(to_lo, lo + 0.4, V))
        side = np.where(to_up, 1, np.where(to_lo, -1, 0)).ravel()
        Zg = self.Z[np.clip(np.round(V), 0, self.h - 1).astype(int), U.astype(int)]
        ny, nx = U.shape
        idx = np.arange(nx * ny).reshape(ny, nx)
        quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        keep = (A.ravel()[quads] > 0.02).any(1)
        qs = side[quads]
        spans = (qs == 1).any(1) & (qs == -1).any(1)
        all_in = (qs != 0).all(1)
        quads = quads[keep & ~spans & ~all_in]
        used = np.unique(quads)
        remap = -np.ones(nx * ny, int)
        remap[used] = np.arange(len(used))
        quads = remap[quads]
        self.U, self.V, self.Zv = U.ravel()[used], V.ravel()[used], Zg.ravel()[used]
        co = self.project(self.U, self.V, self.Zv)

        me = bpy.data.meshes.new("relief")
        me.from_pydata(co.tolist(), [], quads.tolist())
        uv = me.uv_layers.new()
        lv = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", lv)
        uvs = np.stack([self.U / self.w, 1 - np.minimum(self.V, self.h - 0.5) / self.h], -1)[lv]
        uv.data.foreach_set("uv", uvs.astype(np.float32).ravel())
        me.materials.append(unlit("relief", self.img))
        ob = bpy.data.objects.new("relief", me)
        bpy.context.scene.collection.objects.link(ob)
        self.ob = ob
        self.shape_keys()
        self.armature()
        # twins sharing the mesh ride every deformation: the irises (shifted for gaze, clipped
        # to the eye openings) and the closed-eye painting on top of them
        load = lambda f: bpy.data.images.load(str(self.dir / f))  # noqa: E731
        self.iris = self.twin("relief_iris", unlit("iris", load("iris.png"), mask=load("eye_mask.png")), 0.001)
        if self.painted:
            # one twin per painted variant, stacked; their "fade" values are animated
            self.layers, self.gestures = {}, {}
            for k, name in enumerate(self.variants):
                spec = self.rig["variants"][name]
                if "gesture" in spec and self.veo_pool:
                    continue  # Veo frames play the body instead of the painted poses
                mat = unlit(f"var_{name}", load(f"var_{name}.png"))
                if "gesture" in spec:
                    # body poses sit under the face layers; the part outside the reference's
                    # silhouette (a raised hand over the background) gets its own surface
                    tw = self.twin(f"var_{name}", mat, 0.0005 + 0.0001 * k)
                    desk = self.rig.get("desk_y", 860)
                    outside = self.outside_plane(f"var_{name}_out", mat, load(f"var_{name}.png"), rows=(0, desk),
                                                 bone="chest" if spec["gesture"] == "both" else None)
                    on_desk = self.outside_plane(f"var_{name}_desk", mat, load(f"var_{name}.png"), rows=(desk, 10 ** 6), bone="")
                    hide = self.add_hide(name, load(f"var_{name}_hide.png")) if spec.get("hide_green") else None
                    self.gestures[name] = {"kind": spec["gesture"], "via": spec.get("via"), "hide": hide,
                                           "objs": [ob for ob in (tw, outside, on_desk) if ob], "z": k}
                else:
                    tw = self.twin(f"var_{name}", mat, 0.0015 + 0.0002 * k)
                self.layers[name] = tw
            self.lids = self.layers.get("eyes_closed")
            if self.veo_pool:
                self.add_veo(self.veo_pool)
        else:
            self.lids = self.twin("relief_blink", unlit("blink", load("blink.png")), 0.002)
            self.lids.hide_render = True
            self.mouth_layers()
        return ob

    def fade(self, ob):
        return ob.material_slots[0].material.node_tree.nodes["fade"].inputs[1]

    def add_hide(self, name, img):
        """Make the relief see-through where `img` (alpha) says, scaled by an animatable amount:
        relief alpha *= 1 - img.alpha * amount. Returns the amount socket."""
        nt = self.ob.material_slots[0].material.node_tree
        fade = nt.nodes["fade"]
        src = fade.inputs[0].links[0].from_socket
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        img.colorspace_settings.name = "Non-Color"
        tex.extension = "CLIP"
        amt = nt.nodes.new("ShaderNodeMath")
        amt.name = f"hide_{name}"
        amt.operation = "MULTIPLY"
        amt.inputs[1].default_value = 0.0
        nt.links.new(tex.outputs["Color"], amt.inputs[0])
        # fully hidden must mean alpha 0: a 1% remainder shows as a ghost over flat green
        full = nt.nodes.new("ShaderNodeMath")
        full.operation = "MULTIPLY"
        full.use_clamp = True
        full.inputs[1].default_value = 1.04
        nt.links.new(amt.outputs[0], full.inputs[0])
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        nt.links.new(full.outputs[0], inv.inputs[1])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(src, mul.inputs[0])
        nt.links.new(inv.outputs[0], mul.inputs[1])
        nt.links.new(mul.outputs[0], fade.inputs[0])
        self.last_hide_tex = tex
        return amt.inputs[1]

    def add_veo(self, pool_dir):
        """Gesture frames from Veo clips (tools/capture/veo_pool.py): one body layer whose picture can
        be any pool frame, chosen per output frame, with the matching where-the-arm-left mask on the
        relief. The part of a raised arm outside the reference's silhouette gets a surface on the chest."""
        d = (self.dir / pool_dir).resolve()
        meta = json.loads((d / "pool.json").read_text(encoding="utf-8"))

        def seq(path):
            img = bpy.data.images.load(str(path))
            img.source = "SEQUENCE"
            return img

        mat = unlit("veo_body", seq(d / "body" / "0001.png"))
        body_iu = next(nd for nd in mat.node_tree.nodes if nd.type == "TEX_IMAGE").image_user
        tw = self.twin("veo_body", mat, 0.0005)
        out = self.outside_plane("veo_out", mat, bpy.data.images.load(str(d / "outside_union.png")),
                                 rows=(0, self.rig.get("desk_y", 860)), bone="chest")
        amt = self.add_hide("veo", seq(d / "hide" / "0001.png"))
        ius = (body_iu, self.last_hide_tex.image_user)
        for iu in ius:
            # Blender clamps the scene frame to the sequence length *before* adding the offset, so the
            # length must cover the whole episode, not just the pool (else frames past it go missing)
            iu.frame_start, iu.frame_duration, iu.use_auto_refresh = 1, 1_000_000, True
        self.veo = {"takes": meta["takes"], "fps": meta.get("fps", 24), "ius": ius, "hide": amt,
                    "objs": [ob for ob in (tw, out) if ob], "fade": self.fade(tw)}

    def set_veo(self, idx, w, frame):
        """Show pool frame idx (1-based) at amount w on output frame `frame` (w = 0: no Veo body)."""
        V = self.veo
        for iu in V["ius"]:
            iu.frame_offset = int(idx) - frame if idx else 0
            iu.keyframe_insert("frame_offset", frame=frame)
        for sock in (V["fade"], V["hide"]):
            sock.default_value = float(w)
            sock.keyframe_insert("default_value", frame=frame)
        for ob in V["objs"]:
            ob.hide_render = bool(w == 0)
            ob.keyframe_insert("hide_render", frame=frame)

    def outside_plane(self, name, mat, img, step=3, bone=None, rows=None):
        """A surface for the parts of a variant that lie outside the reference's silhouette,
        at the depth of the arm, carried by a bone (bone="" = fixed in place).
        rows=(v0, v1) limits it to those reference rows."""
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        va = px.reshape(h, w, 4)[::-1, :, 3]
        out = (va > 0.02) & (self.alpha < 0.5)
        if rows is not None:
            keep = np.zeros_like(out)
            keep[max(rows[0], 0):min(rows[1], h)] = True
            out &= keep
        ys, xs = np.nonzero(out)
        if not len(xs):
            return None
        us = np.arange(xs.min() - 6, xs.max() + 7, step, dtype=float)
        vs = np.arange(ys.min() - 6, ys.max() + 7, step, dtype=float)
        U, V = np.meshgrid(us, vs)
        Ui, Vi = np.clip(U, 0, w - 1).astype(int), np.clip(V, 0, h - 1).astype(int)
        near = out[Vi, Ui]
        ny, nx = U.shape
        idx = np.arange(nx * ny).reshape(ny, nx)
        quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        grow = np.zeros_like(near)  # keep quads touching the outside area, with a small margin
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                grow |= np.roll(np.roll(near, dy, 0), dx, 1)
        quads = quads[grow.ravel()[quads].any(1)]
        if rows is not None:
            qv = V.ravel()[quads].mean(1)
            quads = quads[(qv >= rows[0]) & (qv < rows[1])]
        z = self.z_at(*self.rig.get("arm_point", self.rig["neck"])) - 0.02
        co = self.project(U.ravel(), V.ravel(), np.full(U.size, z))
        me = bpy.data.meshes.new(name)
        me.from_pydata(co.tolist(), [], quads.tolist())
        uv = me.uv_layers.new()
        lv = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", lv)
        uvs = np.stack([U.ravel() / self.w, 1 - V.ravel() / self.h], -1)[lv]
        uv.data.foreach_set("uv", uvs.astype(np.float32).ravel())
        me.materials.append(mat)
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        ob.location.z = 0.0004
        if bone == "":
            return ob  # fixed: rests on the desk
        bpy.context.view_layer.update()
        mw = ob.matrix_world.copy()
        ob.parent = self.arm
        ob.parent_type = "BONE"
        ob.parent_bone = bone or ("garm" if "garm" in self.arm.data.bones else "chest")
        bpy.context.view_layer.update()
        ob.matrix_world = mw
        return ob

    def set_gesture(self, weights, frame):
        """Show gesture paintings with the given {name: amount} ("rest" = none of them)."""
        for name, g in self.gestures.items():
            on = float(weights.get(name, 0.0))
            s = self.fade(self.layers[name])
            s.default_value = on
            s.keyframe_insert("default_value", frame=frame)
            for ob in g["objs"]:  # a layer that is off is not rendered (or deformed) at all
                ob.hide_render = on == 0.0
                ob.keyframe_insert("hide_render", frame=frame)
            if g["hide"] is not None:
                g["hide"].default_value = on
                g["hide"].keyframe_insert("default_value", frame=frame)

    def twin(self, name, mat, dz):
        tw = bpy.data.objects.new(name, self.ob.data)
        bpy.context.scene.collection.objects.link(tw)
        tw.material_slots[0].link = "OBJECT"
        tw.material_slots[0].material = mat
        tw.location.z = dz
        tw.parent = self.arm
        mod = tw.modifiers.new("Armature", "ARMATURE")
        mod.object = self.arm
        return tw

    def layer(self, name, image, dz, bone, step=2):
        """A flat piece of the reference (mouth interior, teeth) set back by dz behind the face,
        projected so it lines up pixel-for-pixel at rest, and carried by a bone."""
        img = bpy.data.images.load(str(self.dir / image))
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        a = px.reshape(h, w, 4)[::-1, :, 3]
        ys, xs = np.nonzero(a > 0.01)
        us = np.arange(xs.min() - 4, xs.max() + 5, step, dtype=float)
        vs = np.arange(ys.min() - 4, ys.max() + 5, step, dtype=float)
        U, V = np.meshgrid(us, vs)
        # always just behind the skin in front of it (the cheek recedes in 3/4 view), never flat
        zf = self.Z[np.clip(V, 0, self.h - 1).astype(int), np.clip(U, 0, self.w - 1).astype(int)]
        z = box_blur(zf, 2) + dz
        co = self.project(U.ravel(), V.ravel(), z.ravel())
        ny, nx = U.shape
        idx = np.arange(nx * ny).reshape(ny, nx)
        quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        me = bpy.data.meshes.new(name)
        me.from_pydata(co.tolist(), [], quads.tolist())
        uv = me.uv_layers.new()
        lv = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", lv)
        uvs = np.stack([U.ravel() / self.w, 1 - V.ravel() / self.h], -1)[lv]
        uv.data.foreach_set("uv", uvs.astype(np.float32).ravel())
        me.materials.append(unlit(name, img, blend="DITHERED"))
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.update()
        mw = ob.matrix_world.copy()
        ob.parent = self.arm
        ob.parent_type = "BONE"
        ob.parent_bone = bone
        bpy.context.view_layer.update()
        ob.matrix_world = mw  # keep it exactly where it was projected
        return ob

    def mouth_layers(self):
        self.interior = self.layer("mouth_interior", "mouth.png", 0.022, "head")
        self.teeth = self.layer("teeth_lower", "teeth.png", 0.012, "jaw")

    def shape_keys(self):
        R = self.rig
        U, V = self.U, self.V
        M = R["mouth"]
        x0, x1 = M["upper"][0][0], M["upper"][-1][0]
        cx = M["center"][0]
        # jaw: the lower lip, chin and jaw drop together; fades past the corners and down the neck
        lo = curve(M["lower"], np.clip(U, x0, x1))
        side = smoothstep(1 - (np.maximum(x0 - U, U - x1)) / 45.0)
        below = (V > lo - 1) * (1 - smoothstep((V - lo - 45) / 70))
        jaw = R.get("jaw_drop", 24) * side * below
        region = ellipse_w(U, V, M["center"], M["narrow_radius"])
        # the face moves as one: the jaw drags the cheeks and the nose tip, spreading lips lift
        # the cheeks, rounding draws them in
        nose = ellipse_w(U, V, R["nose"]["center"], R["nose"]["radius"])
        cheek = np.clip(sum(ellipse_w(U, V, c["center"], c["radius"]) for c in R["cheeks"]), 0, 1)
        cheek_dir = np.sign(U - cx)
        drop = R.get("jaw_drop", 24)
        shoulders = np.clip(sum(ellipse_w(U, V, s["center"], s["radius"]) for s in R["shoulders"]), 0, 1)
        # breathing stops above the desk (seated) or the hips (standing: the frame edge stays covered)
        torso_end = min(R.get('desk_y', 860), R.get('chest_y', 900) + 40)
        torso = smoothstep((V - R.get('torso_top', 640)) / 60) * (1 - smoothstep((V - torso_end + 80) / 60))
        fingers = ellipse_w(U, V, R["fingers"]["center"], R["fingers"]["radius"])
        fields = {
            "mouth_close": (np.zeros_like(U), close_gap(U, V, M["upper"], M["lower"], M["meet"],
                                                       M["reach_up"], M["reach_down"])),
            "jaw": (np.zeros_like(U), jaw + 0.10 * drop * nose + 0.30 * drop * cheek),
            "wide": ((U - cx) * 0.09 * region + 2.5 * cheek_dir * cheek, -1.0 * region * ((U - cx) / 50) ** 2 - 4.0 * cheek),
            "round": (-(U - cx) * 0.22 * region - 3.5 * cheek_dir * cheek, 1.5 * cheek),
            "brows": (np.zeros_like(U), -R["brow_lift"] * np.clip(
                sum(ellipse_w(U, V, b["center"], b["radius"]) for b in R["brows"]), 0, 1)),
            "shrug": (np.zeros_like(U), -10.0 * shoulders * smoothstep((V - R.get('shrug_top', 650)) / 40)),
            "breathe": ((U - R["neck"][0]) * 0.012 * torso, -3.0 * torso),
            "fingers": (np.zeros_like(U), -6.0 * fingers),
        }
        self.ob.shape_key_add(name="Basis")
        for name, (du, dv) in fields.items():
            kb = self.ob.shape_key_add(name=name, from_mix=False)
            kb.slider_min, kb.slider_max = -1.5, 1.5
            kb.value = 0.0  # Blender 5 starts new keys at 1.0
            kb.data.foreach_set("co", self.project(U + du, V + dv, self.Zv).astype(np.float32).ravel())

    def armature(self):
        R = self.rig
        U, V = self.U, self.V
        rgb = self.rgb[np.minimum(V, self.h - 1).astype(int), U.astype(int)] * 255
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        chair = (b > r + 8) & (r < 90) & (V > 500)
        nu, nv = R["neck"]
        top, bot = R["head_full_above"], R["head_zero_below"]
        bot_eff = bot - 0.6 * np.maximum(0, np.abs(U - nu) - 80)
        w_head = (1 - smoothstep((V - top) / np.maximum(bot_eff - top, 10))) * ~chair
        red = (r > 150) & (g < 120) & (b < 120)
        white = (np.minimum(np.minimum(r, g), b) > 185) & (V < 650)
        cloth = (red | white).astype(float)
        # spread the cloth mask so the whole shemagh moves as one piece
        cu, cv = R["crown"]
        # cloth sway: a smooth field that grows away from the face (no colour mask: a mask leaves loose
        # pixels at the cloth edge that break off when it swings)
        fc, fr_ = R.get("face_center", [750, 460]), R.get("face_radius", [120, 180])
        out_face = smoothstep((np.sqrt(((U - fc[0]) / fr_[0]) ** 2 + ((V - fc[1]) / fr_[1]) ** 2) - 1.0) / 0.6)
        w_sway = w_head * out_face * smoothstep((V - cv - 60) / 380)
        w_sway = self.blur_on_mesh(w_sway, 3)
        u0, v0, u1, v1 = R["hands"]
        w_hands = ((U > u0) & (U < u1) & (V > v0) & (V < v1)).astype(float)
        w_hands = self.blur_on_mesh(w_hands, 2) * ~chair
        # the raised gesture arm (only visible in gesture paintings) swings about the elbow; at rest
        # the bone does not move, so the clasped arm underneath is unaffected
        ga = R.get("gesture_arm")
        if ga:
            ex, ey = ga["elbow"]
            w_garm = ellipse_w(U, V, ga["center"], ga["radius"]) * (1 - smoothstep((V - ey + 10) / 40))
            w_garm = np.clip(self.blur_on_mesh(w_garm, 2) * 1.6, 0, 1) * (1 - w_head)
        else:
            w_garm = np.zeros_like(U)
        w_body = np.clip(1 - w_head - w_hands - w_garm, 0, 1) * ~chair
        # whatever rests on the desk (forearms, hands) is pinned: chest motion fades out toward it, so
        # hands never slide or stretch over the desk (that read as a heat-haze mirage)
        pin = smoothstep((V - R.get("desk_y", 860) + 40) / 80)
        if "hips_y" in R:  # standing: the legs stay planted, the upper body bends at the waist
            pin = np.maximum(pin, smoothstep((V - R["hips_y"] + 60) / 120))
        w_body = w_body * (1 - pin)
        w_hands = w_hands * (1 - pin)

        arm_data = bpy.data.armatures.new("rig")
        arm = bpy.data.objects.new("rig", arm_data)
        bpy.context.scene.collection.objects.link(arm)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        eb = arm_data.edit_bones
        neck = self.point(nu, nv, 0.06)              # pivot sits inside the neck, behind the surface
        crown = self.point(cu, cv, 0.05)
        chest = self.point(nu, R.get("chest_y", 900), 0.08)
        hands = self.point((u0 + u1) / 2, (v0 + v1) / 2, 0.03)

        def bone(name, h, t, parent=None):
            bn = eb.new(name)
            bn.head, bn.tail = h, t
            bn.align_roll(Vector((0, 0, 1)))
            if parent:
                bn.parent = eb[parent]
            return bn

        bone("chest", chest, neck)
        bone("head", neck, neck + Vector((0, 0.3, 0)), "chest")
        bone("shemagh", crown, crown - Vector((0, 0.3, 0)), "head")
        flap = self.point(cu, R.get("shemagh_tail_y", 560), 0.06)
        bone("shemagh_tail", flap, flap - Vector((0, 0.25, 0)), "shemagh")
        bone("hands", hands, hands + Vector((0, 0.1, 0)), "chest")
        if ga:
            elbow = self.point(*ga["elbow"], 0.03)
            bone("garm", elbow, elbow + Vector((0, 0.1, 0)), "chest")
        mouth = self.point(*R["mouth"]["center"], 0.02)
        bone("jaw", mouth, mouth + Vector((0, 0.05, 0)), "head")  # carries the lower teeth; +Y is up
        bpy.ops.object.mode_set(mode="OBJECT")
        for pb in arm.pose.bones:
            pb.rotation_mode = "XYZ"
        tail_y = R.get("shemagh_tail_y", 560)
        w_tail = w_sway * smoothstep((V - tail_y) / 160)
        for name, w in (("head", w_head * (1 - w_sway)), ("shemagh", w_sway - w_tail), ("shemagh_tail", w_tail),
                        ("chest", w_body), ("hands", w_hands), ("garm", w_garm)):
            vg = self.ob.vertex_groups.new(name=name)
            for i in np.nonzero(w > 0.002)[0]:
                vg.add([int(i)], float(w[i]), "REPLACE")
        mod = self.ob.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
        self.ob.parent = arm
        self.arm = arm

    def blur_on_mesh(self, w, passes):
        """Smooth a per-vertex weight through the mesh edges."""
        me = self.ob.data
        e = np.empty(len(me.edges) * 2, np.int32)
        me.edges.foreach_get("vertices", e)
        e = e.reshape(-1, 2)
        for _ in range(passes * 4):
            acc = w.copy()
            cnt = np.ones_like(w)
            np.add.at(acc, e[:, 0], w[e[:, 1]])
            np.add.at(acc, e[:, 1], w[e[:, 0]])
            np.add.at(cnt, e[:, 0], 1)
            np.add.at(cnt, e[:, 1], 1)
            w = acc / cnt
        return w


# ------------------------------------------------------------------ performance

def plates(ref_dir, ep):
    """One background plane per screen image, cut on the cue times."""
    cue_file = ep / "screen.json"
    cues = json.loads(cue_file.read_text(encoding="utf-8")) if cue_file.exists() else [{"t": 0, "image": "screen_default"}]
    planes = {}
    for c in cues:
        stem = Path(c["image"]).stem
        if stem not in planes:
            planes[stem] = plate_plane(f"plate_{stem}", Path(ref_dir) / "plates" / f"{stem}.png",
                                       z=-9.0 - 0.01 * len(planes))
    for ob in planes.values():
        ob.hide_render = True
        ob.keyframe_insert("hide_render", frame=1)
    for c in cues:
        f = max(1, int(round(c["t"] * FPS)) + 1)
        for stem, ob in planes.items():
            ob.hide_render = stem != Path(c["image"]).stem
            ob.keyframe_insert("hide_render", frame=f)


# mouth shape -> painted variants, best first ("base" = the reference's own, slightly open mouth);
# the first one that exists is used, so the rig works with any subset of paintings
PAINTED = {"A": ["mb", "closed"], "B": ["teeth", "ee"], "I": ["ee"], "C": ["base"], "D": ["a"],
           "W": ["a_wide", "a"], "E": ["o", "oo"], "F": ["oo"], "G": ["f"], "H": ["l", "a"],
           "X": ["closed"]}
# how open each painting is: a cut that jumps 2+ levels gets one in-between frame
OPENNESS = {"closed": 0, "mb": 0, "slight": 1, "teeth": 1, "f": 1, "base": 1, "oo": 1,
            "ee": 2, "l": 2, "o": 2, "a": 3, "a_wide": 4}
ROUND = {"o", "oo"}


def inbetween(prev, nxt, names):
    """The painting that sits between two mouths, or None if they are close enough."""
    op, on = OPENNESS.get(prev, 1), OPENNESS.get(nxt, 1)
    if abs(op - on) < 2:
        return None
    mid = (op + on) / 2
    pool = ["o"] if (prev in ROUND or nxt in ROUND) and "o" not in (prev, nxt) else []
    pool += ["slight", "base", "ee", "a"]
    pool = [p for p in pool if p in names and p not in (prev, nxt)]
    return min(pool, key=lambda p: abs(OPENNESS[p] - mid)) if pool else None


def painted_weights(cues, n, layers, hold=2, sigma=0.0):
    """Per-frame alpha for each stacked painted layer.
    Like hand-drawn lip-sync, mouths cut cleanly from drawing to drawing (cross-fading two painted
    mouths double-exposes them) and each drawing holds at least `hold` frames. sigma > 0 adds a
    zero-phase blend. Alphas are 'over'-compositing alphas a_i = w_i / sum(w_0..w_i), so any mix
    of the stack is a proper weighted average."""
    names = ["base"] + list(layers)
    seq = []
    j = 0
    for f in range(n):
        t = f / FPS
        while j + 1 < len(cues) and cues[j]["end"] <= t:
            j += 1
        c = cues[j]
        prefs = PAINTED.get(c["value"] if c["start"] <= t < c["end"] else "X", ["base"])
        target = next((p for p in prefs if p in names), "base")
        seq.append(names.index(target))
    # enforce the minimum hold: a too-short run takes over the next frames (sound stays on time)
    f = 0
    while f < n:
        e = f
        while e < n and seq[e] == seq[f]:
            e += 1
        if e - f < hold:
            for i in range(e, min(f + hold, n)):
                seq[i] = seq[f]
            e = min(f + hold, n)
        f = e
    # in-betweens: on a big jump (closed -> wide open), the first frame of the new drawing shows
    # the shape halfway there, if the new drawing is held long enough to spare a frame
    runs, f = [], 0
    while f < n:
        e = f
        while e < n and seq[e] == seq[f]:
            e += 1
        runs.append((f, e))
        f = e
    for (_, _), (s1, e1) in zip(runs, runs[1:]):
        if e1 - s1 >= 3:
            mid = inbetween(names[seq[s1 - 1]], names[seq[s1]], names)
            if mid:
                seq[s1] = names.index(mid)
    W_ = np.zeros((n, len(names)))
    W_[np.arange(n), seq] = 1.0
    if sigma > 0:
        k = np.arange(-4, 5)
        g = np.exp(-0.5 * (k / sigma) ** 2)
        g /= g.sum()
        W_ = np.stack([np.convolve(np.pad(W_[:, i], 4, mode="edge"), g, mode="valid")
                       for i in range(len(names))], -1)
        W_ /= W_.sum(1, keepdims=True)
    out = []
    for f in range(n):
        acc = W_[f, 0]
        row = {}
        for i, name in enumerate(names[1:], 1):
            acc += W_[f, i]
            row[name] = float(W_[f, i] / acc) if acc > 1e-6 else 0.0
        out.append(row)
    return out


def drift(n, seconds, rng):
    """Smooth random motion that never repeats: white noise smoothed to about `seconds` per swing,
    normalised to unit spread. Replaces sine waves, whose fixed periods read as a loop."""
    x = rng.normal(size=(n + 4 * int(seconds * FPS), 1))
    y = mm.damp(mm.damp(x, FPS, seconds * 0.5), FPS, seconds * 0.5)[2 * int(seconds * FPS):][:n, 0]
    return (y - y.mean()) / max(y.std(), 1e-9)


def breathing(n, phrases_, rng):
    """-1..1 chest motion: an uneven 3.5-5 s rhythm, and a quick inhale just before each phrase."""
    out, phase = np.zeros(n), rng.uniform(0, 2 * np.pi)
    period = rng.uniform(3.5, 5.0)
    for f in range(n):
        if rng.random() < 1 / (4 * FPS):
            period = rng.uniform(3.5, 5.0)
        phase += 2 * np.pi / (period * FPS)
        out[f] = math.sin(phase)
    for a, _ in phrases_:  # speakers fill their lungs right before they talk
        out += 0.8 * bump(n, a - int(0.45 * FPS), int(0.3 * FPS), int(0.1 * FPS), int(0.6 * FPS))
    return np.clip(out, -1, 1.4)


def taps(n, g_phrase, rng, rate=1.1):
    """Finger taps at irregular moments while he talks (each a short lift and drop)."""
    out = np.zeros(n)
    f = 0
    while f < n:
        f += int(rng.exponential(FPS / rate)) + 4
        if f < n and g_phrase[f] > 0.5:
            out = np.maximum(out, bump(n, f, 2, 1, 4) * rng.uniform(0.5, 1.0))
    return out


def spring(impulses, freq=1.8, zeta=0.45):
    """Follow-through: each impulse kicks an underdamped spring, which overshoots and settles."""
    w = 2 * np.pi * freq
    pos, vel, out = 0.0, 0.0, np.zeros(len(impulses))
    for i, k in enumerate(impulses):
        vel += k
        for _ in range(4):
            acc = -w * w * pos - 2 * zeta * w * vel
            vel += acc / (4 * FPS)
            pos += vel / (4 * FPS)
        out[i] = pos
    return out


def body_series(trace, n, fn):
    """Like trace_series, for body values (frames without a tracked body are filled)."""
    return trace_series(trace, n, lambda fr: fn(fr["body"]) if "body" in fr else np.nan)


def highpass(x, seconds):
    k = max(3, int(seconds * FPS) | 1)
    pad = np.pad(x, k // 2, mode="edge")
    return x - np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def gesture_targets(trace, phrases_, n, kinds, look_down=None):
    """Which gesture he should be making each frame ("rest", or a gesture kind such as "open").

    From a traced speaker: how high and visible each wrist is, how open each hand is and whether
    only the index finger is out. Raised + pointing -> "point"; both hands up -> "both"; one hand up
    and open -> "open"; partly up -> "half"; looking down at the desk with hands low -> "tablet".
    Without a trace: a calm gesture on some longer phrases, cycling through the kinds, and now and
    then a touch of the tablet at the start of a sentence."""
    T = np.array(["rest"] * n, dtype=object)
    if trace is not None and any("body" in fr for fr in trace["frames"]):
        def side(sd):
            h = body_series(trace, n, lambda b: b[f"wrist{sd}"][1])
            vis = body_series(trace, n, lambda b: b[f"wrist{sd}"][2])
            raised = mm.damp((np.clip((h + 1.0) / 0.45, 0, 1) * (vis > 0.5))[:, None], FPS, 0.12)[:, 0]
            fing = [body_series(trace, n, lambda b, k=k: b.get(f"fingers{sd}", [1.5] * 4)[k]) for k in range(4)]
            point = np.clip((fing[0] - np.maximum(fing[1], fing[2]) - 0.25) / 0.4, 0, 1)
            return raised, mm.damp(point[:, None], FPS, 0.1)[:, 0]
        rL, pL = side("L")
        rR, pR = side("R")
        level = np.maximum(rL, rR)
        point = np.where(rL >= rR, pL, pR)
        both = np.minimum(rL, rR)
        sided = any(k.endswith(("_L", "_R")) for k in kinds)  # standing: each hand has its own poses
        for f in range(n):
            sd = ("_L" if rL[f] >= rR[f] else "_R") if sided else ""  # the speaker's own left/right
            if level[f] > 0.6:
                T[f] = "both" if both[f] > 0.5 and "both" in kinds else (
                    "point" + sd if point[f] > 0.5 and "point" + sd in kinds else "open" + sd)
            elif level[f] > 0.35:
                T[f] = "half" + sd
        if look_down is not None and "tablet" in kinds:
            down = (look_down > 0.5) & (level < 0.3)
            f = 0
            while f < n:  # a glance down of at least 0.4 s: he reaches for the tablet
                if down[f]:
                    e = f
                    while e < n and down[e]:
                        e += 1
                    if e - f >= int(0.4 * FPS):
                        T[f:e] = "tablet"
                    f = e
                f += 1
    else:
        cycle = [k for k in ("open", "point", "both", "open_R", "open_L", "both", "point_R", "present_L")
                 if k in kinds] or ["half"]
        last, used = -10 * FPS, 0
        for i, (a, b) in enumerate(phrases_):
            if "tablet" in kinds and i % 4 == 3 and b - a >= 1.2 * FPS:
                T[a:a + int(1.0 * FPS)] = "tablet"   # glancing at the next item
            elif b - a >= 2.5 * FPS and a - last >= 6 * FPS:  # at most one gesture per ~6 s
                T[a + int(0.3 * FPS):min(b, a + int(2.4 * FPS))] = cycle[used % len(cycle)]
                last, used = a, used + 1
    return T


BLEND_ARM_CHANGES = False  # a half-transparent arm looks like a ghost; the arm bone carries motion


# gesture and rest lengths are drawn from ranges (seconds), so the rhythm never becomes regular
CALM = {"max_len": (1.8, 3.6), "phrase": (1.6, 3.0), "merge_gap": 0.7, "min_len": 0.9, "cooldown": (0.9, 2.2),
        "beat_window": 0.4, "lead": 0.12, "clasp_rest": 2.5, "clasp_chance": 0.5}
RARE = {"point": 0.5, "both": 0.7}  # emphatic poses come up less often


def family(kind, kinds):
    """Poses that can stand in for `kind`: the other raised poses of the same hand."""
    sd = kind[-2:] if kind.endswith(("_L", "_R")) else ""
    fam = [k for k in sorted(kinds) if not k.startswith(("half", "clasp", "idle", "tablet", "tween"))
           and (k.endswith(sd) if sd else not k.endswith(("_L", "_R")))]
    return fam or [kind]


def calm_gestures(targets, beats_, n, kinds=(), seed=5):
    """A presenter's gestures are few and settled: merge flickers, drop short ones, one kind per
    gesture, rest between gestures, and each gesture's stroke lands on a stressed syllable.
    Varied: gesture and rest lengths are random within ranges, the same pose never comes twice in
    a row (another pose of the same hand stands in), and some longer rests are spent with the
    hands clasped. targets: per-frame kind ("rest", "half", "open", ...). Returns the same, calmed."""
    rng = np.random.default_rng(seed)
    U = lambda r: rng.uniform(*r) * FPS  # noqa: E731
    T = list(targets)
    segs, f = [], 0
    while f < n:  # non-rest segments
        if T[f] != "rest":
            e = f
            while e < n and T[e] != "rest":
                e += 1
            segs.append([f, e])
            f = e
        f += 1
    merged = []
    for a, b in segs:  # bridge short rests inside one gesture
        if merged and a - merged[-1][1] < CALM["merge_gap"] * FPS:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    split = []  # no gesture is held long: long ones become gesture, rest, gesture...
    for a, b in merged:
        while b - a > CALM["max_len"][1] * FPS:
            e = a + int(U(CALM["max_len"]))
            split.append([a, e])
            a = e + int(U(CALM["cooldown"]))
        if b > a:
            split.append([a, b])
    merged = split
    out = ["rest"] * n
    last_end, prev = -10 * FPS, None
    beats_ = np.array(sorted(beats_)) if len(beats_) else np.array([], int)
    for a, b in merged:
        a = max(a, int(last_end + U(CALM["cooldown"])))  # rest first, then gesture
        if b - a < CALM["min_len"] * FPS:
            continue
        if len(beats_):  # move the start so the hand arrives on the nearest stressed syllable
            nb = beats_[np.argmin(np.abs(beats_ - a))]
            if abs(nb - a) <= CALM["beat_window"] * FPS:
                shift = int(nb - CALM["lead"] * FPS) - a
                a, b = max(0, a + shift), min(n, b + shift)
        # long stretches become phrase-length gestures (one kind each), like a presenter's beats
        cuts = [a]
        while cuts[-1] + U(CALM["phrase"]) < b - 0.8 * FPS:
            cuts.append(int(cuts[-1] + U(CALM["phrase"])))
        for i, c0 in enumerate(cuts):
            c1 = cuts[i + 1] if i + 1 < len(cuts) else b
            alias = {} if "both" in kinds else {"both": "open"}
            ks = [alias.get(k, k) for k in targets[c0:c1] if k != "rest" and not k.startswith("half")]
            halves = [k for k in targets[c0:c1] if k.startswith("half")]
            kind = max(sorted(set(ks)), key=ks.count) if ks else (max(sorted(set(halves)), key=halves.count) if halves else "half")
            if ks and kind != "tablet" and (kind == prev or rng.random() < 0.3):  # vary: another pose, same hand
                alts = [k for k in family(kind, kinds) if k != prev] or [kind]
                w = np.array([RARE.get(k.split("_")[0], 1.0) for k in alts])
                kind = alts[rng.choice(len(alts), p=w / w.sum())]
            out[c0:c1] = [kind] * (c1 - c0)
            prev = kind
        last_end = b
    if "clasp" in kinds:  # some longer pauses are spent with the hands folded in front
        f = 0
        while f < n:
            if out[f] == "rest":
                e = f
                while e < n and out[e] == "rest":
                    e += 1
                if e - f >= CALM["clasp_rest"] * FPS and rng.random() < CALM["clasp_chance"]:
                    m = int(0.4 * FPS)
                    out[f + m:e - m] = ["clasp"] * (e - f - 2 * m)
                f = e
            f += 1
    return out


def gesture_mix(seq, f, gestures):
    """Painting amounts for frame f. On the first frame of every change the old and new arm pose
    are mixed 50/50 (the upper layer at half, the lower at full), which reads as motion blur."""
    cur = seq[f]
    w = {} if cur == "rest" else {cur: 1.0}
    if BLEND_ARM_CHANGES and f > 0 and seq[f - 1] != cur:
        prev = seq[f - 1]
        pair = [p for p in (prev, cur) if p != "rest"]
        if len(pair) == 1:
            w = {pair[0]: 0.5}
        else:
            hi, lo = sorted(pair, key=lambda p: gestures[p]["z"], reverse=True)
            w = {hi: 0.5, lo: 1.0}
    return w


def insert_tweens(seq, variants, gestures):
    """At each switch between two poses that have in-between paintings (arm_tweens.py), show them
    on the first two frames of the new pose: A, A>B 1/3, A>B 2/3, B ... (reversed when going back)."""
    tw = {}
    for name, v in variants.items():
        if "tween" in v and name in gestures:
            a, b, t = v["tween"]
            tw.setdefault((a, b), {})[t] = name
    out = list(seq)
    for f in range(1, len(seq) - 1):
        x, y = seq[f - 1], seq[f]
        if x == y:
            continue
        if (x, y) in tw:
            names = [tw[(x, y)][t] for t in sorted(tw[(x, y)])]
        elif (y, x) in tw:
            names = [tw[(y, x)][t] for t in sorted(tw[(y, x)], reverse=True)]
        else:
            continue
        for k, nm in enumerate(names):
            if f + k < len(seq) and seq[f + k] == y:
                out[f + k] = nm
    return out


def gesture_states(targets, gestures, hold=18, hold_between=4, min_target=8):
    """The gesture painting shown each frame. Moves toward the target one painting at a time:
    clasped hands -> in-between ("via") -> gesture, and back the same way; two gestures that share
    an in-between pass through it. In-betweens hold `hold_between` frames, gestures `hold`."""
    kind_of = {g["kind"]: name for name, g in gestures.items()}
    rest = kind_of.get("idle", "rest")  # hands flat on the desk, if that painting exists
    via = {name: g.get("via") for name, g in gestures.items()}
    vias = {v for v in via.values() if v}
    # ignore target blips: a wish shorter than `min_target` frames keeps the previous one
    targets = list(targets)
    f = 0
    while f < len(targets):
        e = f
        while e < len(targets) and targets[e] == targets[f]:
            e += 1
        if e - f < min_target and f > 0:
            targets[f:e] = [targets[f - 1]] * (e - f)
        f = e

    def step(cur, tgt):
        if cur == rest:
            return via[tgt] if via.get(tgt) else tgt
        if cur in vias:  # an in-between: on to the target or back to rest
            return tgt if (tgt == rest or via.get(tgt) == cur) else rest
        v = via.get(cur)  # a full gesture: back out through its in-between
        if tgt == rest or via.get(tgt) != v or tgt == v:
            return v or rest
        return v

    cur, since, out = rest, hold, []
    for k in targets:
        tgt = kind_of.get(k, rest)
        need = hold_between if cur in vias else hold
        if tgt != cur and since >= need:
            cur, since = step(cur, tgt), 0
        out.append(cur)
        since += 1
    return out


# planner gesture kinds <-> Veo take families (tools/capture/veo_pool.py)
# (the clasp take opens with both palms and only folds the hands at its very end: it plays as "both")
VEO_KINDS = {"R": ["open_R"], "point_R": ["point_R"], "L": ["open_L"], "point_L": ["point_L"], "both": ["both"]}
VEO_FAMILY = {"open_R": ["R"], "half_R": ["R"], "point_R": ["point_R", "R"], "open_L": ["L"], "half_L": ["L"],
              "present_L": ["L"], "point_L": ["point_L", "L"], "both": ["both", "clasp"]}
VEO_EDGE = 1  # no fade in/out: Veo's resting hand sits a little off ours, a cross-fade shows both


def veo_track(plan, takes, n, src_fps, seed=11):
    """Per output frame: which Veo pool frame to show (0 = none) and how strongly, for the planned
    gestures. Each planned gesture plays one take of its family (never the same take twice in a row):
    its rise and fall at their natural speed, the hold in between stretched or squeezed to fit, a
    take that never returns to rest comes down by playing its rise backwards. A gesture too short
    for a natural rise and fall borrows time from the rest after it."""
    rng = np.random.default_rng(seed)
    idx, w = np.zeros(n), np.zeros(n)
    gest = ["rest"] * n
    used, last = [], {}
    events, f = [], 0
    while f < n:
        if plan[f] != "rest":
            e = f
            while e < n and plan[e] == plan[f]:
                e += 1
            events.append([f, e, plan[f]])
            f = e
        else:
            f += 1
    rate = src_fps / FPS  # source frames per output frame at natural speed
    for i, (a, b, kind) in enumerate(events):
        cands = [t for t in takes if t["family"] in VEO_FAMILY.get(kind, [])]
        if not cands:
            continue
        pick = [t for t in cands if t is not last.get("take")] or cands  # never the previous take again
        # prefer takes whose natural length fits the planned gesture
        nat = np.array([(t["end"] - t["start"]) / rate for t in pick])
        fit = np.exp(-2 * np.abs(np.log(nat / max(b - a, 1))))
        t = pick[rng.choice(len(pick), p=fit / fit.sum())]
        last["take"] = t
        s0, last_src = t["start"], t["end"] - 1
        rise_o = int(round(t["rise"] / rate))
        fall_o = int(round((t["fall"] if t["returns"] else t["rise"]) / rate))
        nxt = events[i + 1][0] if i + 1 < len(events) else n
        # a gesture shorter than its take borrows from the rest after it (up to half as long again)
        want = min((t["end"] - t["start"]) / rate, 1.5 * (b - a))
        b = max(b, min(nxt - 2, a + int(max(want, rise_o + fall_o + 8))))
        D = b - a
        if rise_o + fall_o > 0.85 * D:  # still too short: move faster
            k = 0.85 * D / (rise_o + fall_o)
            rise_o, fall_o = int(rise_o * k), int(fall_o * k)
        H = D - rise_o - fall_o
        m0 = s0 + t["rise"]
        # a hold that would have to play faster than 1.6x is cut short instead: part of it, then
        # back the way the hand came (no jump, no fast-forward)
        fits = (last_src - t["fall"] - m0) <= 1.6 * H * rate
        if t["returns"] and fits:
            m1 = max(m0, last_src - t["fall"])
            src = np.concatenate([np.linspace(s0, m0, rise_o, endpoint=False),
                                  np.linspace(m0, m1, H, endpoint=False),
                                  np.linspace(m1, last_src, fall_o)])
        else:  # hold by drifting forward and back, then come down along the rise, reversed
            if t["returns"]:
                fall_o = min(rise_o, D - rise_o)
                H = D - rise_o - fall_o
            M = min(last_src - m0, 0.5 * H * rate)
            k = np.arange(H) / max(H - 1, 1)
            src = np.concatenate([np.linspace(s0, m0, rise_o, endpoint=False),
                                  m0 + M * (1 - np.abs(2 * k - 1)),
                                  np.linspace(m0, s0, fall_o)])
        src = np.clip(np.round(src), t["start"], last_src)
        idx[a:a + len(src)] = src
        ramp = np.minimum(np.arange(len(src)) + 1, len(src) - np.arange(len(src))) / VEO_EDGE
        w[a:a + len(src)] = np.clip(ramp, 0, 1)
        gest[a:a + len(src)] = [kind] * len(src)
        used.append((a, a + len(src), t))
    return idx, w, gest, used


EXPR_FADE = 0.1  # seconds to ease an expression in or out (a long cross-fade shows doubled brows)


def expression_track(ep, rig, layers, n):
    """How strongly each expression painting shows, per frame, from the episode's
    expressions.json: [{"t": seconds, "dur": seconds, "name": "frown"}, ...] (names as the rig's
    variants' "expression"). No file: none show."""
    w = {k: np.zeros(n) for k in layers}
    f = ep / "expressions.json"
    if not f.exists():
        return w
    by_name = {rig["variants"][k]["expression"]: k for k in layers}
    fade = EXPR_FADE * FPS
    fr = np.arange(n)
    for c in json.loads(f.read_text(encoding="utf-8")):
        k = by_name.get(c["name"])
        if k is None:
            print("no expression painting for", c["name"])
            continue
        a, b = c["t"] * FPS, (c["t"] + c["dur"]) * FPS
        ramp = np.minimum(smoothstep((fr - a) / fade), 1 - smoothstep((fr - (b - fade)) / fade))
        w[k] = np.maximum(w[k], np.clip(ramp, 0, 1))
    return w


def trace_series(trace, n, fn):
    """One value per output frame from a face trace (gaps where no face was found are filled)."""
    v = np.array([fn(fr) if fr["ok"] else np.nan for fr in trace["frames"]], float)
    good = ~np.isnan(v)
    if not good.any():
        return np.zeros(n)
    v = np.interp(np.arange(len(v)), np.nonzero(good)[0], v[good])
    return np.interp(np.arange(n) / FPS, np.arange(len(v)) / trace["fps"], v)


def perform(rel, ep, n, trace=None):
    """Animate everything. trace: a face trace (tools/capture/trace_face.py) to copy mouth,
    blinks and head from a real speaker; without it, all comes from the speech. A body-only trace
    ("face": false, e.g. from EMAGE) drives head and body but leaves the face to the speech."""
    face_tr = trace if (trace is not None and trace.get("face", True)) else None
    P = rel.arm.pose.bones
    keys = rel.ob.data.shape_keys.key_blocks
    cues = json.loads((ep / "mouth.json").read_text(encoding="utf-8"))["mouthCues"]
    vis = viseme_tracks(cues, n, VISEMES)
    ef, es = envelope(ep / "voice.wav", n)
    ph = phrases(cues, n)
    bt = beats(es, n)
    rnd = random.Random(9)
    phase = [rnd.uniform(0, 6.28) for _ in range(8)]

    g_phrase = np.zeros(n)
    for a, b in ph:
        g_phrase = np.maximum(g_phrase, bump(n, a - 6, 9, max(0, b - a - 12), 14))
    nod = np.zeros(n)
    for f in bt:
        nod += bump(n, f - 3, 3, 0, 8)

    blink = np.zeros(n, bool)
    t = rnd.uniform(0.8, 1.6)
    while t * FPS < n - 5:
        f0 = int(t * FPS)
        blink[f0:f0 + 4] = True
        t += rnd.uniform(2.2, 4.8)
    gest = ["rest"] * n
    if rel.painted:
        # painted blink: a clean cut to the closed-eye painting (fading two eye paintings
        # double-exposes them), 3 frames long
        lid_w = np.zeros(n)
        for f in np.nonzero(blink & ~np.roll(blink, 1))[0]:
            lid_w[f:f + 3] = 1.0
        if face_tr is not None:  # blinks copied from the traced face: real closures only
            bl = trace_series(trace, n, lambda fr: (fr["bs"]["eyeBlinkLeft"] + fr["bs"]["eyeBlinkRight"]) / 2)
            closed = bl - np.median(bl) > 0.28
            lid_w = np.zeros(n)
            f = 0
            while f < n:  # each closure becomes a 2-5 frame blink
                if closed[f]:
                    e = f
                    while e < n and closed[e]:
                        e += 1
                    lid_w[f:f + int(np.clip(e - f, 2, 5))] = 1.0
                    f = e
                f += 1
        # mouth: a point in the mouth space per frame (from sounds or from the traced face),
        # shown as the nearest painting, plus a small jaw warp for the gap to that painting
        # "base" (the reference's own mouth, all layers off) is only a choice if mouth_space lists it;
        # leaving it out means a painted mouth always covers the reference's (e.g. gap-toothed) one
        space = {k: v for k, v in rel.rig["mouth_space"].items()
                 if not k.startswith("_") and (k == "base" or k in rel.layers)}
        mplan = mm.plan(cues=cues, trace=face_tr, n=n, fps=FPS, space=space)
        mouths = [k for k in rel.layers if k in space]
        exprs = [k for k in rel.layers if rel.rig["variants"][k].get("expression")]
        expr_w = expression_track(ep, rel.rig, exprs, n)
        if rel.gestures or rel.veo:
            kinds = ({g["kind"] for g in rel.gestures.values()} if rel.gestures else
                     {k for t in rel.veo["takes"] for k in VEO_KINDS.get(t["family"], [])})
            look_down = None
            if face_tr is not None:  # sustained downward gaze plus head pitch = looking at the desk
                bsd = lambda k: trace_series(trace, n, lambda fr: fr["bs"].get(k, 0.0))  # noqa: E731
                gd = (bsd("eyeLookDownLeft") + bsd("eyeLookDownRight")) / 2
                look_down = np.clip((gd - np.median(gd) - 0.15) / 0.2, 0, 1)
            plan = calm_gestures(gesture_targets(trace, ph, n, kinds, look_down), bt, n, kinds)
            if rel.veo:
                vidx, vw, gest, takes_used = veo_track(plan, rel.veo["takes"], n, rel.veo["fps"])
            else:
                gest = gesture_states(plan, rel.gestures)
                gest = insert_tweens(gest, rel.rig.get("variants", {}), rel.gestures)
            runs = [(gest[0], 0)]
            for f in range(1, n):
                if gest[f] != gest[f - 1]:
                    runs.append((gest[f], f))
            (ep / "out" / f"relief_{rel.dir.name}").mkdir(parents=True, exist_ok=True)
            (ep / "out" / f"relief_{rel.dir.name}" / "gestures.txt").write_text(
                "\n".join(f"{f / FPS:6.2f}  {k}" for k, f in runs) + "\n"  # the gesture timeline, for checking
                + ("".join(f"take {a / FPS:6.2f}-{b / FPS:6.2f}  {t['clip']} {t['family']} {t['start']}-{t['end']}\n"
                           for a, b, t in takes_used) if rel.veo else ""), encoding="utf-8")
        for f in range(n):
            if rel.gestures:
                rel.set_gesture(gesture_mix(gest, f, rel.gestures), f + 1)
            if rel.veo:
                rel.set_veo(vidx[f], vw[f], f + 1)
            if "eyes_closed" in rel.layers:
                s = rel.fade(rel.layers["eyes_closed"])
                s.default_value = float(lid_w[f])
                s.keyframe_insert("default_value", frame=f + 1)
                rel.layers["eyes_closed"].hide_render = bool(lid_w[f] == 0)
                rel.layers["eyes_closed"].keyframe_insert("hide_render", frame=f + 1)
            for name in exprs:
                s = rel.fade(rel.layers[name])
                s.default_value = float(expr_w[name][f])
                s.keyframe_insert("default_value", frame=f + 1)
                rel.layers[name].hide_render = bool(expr_w[name][f] == 0)
                rel.layers[name].keyframe_insert("hide_render", frame=f + 1)
            for name in mouths:
                s = rel.fade(rel.layers[name])
                s.default_value = 1.0 if mplan["paint"][f] == name else 0.0
                s.keyframe_insert("default_value", frame=f + 1)
                rel.layers[name].hide_render = bool(mplan["paint"][f] != name)
                rel.layers[name].keyframe_insert("hide_render", frame=f + 1)
            keys["jaw"].value = float(np.clip(mplan["residual"][f] * 1.2, -0.15, 0.35))
            keys["jaw"].keyframe_insert("value", frame=f + 1)
    else:
        for f in range(n):
            if f == 0 or blink[f] != blink[f - 1]:
                rel.lids.hide_render = not blink[f]
                rel.lids.keyframe_insert("hide_render", frame=f + 1)

    # head angles first, then the shemagh follows them through a damped spring
    # kept small: one image has nothing behind the head to reveal (about +-3 degrees of turn)
    nrng = np.random.default_rng(17)
    yaw = 0.022 * drift(n, 3.5, nrng) + 0.010 * drift(n, 1.0, nrng)
    tilt = 0.016 * drift(n, 4.5, nrng) + 0.006 * drift(n, 1.3, nrng) + 0.015 * g_phrase
    pitch = 0.045 * nod - 0.015 * es + 0.010 * drift(n, 2.5, nrng)
    if trace is not None:
        # head from the traced face: its motion around its own average pose, scaled into the
        # few degrees a single-image relief can turn
        def head(key, scale, lim):
            v = np.array([fr[key] if fr["ok"] else np.nan for fr in trace["frames"]], float)
            good = ~np.isnan(v)
            v = np.interp(np.arange(len(v)), np.nonzero(good)[0], v[good])
            v = np.interp(np.arange(n) / FPS, np.arange(len(v)) / trace["fps"], v)
            v = v - np.convolve(np.pad(v, 45, mode="edge"), np.ones(91) / 91, mode="valid")  # drop slow drift
            v = lim * np.tanh(np.radians(v) * scale / lim)  # soft limit: eases into it, never slams
            return mm.damp(v[:, None], FPS, 0.22)[:, 0]          # settled, never faster than a calm head
        yaw, pitch, tilt = head("yaw", 0.12, 0.03), head("pitch", 0.5, 0.05), head("roll", 0.5, 0.045)
        if rel.rig.get("body_sway", 1.0) < 1.0:
            # standing: a slow change of pitch is the speaker's posture (leaning toward the camera),
            # which on a standing figure reads as the whole man rocking; keep only the quick nods
            pitch = mm.damp(highpass(pitch, 1.2)[:, None], FPS, 0.08)[:, 0]
    # body: uneven breathing tied to the phrases, irregular finger taps, idle hand drift, a body that
    # turns and rolls a little on its own, and a follow-through spring on every change of arm pose
    breath_s = breathing(n, ph, nrng)
    tap_s = taps(n, g_phrase, nrng)
    px_m = rel.z_at(*rel.rig["neck"]) / F_PX / rel.s  # metres per reference pixel at body depth
    hand_drift = np.zeros((n, 2))  # resting hands stay put: any drift reads as shimmer
    chest_turn = 0.004 * drift(n, 6.0, nrng)
    chest_roll = 0.004 * drift(n, 5.0, nrng)
    sway = rel.rig.get("body_sway", 1.0)  # standing: less lean, the legs do not rock
    lean = np.zeros(n)
    if trace is not None and any("body" in fr for fr in trace["frames"]):
        # leaning toward the camera makes the shoulders look wider; follow that, gently
        sw_ = body_series(trace, n, lambda b: b["shoulder_w"])
        lean = np.clip(mm.damp(highpass(sw_ / np.median(sw_), 4.0)[:, None], FPS, 0.25)[:, 0] * 0.4, -0.025, 0.025)
        chest_turn = chest_turn + 0.06 * yaw  # the body turns with the head, much less far
    RESTS = {"rest"} | {k for k, g in rel.gestures.items() if g["kind"] in ("idle", "clasp")} if rel.painted else {"rest"}
    if rel.veo:
        RESTS = RESTS | {"clasp"}
    imp = np.zeros(n)
    for f in range(1, n):
        if gest[f] != gest[f - 1]:
            imp[f] = 0.08 if gest[f] not in RESTS else -0.05  # lean into a gesture, settle out of it
    follow = spring(imp)  # ~0.015 rad peak per gesture change
    # the raised arm is never frozen: it swings about the elbow with the speaker's hand (or with the
    # speech beats), arrives with an overshoot, and does nothing while the hands are clasped
    active = mm.damp(np.array([g not in RESTS | {"hand_tablet"} for g in gest], float)[:, None], FPS, 0.12)[:, 0]
    arm_imp = np.zeros(n)
    for f in range(1, n):
        if gest[f] != gest[f - 1] and gest[f] not in RESTS | {"hand_tablet"}:
            arm_imp[f] = -0.35  # comes up from below and settles
    swing = 0.03 * drift(n, 0.8, nrng)
    if trace is not None and any("body" in fr for fr in trace["frames"]):
        hi = lambda k, i: body_series(trace, n, lambda b: b[f"wrist{k}"][i])  # noqa: E731
        side = "L" if np.mean(hi("L", 1)) > np.mean(hi("R", 1)) else "R"
        wy, wx = highpass(hi(side, 1), 1.5), highpass(hi(side, 0), 1.5)
        swing = np.clip(0.35 * wy + 0.2 * wx, -0.05, 0.05)
        swing = mm.damp(swing[:, None], FPS, 0.25)[:, 0]
    else:
        beat_imp = np.zeros(n)
        for f in bt:
            beat_imp[f] = 0.2 * (1 if nrng.random() < 0.5 else -1)
        swing = swing + spring(beat_imp, 2.2, 0.5)
    arm_rot = active * (swing + spring(arm_imp, 2.0, 0.4))
    # shemagh: cloth with inertia. Each part is a damped pendulum driven by the head's angular
    # acceleration (it lags, swings out, overshoots and settles); the hanging flaps are slower and
    # looser than the cap, and nods swing them forward and back
    def lag(x, freq, zeta, gain):
        a = np.gradient(np.gradient(x)) * FPS * FPS
        w = 2 * np.pi * freq
        pos, vel, out = 0.0, 0.0, np.zeros(n)
        for f in range(n):
            for _ in range(4):
                acc = -w * w * pos - 2 * zeta * w * vel - gain * a[f]
                vel += acc / (4 * FPS)
                pos += vel / (4 * FPS)
            out[f] = pos
        return out
    roll_w = tilt + chest_roll
    yaw_w = yaw + chest_turn
    pitch_w = pitch + follow + lean
    cap_z = np.clip(0.9 * lag(roll_w, 2.2, 0.30, 1.0) + 0.6 * lag(yaw_w, 2.2, 0.30, 1.0), -0.10, 0.10)
    cap_y = np.clip(0.5 * lag(yaw_w, 2.2, 0.30, 1.0), -0.08, 0.08)
    tail_z = np.clip(1.3 * lag(roll_w, 1.3, 0.20, 1.0) + 1.0 * lag(yaw_w, 1.3, 0.20, 1.0), -0.16, 0.16)
    tail_x = np.clip(1.2 * lag(pitch_w, 1.3, 0.20, 1.0), -0.12, 0.12)
    tail_z += 0.010 * drift(n, 2.0, nrng)  # a little air in the cloth, never perfectly still

    # eyes: mostly on the camera, small saccades, now and then a glance down at the tablet
    gaze = np.zeros((n, 2))
    f = 0
    ends = {b for _, b in ph}
    while f < n:
        hold = rnd.randint(16, 55)
        if any(e - 4 <= f <= e + 4 for e in ends) and rnd.random() < 0.5:
            target = (-6.0, 3.0)
        else:
            target = (rnd.uniform(-2.5, 2.5), rnd.uniform(-1.0, 1.5))
        gaze[f:f + hold] = target
        f += hold
    gaze = np.stack([smooth(gaze[:, 0], 0.55), smooth(gaze[:, 1], 0.55)], -1)
    brows = np.clip((es - 0.45) * 2.0, 0, 1) + 0.3 * nod
    if face_tr is not None:
        # gaze and brows copied from the traced face, relative to its own resting values
        bs = lambda k: trace_series(trace, n, lambda fr: fr["bs"].get(k, 0.0))  # noqa: E731
        # +x = screen right: the subject's left eye looks out to screen right, the right eye looks in
        gx = ((bs("eyeLookOutLeft") - bs("eyeLookInLeft")) + (bs("eyeLookInRight") - bs("eyeLookOutRight"))) / 2
        gy = ((bs("eyeLookDownLeft") - bs("eyeLookUpLeft")) + (bs("eyeLookDownRight") - bs("eyeLookUpRight"))) / 2
        gaze = np.stack([np.clip((gx - np.median(gx)) * 16, -6, 6),
                         np.clip((gy - np.median(gy)) * 12, -3, 4)], -1)
        gaze = mm.damp(gaze, FPS, 0.05)  # eyes jump fast (saccades), just remove tracker jitter
        lift = 0.5 * bs("browInnerUp") + 0.25 * (bs("browOuterUpLeft") + bs("browOuterUpRight"))
        down = 0.5 * (bs("browDownLeft") + bs("browDownRight"))
        norm = lambda x: (x - np.median(x)) / max(np.percentile(x, 95) - np.median(x), 1e-3)  # noqa: E731
        brows = mm.damp(np.clip(norm(lift) - 0.5 * np.clip(norm(down), 0, None), -0.4, 1.2)[:, None], FPS, 0.08)[:, 0]
    body = None
    if trace is not None and any("body" in fr for fr in trace["frames"]):
        # shoulders relative to the face centre (in shoulder widths): quick drops are shrugs,
        # the slow 0.15-0.5 Hz swing is breathing; the shoulder line's slope tilts the chest
        face_h = trace_series(trace, n, lambda fr: fr["box"][3] - fr["box"][1])
        sw = body_series(trace, n, lambda b: b["shoulder_w"])
        rel_y = body_series(trace, n, lambda b: b["neck"]) + 0.5 * face_h / sw
        shrug = mm.damp(np.clip(-highpass(rel_y, 2.5) / 0.07, 0, 1)[:, None], FPS, 0.08)[:, 0]
        bp = mm.damp(highpass(rel_y, 4.0)[:, None], FPS, 0.3)[:, 0]
        breath_t = np.clip(-bp / max(2 * bp.std(), 1e-4), -1, 1)
        tilt_t = np.clip(highpass(body_series(trace, n, lambda b: b["tilt"]), 3.0) * 0.3, -0.03, 0.03)
        body = {"shrug": shrug, "breath": breath_t, "tilt": mm.damp(tilt_t[:, None], FPS, 0.15)[:, 0]}
    gaze_loc = rel.iris.material_slots[0].material.node_tree.nodes["gaze"].inputs["Location"]
    jaw_world = rel.rig.get("jaw_drop", 24) * rel.s / F_PX * rel.z_at(*rel.rig["mouth"]["center"])

    for f in range(n):
        F = f + 1
        if not rel.painted:  # painted mode: the mouth comes from the painted variants instead
            close, jaw, wide, rnd_ = vis[f]
            for name, v in (("mouth_close", close), ("jaw", jaw), ("wide", wide), ("round", rnd_)):
                keys[name].value = float(v)
                keys[name].keyframe_insert("value", frame=F)
            jb = P["jaw"]
            jb.location = (0.0, -float(jaw) * jaw_world, 0.0)
            jb.keyframe_insert("location", frame=F)
        gaze_loc.default_value = (-gaze[f, 0] / rel.w, gaze[f, 1] / rel.h, 0.0)
        gaze_loc.keyframe_insert("default_value", frame=F)
        keys["brows"].value = float(brows[f])
        keys["brows"].keyframe_insert("value", frame=F)
        hb = P["head"]
        hb.rotation_euler = (float(pitch[f]), float(yaw[f]), float(tilt[f]))
        hb.keyframe_insert("rotation_euler", frame=F)
        sb = P["shemagh"]
        sb.rotation_euler = (0.0, float(cap_y[f]), float(cap_z[f]))
        sb.keyframe_insert("rotation_euler", frame=F)
        if "shemagh_tail" in P:
            P["shemagh_tail"].rotation_euler = (float(tail_x[f]), 0.0, float(tail_z[f]))
            P["shemagh_tail"].keyframe_insert("rotation_euler", frame=F)
        cb = P["chest"]
        breath = float(breath_s[f])
        cb.rotation_euler = (sway * (0.006 * breath - 0.012 * g_phrase[f] + float(lean[f]) + float(follow[f])),
                             0.06 * float(yaw[f]) + float(chest_turn[f]), sway * float(chest_roll[f]))
        cb.keyframe_insert("rotation_euler", frame=F)
        hb.rotation_euler[0] += 0.5 * float(follow[f])  # the head rides the follow-through too
        hb.keyframe_insert("rotation_euler", index=0, frame=F)
        if "garm" in P:
            P["garm"].rotation_euler = (0.0, 0.0, float(arm_rot[f]))
            P["garm"].keyframe_insert("rotation_euler", frame=F)
        hd = P["hands"]  # resting hands are never perfectly still
        hd.location = (float(hand_drift[f, 0]), float(hand_drift[f, 1]), 0.0)
        hd.keyframe_insert("location", frame=F)
        if body is not None:
            breath = 0.7 * breath + 0.3 * float(body["breath"][f])
            shrug = float(body["shrug"][f])
            cb.rotation_euler[2] = float(body["tilt"][f])
            cb.keyframe_insert("rotation_euler", index=2, frame=F)
        else:
            shrug = float(np.clip(0.8 * nod[f] * g_phrase[f] + 0.25 * (0.5 + 0.5 * breath), 0, 1))
        keys["breathe"].value = 0.5 + 0.5 * breath
        keys["breathe"].keyframe_insert("value", frame=F)
        keys["shrug"].value = shrug
        keys["shrug"].keyframe_insert("value", frame=F)
        # fingers lift off the back of the hand and tap while he talks (only with hands clasped)
        keys["fingers"].value = float(tap_s[f] if gest[f] == "rest" else 0.0)  # taps only fit the clasped hands
        keys["fingers"].keyframe_insert("value", frame=F)


POSES = {"rest": {}, "look": {"_gaze": (-6, 3)}, "blink": {"_blink": True},
         "shrug": {"shrug": 1.0, "breathe": 1.0}, "fingers": {"fingers": 1.0}}
POSES.update({f"v_{k}": dict(zip(("mouth_close", "jaw", "wide", "round"), v)) for k, v in VISEMES.items()})


def main():
    a = args()
    ep = (ROOT / a.episode).resolve()
    import wave
    with wave.open(str(ep / "voice.wav")) as w:
        n = int(math.ceil(w.getnframes() / w.getframerate() * FPS)) + FPS // 2
    if a.frames:
        n = min(n, a.frames)
    s = setup(n)
    rel = Relief(ROOT / a.ref, ROOT / a.image)
    rel.build()
    plates(ROOT / a.ref, ep)
    out = ep / "out" / f"relief_{Path(a.ref).name}"
    out.mkdir(parents=True, exist_ok=True)
    if a.poses and rel.painted:
        for show in ["base"] + list(rel.layers):
            for name, ob in rel.layers.items():
                rel.fade(ob).default_value = 1.0 if name == show else 0.0
            s.frame_set(1)
            s.render.filepath = str(out / f"paint_{show}.png")
            bpy.ops.render.render(write_still=True)
        print("POSES_DONE")
        return
    if a.poses:
        kb = rel.ob.data.shape_keys.key_blocks
        gaze_loc = rel.iris.material_slots[0].material.node_tree.nodes["gaze"].inputs["Location"]
        jaw_world = rel.rig.get("jaw_drop", 24) * rel.s / F_PX * rel.z_at(*rel.rig["mouth"]["center"])
        for name, v in POSES.items():
            for k in kb[1:]:
                k.value = v.get(k.name, 0.0)
            rel.arm.pose.bones["jaw"].location = (0, -v.get("jaw", 0.0) * jaw_world, 0)
            gx, gy = v.get("_gaze", (0, 0))
            gaze_loc.default_value = (-gx / rel.w, gy / rel.h, 0)
            rel.lids.hide_render = not v.get("_blink", False)
            s.frame_set(1)
            s.render.filepath = str(out / f"pose_{name}.png")
            bpy.ops.render.render(write_still=True)
        print("POSES_DONE")
        return
    # a face trace in the episode (recorded alongside voice.wav, same clock) drives mouth, blinks, head
    trace_file = ep / "face_trace.json"
    trace = json.loads(trace_file.read_text(encoding="utf-8")) if trace_file.exists() else None
    perform(rel, ep, n, trace)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    s.render.filepath = str(out / "frames") + "/"
    s.frame_start = max(1, a.from_frame)
    bpy.ops.render.render(animation=True)
    print(f"RELIEF_DONE frames={n}")


main()
