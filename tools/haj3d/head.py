"""Haj Salama's head, matched to the reference sheet (assets/haj_salama/refs):
huge drooping nose, heavy tired lids over brown eyes, worried dark brows, grey-white bushy
mustache, fleshy hanging lower lip, gap teeth, big ears, grey temples, deep wrinkles.

Everything is head-local (origin = head centre, -Y = facing forward, metres). Skin detail
(wrinkles, folds, bags, lip, stubble) is painted into a per-vertex colour attribute."""
import math

import bmesh
import bpy
import numpy as np

from common import (_toon_tree, assign, flat, link, meta_mesh, obj_from_bm, outline, paint, srgb,
                    toon, toon_attr, uv_sphere)

SKIN = (204, 124, 84)
SKIN_LIGHT = (224, 146, 100)
SKIN_DARK = (150, 82, 58)
LIP = (170, 92, 78)
LIP_EDGE = (58, 24, 20)
TEETH = (240, 232, 214)
MOUTH = (66, 18, 18)
TONGUE = (168, 70, 64)
EYE_WHITE = (244, 238, 228)
IRIS = (92, 52, 28)
PUPIL = (18, 12, 10)
BROW = (66, 60, 58)
MUSTACHE = (206, 200, 190)
GREY_HAIR = (170, 164, 158)
STUBBLE = (150, 128, 112)

MOUTH_Z = -0.096                    # mouth line
MOUTH_W, MOUTH_H = 0.030, 0.0065    # half extents of the rest opening (slightly open, like the refs)
EYE_Z = 0.016
EYES = {"R": -1, "L": 1}            # his right eye is at -X (screen left)
OUTLINE = 0.0018                    # thin ink line, the refs are painted 3D rather than flat 2D


def head_elements():
    """Soft shapes that blend into the skull, face and nose (head-local, surface radii)."""
    E = [
        ((0, 0.012, 0.035), (0.090, 0.104, 0.112)),           # cranium
        ((0, -0.032, -0.020), (0.080, 0.068, 0.088)),         # face mass
        ((0, -0.036, -0.088), (0.056, 0.062, 0.050)),         # jaw
        ((0, -0.070, -0.094), (0.044, 0.034, 0.030)),         # muzzle
        ((0, -0.078, -0.134), (0.032, 0.028, 0.028)),         # rounded chin
        ((0, -0.093, MOUTH_Z - 0.014), (0.029, 0.015, 0.011)),  # fleshy hanging lower lip
        ((0, -0.080, 0.046), (0.076, 0.022, 0.020)),          # brow ridge
        ((0, -0.110, -0.006), (0.016, 0.014, 0.014)),         # nose root
        ((0, -0.126, -0.004), (0.019, 0.017, 0.022)),         # bridge
        ((0, -0.146, -0.020), (0.026, 0.023, 0.026)),
        ((0, -0.166, -0.040), (0.047, 0.040, 0.046)),         # the bulb
    ]
    for s in (-1, 1):
        E += [((s * 0.058, -0.060, 0.000), (0.028, 0.024, 0.022)),            # cheekbones
              ((s * 0.034, -0.138, -0.052), (0.019, 0.018, 0.017)),           # nostril wings
              ((s * 0.104, 0.010, -0.012), (0.016, 0.030, 0.046)),            # big ears
              ((s * 0.040, -0.104, EYE_Z), (0.024, 0.022, 0.022), True),      # eye sockets
              ((s * 0.070, -0.056, -0.066), (0.020, 0.022, 0.026), True)]     # hollow cheeks
    E += [((0, -0.124, 0.040), (0.010, 0.008, 0.012), True)]                  # crease above the nose
    return E


def skin_paint(co):
    """Per-vertex skin colour: tan base, warmer nose, lip, stubble, and dark creases."""
    x, y, z = co.T
    c = np.tile(np.array(SKIN, float), (len(co), 1))
    ax = np.abs(x)

    def mix(mask, rgb):
        mask = np.clip(mask, 0, 1)[:, None]
        c[:] = c * (1 - mask) + np.array(rgb, float) * mask

    def darken(mask, amt):
        c[:] = c * (1 - amt * np.clip(mask, 0, 1))[:, None]

    def gauss(d, s):
        return np.exp(-(d / s) ** 2)

    front = np.clip((-y - 0.04) / 0.04, 0, 1)
    mix((-y - 0.125) / 0.05, SKIN_LIGHT)                                    # nose catches the light
    mix(gauss(np.hypot(x / 1.4, z + 0.040), 0.035) * (y < -0.15) * 0.4, (226, 128, 96))  # ruddy tip
    lip = gauss(np.hypot(x / 1.7, (z - (MOUTH_Z - 0.012)) / 0.7), 0.014) * (y < -0.085)
    mix(lip, LIP)
    chin = np.clip((MOUTH_Z - 0.022 - z) / 0.03, 0, 1) * front
    mix(chin * 0.35 + np.clip((MOUTH_Z + 0.0 - z) / 0.05, 0, 1) * (ax > 0.035) * front * 0.2, STUBBLE)
    # forehead wrinkles: wavy horizontal creases
    fz = z - 0.0035 * np.sin(x * 60)
    fore = (z > EYE_Z + 0.040) & (z < 0.105) & (y < -0.05) & (ax < 0.065)
    darken(fore * gauss(((fz - 0.064) % 0.012) - 0.006, 0.0014), 0.32)
    # bags under the eyes: two arcs
    for s in EYES.values():
        ex, d = s * 0.040, np.hypot(x - s * 0.040, z - EYE_Z)
        under = (z < EYE_Z - 0.006) & (y < -0.06)
        for r in (0.030,):
            darken(under * gauss(d - r, 0.0016) * (np.abs(x - ex) < 0.022), 0.22)
    # nasolabial folds from beside the nostril wings to past the mouth corners
    for s in (-1, 1):
        p0, p1 = np.array([s * 0.050, -0.056]), np.array([s * 0.054, MOUTH_Z - 0.030])
        v = p1 - p0
        t = np.clip(((x - p0[0]) * v[0] + (z - p0[1]) * v[1]) / v.dot(v), 0, 1)
        dist = np.hypot(x - (p0[0] + t * v[0]), z - (p0[1] + t * v[1]))
        darken(gauss(dist, 0.0018) * (y < -0.06) * (y > -0.12), 0.30)
    # chin dimple crease and neck-side shadow
    darken(gauss(np.hypot(x, z - (MOUTH_Z - 0.030)), 0.004) * front, 0.18)
    return np.clip(c, 0, 255)


def build_head(parent=None):
    head = meta_mesh("head", head_elements(), resolution=0.0026, parent=parent)
    bm = bmesh.new()
    bm.from_mesh(head.data)

    # mouth opening: delete a thin ellipse on the face, then pull the ragged rim onto the ellipse
    kill = [v for v in bm.verts
            if v.co.y < -0.05 and (v.co.x / MOUTH_W) ** 2 + ((v.co.z - MOUTH_Z) / MOUTH_H) ** 2 < 1]
    bmesh.ops.delete(bm, geom=kill, context="VERTS")
    for v in {v for e in bm.edges if e.is_boundary for v in e.verts}:
        a = math.atan2((v.co.z - MOUTH_Z) / MOUTH_H, v.co.x / MOUTH_W)
        v.co.x, v.co.z = MOUTH_W * math.cos(a), MOUTH_Z + MOUTH_H * math.sin(a)
    # roll the rim inward into the dark of the mouth
    lip_faces = []
    for depth, shrink in ((0.004, 0.94), (0.008, 0.86)):
        rim = [e for e in bm.edges if e.is_boundary]
        ret = bmesh.ops.extrude_edge_only(bm, edges=rim)
        lip_faces += [g_ for g_ in ret["geom"] if isinstance(g_, bmesh.types.BMFace)]
        for v in (g_ for g_ in ret["geom"] if isinstance(g_, bmesh.types.BMVert)):
            v.co.y += depth
            v.co.x *= shrink
            v.co.z = MOUTH_Z + (v.co.z - MOUTH_Z) * shrink
    for f in lip_faces:
        f.material_index = 1
    bm.to_mesh(head.data)
    bm.free()
    head.data.shade_smooth()
    final = np.array([v.co[:] for v in head.data.vertices])
    paint(head, skin_paint(final))
    head.data.materials.append(toon_attr("skin", shade=0.66))
    head.data.materials.append(toon("lip_edge", LIP_EDGE, shade=0.8))
    outline(head, OUTLINE * 1.4)

    front = final[final[:, 1] < -0.04]

    def surf(x, z):
        """Face surface at (x, z): the deepest front point there, so the nose overhang is skipped."""
        d2 = (front[:, 0] - x) ** 2 + (front[:, 2] - z) ** 2
        near = front[d2 < max(np.min(d2), 1e-12) * 1.0 + 0.003 ** 2]
        return near[np.argmax(near[:, 1])].copy()

    parts = {"head": head}
    marks = {"mouth": surf(0, MOUTH_Z + 0.012)}

    # ---- eyes with heavy tired lids and bags
    eye_mat = eye_material()
    lid_mat = toon("lidskin", (192, 112, 78), 0.62)
    bag_mat = toon("bagskin", (176, 100, 72), 0.62)
    for name, side in EYES.items():
        r = 0.021
        c = np.array([side * 0.040, surf(side * 0.040, EYE_Z + 0.03)[1] + 0.012, EYE_Z])
        eye = uv_sphere(f"eye.{name}", r, 32, 16, parent, tuple(c))
        assign(eye, eye_mat)
        outline(eye, OUTLINE)
        parts[f"eye.{name}"] = eye
        marks[f"eye.{name}"] = c
        # lids hang off a pivot aligned with the head, so rotation.x on the lid is always open/close
        pivot = bpy.data.objects.new(f"lidpivot.{name}", None)
        pivot.location = tuple(c)
        link(pivot, parent)
        parts[f"lidpivot.{name}"] = pivot
        eyelid(f"lid.{name}", r * 1.08, pivot, (0, 0, 0), lid_mat)
        eyelid(f"lowlid.{name}", r * 1.07, pivot, (0, 0, 0), bag_mat, lower=True)

    def bar(name, pts, radii, color, lift=0.004, res=0.0016):
        """Soft hair bar (mustache/brow) hugging the face: balls along surface points."""
        els = []
        for (x, z), r in zip(pts, radii):
            p = surf(x, z)
            els.append(((x, p[1] - lift, z), (r[0], r[1], r[2])))
        ob = meta_mesh(name, els, resolution=res, parent=parent)
        assign(ob, toon(f"{name}_hair", color, shade=0.7))
        outline(ob, OUTLINE * 0.8)
        parts[name] = ob

    # ---- mustache: grey-white, bushy, ends drooping past the mouth corners
    mz = MOUTH_Z + 0.012
    xs = np.linspace(-0.050, 0.050, 13)
    bar("mustache", [(x, mz - 0.016 * (abs(x) / 0.05) ** 2.0) for x in xs],
        [(0.010, 0.011, 0.0115 * (1 - 0.6 * (abs(x) / 0.05) ** 2)) for x in xs], MUSTACHE, lift=0.012)

    # ---- brows: thick, dark grey, worried (inner ends lifted)
    for name, s in EYES.items():
        xb = np.linspace(0.014, 0.068, 9)
        tb = np.clip((xb - 0.014) / 0.054, 0, 1)  # 0 at the inner end, 1 at the outer
        pts = [(s * x, EYE_Z + 0.032 + 0.016 * (1 - t) ** 1.5 - 0.004 * t) for x, t in zip(xb, tb)]
        bar(f"brow.{name}", pts, [(0.009, 0.007, 0.0072 - 0.003 * ((x - 0.014) / 0.054)) for x in xb], BROW)

    # ---- grey hair at the temples, showing between the face and the shemagh
    temples = []
    for s in (-1, 1):
        temples += [((s * 0.084, -0.034 + k * 0.010, 0.046 - k * 0.006), (0.008, 0.012, 0.014)) for k in range(3)]
    th = meta_mesh("temple_hair", temples, resolution=0.003, parent=parent)
    assign(th, toon("temple_hair", GREY_HAIR, 0.7))
    outline(th, OUTLINE * 0.8)
    parts["temple_hair"] = th

    # ---- mouth interior, gap teeth, tongue
    m = marks["mouth"]
    cav = uv_sphere("mouth_cavity", 0.040, 24, 12, parent, (0, m[1] + 0.045, MOUTH_Z), (1.0, 0.8, 0.9))
    assign(cav, flat("mouth_dark", MOUTH))
    parts["mouth_cavity"] = cav
    tooth = toon("teeth", TEETH, shade=0.75)
    for row, zc, keep in (("upper", MOUTH_Z + 0.004, (1, 2, 4, 6)), ("lower", MOUTH_Z - 0.006, (2, 3, 5))):
        bm = bmesh.new()
        for i in keep:
            x = -0.024 + i * 0.0068
            y = surf(x, MOUTH_Z)[1] + 0.009 + 0.0025 * (x / 0.024) ** 2 * 3
            bmesh.ops.create_cube(bm, size=1.0, matrix=_mat((x, y, zc), (0.0058, 0.004, 0.009)))
        t = obj_from_bm(f"teeth.{row}", bm, parent, smooth=False)
        assign(t, tooth)
        outline(t, 0.0006)
        parts[f"teeth.{row}"] = t
    tongue = uv_sphere("tongue", 0.022, 24, 12, parent, (0, m[1] + 0.028, MOUTH_Z - 0.012), (1.0, 1.1, 0.45))
    assign(tongue, toon("tongue", TONGUE, 0.7))
    parts["tongue"] = tongue
    return parts, marks


def _mat(loc, size):
    from mathutils import Matrix
    return Matrix.Translation(loc) @ Matrix.Diagonal((*size, 1.0))


def eyelid(name, r, parent, center, mat, lower=False):
    """Hemispherical lid shell around an eyeball; its rotation about X sets how open the eye is."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r)
    cut = [v for v in bm.verts if (v.co.z < -1e-4 if not lower else v.co.z > 1e-4)]
    bmesh.ops.delete(bm, geom=cut, context="VERTS")
    ob = obj_from_bm(name, bm, parent)
    ob.location = tuple(center)
    assign(ob, mat)
    outline(ob, OUTLINE)
    return ob


def eye_material():
    """Eyeball: white, brown iris, black pupil and a catch-light, all centred on local -Y
    (so rotating the ball aims the gaze)."""
    m = bpy.data.materials.new("eye")
    N, L, mul = _toon_tree(m, shade=0.82)
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])

    def radial(ox=0.0, oz=0.0):
        """Distance from the gaze axis (optionally offset), only on the front of the ball."""
        cx = N.new("ShaderNodeMath")
        cx.operation = "SUBTRACT"
        L.new(sep.outputs["X"], cx.inputs[0])
        cx.inputs[1].default_value = ox
        cz = N.new("ShaderNodeMath")
        cz.operation = "SUBTRACT"
        L.new(sep.outputs["Z"], cz.inputs[0])
        cz.inputs[1].default_value = oz
        comb = N.new("ShaderNodeCombineXYZ")
        L.new(cx.outputs[0], comb.inputs["X"])
        L.new(cz.outputs[0], comb.inputs["Z"])
        ln = N.new("ShaderNodeVectorMath")
        ln.operation = "LENGTH"
        L.new(comb.outputs[0], ln.inputs[0])
        return ln.outputs["Value"]

    front = N.new("ShaderNodeMath")
    front.operation = "LESS_THAN"
    front.inputs[1].default_value = 0.0
    L.new(sep.outputs["Y"], front.inputs[0])

    color = None
    rd = radial()
    for radius, rgb, dist in ((0.0105, IRIS, rd), (0.0050, PUPIL, rd), (0.0019, (255, 255, 255), radial(-0.0035, 0.0035))):
        lt = N.new("ShaderNodeMath")
        lt.operation = "LESS_THAN"
        lt.inputs[1].default_value = radius
        L.new(dist, lt.inputs[0])
        both = N.new("ShaderNodeMath")
        both.operation = "MULTIPLY"
        L.new(lt.outputs[0], both.inputs[0])
        L.new(front.outputs[0], both.inputs[1])
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        L.new(both.outputs[0], mix.inputs["Factor"])
        if color is None:
            mix.inputs[6].default_value = srgb(*EYE_WHITE)
        else:
            L.new(color, mix.inputs[6])
        mix.inputs[7].default_value = srgb(*rgb)
        color = mix.outputs[2]
    L.new(color, mul.inputs[1])
    return m
