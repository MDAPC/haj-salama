"""Thobe (torso + sleeves), neck, collar, buttons and hands, in world space (metres, -Y forward).
Returns the parts plus the joint positions the skeleton is built from."""
import math

import bmesh
import numpy as np
from mathutils import Matrix, Vector

from common import assign, meta_mesh, obj_from_bm, outline, toon, uv_sphere

THOBE = (226, 206, 166)        # beige shirt-thobe from the references
THOBE_DARK = (200, 178, 136)
SKIN = (204, 124, 84)
BUTTON = (236, 226, 204)

# rest pose: forearms on the desk, hands clasped in front of him (the anchor pose in every ref)
SHOULDER = {"R": Vector((-0.185, 0.0, 0.955)), "L": Vector((0.185, 0.0, 0.955))}
ELBOW = {"R": Vector((-0.240, -0.070, 0.785)), "L": Vector((0.240, -0.070, 0.780))}
WRIST = {"R": Vector((-0.085, -0.340, 0.805)), "L": Vector((0.085, -0.335, 0.792))}
NECK_BASE = Vector((0.0, -0.005, 0.975))
NECK_TOP = Vector((0.0, -0.030, 1.140))


def capsule(p0, p1, r0, r1, step=0.018):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.linalg.norm(p1 - p0) / step) + 1)
    return [(tuple(p0 + (p1 - p0) * t), r0 + (r1 - r0) * t) for t in np.linspace(0, 1, n)]


def build_thobe(parent=None):
    E = [((0, 0.000, 0.600), (0.165, 0.125, 0.110)),   # hips (hidden by the desk)
         ((0, 0.000, 0.720), (0.150, 0.110, 0.100)),
         ((0, 0.000, 0.840), (0.166, 0.104, 0.100)),   # chest
         ((0, 0.005, 0.935), (0.150, 0.082, 0.060)),   # upper chest / shoulder line
         ((0, -0.020, 0.880), (0.120, 0.080, 0.070))]  # loose front drape
    for s in ("R", "L"):
        sh, el, wr = SHOULDER[s], ELBOW[s], WRIST[s]
        E += [(tuple(sh - Vector((0, 0, 0.012))), (0.050, 0.052, 0.042))]  # sloping old-man shoulders
        E += [(c, r) for c, r in capsule(sh, el, 0.050, 0.046)]
        cuff = wr + (el - wr).normalized() * 0.02
        E += [(c, r) for c, r in capsule(el, cuff, 0.046, 0.056)]  # thobe sleeves widen at the cuff
    thobe = meta_mesh("thobe", E, resolution=0.006, parent=parent)
    assign(thobe, toon("thobe", THOBE, shade=0.72))
    outline(thobe, 0.004)

    # neck: long and thin with an Adam's apple
    NE = [(c, (r, r * 0.95, r)) for c, r in capsule(NECK_BASE, NECK_TOP + Vector((0, 0, 0.03)), 0.037, 0.034)]
    NE += [((0, -0.062, 1.050), (0.013, 0.014, 0.018))]
    neck = meta_mesh("neck", NE, resolution=0.004, parent=parent)
    assign(neck, toon("neckskin", SKIN, shade=0.6))
    outline(neck, 0.003)

    tv = np.array([v.co[:] for v in thobe.data.vertices])

    def chest_y(x, z):
        """Front surface of the shirt at (x, z)."""
        near = tv[(np.abs(tv[:, 0] - x) < 0.008) & (np.abs(tv[:, 2] - z) < 0.008) & (tv[:, 1] < 0)]
        return near[:, 1].min() if len(near) else -0.10

    shirt = toon("shirt_detail", THOBE, shade=0.72)
    dark = toon("shirt_dark", THOBE_DARK, shade=0.72)
    chest_parts = []

    # shirt collar: a band round the neck and two pointed tabs
    collar = torus("collar", 0.047, 0.008, parent)
    collar.location = (0, -0.008, 0.972)
    collar.rotation_euler.x = math.radians(-14)
    assign(collar, shirt)
    outline(collar, 0.002)
    chest_parts.append(collar)
    for s in (-1, 1):
        tip = (s * 0.040, chest_y(s * 0.04, 0.93) - 0.006, 0.928)
        tab = slab(f"collar_tab.{'R' if s < 0 else 'L'}",
                   [(s * 0.010, -0.052, 0.968), (s * 0.052, -0.030, 0.982), tip], 0.004, parent)
        assign(tab, shirt)
        outline(tab, 0.0018)
        chest_parts.append(tab)
    # placket and buttons
    zs = np.linspace(0.955, 0.70, 12)
    pl = slab("placket", [(-0.014, chest_y(0, z) - 0.002, z) for z in zs] +
              [(0.014, chest_y(0, z) - 0.002, z) for z in zs[::-1]], 0.002, parent, strip=True)
    assign(pl, dark)
    chest_parts.append(pl)
    for i, z in enumerate((0.925, 0.865, 0.805, 0.745)):
        b = uv_sphere(f"button.{i}", 0.0055, 16, 8, parent, (0, chest_y(0, z) - 0.0045, z), (1, 0.45, 1))
        assign(b, toon("button", BUTTON, 0.7))
        outline(b, 0.001)
        chest_parts.append(b)
    # chest pocket on his left (screen right)
    px0, px1, pz0, pz1 = 0.045, 0.110, 0.800, 0.868
    grid = [(x, chest_y(x, z) - 0.003, z) for z in (pz1, pz0) for x in (px0, px1)]
    pocket = slab("pocket", [grid[0], grid[1], grid[3], grid[2]], 0.002, parent)
    assign(pocket, shirt)
    outline(pocket, 0.0016)
    chest_parts.append(pocket)

    # cuffs where the sleeves meet the hands
    cuffs = {}
    for s in ("R", "L"):
        el, wr = ELBOW[s], WRIST[s]
        d = (wr - el).normalized()
        c = torus(f"cuff.{s}", 0.050, 0.009, parent)
        c.location = wr - d * 0.022
        c.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        assign(c, shirt)
        outline(c, 0.0018)
        cuffs[s] = c
    return {"thobe": thobe, "neck": neck, "chest_parts": chest_parts, "cuffs": cuffs}


def slab(name, pts, thickness, parent=None, strip=False):
    """A thin flat piece (collar tab, pocket, placket) from a polygon outline."""
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in pts]
    if strip:  # polygon given as left edge down + right edge up: build quads between them
        n = len(vs) // 2
        for i in range(n - 1):
            bm.faces.new((vs[i], vs[i + 1], vs[2 * n - 2 - i], vs[2 * n - 1 - i]))
    else:
        bm.faces.new(vs)
    ob = obj_from_bm(name, bm, parent, smooth=False)
    mod = ob.modifiers.new("thick", "SOLIDIFY")
    mod.thickness = thickness
    return ob


def torus(name, R, r, parent=None, segs=48, rings=12):
    bm = bmesh.new()
    verts = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        row = []
        for j in range(rings):
            b = 2 * math.pi * j / rings
            row.append(bm.verts.new(((R + r * math.cos(b)) * math.cos(a),
                                     (R + r * math.cos(b)) * math.sin(a), r * math.sin(b))))
        verts.append(row)
    for i in range(segs):
        for j in range(rings):
            bm.faces.new((verts[i][j], verts[(i + 1) % segs][j],
                          verts[(i + 1) % segs][(j + 1) % rings], verts[i][(j + 1) % rings]))
    return obj_from_bm(name, bm, parent)


# ------------------------------------------------------------------ hands

FINGERS = {  # name: (knuckle x across the palm, length, radius)
    "index": (0.027, 0.074, 0.0105),
    "middle": (0.008, 0.082, 0.0110),
    "ring": (-0.011, 0.075, 0.0105),
    "pinky": (-0.028, 0.060, 0.0095),
}
CURL = math.radians(42)  # rest curl per joint: fingers wrapped over the other hand
HS = 1.25                # caricature: big knobbly hands


def hand_frame(side):
    """World matrix of a hand: origin at the wrist, +Z along the fingers, +Y out of the palm,
    +X toward the thumb."""
    wr = WRIST[side]
    s = 1 if side == "R" else -1
    # clasped: fingers reach across toward the other hand, palms turned down and inward,
    # thumbs on top pointing forward
    z = Vector((s * 0.85, -0.50, -0.06)).normalized()
    y = Vector((s * 0.40, 0.0, -0.92)).normalized()
    thumb_dir = Vector((0, -0.4, 1.0))
    y = (y - z * y.dot(z)).normalized()
    x = y.cross(z).normalized()
    if x.dot(thumb_dir) < 0:
        x = -x
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = wr
    return m @ Matrix.Scale(HS, 4)


def finger_chain(base, direction, normal, length, n=2, curl=CURL):
    """Joint positions along a finger, curling toward the palm normal."""
    pts, d, p = [base], direction.normalized(), base.copy()
    seg = [0.55, 0.45] if n == 2 else [1.0 / n] * n
    for f in seg:
        axis = d.cross(normal).normalized()
        d = (Matrix.Rotation(-curl, 3, axis) @ d).normalized()
        p = p + d * (length * f)
        pts.append(p.copy())
    return pts


def build_hand(side, parent=None):
    M = hand_frame(side)
    R3 = M.to_3x3()
    def W(v):
        return M @ Vector(v)
    palm_n = (R3 @ Vector((0, 1, 0))).normalized()
    along = (R3 @ Vector((0, 0, 1))).normalized()
    E = [(tuple(W((0, 0.0, 0.050))), (0.044, 0.018, 0.052)),
         (tuple(W((0.004, -0.002, 0.012))), (0.032, 0.016, 0.022))]  # wrist
    joints = {}
    for name, (kx, length, r) in FINGERS.items():
        base = W((kx, -0.004, 0.096))
        pts = finger_chain(base, along, palm_n, length * HS)
        joints[name] = pts
        for a, b in zip(pts[:-1], pts[1:]):
            E += capsule(a, b, r * HS, r * HS * 0.92, step=0.008)
        E += [(tuple(pts[0]), r * HS * 1.25)]  # knobbly knuckle
    tb = W((0.034, 0.010, 0.034))
    tdir = (R3 @ Vector((0.75, 0.35, 0.60))).normalized()
    tpts = finger_chain(tb, tdir, palm_n, 0.066 * HS)
    joints["thumb"] = tpts
    for a, b in zip(tpts[:-1], tpts[1:]):
        E += capsule(a, b, 0.0125 * HS, 0.011 * HS, step=0.008)
    hand = meta_mesh(f"hand.{side}", E, resolution=0.0032, parent=parent)
    assign(hand, toon(f"handskin.{side}", SKIN, shade=0.6))
    outline(hand, 0.0025)
    return hand, M, joints
