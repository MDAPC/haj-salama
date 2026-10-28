"""The news studio from the reference sheet, in 3D around the anchor:
dark walls with blue neon bars, a big screen behind him (story images play there),
a night-city window on the left, a grey desk with a neon edge, navy chair, tablet."""
import math

import bmesh
import bpy
from mathutils import Vector

from common import _toon_tree, assign, flat, outline, srgb, toon

NEON = (40, 110, 255)
WALL = (38, 42, 50)
WALL_DARK = (24, 27, 33)
DESK_TOP = (92, 96, 104)
DESK_FRONT = (32, 35, 42)
CHAIR = (30, 40, 70)
DESK_Z = 0.755
SCREEN_SIZE = (3.0, 1.6875)
SCREEN_LOC = (0.35, 1.72, 1.62)


def box(name, size, loc, mat, rot=(0, 0, 0), line=0.004):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location, ob.rotation_euler = loc, rot
    assign(ob, mat)
    if line:
        outline(ob, line)
    return ob


def glow(name, rgb=NEON, strength=1.6):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = srgb(*rgb)
    e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m


def image_panel(name, path, size, loc, strength=1.0):
    """Self-lit picture (screen, window): unaffected by the toon light."""
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    uvl = bm.loops.layers.uv.new()
    for f in bm.faces:
        for loop in f.loops:
            loop[uvl].uv = (loop.vert.co.x + 0.5, loop.vert.co.y + 0.5)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], 0, v.co.y * size[1]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path))
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], e.inputs["Color"])
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    me.materials.append(m)
    return ob, tex


def build_studio(root):
    S = root / "assets/studio"
    wall = toon("wall", WALL, 0.7)
    wall_dark = toon("wall_dark", WALL_DARK, 0.7)
    neon = glow("neon")

    box("floor", (10, 8, 0.02), (0, 1.0, -0.01), toon("floor", (30, 32, 38), 0.8), line=0)
    box("back_wall", (9, 0.1, 4.0), (0, 1.85, 2.0), wall, line=0)
    # big screen behind him, framed, slightly right of centre as in the refs
    box("screen_bezel", (3.2, 0.06, 1.86), (0.35, 1.76, 1.62), wall_dark, line=0.008)
    screen, screen_tex = image_panel("screen", S / "screen_default.png", SCREEN_SIZE, SCREEN_LOC)
    box("screen_neon_top", (3.4, 0.03, 0.025), (0.35, 1.70, 2.62), neon, line=0)
    # neon-barred pillars either side of the screen
    for x in (-1.75, 2.45):
        box(f"pillar{x}", (0.55, 0.3, 4.0), (x, 1.62, 2.0), wall, line=0.006)
        for k in range(9):
            box(f"bar{x}_{k}", (0.42, 0.02, 0.028), (x, 1.46, 0.75 + k * 0.24), neon, line=0)
    # window with the night city on the far left
    window, _ = image_panel("window", S / "city_night.png", (1.2, 1.4), (-2.75, 1.70, 1.55), strength=0.9)
    box("window_frame", (1.3, 0.05, 1.5), (-2.75, 1.74, 1.55), wall_dark, line=0.006)

    # desk: grey top with a glowing front edge, dark front with a neon trapezoid
    top = box("desk_top", (2.2, 0.75, 0.05), (0, -0.52, DESK_Z - 0.025), toon("desk_top", DESK_TOP, 0.72), line=0.004)
    box("desk_edge", (2.2, 0.02, 0.018), (0, -0.90, DESK_Z - 0.03), neon, line=0)
    front = box("desk_front", (2.0, 0.05, DESK_Z - 0.05), (0, -0.84, (DESK_Z - 0.05) / 2),
                toon("desk_front", DESK_FRONT, 0.75), line=0.004)
    for (x0, z0), (x1, z1) in (((-0.55, 0.12), (-0.35, 0.55)), ((-0.35, 0.55), (0.35, 0.55)),
                               ((0.35, 0.55), (0.55, 0.12)), ((0.55, 0.12), (-0.55, 0.12))):
        c = Vector(((x0 + x1) / 2, -0.875, (z0 + z1) / 2))
        ln = math.hypot(x1 - x0, z1 - z0)
        box("desk_neon", (ln, 0.01, 0.014), c, neon, rot=(0, -math.atan2(z1 - z0, x1 - x0), 0), line=0)

    # navy high-back chair behind him
    chair = toon("chair", CHAIR, 0.65)
    box("chair_back", (0.50, 0.08, 0.62), (0, 0.20, 0.98), chair, rot=(math.radians(-6), 0, 0), line=0.005)
    box("chair_seat", (0.50, 0.46, 0.08), (0, 0.02, 0.50), chair, line=0.005)

    # tablet on its stand, screen toward him
    dark = toon("tablet", (26, 26, 30), 0.8)
    box("tablet", (0.30, 0.012, 0.21), (0.52, -0.58, DESK_Z + 0.09), dark,
        rot=(math.radians(-62), 0, math.radians(-18)), line=0.003)
    return {"screen": screen, "screen_tex": screen_tex}
