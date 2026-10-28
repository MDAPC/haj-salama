"""Red-and-white shemagh as a simulated cloth, pinned at the crown under a black agal.

The cloth starts as a triangle (the folded shemagh) wrapped over the skull, is pre-draped by
simulating a few dozen frames against collision proxies, and that draped shape becomes its
rest state. It stays a live cloth parented to the head bone, so head turns make it swing."""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from body import torus
from common import _toon_tree, assign, outline, srgb, toon, uv_sphere
from rig import armature_first, parent_to_bone

RED = (204, 46, 48)
WHITE = (246, 240, 232)
AGAL = (22, 18, 18)

CRANIUM = ((0, 0.012, 0.035), (0.090, 0.104, 0.112))   # head-local, from head.head_elements
FACE = ((0, -0.032, -0.020), (0.080, 0.068, 0.088))


def shemagh_material():
    """Red-and-white check, as in the reference sheet (UVs are in metres)."""
    m = bpy.data.materials.new("shemagh")
    N, L, mul = _toon_tree(m, shade=0.64)
    uv = N.new("ShaderNodeTexCoord")
    chk = N.new("ShaderNodeTexChecker")
    chk.inputs["Scale"].default_value = 70.0  # ~1.4 cm squares
    chk.inputs["Color1"].default_value = srgb(*RED)
    chk.inputs["Color2"].default_value = srgb(*WHITE)
    L.new(uv.outputs["UV"], chk.inputs["Vector"])
    L.new(chk.outputs["Color"], mul.inputs[1])
    return m


def collision_proxies(head_root, arm, body):
    """Cheap smooth shapes for the cloth to collide with (hidden from render)."""
    H = head_root.matrix_world
    hs = head_root.scale[0]
    proxies = []
    shapes = [("col_cranium", CRANIUM, 0.010), ("col_face", FACE, 0.006)]
    # ears (never seen, always covered) tent the cloth out beside the face like a real ghutra
    shapes += [(f"col_ear{k}", ((s * 0.098, 0.004, -0.004), (0.026, 0.034, 0.042)), 0.004)
               for k, s in enumerate((-1, 1))]
    for name, (c, r), pad in shapes:
        ob = uv_sphere(name, 1.0, 32, 16, None, tuple(H @ Vector(c)),
                       tuple(v * hs + pad for v in r))
        parent_to_bone(ob, arm, "head")
        proxies.append(ob)
    for ob in proxies:
        col = ob.modifiers.new("collision", "COLLISION")
        ob.collision.thickness_outer = 0.004
        ob.collision.cloth_friction = 8
        ob.hide_render = True
        ob.display_type = "WIRE"
    for ob in (body["neck"], body["thobe"]):
        ob.modifiers.new("collision", "COLLISION")
        mods = list(ob.modifiers)
        ob.modifiers.move(len(mods) - 1, 1)  # right after the armature, before the outline shell
        ob.collision.thickness_outer = 0.005
        ob.collision.cloth_friction = 8
    return proxies


def cloth_mesh(head_root, width=1.15, depth=0.62, nx=64, ny=34):
    """Folded cloth wrapped over the skull: the long edge across the forehead, the back
    falling behind the neck, full-width side panels falling past the cheeks to the chest."""
    H = head_root.matrix_world
    hs = head_root.scale[0]
    S = H @ Vector(CRANIUM[0])
    R = max(CRANIUM[1]) * hs + 0.016
    a0 = math.radians(18)              # front edge sits just above the brows
    amax, bmax = math.radians(165), math.radians(78)
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new()
    grid = {}
    pin = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            s = (i / nx - 0.5) * width                   # along the forehead edge
            d = (j / ny) * depth                         # from the edge toward the back
            a, b = a0 + d / R, s / R
            over_a = max(0.0, a - amax) * R
            over_b = max(0.0, abs(b) - bmax) * R
            a, b = min(a, amax), max(-bmax, min(bmax, b))
            p = S + R * Vector((math.sin(b), -math.cos(b) * math.cos(a), math.cos(b) * math.sin(a)))
            # overhangs start clear of the body and fall into place in the pre-drape:
            # the back point flares backward, the side flaps sweep forward over the chest
            p.x += math.copysign(0.38 * over_b, b)
            p.y += 0.55 * over_a - 0.55 * over_b
            p.z -= 0.75 * over_a + 0.72 * over_b
            grid[i, j] = bm.verts.new(p)
            pin[i, j] = 1.0 if (a < math.radians(125) and abs(b) < math.radians(52)) else 0.0
    for i in range(nx):
        for j in range(ny):
            q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            f = bm.faces.new([grid[k] for k in q])
            for loop, k in zip(f.loops, q):
                loop[uvl].uv = (k[0] / nx * width, k[1] / ny * depth)  # metres, so the lattice stays square
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    f0 = bm.faces[len(bm.faces) // 2]
    if f0.normal.dot(f0.calc_center_median() - S) < 0:  # normals must face out, away from the skull
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("shemagh")
    bm.to_mesh(me)
    order = list(grid.keys())
    bm.free()
    ob = bpy.data.objects.new("shemagh", me)
    bpy.context.scene.collection.objects.link(ob)
    vg = ob.vertex_groups.new(name="pin")
    for idx, k in enumerate(order):
        if pin[k] > 0:
            vg.add([idx], pin[k], "REPLACE")
    me.shade_smooth()
    return ob


def add_cloth(ob, frames):
    mod = ob.modifiers.new("cloth", "CLOTH")
    cs = mod.settings
    cs.quality = 10
    cs.mass = 0.15
    cs.air_damping = 2.0
    cs.tension_stiffness = cs.compression_stiffness = 18
    cs.shear_stiffness = 8
    cs.bending_stiffness = 1.5
    cs.vertex_group_mass = "pin"
    cs.pin_stiffness = 1.0
    cc = mod.collision_settings
    cc.use_collision = True
    cc.distance_min = 0.004
    cc.collision_quality = 3
    cc.use_self_collision = False
    mod.point_cache.frame_start = 1
    mod.point_cache.frame_end = frames
    return mod


def predrape(ob, frames=60):
    """Let the cloth settle once, then keep that shape as the new rest mesh."""
    scene = bpy.context.scene
    mod = add_cloth(ob, frames)
    for f in range(1, frames + 1):
        scene.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    draped = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    old = ob.data
    ob.modifiers.remove(mod)
    ob.data = draped
    draped.name = "shemagh"
    bpy.data.meshes.remove(old)
    scene.frame_set(1)


def build_shemagh(head_root, arm, body):
    proxies = collision_proxies(head_root, arm, body)
    ob = cloth_mesh(head_root)
    parent_to_bone(ob, arm, "head")
    predrape(ob)
    add_cloth(ob, 250)
    th = ob.modifiers.new("thickness", "SOLIDIFY")
    th.thickness = 0.003
    th.offset = 1
    assign(ob, shemagh_material())
    outline(ob, 0.003)

    # agal: two black cords hugging the cloth round the crown, front higher than the back.
    # Each cord is the cross-section of the (padded) cranium ellipsoid by a tilted plane.
    H = head_root.matrix_world
    hs = head_root.scale[0]
    S = H @ Vector(CRANIUM[0])
    pad = 0.010 + 0.006 + 0.008                        # proxy pad + cloth + half a cord
    ax, ay, az = (v * hs + pad for v in CRANIUM[1])
    tilt = math.radians(-14)
    n = Vector((0, -math.sin(tilt), math.cos(tilt)))
    agal = []
    for k, h in enumerate((0.40, 0.52)):
        hz = h * az
        k_ = math.sqrt(max(1 - (hz / az) ** 2, 0.05))
        t = torus(f"agal.{k}", ay * k_, 0.0105, None, 64, 12)
        t.location = S + n * hz
        t.rotation_euler.x = tilt
        t.scale.x = ax / ay
        assign(t, toon(f"agal{k}", AGAL, shade=0.8))
        outline(t, 0.0015)
        parent_to_bone(t, arm, "head")
        agal.append(t)
    return ob, agal, proxies
