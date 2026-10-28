"""Skeleton, skin weights and face shape keys for the 3D Haj Salama.

Bones (all rotation_mode XYZ; spine/neck/head rolled so X = nod, Y = turn, Z = tilt):
  root > spine > chest > neck > head > jaw, eye.R/L, brow.R/L
  chest > shoulder.S > upperarm.S > forearm.S > hand.S > {index,middle,ring,pinky,thumb}.{1,2}.S
Head mesh shape keys: wide, round, press, fv (lip shapes on top of the jaw bone).
"""
import math

import bpy
import numpy as np
from mathutils import Vector

from body import ELBOW, FINGERS, NECK_BASE, NECK_TOP, SHOULDER, WRIST
from head import MOUTH_H, MOUTH_W, MOUTH_Z

FWD = Vector((0, -1, 0))


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def build_armature(head_root, parts, hands):
    hs = head_root.scale[0]
    H = head_root.matrix_world

    def hw(v):  # head-local -> world
        return H @ Vector(v)

    arm_data = bpy.data.armatures.new("rig")
    arm = bpy.data.objects.new("rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, deform=True, roll_to=FWD):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if parent:
            b.parent = eb[parent]
        b.use_deform = deform
        if roll_to is not None:
            b.align_roll(roll_to)
        return b

    bone("root", (0, 0, 0.42), (0, 0, 0.55))
    bone("spine", (0, 0, 0.55), (0, 0, 0.80), "root")
    bone("chest", (0, 0, 0.80), NECK_BASE, "spine")
    bone("neck", NECK_BASE, NECK_TOP, "chest")
    bone("head", NECK_TOP, NECK_TOP + Vector((0, 0, 0.30)), "neck")
    bone("jaw", hw((0, 0.010, -0.040)), hw((0, -0.090, -0.150)), "head")
    for s in ("R", "L"):
        c = Vector(parts[f"eye.{s}"].matrix_world.translation)
        bone(f"eye.{s}", c, c + FWD * 0.05, "head", deform=False, roll_to=Vector((0, 0, 1)))
        bc = sum((parts[f"brow.{s}"].matrix_world @ v.co for v in parts[f"brow.{s}"].data.vertices),
                 Vector()) / len(parts[f"brow.{s}"].data.vertices)
        bone(f"brow.{s}", bc, bc + Vector((0, 0, 0.03)), "head", deform=False)
        sign = -1 if s == "R" else 1
        bone(f"shoulder.{s}", (sign * 0.03, 0.0, 0.95), SHOULDER[s], "chest", roll_to=Vector((0, 0, 1)))
        bone(f"upperarm.{s}", SHOULDER[s], ELBOW[s], f"shoulder.{s}", roll_to=FWD)
        bone(f"forearm.{s}", ELBOW[s], WRIST[s], f"upperarm.{s}", roll_to=Vector((0, 0, 1)))
        hand, M, joints = hands[s]
        palm_n = (M.to_3x3() @ Vector((0, 1, 0))).normalized()
        along = (M.to_3x3() @ Vector((0, 0, 1))).normalized()
        bone(f"hand.{s}", WRIST[s], WRIST[s] + along * 0.11, f"forearm.{s}", roll_to=palm_n)
        for f, pts in joints.items():
            bone(f"{f}.1.{s}", pts[0], pts[1], f"hand.{s}", roll_to=palm_n)
            bone(f"{f}.2.{s}", pts[1], pts[2], f"{f}.1.{s}", roll_to=palm_n)
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
    arm.show_in_front = True
    arm.hide_render = True
    return arm


def armature_first(ob, arm):
    """Armature deformation has to come before the outline shell (and cloth/collision)."""
    mod = next((m for m in ob.modifiers if m.type == "ARMATURE"), None)
    if mod is None:
        mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    ob.modifiers.move(list(ob.modifiers).index(mod), 0)
    return mod


def auto_weight(ob, arm, bones):
    """Heat-diffusion weights, limited to the listed bones."""
    saved = {b.name: b.use_deform for b in arm.data.bones}
    for b in arm.data.bones:
        b.use_deform = b.name in bones
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)
    for b in arm.data.bones:
        b.use_deform = saved[b.name]
    armature_first(ob, arm)


def parent_to_bone(ob, arm, bone):
    bpy.context.view_layer.update()
    mw = ob.matrix_world.copy()
    ob.parent = arm
    ob.parent_type = "BONE"
    ob.parent_bone = bone
    ob.matrix_parent_inverse.identity()
    bpy.context.view_layer.update()
    ob.matrix_world = mw


def weight_head(head, arm):
    """Head mesh: jaw takes everything below the mouth line on the front of the face."""
    co = np.array([v.co[:] for v in head.data.vertices])  # head-local
    x, y, z = co.T
    front = smoothstep((-y - 0.02) / 0.05)
    spread = 0.004 + 0.035 * smoothstep((np.abs(x) - MOUTH_W) / 0.04)
    jaw = smoothstep((MOUTH_Z - z + spread * 0.5) / spread) * front
    jaw *= 1 - smoothstep((np.abs(x) - 0.075) / 0.02)  # nothing past the jaw hinge
    gh = head.vertex_groups.new(name="head")
    gj = head.vertex_groups.new(name="jaw")
    for i, w in enumerate(jaw):
        if w < 0.999:
            gh.add([i], float(1 - w), "REPLACE")
        if w > 0.001:
            gj.add([i], float(w), "REPLACE")
    mw = head.matrix_world.copy()
    head.parent = arm
    head.parent_type = "OBJECT"
    head.matrix_world = mw
    armature_first(head, arm)


def lip_keys(head):
    """Lip shapes in head-local space, centred on the mouth."""
    co = np.array([v.co[:] for v in head.data.vertices])
    x, y, z = co.T
    dz = z - MOUTH_Z
    w = np.exp(-((x / (MOUTH_W * 1.6)) ** 2 + (dz / 0.028) ** 2)) * smoothstep((-y - 0.04) / 0.03)
    rim = np.exp(-((x / (MOUTH_W * 1.3)) ** 2 + (dz / (MOUTH_H * 3)) ** 2)) * smoothstep((-y - 0.04) / 0.03)
    keys = {
        "wide": np.stack([x * 0.16 * w, np.zeros_like(x), -dz * 0.3 * rim], -1),
        "round": np.stack([-x * 0.38 * w, -0.010 * w, dz * 0.9 * rim], -1),
        "press": np.stack([np.zeros_like(x), 0.002 * rim, -dz * 0.95 * rim], -1),
        "fv": np.stack([np.zeros_like(x), 0.005 * rim * (dz < 0), np.where(dz < 0, 0.006, 0) * rim], -1),
    }
    if head.data.shape_keys is None:
        head.shape_key_add(name="Basis")
    for name, d in keys.items():
        kb = head.shape_key_add(name=name, from_mix=False)
        kb.data.foreach_set("co", (co + d).astype(np.float32).ravel())


def rig_character(head_root, parts, body, hands):
    bpy.context.view_layer.update()
    arm = build_armature(head_root, parts, hands)
    head = parts["head"]
    lip_keys(head)
    weight_head(head, arm)
    for name, ob in parts.items():
        if name == "head":
            continue
        if name in ("teeth.lower", "tongue"):
            parent_to_bone(ob, arm, "jaw")
        elif name.startswith("eye."):
            parent_to_bone(ob, arm, name)
        elif name.startswith("brow."):
            parent_to_bone(ob, arm, name)
        else:
            parent_to_bone(ob, arm, "head")
    auto_weight(body["thobe"], arm, {"spine", "chest", "shoulder.R", "shoulder.L", "upperarm.R",
                                     "upperarm.L", "forearm.R", "forearm.L"})
    auto_weight(body["neck"], arm, {"chest", "neck", "head"})
    for s in ("R", "L"):
        bones = {f"forearm.{s}", f"hand.{s}"} | {f"{f}.{i}.{s}" for f in list(FINGERS) + ["thumb"] for i in (1, 2)}
        auto_weight(hands[s][0], arm, bones)
    for ob in body["chest_parts"]:
        parent_to_bone(ob, arm, "chest")
    for s, ob in body["cuffs"].items():
        parent_to_bone(ob, arm, f"forearm.{s}")
    return arm
