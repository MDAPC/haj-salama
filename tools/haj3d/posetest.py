"""Render the rig in a few test poses to check deformation.

    blender -b assets/haj_salama_3d/haj.blend -P tools/haj3d/posetest.py -- OUT_DIR
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

out = Path(sys.argv[sys.argv.index("--") + 1])
s = bpy.context.scene
arm = bpy.data.objects["rig"]
P = arm.pose.bones
head = bpy.data.objects["head"]
print("GROUPS", {o.name: len(o.vertex_groups) for o in bpy.data.objects if o.type == "MESH" and o.vertex_groups})

cam_data = bpy.data.cameras.new("tcam")
cam = bpy.data.objects.new("tcam", cam_data)
s.collection.objects.link(cam)
s.camera = cam
s.render.resolution_x, s.render.resolution_y = 900, 1000
s.world.color = (0.12, 0.2, 0.3)


def shoot(name, loc, target, lens=50):
    cam.location = loc
    cam_data.lens = lens
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
    s.render.filepath = str(out / f"{name}.png")
    bpy.ops.render.render(write_still=True)


def reset():
    for pb in P:
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)
    for kb in head.data.shape_keys.key_blocks[1:]:
        kb.value = 0


reset()
shoot("rest", (0, -2.4, 1.1), (0, 0, 1.0))
# talking pose: head turned and tilted, jaw open, round lips, brows up, arm raised, fingers curled
P["head"].rotation_euler = (math.radians(-8), math.radians(25), math.radians(6))
P["neck"].rotation_euler = (0, math.radians(8), 0)
P["jaw"].rotation_euler.x = math.radians(16)
head.data.shape_keys.key_blocks["round"].value = 0.6
P["brow.R"].location.y = 0.012
P["brow.L"].location.y = 0.012
P["upperarm.R"].rotation_euler = (math.radians(25), 0, math.radians(-10))
P["forearm.R"].rotation_euler = (math.radians(-35), 0, 0)
P["hand.R"].rotation_euler = (math.radians(30), 0, 0)
for f in ("index", "middle", "ring", "pinky"):
    P[f"{f}.1.L"].rotation_euler.x = math.radians(50)
    P[f"{f}.2.L"].rotation_euler.x = math.radians(50)
P["eye.R"].rotation_euler.z = math.radians(20)
P["eye.L"].rotation_euler.z = math.radians(20)
P["spine"].rotation_euler.x = math.radians(6)
shoot("pose", (0, -2.4, 1.1), (0, 0, 1.0))
shoot("pose_face", (0.15, -1.0, 1.32), (0, 0, 1.28), 60)
print("POSETEST_DONE")
