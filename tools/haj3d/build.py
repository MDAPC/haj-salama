"""Build the 3D Haj Salama character into a .blend.

    blender -b -P tools/haj3d/build.py -- [--preview DIR]

Writes assets/haj_salama_3d/haj.blend (and preview stills when asked).
"""
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from body import build_hand, build_thobe  # noqa: E402
from head import build_head  # noqa: E402
from rig import rig_character  # noqa: E402
from shemagh import build_shemagh  # noqa: E402

ROOT = HERE.parent.parent
OUT = ROOT / "assets/haj_salama_3d"
HEAD_POS = (0.0, -0.02, 1.225)
HEAD_SCALE = 1.3  # caricature: big head on a thin neck


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--preview")
    return p.parse_args(argv)


def scene_setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "BLENDER_EEVEE"
    s.view_settings.view_transform = "Standard"
    s.eevee.taa_render_samples = 16
    w = bpy.data.worlds.new("world")
    w.color = (0.02, 0.02, 0.025)
    s.world = w
    sun = bpy.data.lights.new("key", "SUN")
    sun.energy = 3.2
    sun.use_shadow = False  # ink-outline shells would shadow everything; toon shading needs N.L only
    key = bpy.data.objects.new("key", sun)
    key.rotation_euler = (math.radians(50), math.radians(-25), math.radians(-30))
    s.collection.objects.link(key)
    return s


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def preview(s, out, shots):
    s.render.resolution_x, s.render.resolution_y = 900, 1000
    s.render.film_transparent = False
    s.world.color = (0.12, 0.2, 0.3)
    cam_data = bpy.data.cameras.new("pcam")
    cam = bpy.data.objects.new("pcam", cam_data)
    s.collection.objects.link(cam)
    s.camera = cam
    for name, loc, target, lens in shots:
        cam.location = loc
        cam_data.lens = lens
        look_at(cam, target)
        s.render.filepath = str(Path(out) / f"{name}.png")
        bpy.ops.render.render(write_still=True)


def main():
    a = args()
    s = scene_setup()
    head_root = bpy.data.objects.new("head_root", None)
    head_root.location = HEAD_POS
    head_root.scale = (HEAD_SCALE,) * 3
    s.collection.objects.link(head_root)
    bpy.context.view_layer.update()
    parts, marks = build_head(head_root)

    # rest expression: right eye wide, left droopy
    # rest expression from the refs: heavy, tired lids over both eyes, bags beneath
    for name, deg in (("lid.R", -20), ("lid.L", -14), ("lowlid.R", 58), ("lowlid.L", 56)):
        bpy.data.objects[name].rotation_euler.x = math.radians(deg)

    body = build_thobe()
    hands = {s: build_hand(s) for s in ("R", "L")}
    arm = rig_character(head_root, parts, body, hands)
    build_shemagh(head_root, arm, body)

    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "haj.blend"))
    if a.preview:
        h = Vector(HEAD_POS)
        preview(s, a.preview, [
            ("front", h + Vector((0, -0.85, 0.02)), h, 70),
            ("three_q", h + Vector((-0.55, -0.62, 0.05)), h, 70),
            ("side", h + Vector((-0.85, 0, 0.02)), h, 70),
            ("body", Vector((0.0, -2.4, 1.05)), Vector((0, 0, 0.98)), 50),
            ("body_q", Vector((-1.5, -1.9, 1.15)), Vector((0, 0, 0.98)), 50),
        ])
    print("BUILD_DONE")


main()
