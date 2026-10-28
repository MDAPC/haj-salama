"""Turn a rig pose render back into a reference-space image, as the starting point for inpainting.

    python from_pose.py REF.png FRAME_JSON POSE.png OUT.png --box x0 y0 x1 y1

The pose render is a 1920x1080 frame; frame.json says how reference pixels map onto it
(frame = offset + scale * ref). Only the box (reference px) is taken from the pose.
"""
import argparse
import json

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("ref")
p.add_argument("frame_json")
p.add_argument("pose")
p.add_argument("out")
p.add_argument("--box", type=int, nargs=4, required=True)
a = p.parse_args()

fr = json.load(open(a.frame_json, encoding="utf-8"))
s, (ox, oy) = fr["scale"], fr["offset"]
ref = Image.open(a.ref).convert("RGB")
pose = Image.open(a.pose).convert("RGB")
x0, y0, x1, y1 = a.box
fx0, fy0, fx1, fy1 = ox + s * x0, oy + s * y0, ox + s * x1, oy + s * y1
patch = pose.transform((x1 - x0, y1 - y0), Image.EXTENT, (fx0, fy0, fx1, fy1), Image.BICUBIC)
out = ref.copy()
out.paste(patch, (x0, y0))
out.save(a.out)
