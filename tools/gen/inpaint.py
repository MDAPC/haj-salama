"""Local, open-source image edits of a reference (CPU): repaint one region, keep every other pixel.

    python tools/gen/inpaint.py REF.png OUT.png --box x0 y0 x1 y1 \
        --mask ellipse cx cy rx ry --prompt "..." [--seeds 1 2 3] [--strength 0.9] [--steps 30]

The box is cropped, upscaled to 512 for detail, repainted inside the (feathered) mask by
DreamShaper 8 inpainting (SD 1.5, CreativeML OpenRAIL-M), then pasted back. One output per seed:
OUT_s<seed>.png.
"""
import argparse
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw, ImageFilter

MODEL = "Lykon/dreamshaper-8-inpainting"
STYLE = ("detailed digital caricature painting, old arab man, warm skin, bold black ink outlines, "
         "soft painterly shading, high quality")
NEG = "photo, realistic, blurry, deformed, extra teeth, extra lips, text, watermark, lowres"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("out")
    p.add_argument("--box", type=int, nargs=4, required=True)
    p.add_argument("--mask", nargs="+", action="append", required=True,
                   help="ellipse cx cy rx ry  (image px; repeatable)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--strength", type=float, default=0.9)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--feather", type=float, default=4)
    a = p.parse_args()

    torch.set_num_threads(12)
    ref = Image.open(a.ref).convert("RGB")
    x0, y0, x1, y1 = a.box
    crop = ref.crop(a.box)
    size = 512
    sx, sy = size / crop.width, size / crop.height
    mask = Image.new("L", ref.size, 0)
    d = ImageDraw.Draw(mask)
    for m in a.mask:
        _, cx, cy, rx, ry = m[0], *map(float, m[1:])
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    mask_c = mask.crop(a.box)

    pipe = AutoPipelineForInpainting.from_pretrained(MODEL, torch_dtype=torch.float32, variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    src = crop.resize((size, size), Image.LANCZOS)
    msk = mask_c.resize((size, size), Image.LANCZOS)
    for seed in a.seeds:
        t = time.time()
        g = torch.Generator().manual_seed(seed)
        res = pipe(prompt=f"{a.prompt}, {STYLE}", negative_prompt=NEG, image=src, mask_image=msk,
                   width=size, height=size, strength=a.strength, num_inference_steps=a.steps,
                   guidance_scale=7.0, generator=g).images[0]
        patch = res.resize(crop.size, Image.LANCZOS)
        soft = mask_c.filter(ImageFilter.GaussianBlur(a.feather))
        out = ref.copy()
        out.paste(Image.composite(patch, crop, soft), (x0, y0))
        path = Path(a.out).with_name(f"{Path(a.out).stem}_s{seed}.png")
        out.save(path)
        print(f"{path.name} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
