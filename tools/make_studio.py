"""Paint studio sets, Arabic news graphics and layout mockups for a single-image anchor.

    python make_studio.py

Writes  assets/studio/*.png          set pieces and graphics (separate layers for Blender)
        assets/haj_salama/figure_nosmoke.png
        layouts/<option>.json        where each layer sits on the 1920x1080 frame
        mockups/<option>.png, mockups/sheet.png
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from make_placeholder import painterly  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STUDIO = ROOT / "assets/studio"
FIG = ROOT / "assets/haj_salama"
W, H = 1920, 1080

NAVY = (15, 30, 54)
TEAL = (19, 111, 122)
GOLD = (224, 166, 58)
CRIMSON = (179, 38, 46)
SAND = (233, 215, 176)
INK = (16, 12, 12)
LINE = 7  # outline weight, matched to the painting's black contour

FONT = "/usr/share/fonts/truetype/noto/"
KUFI_B = FONT + "NotoKufiArabic-Bold.ttf"
NASKH_B = FONT + "NotoNaskhArabic-Bold.ttf"
NASKH = FONT + "NotoNaskhArabic-Regular.ttf"

CHANNEL = "أخبار الفريج"
NAME = "الحاج سلامة"
TITLE = "مذيع نشرة الفريج المسائية"
HEADLINES = ["موجة غبار تجتاح المنطقة وتغلق الطرق السريعة",
             "أسعار التمور ترتفع في سوق الجمعة",
             "مجلس الحي يقرر أخيراً إصلاح الدوار القديم"]


def font(path, size):
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


def ar_text(d, xy, text, path, size, fill, anchor="rm"):
    d.text(xy, text, font=font(path, size), fill=fill, anchor=anchor, direction="rtl", language="ar")


def ar_width(text, path, size):
    return font(path, size).getlength(text, direction="rtl", language="ar")


def gradient(w, h, top, bottom):
    t = np.linspace(0, 1, h)[:, None, None]
    a = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    return Image.fromarray(np.broadcast_to(a, (h, w, 3)).astype(np.uint8), "RGB").convert("RGBA")


def outlined_poly(d, pts, fill, width=LINE):
    d.polygon(pts, fill=fill)
    d.line(pts + [pts[0]], fill=INK, width=width, joint="curve")


# ------------------------------------------------------------------ set pieces

def skyline(img, box, color, seed=4):
    """Generic Gulf skyline: towers with spires, a dome + minaret, palms."""
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    rnd = random.Random(seed)
    x = x0
    while x < x1:
        w = rnd.randint(34, 90)
        h = rnd.randint(int((y1 - y0) * 0.25), int((y1 - y0) * 0.9))
        d.rectangle((x, y1 - h, x + w, y1), fill=color)
        if rnd.random() < 0.35:
            d.polygon([(x + w * 0.3, y1 - h), (x + w * 0.5, y1 - h - rnd.randint(30, 90)),
                       (x + w * 0.7, y1 - h)], fill=color)
        x += w + rnd.randint(4, 30)
    cx = (x0 + x1) * 0.62
    d.ellipse((cx - 70, y1 - 170, cx + 70, y1 - 30), fill=color)  # dome
    d.rectangle((cx - 80, y1 - 100, cx + 80, y1), fill=color)
    d.rectangle((cx + 110, y1 - 260, cx + 128, y1), fill=color)  # minaret
    d.polygon([(cx + 104, y1 - 260), (cx + 119, y1 - 300), (cx + 134, y1 - 260)], fill=color)
    for px in (x0 + 60, x1 - 90):
        palm(d, px, y1, 150, color)


def palm(d, x, base, h, color):
    pts = [(x + 12 * math.sin(t * 1.4), base - h * t) for t in np.linspace(0, 1, 12)]
    d.line(pts, fill=color, width=10)
    tx, ty = pts[-1]
    for a in range(0, 360, 45):
        r = math.radians(a)
        d.line([(tx, ty), (tx + 70 * math.cos(r), ty + 30 * math.sin(r) + 25)], fill=color, width=8)


def studio_bg(warm=False):
    top, bottom = ((44, 30, 26), (120, 70, 40)) if warm else (NAVY, (22, 70, 92))
    img = gradient(W, H, top, bottom)
    d = ImageDraw.Draw(img)
    # LED wall
    wall = (150, 90, 1770, 760)
    d.rounded_rectangle(wall, 26, fill=(28, 58, 92) if not warm else (90, 52, 34))
    glow = gradient(wall[2] - wall[0], wall[3] - wall[1],
                    (60, 140, 170) if not warm else (200, 120, 60), (20, 50, 80) if not warm else (80, 40, 26))
    mask = Image.new("L", glow.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *glow.size), 26, fill=255)
    img.paste(glow, wall[:2], mask)
    sky = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    skyline(sky, (wall[0] + 30, wall[1] + 120, wall[2] - 30, wall[3] - 10),
            (14, 40, 66, 230) if not warm else (60, 30, 20, 230))
    img.alpha_composite(sky)
    d.rounded_rectangle(wall, 26, outline=INK, width=LINE)
    # light columns either side
    for x in (60, 1800):
        d.rounded_rectangle((x, 60, x + 60, 1000), 18, fill=(70, 160, 190) if not warm else (230, 160, 80))
        d.rounded_rectangle((x, 60, x + 60, 1000), 18, outline=INK, width=LINE)
    # floor
    floor = gradient(W, H - 800, (20, 46, 70) if not warm else (110, 50, 34), (8, 18, 30) if not warm else (60, 24, 18))
    img.paste(floor, (0, 800))
    if warm:
        rug(img)
    soft = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(soft)
    for i in range(5):
        sd.polygon([(300 + i * 330, 0), (420 + i * 330, 0), (620 + i * 330, H), (380 + i * 330, H)],
                   fill=(255, 255, 255, 12))
    img.alpha_composite(soft.filter(ImageFilter.GaussianBlur(30)))
    return painterly(img)


def rug(img):
    d = ImageDraw.Draw(img)
    d.rectangle((0, 860, W, H), fill=(140, 28, 34))
    for y in (880, 1050):
        d.rectangle((0, y, W, y + 14), fill=GOLD)
    for x in range(0, W, 120):
        d.polygon([(x + 60, 905), (x + 110, 965), (x + 60, 1025), (x + 10, 965)], fill=(30, 40, 70))
        d.polygon([(x + 60, 930), (x + 85, 965), (x + 60, 1000), (x + 35, 965)], fill=SAND)
    d.line([(0, 860), (W, 860)], fill=INK, width=LINE)


def desk(width, height):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = LINE
    top = 46
    outlined_poly(d, [(m, top), (width - m, top), (width - 60, height - m), (60, height - m)], NAVY)
    outlined_poly(d, [(m, m), (width - m, m), (width - m, top), (m, top)], SAND)
    d.rectangle((70, top + 40, width - 70, top + 54), fill=GOLD)
    ar_text(d, (width / 2, top + 120), CHANNEL, KUFI_B, 64, SAND, anchor="mm")
    return painterly(img)


def city_night(w=1200, h=1400, seed=12):
    """Night skyline for the studio window: towers with lit windows over dark water."""
    img = gradient(w, h, (6, 10, 28), (18, 30, 70))
    d = ImageDraw.Draw(img)
    rnd = random.Random(seed)
    water = int(h * 0.78)
    x = -20
    while x < w:
        bw, bh = rnd.randint(50, 140), rnd.randint(int(h * 0.25), int(h * 0.7))
        top = water - bh
        d.rectangle((x, top, x + bw, water), fill=(10, 14, 30))
        if rnd.random() < 0.3:
            d.polygon([(x + bw * 0.4, top), (x + bw * 0.5, top - rnd.randint(40, 120)), (x + bw * 0.6, top)],
                      fill=(10, 14, 30))
        for wy in range(top + 10, water - 8, 16):
            for wx in range(x + 6, x + bw - 6, 12):
                if rnd.random() < 0.35:
                    c = rnd.choice([(255, 214, 140), (200, 220, 255), (255, 240, 200)])
                    d.rectangle((wx, wy, wx + 5, wy + 7), fill=c)
        if rnd.random() < 0.25:
            d.ellipse((x + bw / 2 - 5, top - 10, x + bw / 2 + 5, top), fill=(120, 170, 255))
        x += bw + rnd.randint(4, 30)
    sky = img.crop((0, 0, w, water))
    refl = sky.transpose(Image.FLIP_TOP_BOTTOM).resize((w, h - water)).filter(ImageFilter.GaussianBlur(4))
    refl = Image.blend(refl, Image.new("RGBA", refl.size, (8, 14, 36, 255)), 0.45)
    img.paste(refl, (0, water))
    return img.convert("RGB")


def screen_default(w=1920, h=1080):
    """What the big studio screen shows between stories: channel ident on a dotted field."""
    img = gradient(w, h, (12, 40, 90), (8, 22, 56))
    d = ImageDraw.Draw(img)
    rnd = random.Random(5)
    for y in range(40, h - 200, 26):
        for x in range(40, w - 40, 26):
            n = math.sin(x * 0.004) * math.cos(y * 0.006) + 0.3 * math.sin(x * 0.011 + y * 0.004)
            if n > 0.15 and rnd.random() < 0.9:
                d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(60, 120, 200))
    d.rectangle((0, h - 190, w, h), fill=CRIMSON)
    ar_text(d, (w / 2, h - 95), CHANNEL, KUFI_B, 96, (255, 255, 255), anchor="mm")
    return img.convert("RGB")


def desk_front(w=1500, h=760):
    """Texture for the front of the 3D desk."""
    img = Image.new("RGBA", (w, h), NAVY + (255,))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 70, w, 90), fill=GOLD)
    ar_text(d, (w / 2, 260), CHANNEL, KUFI_B, 150, SAND, anchor="mm")
    d.rectangle((0, 0, w - 1, h - 1), outline=INK, width=10)
    return painterly(img).convert("RGB")


def logo():
    img = Image.new("RGBA", (420, 150), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((6, 6, 414, 96), 18, fill=CRIMSON, outline=GOLD, width=5)
    ar_text(d, (210, 50), CHANNEL, KUFI_B, 44, (255, 255, 255), anchor="mm")
    d.rounded_rectangle((280, 104, 414, 146), 10, fill=(220, 20, 30))
    ar_text(d, (400, 124), "مباشر", KUFI_B, 26, (255, 255, 255))
    d.ellipse((294, 117, 308, 131), fill=(255, 255, 255))
    return img


def lower_third(name=NAME, title=TITLE):
    w = int(max(ar_width(name, KUFI_B, 50), ar_width(title, NASKH_B, 32)) + 120)
    img = Image.new("RGBA", (w + 30, 150), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, 84), fill=CRIMSON)
    d.rectangle((w, 0, w + 30, 84), fill=GOLD)
    ar_text(d, (w - 30, 40), name, KUFI_B, 50, (255, 255, 255))
    d.rectangle((60, 84, w, 140), fill=SAND)
    ar_text(d, (w - 30, 111), title, NASKH_B, 32, NAVY)
    return img


def ticker_bar():
    img = Image.new("RGBA", (W, 70), NAVY + (255,))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 5), fill=GOLD)
    return img


def ticker_tag():
    """'Breaking' tag; sits above the scrolling text so headlines slide under it."""
    img = Image.new("RGBA", (190, 65), CRIMSON + (255,))
    ar_text(ImageDraw.Draw(img), (160, 33), "عاجل", KUFI_B, 36, (255, 255, 255))
    return img


def ticker_text():
    """Long strip of headlines; the video slides it left-to-right under the bar's label."""
    gap = 90  # room for a gold diamond between headlines (Naskh has no bullet glyphs)
    items = HEADLINES * 4
    widths = [int(ar_width(t, NASKH_B, 34)) for t in items]
    w = sum(widths) + gap * len(items)
    img = Image.new("RGBA", (w, 70), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = w  # right-to-left
    for t, tw in zip(items, widths):
        ar_text(d, (x - gap / 2, 38), t, NASKH_B, 34, (255, 255, 255))
        x -= tw + gap
        cx = x
        d.polygon([(cx, 26), (cx + 11, 37), (cx, 48), (cx - 11, 37)], fill=GOLD)
    return img


def news_image_dust(w=1600, h=900):
    """Placeholder story image: dust storm over dunes and a city."""
    img = gradient(w, h, (196, 128, 62), (236, 190, 120))
    d = ImageDraw.Draw(img)
    d.ellipse((w * 0.62, h * 0.12, w * 0.62 + 160, h * 0.12 + 160), fill=(250, 226, 170))
    sky = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    skyline(sky, (100, int(h * 0.3), w - 100, int(h * 0.72)), (150, 90, 52, 200), seed=9)
    img.alpha_composite(sky)
    for i, (y, c) in enumerate([(0.68, (214, 150, 80)), (0.78, (190, 118, 56)), (0.9, (160, 92, 40))]):
        pts = [(x, h * y + 40 * math.sin(x / (180 + 60 * i) + i)) for x in range(0, w + 20, 20)]
        d.polygon(pts + [(w, h), (0, h)], fill=c)
    haze = Image.new("RGBA", (w, h), (226, 170, 100, 0))
    hd = ImageDraw.Draw(haze)
    rnd = random.Random(2)
    for _ in range(40):
        x, y, r = rnd.uniform(0, w), rnd.uniform(h * 0.2, h), rnd.uniform(80, 260)
        hd.ellipse((x - r, y - r * 0.5, x + r, y + r * 0.5), fill=(230, 176, 110, 60))
    img.alpha_composite(haze.filter(ImageFilter.GaussianBlur(40)))
    return painterly(img)


def news_image_dates(w=1600, h=900):
    """Placeholder story image: date stalls at the Friday market."""
    img = gradient(w, h, (236, 196, 140), (196, 140, 86))
    d = ImageDraw.Draw(img)
    rnd = random.Random(4)
    for k in range(3):  # awning
        d.polygon([(k * w / 3, 0), ((k + 1) * w / 3, 0), ((k + 1) * w / 3 - 40, 160), (k * w / 3 + 40, 160)],
                  fill=(CRIMSON, SAND, (40, 90, 120))[k])
    for row in range(3):
        y0 = 360 + row * 170
        for c in range(5):
            x0 = 80 + c * 300
            d.rectangle((x0, y0 + 70, x0 + 260, y0 + 150), fill=(140, 92, 50), outline=INK, width=4)
            for _ in range(140):  # heap of dates
                x = rnd.gauss(x0 + 130, 55)
                y = y0 + 70 - abs(rnd.gauss(0, 30))
                d.ellipse((x - 9, y - 6, x + 9, y + 6), fill=rnd.choice([(92, 40, 22), (120, 56, 26), (70, 30, 18)]))
    return painterly(img)


def news_image_roundabout(w=1600, h=900):
    """Placeholder story image: the old roundabout, cracked and waiting for repair."""
    img = gradient(w, h, (226, 204, 160), (206, 180, 132))
    d = ImageDraw.Draw(img)
    cx, cy = w / 2, h / 2 + 40
    for r, c in ((380, (90, 92, 96)), (230, (226, 204, 160)), (210, (120, 160, 90))):
        d.ellipse((cx - r * 1.4, cy - r * 0.6, cx + r * 1.4, cy + r * 0.6), fill=c)
    for ang in (0, 90, 180, 270):
        a = math.radians(ang)
        d.line([(cx + 380 * 1.4 * math.cos(a), cy + 380 * 0.6 * math.sin(a)),
                (cx + 900 * math.cos(a), cy + 600 * math.sin(a))], fill=(90, 92, 96), width=110)
    d.rectangle((cx - 20, cy - 260, cx + 20, cy), fill=(200, 170, 110), outline=INK, width=4)  # monument
    d.ellipse((cx - 60, cy - 330, cx + 60, cy - 240), fill=GOLD, outline=INK, width=4)
    rnd = random.Random(8)
    for _ in range(14):  # cracks
        x, y = cx + rnd.uniform(-500, 500), cy + rnd.uniform(-200, 200)
        pts = [(x, y)]
        for _ in range(5):
            x, y = x + rnd.uniform(-40, 40), y + rnd.uniform(-25, 25)
            pts.append((x, y))
        d.line(pts, fill=(40, 40, 42), width=3)
    return painterly(img)


def story_card(story, caption, w=1920, h=1080):
    """Full-screen story card for the studio screen: picture plus an Arabic caption band."""
    img = Image.new("RGBA", (w, h), INK + (255,))
    img.paste(story.resize((w, h - 170)), (0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, h - 170, w, h), fill=CRIMSON)
    d.rectangle((0, h - 176, w, h - 170), fill=GOLD)
    ar_text(d, (w - 70, h - 85), caption, KUFI_B, 78, (255, 255, 255))
    return img.convert("RGB")


def screen_frame(story, caption):
    """LED screen showing a story image with an Arabic caption strip."""
    w, h = 900, 560
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w, h), 20, fill=INK)
    pic = story.resize((w - 24, h - 24 - 70))
    m = Image.new("L", pic.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, *pic.size), 14, fill=255)
    img.paste(pic, (12, 12), m)
    d.rectangle((12, h - 12 - 70, w - 12, h - 12), fill=CRIMSON)
    ar_text(d, (w - 34, h - 12 - 35), caption, KUFI_B, 34, (255, 255, 255))
    return img


# ------------------------------------------------------------------ figure

def figure_nosmoke(src):
    """Hide the cigarette smoke column for desk shots (the hand is behind the desk)."""
    im = Image.open(src)
    a = np.asarray(im).copy()
    a[1040:1440, :765, 3] = 0
    Image.fromarray(a).save(FIG / "figure_nosmoke.png")


# ------------------------------------------------------------------ layouts

def layouts():
    """Each option: ordered layers, pos = top-left on the 1920x1080 frame, scale for the figure."""
    lt = {"file": "assets/studio/lower_third.png", "role": "lower_third"}
    tick = [{"file": "assets/studio/ticker_bar.png", "pos": [0, 1010]},
            {"file": "assets/studio/ticker_text.png", "pos": [1730, 1010], "align": "right", "role": "ticker"},
            {"file": "assets/studio/ticker_tag.png", "pos": [1730, 1015]}]
    lg = {"file": "assets/studio/logo.png", "pos": [1460, 40]}
    return {
        "A_desk": [
            {"file": "assets/studio/bg_studio.png", "pos": [0, 0]},
            {"figure": "assets/haj_salama", "image": "figure_nosmoke.png", "pos": [82, -260], "scale": 0.75},
            {"file": "assets/studio/desk.png", "pos": [330, 770]},
            lg, dict(lt, pos=[1920 - 60 - 0, 860], align="right"), *tick],
        "B_closeup": [
            {"file": "assets/studio/bg_studio_close.png", "pos": [0, 0]},
            {"figure": "assets/haj_salama", "image": "figure_nosmoke.png", "pos": [-270, -380], "scale": 1.0},
            lg, dict(lt, pos=[1920 - 60, 860], align="right"), *tick],
        "C_wall": [
            {"file": "assets/studio/bg_studio.png", "pos": [0, 0]},
            {"file": "assets/studio/screen_dust.png", "pos": [120, 130], "role": "screen"},
            {"figure": "assets/haj_salama", "image": "figure_nosmoke.png", "pos": [806, -82], "scale": 0.55},
            {"file": "assets/studio/desk_small.png", "pos": [1080, 680]},
            lg, dict(lt, pos=[120, 860], align="left"), *tick],
        "D_majlis": [
            {"file": "assets/studio/bg_majlis.png", "pos": [0, 0]},
            {"figure": "assets/haj_salama", "image": "source.png", "pos": [702, 50], "scale": 0.5},
            {"file": "assets/studio/logo.png", "pos": [80, 40]},
            dict(lt, pos=[60 + 0, 860], align="left"), *tick],
    }


def place(entry, size):
    """Resolve right-aligned positions now that the image size is known."""
    x, y = entry["pos"]
    if entry.get("align") == "right":
        x -= size[0]
    return int(x), int(y)


def compose(layout):
    frame = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    for L in layout:
        if "figure" in L:
            im = Image.open(ROOT / L["figure"] / L["image"])
            s = L["scale"]
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            _paste(frame, im, L["pos"])
        else:
            im = Image.open(ROOT / L["file"])
            _paste(frame, im, place(L, im.size))
    return frame


def _paste(frame, im, pos):
    x, y = int(pos[0]), int(pos[1])
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    layer.paste(im, (x, y))
    frame.alpha_composite(layer)


def main():
    STUDIO.mkdir(parents=True, exist_ok=True)
    studio_bg().save(STUDIO / "bg_studio.png")
    studio_bg().resize((int(W * 1.3), int(H * 1.3))).filter(ImageFilter.GaussianBlur(6)) \
        .crop((int(W * 0.15), int(H * 0.05), int(W * 0.15) + W, int(H * 0.05) + H)).save(STUDIO / "bg_studio_close.png")
    studio_bg(warm=True).save(STUDIO / "bg_majlis.png")
    desk(1260, 250).save(STUDIO / "desk.png")
    desk(800, 340).save(STUDIO / "desk_small.png")
    desk_front().save(STUDIO / "desk_front.png")
    city_night().save(STUDIO / "city_night.png")
    screen_default().save(STUDIO / "screen_default.png")
    logo().save(STUDIO / "logo.png")
    lower_third().save(STUDIO / "lower_third.png")
    ticker_bar().save(STUDIO / "ticker_bar.png")
    ticker_text().save(STUDIO / "ticker_text.png")
    ticker_tag().save(STUDIO / "ticker_tag.png")
    dust = news_image_dust()
    dust.save(STUDIO / "story_dust.png")
    screen_frame(dust, "موجة غبار تجتاح المنطقة").save(STUDIO / "screen_dust.png")
    story_card(dust, HEADLINES[0]).save(STUDIO / "card_dust.png")
    story_card(news_image_dates(), HEADLINES[1]).save(STUDIO / "card_dates.png")
    story_card(news_image_roundabout(), HEADLINES[2]).save(STUDIO / "card_roundabout.png")
    figure_nosmoke(FIG / "source.png")

    (ROOT / "layouts").mkdir(exist_ok=True)
    (ROOT / "mockups").mkdir(exist_ok=True)
    thumbs = []
    for name, layout in layouts().items():
        (ROOT / "layouts" / f"{name}.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
        frame = compose(layout).convert("RGB")
        frame.save(ROOT / "mockups" / f"{name}.png")
        t = frame.resize((960, 540))
        ImageDraw.Draw(t).text((16, 12), name, fill=(255, 255, 0),
                               font=ImageFont.truetype("DejaVuSans-Bold.ttf", 30))
        thumbs.append(t)
    sheet = Image.new("RGB", (1920, 1080))
    for i, t in enumerate(thumbs):
        sheet.paste(t, (960 * (i % 2), 540 * (i // 2)))
    sheet.save(ROOT / "mockups" / "sheet.png")
    print("ok")


if __name__ == "__main__":
    main()
