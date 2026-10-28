# Veo prompts for gesture takes

Gemini Pro (gemini.google.com), Create video, start frame uploaded. About 3–4 videos a day;
the limit message gives the reset time. After downloading:

    python tools/capture/veo_add.py Downloads/gemini_generated_video_XXXX.mp4 NAME [--mirrored] [--about "..."]

Start frames in `assets/haj_salama/veo/start/`:
- `stand01_green.jpg`: refs/stand01 cut out (haj_relief/stand01/rgba.png) over RGB (0, 177, 64)
- `stand01_close.jpg`: face close-up (crop 480,40-1440,580, doubled) for expressions
- `stand01_mirror.jpg`: stand01_green mirrored. Veo prefers the right hand; ask for "his right hand"
  on the mirrored picture, and `--mirrored` flips the clip back -> a true left-hand take.

Shared opening for every prompt:
> Animate this exact cartoon character, same painted style. Locked-off static camera, no zoom, no
> cuts, same framing, plain flat green background. A calm TV news presenter standing in place,
> talking to the camera. Keep his face, red and white shemagh, beige jacket and white thobe identical.
> No text on screen. Slow, settled, natural movements.

One gesture per clip, done twice (a full version, then back to rest, then a smaller, shorter one),
so each clip gives two takes of different lengths.

## Used (Oct 24)
| clip | start frame | asked for |
|---|---|---|
| veo01 | green | R open palm, L side present (Veo used R), both palms, clasp |
| veo02 | green | frown + head shake, L open palm (Veo used R), R pointing, shrug, sad sigh |
| veo03_face | close-up | expressions (refused once; gentle wording went through) |
| veo04_left | mirrored | R open palm, side present, calming palm, hand on chest -> L after flip |

## Used (Oct 25)
| clip | start frame | result |
|---|---|---|
| veo05_count | green | R counts two on fingers (3 s take) + short one-finger (point_R) |
| veo06_both | green | hands fold, then both palms up (1.7 s) + smaller both palms |
| (left beats) | mirrored | failed 3 times: "beats" was refused by policy; reworded as "gently moves down and up to emphasize" it still failed twice (generic error / "hard time") |

Lessons (Oct 25): one video at a time (parallel ones all failed); generic errors and "hard time
fulfilling your request" often meant Veo was at capacity ("unusually high traffic... try again
later"); "beats" reads as violence, avoid it; refusals and errors do not use up the daily count;
the flowing one-paragraph style ("...comes back down to his side and rests; a little later the hand
rises again briefly...") works, "Then a shorter, smaller version:" as a separate sentence did not.

## Next
1. green: his right hand counts points on his fingers (one, two), then rests; then a short "one".
2. mirrored: his right hand makes small beats at chest height, palm sideways, with the words; rests; a smaller beat. (-> left)
3. green: both hands open low in front, palms up, a small "what can we do"; rests; a smaller one.
4. green: his right hand lifts slightly from his side, palm up, a short low gesture; rests; again, even smaller.
5. mirrored: his right hand points out to the side behind him, as if at a screen; rests; a shorter point. (-> left)
6. green: both hands come together in front of his chest, fingertips touching, then apart and down; rests; a shorter one.
