# Generated variants of ref10

Made with Gemini (a Gemini Pro account, in the browser), each as an edit of
`refs/ref10.png` attached fresh to every prompt. Output is 2242x1888 (same aspect as ref10, 1367x1151);
resize to ref10 size and the framing lines up, though Gemini slightly re-renders the whole picture, so
take only the changed region and blend it in.

| File | Use | Quality |
|---|---|---|
| mouth_closed_g1 | viseme A / X (m b p, rest) | very good |
| mouth_a_g2 | viseme C/D (calm "a") | very good |
| mouth_aa_g1 | — | too extreme (shouting face); only for exaggeration |
| mouth_ee_g1 | viseme B (ee, teeth) | very good |
| mouth_oo_g1 | viseme F (oo, pursed) | very good |
| mouth_f_g1 | viseme G (f v) | usable (teeth show, lip contact weak) |
| eyes_closed_g1 | blink | very good |
| smile_g1 | expression | very good |
| hand_open_g1 | right-hand open gesture | very good |
| hands_spread_g1 | — | only one hand moved; same as hand_open |
| head_left20_g1, head_right35_g2 | — | failed: Gemini kept the head pose (only eyes moved) |

Findings: Gemini edits faces and hands reliably but refuses to re-pose the head in an edit. Head turns
need a different route (new generation as a separate shot, or layered inpainting, see docs/RESEARCH.md).

`../gen_local/` holds local SD1.5 (DreamShaper 8 inpainting) tries: closed mouth and closed eyes are good;
open mouths made from rig poses came out distorted.

## Gap fill (from a traced real speaker)

`tools/capture` traced 2 min of a real Arabic speaker and showed where her mouth went that no painting
covered (`assets/haj_relief/ref10/mouth_gaps.png`). Five new edits fill those gaps, all very good:
mouth_pout_g1 (almost closed, slightly forward), mouth_midnarrow_g1 (half open, relaxed narrow oval),
mouth_slightwide_g1 (barely parted, corners wide), mouth_pursed_g1 (closed, tightly pursed),
mouth_osmall_g1 (small round opening). Poorly covered real-speaker frames fell from 49% to 25%
(`mouth_gaps_after.png`).

Gemini note: image mode switches itself off after some generations; check the prompt says
"Describe your image" before sending, or Gemini replies in text.

## Gestures

hand_half_g1 (halfway in-between), hand_open_g1, hand_point_g1, hands_both_g1, hand_tablet_g1: all edits of
ref10, all usable. Cut with the wide "body" mask and only where the pose changed (`only_changed`), so the
studio behind stays ref10's own. hands_both worked once the prompt insisted that BOTH hands leave the desk.
hands_rest_g1: both hands resting flat on the desk; edited from hand_open_g1 (not ref10) so its left hand
matches the gesture paintings. Used as the resting pose.
hands_both_g1 is no longer used: it is the only pose that moves both arms, so every switch in or out of it
popped the left hand. Two tries at a low in-between (hands_both_half_g1, hands_both_low_g1) failed: Gemini
either kept the hands high or re-framed the whole picture.
