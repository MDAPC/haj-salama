# Performance capture: a real speaker drives Haj Salama

Status: working end to end on a stock clip (`episodes/capture01`). Free, local, CPU only.

## How it works now

1. **Trace** a face video with MediaPipe Face Landmarker (Apache-2.0, ~2x real time on this CPU):
   `python tools/capture/trace_face.py VIDEO face_trace.json --face largest [--body]`
   Per frame: lip gap and mouth width (normalised by eye distance), 27 ARKit-style blendshapes
   (jaw, funnel/pucker, press, smile, blinks, gaze, brows), head yaw/pitch/roll.
2. **Mouth space** (`tools/capture/mouth_model.py`): every mouth, whether a speech sound, a frame of
   real face or a painting, is a point (open, spread, round). Paintings are placed in `rig.json`
   `mouth_space`. The traced face is mapped using the speaker's own ranges, so any person works.
3. **Rig** (`tools/relief/build_relief.py`): if the episode folder has `face_trace.json`, the mouth,
   blinks and head motion come from it; otherwise from the speech sounds. Each frame shows the
   nearest painting (with hysteresis and a 2-frame minimum hold), and a small jaw warp covers the
   gap to that painting. Paintings tied to one sound (m/b, f, l, oo) are reserved for that sound.

## What the stock trace taught us (Zeinab Takouti, 2 min, Arabic)

- Mouth opens and closes about **2.2 times a second** (syllable rate).
- Median open-axis speed ~1.1/s in our units; speech-driven Haj Salama is now tuned slower
  (median ~0.6/s) because a calm anchor should move less than a lively interviewee.
- Real mouths spend most time **half open**, not at the extremes; full closures are brief.
- Rounding (funnel/pucker) is short and spiky, matching u/w/o, which is why rounded paintings are
  only shown on their sounds.

## A filmed speaker in one command (Oct 25)

    python tools/capture/from_video.py VIDEO NAME [--start S] [--dur D] [--ref stand01] [--frames N]

(`--ref stand01` by default, `--frames N` for a quick partial render). Makes `episodes/NAME/` with
`source.mp4` (trimmed, upright, 30 fps), `voice.wav`, `face_trace.json` (face + body), prints a QC
line (face / shoulders / each wrist / fingers found, with advice when too low), renders the figure
and writes `out/side_by_side.mp4` (speaker left, Haj Salama right). Lips follow the face trace and
Rhubarb on the recorded voice; head, brows, blinks from the face; wrists choose the gestures (Veo
takes on the standing rig). Tested on the Wales clip (213-221 s): 6 s in ~6 min. The guide for the
person filming is a bilingual page (Artifact "Haj Salama Filming Guide").

## Recording a performance (for later)

- Phone or webcam, face to camera, chest up, even light, 30 fps or more. Read the Arabic script.
- Put the recording's audio as `episodes/<name>/voice.wav` and run the trace on the video into
  `episodes/<name>/face_trace.json` (same start time). Then the normal command:
  `python anchor.py episodes/<name> --lang ar --relief assets/haj_relief/ref10`
- Your voice becomes his voice; your mouth timing, blinks and nods become his.

## Next steps

1. **Fill the gaps in the mouth space.** Done once (5 paintings, 17 in total): poorly covered
   frames 49% -> 25% for the real speaker, 14% -> 6% for speech. Repeat with each new speaker:
   the remaining gaps are very wide open spread and very narrow open shapes.
2. **Hands and shoulders**: done (`trace_face.py --body`, MediaPipe Pose + Hands, ~0.5x real time).
   Per frame: shoulder line relative to the chin/face, shoulder tilt, both wrists' height and
   visibility, and each hand's openness (fingertip reach / palm length).
   - Shrug: quick drops of the shoulder line relative to the face (2.5 s high-pass).
   - Breathing: the slow shoulder swing, mixed 30/70 with his idle breathing (in interview footage
     the swing is mostly gesture motion, so the trace only colours it).
   - Chest tilt: follows the shoulder line's slope.
   - Gestures (6 paintings: clasped, halfway, open palm, pointing finger, both palms up, touching
     the tablet). The trace picks one per frame: a hand raised with only the index finger out
     (per-finger reach / palm length) -> point; both wrists up -> both; one up -> open; partly up
     -> halfway; a downward glance of 0.4 s+ with hands low -> tablet. Paintings change one step at
     a time through their in-between (clasped -> halfway -> gesture), in-betweens held 3 frames,
     gestures 6. Without a trace: a calm gesture on some longer phrases (at most one per ~6 s,
     cycling open / point / both) and a tablet touch at some sentence starts.
   - Body paintings (`hand_half`, `hand_open`) use a "body" mask; where they show the green screen
     the reference's own arm is made transparent (`var_*_hide.png`), and the raised hand outside his
     outline gets its own surface on the chest bone.
3. **Brows and gaze**: done. Gaze from both eyes' look blendshapes (relative to the speaker's own
   resting gaze, up to 6 px of iris travel), brows from browInnerUp/OuterUp minus browDown. Blinks are
   real closures only, shown as clean 2-5 frame cuts (fading two eye paintings double-exposes them).
4. **Live preview**: run the trace on a webcam stream and show the chosen painting live, for
   rehearsing a performance before rendering.
5. Mix: lips from the audio (exact for TTS), head/brows/blinks from a recorded performance.

Source clip: Wikimedia Commons, "Wikimedia Strategy 2030 - Roles & Responsibilities - Interview
Zeinab Takouti (AR)", Anass (WMF), CC BY 3.0 (`assets/capture/stock/SOURCE.md`).

## Less mechanical motion (Oct 24)

- No more sine waves: head drift, tilt, chest turn/roll and resting-hand drift are smooth random
  motion (`drift()`), so nothing repeats on a fixed period.
- Breathing has an uneven 3.5-5 s rhythm plus a quick inhale just before each phrase.
- Finger taps happen at irregular moments while talking.
- Follow-through: every change of arm pose kicks an underdamped spring that leans the chest (and
  head) into the gesture, overshoots and settles; settling back to clasped hands does the reverse.
- With a trace, the chest also leans with the speaker (shoulder width grows when he leans in) and
  turns partly with the head.
- Arm-pose changes get one 50/50 blended frame (reads as motion blur at 30 fps; `gesture_mix`).
- Tested on a second speaker so the tuning isn't fitted to one person: Jimmy Wales (Wikimania 2025,
  CC BY 4.0), `episodes/capture03`. He gestures over half the time; all four gestures show up.

## Oct 24, second pass

- Head-turn glitch fixed: the tracker exaggerates fast turns (Wales read as +-40 deg at 370 deg/s), and
  fine depth detail tore at the cheek edge. Head angles now go through a soft limit and a speed limit,
  and the depth surface is smoothed more.
- Raised arm has its own bone at the elbow (`gesture_arm` in rig.json): while a gesture holds, the arm
  swings with the speaker's wrist (or with speech beats), and arrives with an overshoot.
- Shemagh sway weight is a smooth field around the face (a colour mask left loose pixels behind).
- Gesture patches exclude the head and shemagh, so Gemini's repainted cloth edges can't ride the arm.
- The 50/50 blend frame on arm changes is off (it looked like a ghost arm); `BLEND_ARM_CHANGES`.

## Oct 24, third pass

- Resting pose is a new painting with both hands flat on the desk (`hands_rest`, edited from
  hand_open in Gemini), and every pose that keeps the left hand down gets the exact same left-hand
  pixels (`left_hand_from`), so the left hand never jumps between paintings. The clasped hands of
  ref10 are no longer used while gesture paintings exist.
- Gemini's green desk tint is removed from all patches below `green_above`.
- Gestures need a wish that lasts 8+ frames and hold at least 12, so no more 5-frame pops.
- Shemagh is two-part cloth: cap at the crown and hanging flaps (`shemagh_tail`, pivot at
  `shemagh_tail_y`), each a damped pendulum driven by the head's angular acceleration (turns,
  tilts, nods, gesture follow-through); the flaps are slower and looser, plus a little air.

## Oct 24, fourth pass: the "mirage"

Measured with optical flow (non-rigid motion = what an affine move can't explain), median per frame:
torso 0.18 -> 0.07 px, left hand 1.1 -> 0 px, shemagh 0.73 -> 0.42 px, face 1.0 -> 0.5 px.
- Forearms and hands on the desk are pinned (`desk_y`): chest motion and breathing fade out above the desk.
- The part of a gesture painting outside his outline is split at the desk line: above it rides the
  arm bone, below it is fixed. Before, the flat left hand partly rode the right arm's swing.
- Body turn/roll drift cut to a quarter (turning a depth relief warps it everywhere), resting-hand
  drift removed, finger taps only with clasped hands.
- Anti-aliasing 8 -> 24 samples so fine painted lines don't shimmer.
- "Both palms up" is out of the chooser: it is the only pose that moves both arms, so every switch
  popped the left hand, and Gemini would not paint a usable low in-between.

## Oct 24, arm in-betweens

Tried three ways to fill the jump between arm paintings (`tools/relief/arm_tweens.py`):
- Optical flow (Farneback) morph: the hand travels too far, so it cross-fades into two ghost hands.
- RIFE v4.6 (rife-ncnn-vulkan, MIT, `python tools/setup.py --rife` puts it in `.tools/`, ~1 s per frame):
  good for small changes where the hand mostly changes shape (halfway -> open palm, halfway ->
  pointing); used there, 2 frames per switch (1/3, 2/3).
- Cut-out: rotate the raised forearm about the elbow back toward the lower pose. Tore in the
  render (the removed arm leaves outline residue, patch masks clip the rotated hand); not used.
Big swings (resting <-> halfway, halfway <-> tablet) stay clean cuts. Getting those smooth needs more
painted poses (e.g. a quarter-raised arm from Gemini), not interpolation.
The tablet gesture now goes via the halfway pose.

## Oct 24, calm gestures (after docs/MOTION_RESEARCH.md)

Measured: Haj Salama's continuous motion was already slower than the speaker's (head 0.20 vs 0.79
face-widths/s); what read as exaggerated was ~38 instant arm-pose changes in 8 s. Now gestures are
events (`calm_gestures`, settings in `CALM`): flickers merged (gaps < 0.7 s), gestures < 0.9 s dropped,
1.5 s of rest between gestures, long stretches split into ~2.4 s phrase gestures with one kind each,
and each gesture's start moved so the hand arrives on the nearest stressed syllable (within 0.4 s).
Poses hold at least 18 frames. Elbow swing cut to +-0.05 rad and slowed; lean-in and arm overshoot
halved; head from the trace smoothed (0.22 s). Result on capture03: 3 arm changes in 8 s instead of ~38.
Gemini would not paint a low-raised hand (3 tries: hand_quarter_g1/g2 came out at the halfway height).

## EMAGE as a speech-only performer (NON-COMMERCIAL)

`tools/capture/emage_trace.py` runs EMAGE (PantoMatrix, via the MIT VolgaGerm/emage-onnx-export) on CPU
with onnxruntime: 29 s of Arabic speech -> 30 fps SMPL-X motion in ~19 s. Forward kinematics on an
approximate SMPL skeleton (the 6D rotations are rows, not columns), measured in the upper-torso frame,
written as a body-only trace ("face": false) so lips, blinks, brows and gaze stay on the speech.
EMAGE's standing speakers gesture almost continuously with hands near the chest, so wrist height is
rescaled from its own range, and the calm rules cap gestures at 3 s with 1.5 s rest between.
Install: `.tools/emage/emage-onnx-export` (its own venv, Python 3.10, CPU torch for the one-time export; see docs/SETUP.md).
Licence: BEAT2 training data and SMPL-X are non-commercial; experiments only. Episode: episodes/emage01.

## Standing figure for EMAGE, on green screen (Oct 24)

EMAGE is trained on standing speakers, so Haj Salama now also exists standing: `refs/stand01.png`
(Gemini, full body to the knees, arms hanging) rigged as `assets/haj_relief/stand01`. The owner wants
plain chroma green behind him, no studio: the plate is solid RGB (0, 177, 64), `frame.json` has
`"green": true`, and anchor.py then skips the channel graphics (ticker, logo, strap).
Paintings (gen/s_*.jpeg): mouths closed, a, ee, oo, o (the reference's own mouth is "base", slightly
open with teeth); eyes closed; gestures per hand (_R = his right, screen left; _L = his left):
half_R -> open_R / point_R, half_L -> open_L / present_L, both, and clasp (hands folded, used in some
longer pauses). Gemini paints studio floor (not green) under the screen, so gesture paintings are cut
out with rembg (`"matte": true` in prep_variants.py) before the "hide where the arm left" step.
Body constants that were seated-only are rig keys now: chest_y (pivot), torso_top, shrug_top.

Varied gestures (`calm_gestures`): gesture length, rest length and phrase chunks are drawn from ranges
(CALM: 1.8-3.6 s, 0.9-2.2 s, 1.6-3.0 s, seeded) so there is no regular rhythm; the same pose never
comes twice in a row and 30% of the time another pose of the same hand stands in (point and both
weighted down as emphatic); rests of 2.5 s+ have a 50% chance to be spent with hands clasped.
The gesture timeline of each render is written to out/relief_<ref>/gestures.txt.
Episode: episodes/emage02.

## Veo (Gemini video) as a reference performance (Oct 24)

`assets/haj_salama/veo/veo01.mp4`: Gemini "Create video" (Veo, Pro account, 3 videos/day) from
stand01 on plain green, 10 s, 1280x720, 24 fps. Prompted script: right open palm, left side present,
both palms, hands folded. Veo kept his likeness and framing; it used the right hand for the side
gesture too. MediaPipe traces the caricature fine (face in 239/240 frames; `veo01_trace.json`,
`--body`), and the trace separates raise, both-palms and clasp (wrists move to the centre).
Measured: an arm move takes ~0.5-0.7 s, holds ~1-1.5 s (ours switch in one frame).
Its in-between frames (`veo01_frames/`, upscaled to 1920x1080; `veo01_rise_strip.png`) are the
low/quarter-raised arm poses Gemini stills refused to paint. He leans while gesturing (up to ~37 px
at the face), so frames must be aligned on the torso, not the face, before use as paintings.

### Veo library (Oct 24, clips 2-4)

Gemini allows a few Veo videos a day (limit hit after 4 on Oct 24; it resets the next morning).
A close-up prompt was refused once ("reacting to bad news... annoyed, eyes rolling"); a gentler
wording ("thoughtful cartoon presenter... doubtful, concerned, disappointed") went through.
Veo prefers his right hand even when asked for the left: give it the reference mirrored, ask for the
right hand, flip the result back (`ffmpeg -vf hflip`) -> true left-hand gestures (veo04_left.mp4).

| clip | what is in it |
|---|---|
| veo01.mp4 | R open palm, R side present, both palms, hands clasped |
| veo02.mp4 | frown + head shake, R stop palm, R stern pointing finger, both-hands shrug, sad sigh |
| veo03_face.mp4 | close-up: frown, stern, doubtful side glance, skeptic brow, worried, sad, eyes closed |
| veo04_left.mp4 | L open palm -> present to the side -> calming palm, both calming, L hand on chest |

`tools/capture/veo_library.py CLIP TRACE REF REF_DIR OUT` -> `lib0N/`: per frame the body below the
head keyed off the green, moved so the torso sits on the reference's (Veo's torso moves only a few px;
the head nods), colour-matched; a hide mask (where the reference's arm must vanish); gestures.json
(segments found from the trace: R, L, both, clasp, point_R/L) and gestures.png.
`tools/capture/veo_expressions.py` picks the peak frame of each expression from the close-up's
blendshapes and maps it back onto the reference; rig variants expr_frown, expr_stern, expr_worried,
expr_skeptic, expr_sad, expr_closed (region upper_face: brows, eyes, forehead; the mouth stays on
lip-sync). An episode's expressions.json ([{"t", "dur", "name"}]) fades them in and out (0.1 s; a longer cross-fade doubles the brows).

### Veo gestures played in the rig (Oct 25)

`tools/capture/veo_pool.py OUT REF_DIR LIB...` numbers the frames of all libraries into one pool
(`assets/haj_salama/veo/pool`, symlinks) and cuts "takes": one gesture from rest back to rest
(families R, L, both, clasp; point/shrug chains stay one take when the hand never returns between).
With rig key `veo_pool`, build_relief.py skips the painted gesture poses and adds one Veo body layer
whose texture is an image sequence; its frame offset is keyframed per output frame, so any pool frame
can show on any frame (the matching hide mask is a second sequence on the relief). `veo_track` turns
the planner's gestures into takes: never the same take twice in a row, takes whose natural length
fits are preferred, a short gesture borrows up to half again from the rest after it; rise and fall
play at natural speed (24 -> 30 fps), the hold is stretched or squeezed (at most 1.6x); a hold that
does not fit, or a take that never returns to rest, plays part of its hold and comes down along its
rise reversed. No fade at take edges (Veo's resting hand sits a little off ours; a cross-fade showed
both). Each render lists the takes used in out/relief_<ref>/gestures.txt.

### First real speaker (Oct 25): episodes/speaker01

A 37 s WhatsApp video (480x848 portrait, seated, chest up; sent as a video, so compressed).
QC: face 100%, shoulders 100%, wrists 35-45% (hands below the frame), fingers 32%. Still usable:
his two-handed chest gestures became both-palms takes, a few one-hand moves became R/L takes.
Full render 37 s in ~35 min. Fixed on the way: frames past the pool length (720) showed Blender's
missing-texture magenta (an image sequence's frame is clamped to its length before the offset is
added; the declared length is now huge), and set-order randomness in the gesture choice (now sorted).
`--from N` on build_relief.py re-renders only frames N onward. With 2 two-handed takes in the pool
they alternate 5 times in 37 s: more Veo takes needed, especially two-handed ones.

### Standing fixes after speaker01 (Oct 25)

- **Rocking like a stick**: the body bone pivoted near the feet, so every lean tilted the whole man.
  Rig keys now: `chest_y` 800 (pivot at the waist), `hips_y` 800 (everything below stays planted,
  soft 120 px band), `body_sway` 0.5 (lean and side sway halved). The bigger part was the speaker's
  slow posture drift (seated, leaning toward the phone) copied as head pitch: on a standing figure
  with `body_sway` < 1 only quick nods are kept (pitch high-passed over 1.2 s). Head-top travel on
  frames 80-340: 52 px -> 21 px (std 15 -> 5.4).
- **Hands popping at take start/end**: take edges came from wrist positions, but in some gestures
  the wrists barely move while the palms turn. `veo_pool.py` now widens each take to where the body
  picture itself is back at rest (`deviation`: pixels the arm left + pixels outside the silhouette).
- **Shemagh scraps riding on the body layer**: Veo's head nods, so bits of its shemagh fell outside
  the head cut-out. `veo_library.py` cuts a wider head (VEO_HEAD) and removes shemagh red above the
  shoulders that is connected to the head area (colour alone also took fingers: skin overlaps
  shemagh red; a raised hand is separated from the head by background, so it is kept).
- **No take twice in a row**: the "never the same take again" rule was per gesture kind, so open_L
  and half_L could play the same left-hand take back to back; it now remembers the previous take
  whatever the kind. speaker01 re-rendered in full (episodes/speaker01_fix, 11-take pool): no
  missing textures, edge clean, head-top travel 22 px over 37 s, two-handed gestures use 4 takes.
