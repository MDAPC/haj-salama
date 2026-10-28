# Research: natural, settled talking motion for Haj Salama

Date: 2026-10-24. The problem: body motion is exaggerated, too fast and uncoordinated. It follows the traced human but amplifies speed and distance. Constraints: free/open (licence stated exactly), local Linux, CPU or AMD iGPU, no paid APIs. Every link below was opened. Lines marked **Our assessment** are inferences, not claims from a source.

## TL;DR

- **No co-speech gesture model has commercially clean weights.** Every trained model sits on non-commercial data or a non-commercial body model: BEAT/BEAT2 are NC, ZEGGS is CC BY-NC-ND, Talking With Hands is CC BY-NC, and SMPL-X is NC. Code licences (MIT, Apache, BSD) don't change that. For a personal or non-commercial project, **EMAGE** is the best fit: audio-only (so Arabic audio goes in), and a community ONNX export already runs on CPU (even in a browser via WASM).
- **The evidence backs "less is more."** Doubling natural head motion threw away its benefit to intelligibility (Munhall 2004). Teleprompter readers move their heads less than people in conversation (Graf 2002). Presenter coaching says: "keep gestures small … avoid sudden body movement".
- **The biggest win needs no new model.** Stop replaying the trace frame by frame. Use the trace (or the TTS) only to choose events: *which* gesture, *roughly when*, *how strong*. Then **re-synthesise** each event with fixed, calm timing: minimum-jerk transitions, the stroke snapped to a stressed syllable, minimum hold, cooldown. Scale head/chest to ~30–50 % of the tracker's values and pass them through a critically damped spring.
- **The idle layer: copy Live2D's defaults.** Breath period ~3.2 s; head sway with periods 3.5 / 5.5 / 6.5 s; body sway 15.5 s; blinks 0.1 s closing / 0.05 s closed / 0.15 s opening.

---

## 1. Speech-driven gesture models (code + weights)

| Model | Code licence | Weights / data licence | CPU? | Input | Output |
|---|---|---|---|---|---|
| **EMAGE** (PantoMatrix, CVPR 2024) [repo](https://github.com/PantoMatrix/PantoMatrix) | none in repo (GitHub API: no licence → all rights reserved by default) | HF weights tagged apache-2.0 [card](https://huggingface.co/H-Liu1997/emage_audio), but trained on BEAT2, and the BEAT project page says "Licensed under the Non-commercial license" [BEAT](https://pantomatrix.github.io/BEAT/) | Yes. README uses `.cuda()`, but an MIT ONNX export runs it in onnxruntime-web WASM [emage-onnx-export](https://github.com/VolgaGerm/emage-onnx-export) | **Audio only** in the current codebase ("This new codebase only have the audio-only version") | SMPL-X 55-joint rotations (.npz) + FLAME face; 30 fps, 64-frame windows |
| **CaMN, DisCo** (same repo) | same | same (BEAT) | as EMAGE | audio | upper body + hands |
| **GestureLSM** (ICCV 2025) [repo](https://github.com/andypinxinliu/GestureLSM) | MIT | trained on BEAT2 (NC) | Not documented; 1-step "MeanFlow" variant is pitched for real time | audio + text | SMPL-X (BEAT2 format) |
| **SynTalker** [repo](https://github.com/RobinWitch/SynTalker) | MIT | BEAT-X + AMASS + HumanML3D (all research-licensed) | GPU assumed | audio + **text prompts** for upper/lower body ("sit", etc.) | SMPL-X .npy/.npz |
| **MambaTalk** (NeurIPS 2024) | Apache-2.0 (GitHub API) | BEAT2 (NC) | not checked | audio | SMPL-X |
| **DiffSHEG** (CVPR 2024) | BSD-3 (GitHub API) | BEAT (NC) | GPU real-time claim | audio | body + face |
| **DiffuseStyleGesture / +** [repo](https://github.com/YoungSeng/DiffuseStyleGesture) | MIT | ZEGGS (NC-ND), BEAT (NC), TWH (NC) | has `--no_cuda` flag, untested; needs WavLM-Large | audio, style label in filename ("Neutral", "Happy") | **BVH** |
| **ZeroEGGS** (Ubisoft) [repo](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS) | **CC BY-NC-ND 4.0** ([License.md](https://raw.githubusercontent.com/ubisoft/ubisoft-laforge-ZeroEGGS/main/License.md)) covers the repo | same | CUDA in instructions; model is small | audio + example-style clip (zero-shot style) | **BVH** |
| **TalkSHOW** [repo](https://github.com/yhw-yhw/TalkSHOW) | no licence; "For commercial licensing, please contact ps-licensing@tue.mpg.de" | SHOW data + SMPL-X | needs CUDA GPU | audio ("English, French, Songs so far") | SMPL-X |
| **Audio2Photoreal** (Meta) [repo](https://github.com/facebookresearch/audio2photoreal) | CC-NC 4.0, archived Aug 2025 | per-person models | CUDA + pytorch3d | 2-channel dyadic audio | photoreal avatar params |
| **Gesticulator** | GPL-3.0 (GitHub API) | Trinity data | – | audio + English text (BERT) | joint angles |
| **Semantic Gesticulator** | no licence | – | – | – | – |

The SMPL-X body model licence is "non-commercial scientific research, non-commercial education, or non-commercial artistic projects" only, with commercial licensing via Meshcapade ([SMPL-X licence](https://smpl-x.is.tue.mpg.de/modellicense.html)).

NVIDIA's Audio2Face-3D weights (NVIDIA Open Model License, commercial use allowed) are **face only**, not body ([HF](https://huggingface.co/nvidia/Audio2Face-3D-v3.0)). No open Audio2Gesture release was found.

**Our assessment:**
- **EMAGE via ONNX is the only option that is both CPU-proven and audio-only.** Rhythmic "beat" gestures come from prosody, so they should carry over to Arabic speech reasonably well. Semantic gestures (iconic, pointing on a keyword) will not, because training was on English (BEAT2 "English (Speaker 2)").
- **Mapping EMAGE output onto our rig:**
  - Run forward kinematics on the SMPL-X joint rotations with a generic skeleton. Approximate bone lengths are enough, so we don't ship the SMPL-X mesh.
  - Take wrist height and hand openness, and feed them through **the same chooser we already use for MediaPipe** (hands rest / halfway / open / point / tablet).
  - Take head (joint 15) and spine3/neck rotations, scaled ×0.3–0.5, into head yaw/pitch/roll and chest lean/turn/roll. Take collar rotation into shrug.
  - Use the output to pick events, not to play it back literally (see §5).
- **Licence bottom line:** fine for non-commercial art. For a monetised channel, treat every row above as non-commercial unless the owner gets permission.

## 2. Ready motion libraries

| Source | Licence (exact wording where found) | Talking / seated content |
|---|---|---|
| **BEAT** (76 h, 30 speakers, EN/ZH/ES/JA, 8 emotions, per-frame gesture semantic labels) [page](https://pantomatrix.github.io/BEAT/) | "Licensed under the Non-commercial license." HF dataset pages are tagged apache-2.0 ([BEAT](https://huggingface.co/datasets/H-Liu1997/BEAT), [BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2)), which **conflicts** with the project page; assume NC | standing talking mocap, BVH + 52 blendshapes |
| **CMU Mocap** [home](http://mocap.cs.cmu.edu/), [FAQ](http://mocap.cs.cmu.edu/faqs.php) | "free for all uses … You may include this data in commercially-sold products, but you may not resell this data directly, even in converted form." | category "communication gestures and signals" (mostly waving/signals, standing). Weak for seated talking |
| **Mixamo** ([Adobe FAQ thread](https://community.adobe.com/questions-696/mixamo-faq-licensing-royalties-ownership-eula-and-tos-589400)) | royalty-free for commercial and non-commercial projects including films/videos; "the only thing you can't do is distribute the raw character and animation files"; "can't be used in … training machine-learning models". Free with Adobe ID. (Official helpx page returned 403 to us; the thread quotes it) | has "Talking", "Sitting Talking", idle clips. Rendered video use is OK |
| **Rokoko Motion Library** ([CG Channel, 2020](https://www.cgchannel.com/2020/03/get-150-free-mocap-moves-from-rokokos-motion-library)) | "The free moves are licensed for use in commercial projects"; FBX/BVH/CSV export via free Rokoko Studio account. Current terms not verified | 150 free moves, content mixed |
| **Bandai Namco Research Motion dataset** [repo](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset) | **CC BY-NC 4.0** (scripts MIT) | bow, bye, guide, respond, call, in 15 styles; standing, no talking |
| **Talking With Hands 16.2M** (Meta) [repo](https://github.com/facebookresearch/TalkingWithHands32M) | "CC-BY-NC 4.0" | conversational mocap |
| **ActorCore** (Reallusion) | licence page returned empty; not verified | has "Sit Talk"/"Stand Talk" clips, mostly paid |

**Our assessment:**
- **Mixamo is the only commercially usable source with seated-talking clips.** Rendering it into our videos is allowed. Using it to train a model is not.
- **What it's good for:** retarget a "Sitting Talking" clip onto a generic skeleton and read head/chest/wrist curves as a calm reference. Copy the timing statistics (how long holds last, how slowly the chest drifts); don't copy the frames.

## 3. VTuber / 2D puppet tooling: parameters worth copying

- **Live2D Cubism SDK idle** ([sample lappmodel.ts](https://raw.githubusercontent.com/Live2D/CubismWebSamples/develop/Samples/TypeScript/Demo/src/lappmodel.ts), [cubismbreath.ts](https://raw.githubusercontent.com/Live2D/CubismWebFramework/develop/src/effect/cubismbreath.ts)):
  - Each parameter gets `offset + peak·sin(2π·t/cycle)`, added with a weight.
  - Defaults (offset, peak, cycle in s, weight):
    - AngleX 0, 15, **6.5345**, 0.5
    - AngleY 0, 8, **3.5345**, 0.5
    - AngleZ 0, 10, **5.5345**, 0.5
    - BodyAngleX 0, 4, **15.5345**, 0.5
    - Breath 0.5, 0.5, **3.2345**, 1
  - With weight 0.5, head yaw sways about ±7.5 units, on a model whose AngleX range is ±30.
  - The .5345 tails keep the periods out of sync, so the sum never visibly repeats.
- **Live2D eye blink** ([cubismeyeblink.ts](https://raw.githubusercontent.com/Live2D/CubismWebFramework/develop/src/effect/cubismeyeblink.ts)):
  - Phases: closing **0.1 s**, closed **0.05 s**, opening **0.15 s**.
  - The next blink comes at `random·(2·4.0 − 1)` s, i.e. uniform 0–7 s.
- **Live2D licences:**
  - SDK release licence is required only for businesses with revenue over ¥10M ([LICENSE](https://raw.githubusercontent.com/Live2D/CubismWebFramework/develop/LICENSE.md)).
  - Editor FREE may be used commercially by "General Users and Small-Scale Enterprises (annual sales less than 10 million yen)", with limits of 100 ArtMeshes and 30 parameters ([comparison](https://www.live2d.com/en/cubism/comparison/)).
  - Editor runs on Windows/macOS only.
- **VTube Studio** ([model settings wiki](https://github.com/DenchiSoft/VTubeStudio/wiki/VTS-Model-Settings)):
  - Per-parameter smoothing; "More smoothing will make the movements less shaky but will introduce lag".
  - Auto-breath and auto-blink.
  - "Limit Range" makes values "stop smoothly when approaching the limits" (a soft clamp, which we already do for the head).
  - Playing an idle animation overrides tracking.
- **Inochi2D** (BSD-2: [inochi2d](https://github.com/Inochi2D/inochi2d), [inochi-session](https://github.com/Inochi2D/inochi-session)):
  - Breathing is an expression, `usin(time())`.
  - A "Dampening" setting smooths expression output ([wiki](https://github.com/Inochi2D/inochi-session/wiki/Expressions)).
- **VSeeFace** ([site](https://www.vseeface.icu/)): free, commercial use allowed, not open source, Linux only via wine. Its tracker OpenSeeFace is BSD-2.
- **veadotube mini** ([site](https://veado.tube/)): free, Linux build, mic-volume "bounce" PNGtuber. Too crude for us.

**Our assessment:**
- **VTuber tools look calm because of a pattern, not a model.** They use:
  - a slow, low-amplitude idle layer that is always on;
  - tracking that is smoothed and range-limited;
  - discrete expression/pose states blended over a fixed time.
- That is the architecture to copy. Our `drift()` noise already replaces the sines; the Live2D periods give sensible time scales for it.

## 4. Craft and measurements on natural talking motion

- **Graf, Cosatto et al. 2002, "Visual Prosody"** ([PDF](https://www.cstr.ed.ac.uk/downloads/publications/2002/paper.vtts.pdf)). Speakers **reading from a teleprompter**, our exact case.
  - Head motion splits into 0–2 Hz posture drift ("caused by a change of posture … rather than being related to the speech") and 2–15 Hz speech-linked motion.
  - Accents are underlined by nods "that extend typically over two to four phones", synchronised with pitch accents.
  - In one corpus, 42 % of pitch accents got a nod, 18 % a nod with overshoot, 20 % a one-way swing.
  - Nods dominate, yaw is common, roll is rare.
  - A small dip then upward nod often starts a sentence after a pause (in >70 % of one speaker's sentences).
  - When reading Wall Street Journal text, "head movements were typically less pronounced".
  - "Sequences where … the head moves randomly look very synthetic … ‘eerie’", while repeated, prosody-timed primitives look engaged.
- **Munhall et al. 2004** ([PDF](https://www.queensu.ca/psychology/sites/psycwww/files/uploaded_files/Faculty/Kevin%20Munhall/Munhall_Psyc_Sci.pdf)):
  - Head motion predicted more than 63 % of F0 variance.
  - Natural head motion gave the best speech perception in noise.
  - **Doubled** head motion was worse than natural, and statistically no better than zero motion.
- **Tiede et al. 2019** ([Frontiers](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.02459/full)), citing Hadar 1983:
  - The head is moving in 89.9 % of frames while speaking, versus 12.8 % in pauses or while listening.
  - Amplitude correlates with loudness, driven mostly by fast, loud events.
  - So: something always moves a little, and big moves are reserved for emphasis.
- **Gesture timing** ([Ghaleb et al., arXiv 2404.14952](https://arxiv.org/html/2404.14952)):
  - 6,106 annotated strokes have **mean 0.58 s, median 0.42 s**.
  - Representational stroke onsets precede their word by about **200–500 ms**.
  - Beat gestures align with pitch accents.
  - Phases are preparation → stroke → (hold) → retraction.
- **Arm kinematics:** Flash & Hogan 1985 (J. Neurosci. 5(7):1688) showed point-to-point arm moves follow a **minimum-jerk** path with bell-shaped velocity. We did not open the paper itself; this comes from its abstract as quoted in search results. Velocity goes s(t) = 10t³ − 15t⁴ + 6t⁵.
- **Presenter coaching:**
  - "Keep gestures small and in front of you, and avoid sudden body movement"; "Remain neutral and become animated only when you begin to speak" ([PR News](https://prnewsonline.com/8-media-training-tips)).
  - Former BBC/Al Jazeera anchor Riz Khan says framing makes side-to-side or up-down movement look unnatural. Use subtle **lean forward/back plus a slight head tilt** instead; when glancing at notes, lean forward, then tilt up to camera before straightening ([MediaHelpingMedia](https://mediahelpingmedia.org/advanced/the-art-of-engaging-viewers-and-listeners/)).
- **Gesture rate:**
  - No measured rate for news anchors was found.
  - Commonly cited TED-speaker counts are 465 vs 272 gestures per 18-minute talk (about 15–26 per minute). This figure comes from search snippets only; we did not verify it at source. These count every hand movement of standing, animated speakers.
  - **Our assessment:** a seated anchor reading news should show far fewer *visible pose changes*. Aim for about **2–4 arm-pose changes per minute**, each held 1.5–4 s, with beats done as small elbow/wrist pulses inside a held pose, not as pose switches.
- **Head amplitude in degrees:** no source we opened gives it. **Our assessment:**
  - Speech nods of ~1–3° pitch, yaw drift within ±3°, roll ±1.5°, posture drift under 0.3 Hz.
  - Check against our own traces after scaling (§5).

## 5. Practical recommendation

### (a) Calming a traced human

Why it looks wrong now (**our assessment**): MediaPipe angles are noisy, and the tracker exaggerates fast turns (we saw ±40° at 370°/s on Wales). Each frame's value is shown directly, so tracker jitter and the speaker's own large gestures arrive at full gain. On top of that, our gesture switches are cuts, which read as teleports.

1. **Filter first, scale second.**
   - Run a 1€ filter on the raw landmarks ([1€ filter](https://gery.casiez.net/1euro/); tune `mincutoff` with the head still, then raise `beta` from 0.001; reference code is BSD/MIT).
   - Split each angle into posture (low-pass under 0.5 Hz) and accent (band 0.5–3 Hz). Apply gains of about **0.4 to posture and 0.6 to accent**, then the existing soft limits.
2. **Critically damped springs on every continuous channel**, using half-life parameters ([Holden, "Spring-It-On"](https://theorangeduck.com/page/spring-roll-call); code MIT, [repo](https://github.com/orangeduck/Spring-It-On)). Half-lives:
   - head ~120–180 ms
   - chest ~300–450 ms
   - shrug ~200 ms
   - breathing has none (it is generated)

   Keep our underdamped follow-through only where a deliberate gesture arrives, and with a small overshoot.
3. **Speed caps per channel**, enforced after the spring: e.g. head ≤ 25°/s, chest ≤ 6°/s (**our assessment**; tune by eye). This is what removes "sudden".
4. **Gestures as events, not samples.** From the trace keep only (pose, start, end, intensity). Then:
   - **Drop** events shorter than ~0.4 s (shorter than a median stroke) and merge events closer than 0.8 s.
   - **Re-time:** move the stroke so its apex lands on the nearest stressed syllable or energy peak within ±250 ms. For TTS, Piper's espeak phonemes carry stress marks; for recorded audio, use RMS/F0 peaks. Start the preparation ~0.3 s before.
   - **Animate** preparation and retraction with minimum-jerk easing over fixed durations: ~0.35–0.5 s up, 0.5–0.7 s down (retraction slower). Hold at least 1 s, and cool down at least 1.5 s before the next pose change.
   - **Budget:** at most one large channel at a time. While an arm gesture is in its stroke, halve head accent gain, and vice versa.
   - With painted discrete poses, "minimum-jerk" means: use the in-between painting at the right time fraction (e.g. halfway at ~40 % of the transition), plus a minimum-jerk *offset* on the arm bone so the cut is hidden inside motion. This is inertialization ([same Holden article](https://theorangeduck.com/page/spring-roll-call)): at a pose cut, record the visual offset, then decay it with a spring.
5. **Head nods from prosody, even with a trace.** On accented syllables add a Graf-style primitive (^ nod, or ~ nod with overshoot) of 1–2° pitch over 2–4 phones (~150–300 ms), and a dip-then-lift at sentence starts. The trace only modulates intensity.

### (b) Motion from speech alone

- **Recommended (no licence risk):**
  - Extend the current procedural generator with the rules in (a)4–5: Graf nod primitives on stressed syllables, Live2D-scale idle periods, 2–4 pose events per minute placed on phrase-initial stressed words, with minimum-jerk prep and retraction.
  - Piper's phoneme timings plus espeak stress give everything needed. All of it is our own code, so there are no licence questions.
- **Optional ML layer (non-commercial only):**
  1. Run **EMAGE via the ONNX export** on CPU over the Piper WAV.
  2. Use FK on its SMPL-X rotations and derive wrist height, hand openness, head and spine curves.
  3. Feed these through the *same* event extractor and calmer as a traced human. EMAGE becomes a "virtual performer", and steps (a)1–4 keep it settled.
  4. Expected cost: seconds per minute of audio on 12 cores (**our assessment**, untested).

## Ranked list: what to try next

1. **Event-based gesture re-synthesis** from the existing trace: drop/merge, stroke snapped to stress, minimum-jerk transitions, minimum hold, cooldown, one-big-channel budget. It directly addresses "too fast, too far, uncoordinated", needs no new dependency, and works for both trace and TTS.
2. **Filter → split → scale → spring → speed-cap chain** for head and chest (1€ + half-life springs, gains ~0.4–0.6). Cheap, MIT/BSD code, and Munhall's result says exaggerated head motion actively hurts.
3. **Prosodic nod primitives** (Graf) on stressed syllables, plus a sentence-start dip-lift, with idle periods taken from Live2D. This gives a teleprompter reader's restrained, rhythmic head.
4. **Mixamo "Sitting Talking" clips as a calm reference.** Measure their hold lengths, sway amplitudes and gesture rate, and set our parameters from them. Commercial video use is allowed; ML training is not.
5. **EMAGE ONNX on CPU** as an optional speech-only performer feeding step 1. Worth a one-afternoon test, but it is non-commercial (BEAT2 data, SMPL-X).
6. **ZeroEGGS / DiffuseStyleGesture** only if a "style" knob (e.g. "calm/neutral") is wanted. Both output BVH, but ZeroEGGS is CC BY-NC-ND and both lean on CUDA instructions.
