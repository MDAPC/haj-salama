# Research: open, CPU-friendly ways to animate Haj Salama

Date: 2026-10-23. Scope: free/open tools that run locally on our box (Ryzen with Radeon 680M iGPU, 12 cores, 29 GB RAM, no NVIDIA). Every claim is linked to a page that was actually opened. Where a line is **our assessment** (an inference, not something a source says), it is marked as such.

## TL;DR

- **LivePortrait is the one neural method worth testing now.** The code is MIT and the weights are tagged MIT, but its default face detector (InsightFace) is research-only. There is a CPU flag, community ONNX exports, and a parametric "expression editor" variant with `aaa`/`eee`/`woo`/blink/pupil/yaw sliders that avoids InsightFace. Use it **offline, to paint new frames and sprites for our Blender rig**, not as a real-time renderer.
- **The diffusion talking heads are a poor fit.** SadTalker, Hallo, EchoMimic, AniPortrait, MuseTalk and Wav2Lip all assume CUDA and photoreal faces, and some carry licence problems.
- **For bigger head turns,** use layered inpainting (LaMa ONNX, or 3D-Photo-Inpainting, which has a CPU code path) rather than a single relief sheet.
- **For Arabic lip-sync,** there is a published 13-class Arabic viseme table to map onto our 9 shapes. Replace nearest-shape switching with JALI-style timing curves (120 ms ramp in, 75 % hold, 120 ms decay).
- **Use stable-diffusion.cpp** (MIT; CPU and Vulkan backends; IP-Adapter for SD1.5; LCM-LoRA) to make painted variants faster and more on-model. The 12–20B editing models (FLUX Kontext, Qwen-Image-Edit) are technically loadable as GGUF but heavy, and FLUX Kontext dev is non-commercial.
- **For mocap,** MediaPipe Face Landmarker outputs ARKit-named blendshapes (Apache-2.0, CPU). A small mapping table drives our shape keys.

---

## 0. Hardware reality check

- **PyTorch on the 680M (ROCm) is unofficial and fragile.** Community guides set `HSA_OVERRIDE_GFX_VERSION=10.3.0` for RDNA 2 GPUs and APUs, sometimes with `HSA_ENABLE_SDMA=0` ([linuxcontainers guide](https://discuss.linuxcontainers.org/t/rocm-and-pytorch-on-amd-apu-or-gpu-ai/19743/1)). On an older RDNA1 card (gfx1010), the same override worked with PyTorch 1.13.1 but newer ROCm builds segfaulted, and AMD said the card was not officially supported ([ROCm#2527](https://github.com/ROCm/ROCm/issues/2527)). **Our assessment:** expect the same fragility on the 680M (gfx1035).
- **ONNX Runtime has no supported AMD route for us.** Its only AMD GPU providers are MIGraphX and ROCm, and ROCm is marked deprecated. **WebGPU** is listed as a provider ([ORT EP list](https://onnxruntime.ai/docs/execution-providers/)), and on Linux it runs over Vulkan ([Vulkan docs](https://docs.vulkan.org/tutorial/latest/ML_Inference/Third_Party_Libraries/02_onnx_runtime.html)).
- **Our assessment:** plan everything as CPU ONNX/PyTorch first. The realistic way to use the iGPU for ML is Vulkan, through stable-diffusion.cpp (§4) or ORT-WebGPU.

## 1. Single-image talking-head / portrait animation

| Method | Licence | CPU / AMD path | Cartoon fit | Verdict |
|---|---|---|---|---|
| **LivePortrait** ([repo](https://github.com/KwaiVGI/LivePortrait), [paper](https://arxiv.org/abs/2407.03168)) | Code MIT; InsightFace models "non-commercial research purposes only" ([LICENSE](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/LICENSE)); HF weights tagged MIT ([HF](https://huggingface.co/KwaiVGI/LivePortrait)) | `flag_force_cpu` exists but is marked "WIP" ([config](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/src/config/argument_config.py)); full ONNX set exists (below) | Shown on oil painting, sculpture and 3D renders, not anime ([project page](https://liveportrait.github.io/)) | **Try first** |
| **THA3 / THA4** (pkhungurn) | THA3: code MIT, models CC-BY-4.0 ([THA3](https://github.com/pkhungurn/talking-head-anime-3-demo)); THA4 models CC-BY-NC-4.0 ([THA4](https://github.com/pkhungurn/talking-head-anime-4-demo)) | Requires "a recent and powerful Nvidia GPU"; a THA4 student distil took ~30 h on an RTX A6000 | Anime-only input contract: 512², alpha, upright, head in a fixed box | Skip (GPU-bound) |
| **SadTalker** | Apache-2.0 ([repo](https://github.com/OpenTalker/SadTalker)) | CUDA install; CPU not documented | Not documented; relies on dlib/face-alignment/3DMM | Low |
| **Wav2Lip** | Non-commercial only; LRS2-trained ([repo](https://github.com/Rudrabha/Wav2Lip)) | Not documented | Claims CGI faces; cartoons not addressed | Avoid (licence) |
| **MuseTalk 1.5** | Code MIT; weights usable commercially ([repo](https://github.com/TMElyralab/MuseTalk)) | Lowest config cited is a 4 GB laptop GPU, ~5 min for 8 s of video | 256×256 mouth region; zh/en/ja audio | Low |
| **EchoMimic v2** | Apache-2.0 ([repo](https://github.com/antgroup/echomimic_v2)) | Tested on A100/4090/V100; ~1 min per 120 frames on an A100 even when accelerated | Human demos only | Skip |
| **Hallo** | MIT, but needs InsightFace ([repo](https://github.com/fudan-generative-vision/hallo)) | A100-tested | English audio only; face 50–70 % of frame | Skip |
| **AniPortrait** | Apache-2.0 ([repo](https://github.com/Zejun-Yang/AniPortrait)) | CUDA 11.7 | "photorealistic" | Skip |
| **JoyVASA** (audio → LivePortrait motion) | MIT ([repo](https://github.com/jdh-algo/JoyVASA)) | Smallest tested GPU: 8 GB RTX 4060 laptop | Trained on Chinese and English; has an animal mode | Later, maybe |

### LivePortrait in detail

**How it works.** LivePortrait uses compact implicit keypoints that "act like blendshapes", plus small MLP stitching and eye/lip retargeting modules ([arXiv](https://arxiv.org/abs/2407.03168)).

**Driving it from a video.** You can drive it with a video or with saved `.pkl` motion templates. The first driving frame should be frontal and neutral ([repo](https://github.com/KwaiVGI/LivePortrait)). Useful config options ([config](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/src/config/argument_config.py)):
- `animation_region` (`exp` / `pose` / `all`)
- `driving_option: expression-friendly` with a `driving_multiplier`
- relative motion
- `flag_stitching`, which the config recommends turning off for large head motion

The eye and lip retargeting flags are labelled "not recommend … WIP".

**ONNX / CPU route.** FasterLivePortrait (MIT code) ships ONNX exports and says onnxruntime CPU works but "cpu inference is extremely slow and not recommended" ([repo](https://github.com/warmshao/FasterLivePortrait)). It also offers a MediaPipe face path (`--mp`) to avoid InsightFace licensing. The HF repo holds `appearance_feature_extractor` (3.4 MB), `motion_extractor` (113 MB), `warping_spade(-fix)` (421 MB), `stitching_{eye,lip}` and `landmark.onnx` ([file list](https://huggingface.co/warmshao/FasterLivePortrait)). **Our assessment:** "slow" is fine for offline 25 fps news segments rendered overnight, but measure seconds per frame before committing.

**Parametric control (the important part for us).** ComfyUI-AdvancedLivePortrait's ExpressionEditor ([nodes.py](https://raw.githubusercontent.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait/main/nodes.py)) exposes:
- `rotate_pitch/yaw/roll` ±20
- `blink`, `eyebrow`, `wink`
- `pupil_x/y`
- `aaa`, `eee`, `woo`, `smile`

It falls back from CUDA/MPS to CPU. It detects faces with a YOLOv8 face model and **falls back to the full image when no face is found**, which matters for caricatures. Two licence caveats:
- The repo has no licence file (GitHub API reports none).
- Ultralytics is AGPL-3.0.

**Our assessment:** reimplement this keypoint-offset idea on top of the MIT LivePortrait code, with our own hand-placed crop instead of a detector.

**Cartoon caveat.** None of the official pages claim cartoon or anime support. ComfyUI-LivePortraitKJ offers MediaPipe and blazeface/SFD detectors as InsightFace replacements under MIT/Apache terms ([repo](https://github.com/kijai/ComfyUI-LivePortraitKJ)). Haj Salama is semi-realistic painted, so the odds are better than for anime, but it must be tested.

### Live2D-like rigs

Inochi2D's Inochi Creator is a BSD-2 open rigging editor for layered 2D puppets, built for Windows and Linux ([repo](https://github.com/Inochi2D/inochi-creator)). **Our assessment:** our Blender relief rig already covers this, and Inochi2D would mean repainting into layers. Worth knowing as a format, not a switch.

## 2. 2.5D parallax from one image without tearing

- **3D Photo Inpainting** (Shih et al., CVPR 2020; MIT) ([paper](https://arxiv.org/abs/2004.04727), [repo](https://github.com/vt-vl-lab/3d-photo-inpainting)):
  - Builds a Layered Depth Image with explicit pixel connectivity, then inpaints colour *and depth* behind depth edges.
  - Exports a `.ply` mesh, about 2–3 min per image.
  - The code falls back to `device = "cpu"` when `gpu_ids` is negative, and `offscreen_rendering: True` switches vispy to EGL ([main.py](https://raw.githubusercontent.com/vt-vl-lab/3d-photo-inpainting/master/main.py)).
  - Tunables include `depth_threshold`, `background_thickness` and `context_thickness` ([argument.yml](https://raw.githubusercontent.com/vt-vl-lab/3d-photo-inpainting/master/argument.yml)).
  - It defaults to MiDaS depth. **Our assessment:** feed it our Depth Anything V2 depth map instead, and import the PLY into Blender as an occlusion-filled back layer.
- **LaMa** (Apache-2.0) generalises from 256² training to ~2K images ([repo](https://github.com/advimman/lama), [LICENSE](https://raw.githubusercontent.com/advimman/lama/main/LICENSE)). An Apache-2.0 ONNX export `lama_fp32.onnx` exists at a fixed 512×512 input ([Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)). **Our assessment, a manual LDI that fits our rig:**
  1. Split the cut-out into layers: background, torso, shemagh drape, head/face, nose and ears.
  2. Inpaint each layer's hidden region with LaMa (fast; flat colours) or SD1.5 inpainting (painterly, slow).
  3. Give each layer its own relief. Turning the head then reveals painted pixels instead of stretched ones.
- **Generated side views.** Qwen-Image-Edit-2511 lists built-in "new viewpoint generation" LoRAs ([HF](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)). That could paint ±30° reference views for texture projection, but it costs 20B parameters (§4).

## 3. Arabic visemes and coarticulation

**Arabic data.**
- **A 13-viseme Arabic mapping** is published by Abou Zliekha et al., EUSIPCO 2006 ([PDF](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2006/papers/1568982208.pdf), Table 1, MBROLA notation):
  - (1) aa, q
  - (2) b, m
  - (3) t d z s sˤ dˤ tˤ k n
  - (4) θ ð ðˤ
  - (5) ʒ ʃ
  - (6) ħ x h
  - (7) r
  - (8) ʕ ɣ
  - (9) f
  - (10) l
  - (11) w, o
  - (12) j, e
  - (13) pause

  They blend between visemes linearly, keyed by TTS durations, which is the baseline we already beat.
- **Lipreading results for Arabic** come from Ouni & Ouni, Interspeech 2007 ([PDF](https://www.isca-archive.org/interspeech_2007/ouni07_interspeech.pdf)):
  - Back consonants (velar, uvular, pharyngeal, glottal) are "not visible, or very hard to see".
  - Bilabials, /f/ and /w/ are identified near 100 %, and vowels at over 95 %.
  - Vowels become pharyngealised next to emphatic or pharyngeal consonants.
  - /n/ is confused with /l/ and /t/.
- **Forced alignment for recorded Arabic.** Rhubarb's default PocketSphinx recogniser is English-only. Its `phonetic` recogniser is language-independent but less precise, and passing the script with `--dialogFile` helps ([Rhubarb](https://github.com/DanielSWolf/rhubarb-lip-sync)). An alternative is the multilingual phoneme CTC model `wav2vec2-xlsr-53-espeak-cv-ft` (Apache-2.0) ([HF](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)). torchaudio's CTC forced-alignment tutorial says its APIs were deprecated in 2.8 and removed in 2.9 ([docs](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)), so pin torchaudio ≤ 2.8 if we use it.

**Proposed Arabic → our 9 shapes** (our assessment, derived from the two papers above):

| Ours | Arabic phonemes |
|---|---|
| A (closed) | b m |
| G (lip on teeth) | f |
| F (pucker) | w, u, uː, o |
| E (rounded-open) | ʃ ʒ (lip-heavy per JALI's S/Z/J class) |
| D (wide open) | aː, and a next to emphatics/pharyngeals |
| C (open) | short a, e |
| B (teeth, spread) | i iː j, t d s z sˤ dˤ tˤ k n r |
| H (tongue) | l, θ ð ðˤ (tongue tip visible) |
| *no own shape* | q x ɣ ħ ʕ h ʔ: take the lip shape of the neighbouring vowel |
| X | pause |

Add an "emphatic" modifier: slightly more jaw and less lip spread on vowels next to sˤ dˤ tˤ ðˤ q (our assessment, from Ouni's pharyngealisation note).

**Better blending than nearest-shape.**
- **Cohen–Massaro dominance functions.** Each phoneme has a target value and a negative-exponential dominance function. Consecutive dominance functions overlap in time, so a segment influences the articulators before and after it ([Cosi et al. 2002](https://www3.pd.istc.cnr.it/piero/Papers/PieroCosi/cp-ICMI2002.pdf) summarises the model). JALI notes the model's known failure is not closing on bilabials ([JALI, SIGGRAPH 2016](https://www.dgp.toronto.edu/~elf/JALISIG16.pdf)).
- **JALI's procedural rules** ([PDF](https://www.dgp.toronto.edu/~elf/JALISIG16.pdf)) map directly onto our rig:
  - Viseme onset starts **120 ms before** the apex, and the apex is at sound onset.
  - Hold the apex to **75 %** of the phoneme, then decay over **120 ms**.
  - Lip protrusion uses 150 ms onset and offset.
  - Merge duplicate visemes.
  - Lip-heavy visemes start early and end late, and override neighbours' lips.
  - Tongue-only phonemes (l n t d g k ŋ) don't move the lips.
  - Separate jaw (JA) and lip (LI) channels.
  - The last phoneme of a word looks back.

  **Our assessment:** implement this as a small Python pass over the Piper/Rhubarb timeline. It outputs per-frame weights for `jaw`, `wide`, `round` and `close` (we already have those shape keys), with a hard constraint that `close` = 1 at the bilabial apex. Feed the Arabic back consonants through the "tongue-only" rule.

## 4. Character consistency for painted variants

| Option | Size / licence | CPU feasibility |
|---|---|---|
| **IP-Adapter** (SD1.5, Plus, Plus-Face) | 22M-param adapter, Apache-2.0; works with ControlNet and inpainting ([repo](https://github.com/tencent-ailab/IP-Adapter)) | Small overhead on top of SD1.5. **Yes** |
| **Reference-only ControlNet** | No extra model; injects the reference image into attention; "Style Fidelity" slider; some users hit errors with inpainting checkpoints ([discussion](https://github.com/Mikubill/sd-webui-controlnet/discussions/1236)) | **Yes** |
| **InstantID** | SDXL; needs InsightFace antelopev2, "non-commercial research purposes only" ([repo](https://github.com/instantX-research/InstantID)) | Avoid (licence; photo-face embedding) |
| **FLUX.1 Kontext [dev]** | 12B; FLUX.1 [dev] Non-Commercial License; gated ([HF](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)) | GGUF loadable; licence blocks commercial use |
| **Qwen-Image-Edit-2509 / 2511** | 20B, Apache-2.0; 2511 claims less drift and better character consistency ([2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509), [2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)) | GGUF quants of 2509 run from Q2_K at 7.15 GB to Q4_K_M at 13.1 GB and Q8_0 at 21.8 GB ([QuantStack](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/main)), so they fit in 29 GB RAM. **Our assessment:** expect many minutes per image on CPU; use only for a few hero variants |

**The runtime that ties this together is stable-diffusion.cpp** (MIT) ([README](https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/README.md)). It offers:
- CPU (AVX2/AVX512) and **Vulkan** backends, plus GGUF weights
- SD1.5 with ControlNet and **IP-Adapter (incl. Plus)** ([doc](https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/ip_adapter.md)), and LCM-LoRA
- FLUX.1-Kontext-dev and Qwen Image Edit

**Our assessment:** benchmark our DreamShaper-8-inpainting job under sd.cpp on Vulkan (680M) and on CPU with LCM-LoRA (4–8 steps). Either could cut the ~2 min per image substantially. Then add IP-Adapter-Plus with ref10 as the reference so the mouth and eye variants stay on-model.

## 5. Webcam/phone mocap into our rig

- **MediaPipe Face Landmarker** (Apache-2.0, [repo](https://github.com/google-ai-edge/mediapipe)):
  - Outputs 478 3D landmarks, **52 blendshape scores**, and a facial transformation matrix (head pose).
  - Has IMAGE, VIDEO and LIVE_STREAM modes and a Python API.
  - Left/right labels refer to the subject, so mirrored input swaps them ([docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)).
  - The model card lists the names, which follow ARKit naming (browDownLeft … tongueOut). It warns that jitter rises in poor light or with noise, and that it fails past 80° of head turn ([model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)).
- **MediaPipe Pose Landmarker** gives 33 body landmarks plus world coordinates, which we can use for shrug and lean ([docs](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)).
- **BlendArMocap** (GPL-3.0) runs MediaPipe inside Blender and transfers the results to Rigify rigs, but the author has **discontinued** it ([repo](https://github.com/cgtinker/BlendArMocap)). **Our assessment:** a 100-line offline script is simpler: run MediaPipe in VIDEO mode, smooth the results, and write shape-key keyframes.

**Mapping to our shape keys** (our assessment):

| Our key | From blendshapes |
|---|---|
| jaw | jawOpen |
| close | mouthClose, plus max(mouthPressL/R) and mouthRollLower/Upper, gated by low jawOpen |
| wide | mean(mouthSmileL/R, mouthStretchL/R) |
| round | max(mouthPucker, mouthFunnel) |
| brows | browInnerUp, mean(browOuterUpL/R), minus mean(browDownL/R) |
| blink overlay | eyeBlinkL/R (threshold ≈0.5 with hysteresis) |
| iris layer | eyeLookIn/Out/Up/Down → x/y offset |
| head | yaw/pitch/roll from the transformation matrix, scaled to our safe parallax range |
| shrug | Pose: shoulder-to-ear distance relative to the neutral frame |

Fit per-performer gains with a 5-second calibration clip: neutral, "aaa", "ooo", "eee", "mmm", blink, brows up. A hybrid is also possible: drive the mouth from audio (exact phonemes) and everything else (brows, gaze, blinks, head) from the video.

---

## Ranked recommendations

1. **JALI-style coarticulation plus the Arabic viseme table (§3).** Pure Python on our existing timeline: zero new dependencies, and the largest visible gain in lip-sync quality per hour of work. Put the bilabial-closure and back-consonant rules in first.
2. **stable-diffusion.cpp for the inpainting jobs (§4).** Benchmark Vulkan on the 680M against CPU with LCM-LoRA, and add IP-Adapter-Plus from ref10. This makes the painted mouth/eye swap plan affordable and keeps it on-model.
3. **LivePortrait as an offline painter (§1).**
   - Run the FasterLivePortrait ONNX models on CPU, with a manual crop or a MediaPipe detector, so no InsightFace.
   - Apply ExpressionEditor-style keypoint offsets (`aaa`, `eee`, `woo`, blink, pupil, ±yaw) to ref10.
   - Use the outputs as identity-preserving painted mouth/eye/turn variants for the Blender rig.
   - Measure seconds per frame. If tolerable, test full-frame driving from a webcam video with `animation_region=exp`.
4. **MediaPipe mocap to shape keys (§5).** A cheap way to get natural brows, gaze, blinks and head motion from a phone clip of a performer, layered over audio-driven lips.
5. **Layered inpainting for head turns (§2).** Start with a LaMa-ONNX back layer behind the head and shemagh. Try 3D-Photo-Inpainting on CPU with our own depth if seams persist.
6. **Qwen-Image-Edit-2511 GGUF** (Apache-2.0) for a handful of hero variants or side views only, run overnight. Skip FLUX Kontext dev (non-commercial) and the diffusion talking-head stacks (CUDA-bound, photoreal-trained, and in some cases licence-encumbered).
