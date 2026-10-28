# Third-party material

Nothing below is bundled in this repository except where marked; `tools/setup.py` downloads the
models and tools from their publishers.

| Component | Use | Licence |
|---|---|---|
| Blender 5.2 | rendering (not bundled) | GPL-3.0 |
| Piper TTS (`piper-tts`) | speech with phoneme timings | MIT (code); voices: see each voice's MODEL_CARD at huggingface.co/rhasspy/piper-voices |
| Rhubarb Lip Sync 1.14 | mouth shapes from recorded audio | MIT |
| MediaPipe + landmarker models | face, pose and hand tracking | Apache-2.0 |
| Depth Anything V2 **Small** (ONNX) | relief depth | Apache-2.0 (Base/Large are CC BY-NC: do not use) |
| rembg + isnet-general-use | background removal | MIT / Apache-2.0 |
| RIFE ncnn-vulkan (optional) | in-between frames | MIT |
| EMAGE / BEAT2 / SMPL-X (optional, not bundled) | speech-to-gesture experiments | **non-commercial** |
| DreamShaper 8 inpainting (optional, not bundled) | local SD experiments | CreativeML OpenRAIL-M |
| Stock interview clips (optional download) | motion-capture references | CC BY 3.0 / 4.0, Wikimedia Commons; see `assets/capture/stock/SOURCE.md` |

**Bundled, generated with Google Gemini / Veo** from this project's own reference images: the
paintings in `assets/haj_salama/gen/` and the clips in `assets/haj_salama/veo/`. Their use is subject
to Google's terms for generated content.

The Haj Salama character and the reference paintings in `assets/haj_salama/` belong to the project
owner. No licence file is included, so all rights are reserved unless one is added.
