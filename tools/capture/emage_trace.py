"""Speech-only body performance from EMAGE (PantoMatrix, via the MIT emage-onnx-export), on CPU.

    <emage venv python> tools/capture/emage_trace.py voice.wav face_trace.json   (see docs/SETUP.md)

EMAGE predicts SMPL-X joint rotations from audio at 30 fps. We run its windowed autoregressive
inference with onnxruntime, do forward kinematics on an approximate SMPL skeleton, and write the
result in the same format as tools/capture/trace_face.py (head angles, shoulders, wrists, finger
reach), so the rig's calm gesture pipeline uses it unchanged. The file is marked
"source": "emage", "face": false: lips, blinks, brows and gaze still come from the speech.

NON-COMMERCIAL: EMAGE is trained on BEAT2 (non-commercial licence) and uses the SMPL-X body model
(non-commercial). Use only for experiments / non-commercial work.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toolpaths  # noqa: E402

ONNX = toolpaths.emage() / "onnx"
WINDOW, SEED, FPS, SR = 64, 4, 30, 16000
EFF, SPF = WINDOW - SEED, round(SR / FPS)
WAUDIO, MDIM = WINDOW * SPF, 337
PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28,
          29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]
# approximate SMPL T-pose offsets from parent (metres; x = subject's left, y up, z forward)
OFFSET = {0: (0, 0, 0), 3: (0, .11, -.02), 6: (0, .14, .02), 9: (0, .06, 0), 12: (0, .21, -.03),
          13: (.08, .12, -.02), 14: (-.08, .12, -.02), 15: (0, .09, .05), 16: (.12, .04, 0), 17: (-.12, .04, 0),
          18: (.26, 0, 0), 19: (-.26, 0, 0), 20: (.25, 0, 0), 21: (-.25, 0, 0)}
FINGERS = {"L": [(25, 26, 27), (28, 29, 30), (34, 35, 36), (31, 32, 33)],   # index, middle, ring, pinky
           "R": [(40, 41, 42), (43, 44, 45), (49, 50, 51), (46, 47, 48)]}


def rot6d_to_mat(r):
    a1, a2 = r[..., :3], r[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], -2)  # rows (columns put the arms up and behind the body)


def run(audio):
    s = {k: ort.InferenceSession(str(ONNX / f"{k}.onnx"), providers=["CPUExecutionProvider"])
         for k in ("emage_step", "vq_face", "vq_upper_idx", "vq_hands_idx", "vq_lower_idx", "postprocess")}
    ident = np.zeros((WINDOW, MDIM), np.float32)
    ident[:, 0:330:6] = 1.0
    ident[:, 4:330:6] = 1.0
    N = len(audio) // SPF
    nw = max(1, math.ceil((N - SEED) / EFF))
    buf = {k: np.zeros((N, 256), np.float32) for k in ("rec_face", "cls_upper", "cls_hands", "cls_lower")}
    seed, off = None, 0
    for w in range(nw):
        wa = np.zeros(WAUDIO, np.float32)
        seg = audio[w * EFF * SPF: w * EFF * SPF + WAUDIO]
        wa[:len(seg)] = seg
        mm, mask = ident.copy(), np.ones((WINDOW, MDIM), np.float32)
        if seed is not None:
            mm[:SEED] = seed.reshape(SEED, MDIM)
            mask[:SEED] = 0
        out = s["emage_step"].run(None, {"audio": wa[None], "speaker_id": np.zeros((1, 1), np.int64),
                                         "masked_motion": mm[None], "mask": mask[None]})
        names = [o.name for o in s["emage_step"].get_outputs()]
        o = dict(zip(names, out))
        seed = o["seed"]
        keep = min(WINDOW if w == nw - 1 else EFF, N - off)
        for k in buf:
            buf[k][off:off + keep] = o[k][0, :keep]
        off += keep
    idx = lambda k: buf[k].argmax(-1)[None].astype(np.int64)  # noqa: E731
    fd = s["vq_face"].run(None, {"latent": buf["rec_face"][None]})[0]
    ud = s["vq_upper_idx"].run(None, {"indices": idx("cls_upper")})[0]
    hd = s["vq_hands_idx"].run(None, {"indices": idx("cls_hands")})[0]
    ld = s["vq_lower_idx"].run(None, {"indices": idx("cls_lower")})[0]
    motion = s["postprocess"].run(["motion_inference"], {"face_dec": fd, "upper_dec": ud, "hands_dec": hd,
                                                         "lower_dec": ld})[0]
    return motion[0, :, :330].reshape(-1, 55, 6)


def fk(local):
    """World rotations and positions of the body joints (no fingers) per frame."""
    T = len(local)
    R = rot6d_to_mat(local)  # T,55,3,3
    W = np.zeros_like(R)
    P = np.zeros((T, 55, 3))
    for j in range(55):
        p = PARENT[j]
        if p < 0:
            W[:, j] = R[:, j]
        else:
            W[:, j] = W[:, p] @ R[:, j]
            P[:, j] = P[:, p] + (W[:, p] @ np.array(OFFSET.get(j, (0, 0, 0)), float))
    return R, W, P


def euler_deg(M):
    """yaw (about y), pitch (about x), roll (about z) of a world rotation."""
    yaw = np.degrees(np.arctan2(M[..., 0, 2], M[..., 2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-M[..., 1, 2], -1, 1)))
    roll = np.degrees(np.arctan2(M[..., 1, 0], M[..., 1, 1]))
    return yaw, pitch, roll


def main(wav, out):
    a, sr = sf.read(wav, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    local = run(a)
    R, W, P = fk(local)
    # measure everything in the upper-torso frame (spine3), so hips and root rotation don't matter
    W9i = np.linalg.inv(W[:, 9])
    P = np.einsum("tij,tkj->tki", W9i, P - P[:, 9:10])
    W = np.einsum("tij,tkjl->tkil", W9i, W)
    # SMPL rest pose has arms out sideways (T-pose); EMAGE's rotations bring them down
    yaw, pitch, roll = euler_deg(W[:, 15])  # head relative to the upper torso
    frames = []
    for t in range(len(P)):
        sl, sr_ = P[t, 16], P[t, 17]
        sw = max(np.linalg.norm(sl - sr_), 1e-3)
        sy = (sl[1] + sr_[1]) / 2
        cx = (sl[0] + sr_[0]) / 2
        body = {"shoulder_w": 100.0, "neck": round(float((P[t, 15][1] - sy) / sw), 4),
                "tilt": round(float((sl[1] - sr_[1]) / sw), 4), "vis": 1.0}
        for side, wj in (("L", 20), ("R", 21)):
            wr = P[t, wj]
            body[f"wrist{side}"] = [round(float((wr[0] - cx) / sw), 4), round(float((wr[1] - sy) / sw), 4), 1.0]
            reach = []
            for chain in FINGERS[side]:
                curl = sum(np.arccos(np.clip((np.trace(R[t, j]) - 1) / 2, -1, 1)) for j in chain)
                reach.append(round(float(np.clip(2.1 - 0.45 * curl, 0.8, 2.1)), 3))
            body[f"fingers{side}"] = reach
            body[f"open{side}"] = round(float(np.mean(reach)), 3)
        frames.append({"t": round(t / FPS, 4), "ok": True, "open": 0.0, "width": 0.0,
                       "yaw": round(float(yaw[t]), 2), "pitch": round(float(pitch[t]), 2), "roll": round(float(roll[t]), 2),
                       "box": [0, 0, 100, 100], "bs": {}, "body": body})
    # EMAGE's (standing) speakers keep their hands near chest height; rescale wrist height from its
    # own range (20th percentile = resting, 95th = a full gesture) to the seated range a traced
    # speaker shows (-1.1 resting .. -0.4 raised, in shoulder widths) that the gesture chooser expects
    hs = np.array([[f["body"][f"wrist{s_}"][1] for s_ in "LR"] for f in frames])
    lo, hi = np.percentile(hs, 20), np.percentile(hs, 95)
    for f, (hl, hr) in zip(frames, hs):
        for s_, h in (("L", hl), ("R", hr)):
            f["body"][f"wrist{s_}"][1] = round(float(-1.1 + 0.7 * np.clip((h - lo) / max(hi - lo, 1e-3), 0, 1.2)), 4)
    Path(out).write_text(json.dumps({"source": "emage", "face": False, "fps": FPS, "start": 0.0, "frames": frames}), encoding="utf-8")
    wy = np.array([max(f["body"]["wristL"][1], f["body"]["wristR"][1]) for f in frames])
    print(f"{len(frames)} frames; wrist height p10 {np.percentile(wy, 10):.2f} p50 {np.median(wy):.2f} p90 {np.percentile(wy, 90):.2f} (shoulder widths)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
