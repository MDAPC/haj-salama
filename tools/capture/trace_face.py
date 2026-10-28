"""Trace a speaking face in a video: mouth, eyes, brows and head pose per frame (MediaPipe, CPU).

    python tools/capture/trace_face.py VIDEO OUT.json [--start S] [--dur D] [--face right|left|largest]

Per frame (at the video's own rate) it stores:
  mouth  open (inner-lip gap / face size), width (mouth corners / face size), and the ARKit-style
         blendshapes jawOpen, mouthClose, mouthFunnel, mouthPucker, mouthPress*, mouthStretch*, mouthSmile*,
         mouthRoll*, mouthLowerDown*, mouthUpperUp*
  eyes   eyeBlinkLeft/Right, eyeLook* (gaze)
  brows  browInnerUp, browDown*, browOuterUp*
  head   yaw, pitch, roll in degrees (from MediaPipe's facial transformation matrix)
This is the raw performance; tools/capture/mouth_model.py turns it into rig controls.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toolpaths  # noqa: E402

MODEL = toolpaths.model("mediapipe/face_landmarker.task")
POSE_MODEL = toolpaths.model("mediapipe/pose_landmarker_full.task")
HAND_MODEL = toolpaths.model("mediapipe/hand_landmarker.task")
KEEP = ("jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthPressLeft", "mouthPressRight",
        "mouthStretchLeft", "mouthStretchRight", "mouthSmileLeft", "mouthSmileRight", "mouthRollLower",
        "mouthRollUpper", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
        "eyeBlinkLeft", "eyeBlinkRight", "eyeLookUpLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
        "eyeLookUpRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
        "browInnerUp", "browDownLeft", "browDownRight", "browOuterUpLeft", "browOuterUpRight")


def euler(m):
    """Yaw, pitch, roll (degrees) of a 4x4 head transform."""
    r = np.asarray(m)[:3, :3]
    pitch = math.degrees(math.atan2(-r[1, 2], r[2, 2]))
    yaw = math.degrees(math.asin(np.clip(r[0, 2], -1, 1)))
    roll = math.degrees(math.atan2(-r[0, 1], r[0, 0]))
    return yaw, pitch, roll


def trace_body(pres, hres, face_lm, shape):
    """Shoulders, wrists and hand openness of the person whose face we track (all lengths in
    shoulder widths, y up). Picks the pose whose nose is nearest that face."""
    if not pres.pose_landmarks:
        return None
    h, w = shape[:2]
    fc = face_lm.mean(0)

    def px(q):
        return np.array([q.x * w, q.y * h])

    j = int(np.argmin([np.linalg.norm(px(P[0]) - fc) for P in pres.pose_landmarks]))
    P = pres.pose_landmarks[j]
    sl, sr = px(P[11]), px(P[12])
    sw = max(np.linalg.norm(sl - sr), 1.0)
    sy = (sl[1] + sr[1]) / 2
    chin = face_lm[152]
    out = {"shoulder_w": round(float(sw), 1),
           # neck: chin to shoulder line; shrinks when the shoulders come up
           "neck": round(float((sy - chin[1]) / sw), 4),
           "tilt": round(float((sr[1] - sl[1]) / sw), 4),
           "vis": round(float(min(P[11].visibility, P[12].visibility)), 3)}
    for side, wi, ei in (("L", 15, 13), ("R", 16, 14)):
        wr = px(P[wi])
        out[f"wrist{side}"] = [round(float((wr[0] - (sl[0] + sr[0]) / 2) / sw), 4),
                               round(float((sy - wr[1]) / sw), 4), round(float(P[wi].visibility), 3)]
    # hand openness: fingertips' reach from the wrist, relative to palm length (closed ~1.2, open ~2)
    best = {}
    for H in hres.hand_landmarks or []:
        pts = np.array([px(q) for q in H])
        palm = max(np.linalg.norm(pts[9] - pts[0]), 1.0)
        fingers = [float(np.linalg.norm(pts[t] - pts[0]) / palm) for t in (8, 12, 16, 20)]
        reach = float(np.mean(fingers))
        for side, wi in (("L", 15), ("R", 16)):
            d = np.linalg.norm(pts[0] - px(P[wi])) / sw
            if d < 0.35 and (side not in best or d < best[side][0]):
                best[side] = (d, reach, fingers)
    for side, (_, reach, fingers) in best.items():
        out[f"open{side}"] = round(reach, 3)
        # per finger (index, middle, ring, pinky) reach / palm: pointing = index long, others short
        out[f"fingers{side}"] = [round(v, 3) for v in fingers]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--dur", type=float, default=0.0, help="seconds; 0 = to the end")
    p.add_argument("--face", default="largest", choices=("largest", "left", "right"))
    p.add_argument("--max-width", type=int, default=1280, help="downscale wide frames for speed")
    p.add_argument("--body", action="store_true", help="also trace shoulders, arms and hands")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    pose = hands = None
    if a.body:
        pose = mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=2))
        hands = mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=4))

    frames, i, t0 = [], 0, time.time()
    while True:
        ok, img = cap.read()
        if not ok or (a.dur and i / fps >= a.dur):
            break
        h, w = img.shape[:2]
        if w > a.max_width:
            img = cv2.resize(img, (a.max_width, int(h * a.max_width / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(1000 * i / fps))
        rec = {"t": round(i / fps, 4), "ok": False}
        if res.face_landmarks:
            k = range(len(res.face_landmarks))
            xs = [np.mean([q.x for q in res.face_landmarks[j]]) for j in k]
            sz = [np.ptp([q.x for q in res.face_landmarks[j]]) for j in k]
            j = {"largest": int(np.argmax(sz)), "left": int(np.argmin(xs)), "right": int(np.argmax(xs))}[a.face]
            L = np.array([[q.x * img.shape[1], q.y * img.shape[0]] for q in res.face_landmarks[j]])
            face = np.linalg.norm(L[33] - L[263])           # outer eye corners: face scale
            bs = {c.category_name: round(c.score, 4) for c in res.face_blendshapes[j] if c.category_name in KEEP}
            yaw, pitch, roll = euler(res.facial_transformation_matrixes[j])
            rec.update(ok=True, open=round(float(np.linalg.norm(L[13] - L[14]) / face), 4),
                       width=round(float(np.linalg.norm(L[61] - L[291]) / face), 4),
                       yaw=round(yaw, 2), pitch=round(pitch, 2), roll=round(roll, 2),
                       box=[round(float(v), 1) for v in (*L.min(0), *L.max(0))], bs=bs)
            if pose is not None:
                ts = int(1000 * i / fps)
                body = trace_body(pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts),
                                  hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts),
                                  L, img.shape)
                if body:
                    rec["body"] = body
        frames.append(rec)
        i += 1
    Path(a.out).write_text(json.dumps({"video": a.video, "fps": fps, "start": a.start, "frames": frames}), encoding="utf-8")
    ok = sum(f["ok"] for f in frames)
    print(f"{len(frames)} frames, face found in {ok}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
