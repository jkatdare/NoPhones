"""Branch B: posture and gaze features from MediaPipe.

Branch A answers "is there a phone?". This answers "what is the person doing?"
- which is the question that actually separates working from distracted. It
needs no phone in frame at all, so it covers exactly the cases Branch A lost:
the phone at arm's length, the phone against dark clothing, and the phone
face-down on the desk while you work.

Run it and watch the numbers move. Look down at your lap and head_pitch should
swing; drop your hands off the keyboard and the wrist features should follow.
That is the whole point of this step - see what each feature actually measures
before trusting it.

    python src/pose.py

MediaPipe 1.0.x removed the old mp.solutions API, so this uses the Tasks API
with model bundles in models/.
"""

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import protocol
from capture import open_camera

POSE_MODEL = "models/pose_landmarker_full.task"
FACE_MODEL = "models/face_landmarker.task"

# MediaPipe's 33-point pose topology. "Left" is the subject's left, so it
# appears on the right of an un-mirrored frame.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
]

FEATURE_NAMES = [
    "face_present", "head_pitch", "head_yaw", "head_roll", "eye_look_down",
    "pose_present", "wrist_l_y", "wrist_r_y", "wrist_l_vis", "wrist_r_vis",
    "wrist_l_pres", "wrist_r_pres", "wrist_l_oob", "wrist_r_oob",
    "wrist_sep", "nose_y", "shoulder_w",
]


def out_of_bounds(p):
    """How far outside the image a landmark sits, 0 if inside.

    MediaPipe regresses the whole 33-point skeleton whether or not each joint
    is actually in shot, and it extrapolates past the image edge - so a wrist
    under the desk comes back at y > 1.0. That is a far more literal
    "not in frame" signal than visibility, which stays high because the model
    can infer a wrist perfectly well from the shoulder and elbow above it.
    """
    return max(0.0, -p.x, p.x - 1.0, -p.y, p.y - 1.0)


def euler_from_matrix(m):
    """4x4 facial transformation matrix -> (pitch, yaw, roll) in degrees.

    Standard XYZ Euler extraction from the rotation block. Which sign means
    "looking down" depends on the camera's orientation, so verify it against
    the live readout rather than trusting a convention.
    """
    r = np.asarray(m)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def blendshape(cats, name):
    for c in cats:
        if c.category_name == name:
            return float(c.score)
    return 0.0


def extract(pose_res, face_res):
    """Turn raw landmarks into a fixed-length feature vector.

    Everything positional is divided by shoulder width. Raw coordinates encode
    how far you happen to be sitting from the camera; dividing by a body-scale
    reference makes the features mean the same thing across sessions, which is
    what lets a model trained today still work next week.
    """
    f = {k: 0.0 for k in FEATURE_NAMES}

    if face_res.facial_transformation_matrixes:
        f["face_present"] = 1.0
        pitch, yaw, roll = euler_from_matrix(face_res.facial_transformation_matrixes[0])
        f["head_pitch"], f["head_yaw"], f["head_roll"] = pitch, yaw, roll
        if face_res.face_blendshapes:
            bs = face_res.face_blendshapes[0]
            f["eye_look_down"] = 0.5 * (blendshape(bs, "eyeLookDownLeft")
                                        + blendshape(bs, "eyeLookDownRight"))

    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks[0]
        f["pose_present"] = 1.0
        ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
        # Shoulder width doubles as a distance-from-camera proxy.
        sw = max(math.hypot(ls.x - rs.x, ls.y - rs.y), 1e-3)
        mid_x, mid_y = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
        lw, rw = lm[L_WRIST], lm[R_WRIST]

        # Positive = below the shoulder line (hands down); negative = raised.
        f["wrist_l_y"] = (lw.y - mid_y) / sw
        f["wrist_r_y"] = (rw.y - mid_y) / sw
        # visibility = "is it unoccluded", presence = "is it in frame". They
        # are separate heads and can disagree; log both and let the fusion
        # model in Step 6 decide which one carries information.
        f["wrist_l_vis"] = float(getattr(lw, "visibility", 0.0) or 0.0)
        f["wrist_r_vis"] = float(getattr(rw, "visibility", 0.0) or 0.0)
        f["wrist_l_pres"] = float(getattr(lw, "presence", 0.0) or 0.0)
        f["wrist_r_pres"] = float(getattr(rw, "presence", 0.0) or 0.0)
        f["wrist_l_oob"] = out_of_bounds(lw)
        f["wrist_r_oob"] = out_of_bounds(rw)
        f["wrist_sep"] = math.hypot(lw.x - rw.x, lw.y - rw.y) / sw
        f["nose_y"] = (lm[NOSE].y - mid_y) / sw
        f["shoulder_w"] = sw

    return f


def draw_skeleton(frame, pose_res):
    if not pose_res.pose_landmarks:
        return
    lm = pose_res.pose_landmarks[0]
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in SKELETON:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for i in (NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST):
        cv2.circle(frame, pts[i], 5, (0, 255, 0), -1)


def draw_panel(frame, feats):
    """Feature readout. Bars are scaled per-feature so motion is visible."""
    h, w = frame.shape[:2]
    x0, y0 = w - 250, 20
    step = 18
    panel = frame.copy()
    cv2.rectangle(panel, (x0 - 10, y0 - 14), (w - 5, y0 + step * len(FEATURE_NAMES) + 6),
                  (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

    ranges = {"head_pitch": 60, "head_yaw": 60, "head_roll": 60, "eye_look_down": 1,
              "wrist_l_y": 3, "wrist_r_y": 3, "wrist_l_vis": 1, "wrist_r_vis": 1,
              "wrist_l_pres": 1, "wrist_r_pres": 1, "wrist_l_oob": 0.5,
              "wrist_r_oob": 0.5, "wrist_sep": 3, "nose_y": 3, "shoulder_w": 0.5,
              "face_present": 1, "pose_present": 1}

    for i, name in enumerate(FEATURE_NAMES):
        v = feats[name]
        y = y0 + i * step
        cv2.putText(frame, f"{name:<14}{v:>7.2f}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        span = ranges.get(name, 1)
        frac = max(-1.0, min(1.0, v / span))
        cx = x0 + 175
        cv2.line(frame, (cx, y - 4), (cx, y + 2), (90, 90, 90), 1)
        end = int(cx + frac * 55)
        colour = (80, 200, 255) if frac >= 0 else (255, 140, 80)
        cv2.line(frame, (cx, y - 1), (end, y - 1), colour, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--mirror", action="store_true",
                    help="flip horizontally so the preview reads like a mirror")
    ap.add_argument("--protocol", action="store_true",
                    help="guided mode: same scenario sequence detect.py uses")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--lead", type=float, default=6)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomise scenario order so elapsed time stops "
                         "correlating with the label")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed; record it, it identifies the session")
    args = ap.parse_args()
    order = protocol.make_order(args.shuffle, args.seed)
    source = int(args.source) if str(args.source).isdigit() else args.source

    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True))

    writer = handle = None
    if not args.no_log:
        Path("data").mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_seed{args.seed}" if args.shuffle else ""
        out = Path("data") / f"pose_{stamp}{tag}.csv"
        handle = open(out, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            handle, fieldnames=["ts_iso", "t_elapsed", "frame_idx", "scenario",
                                "seed", "infer_ms"] + FEATURE_NAMES)
        writer.writeheader()
        print(f"logging to {out}")

    protocol.print_menu(args.protocol, args.seconds, args.lead, order)

    cap = open_camera(source)
    t0 = time.perf_counter()
    last_report, frames_since, fps = t0, 0, 0.0
    last_ts = -1
    frame_idx = 0
    scenario = "idle"
    phase, remaining, announced = "record", 0.0, None

    if not args.protocol:
        print("\nWatch the panel while you move:")
        print("  look down at your lap        -> head_pitch and eye_look_down move")
        print("  hands off keyboard, phone up -> wrist_l_y / wrist_r_y go negative")
        print("  drop hands below the desk    -> watch raw wrist y vs wrist_*_vis")
        print("  lean back                    -> shoulder_w shrinks\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            t_elapsed = now - t0

            if args.protocol:
                phase, sched, remaining = protocol.schedule(
                    t_elapsed, args.seconds, args.lead, order)
                if phase == "done":
                    print("\nprotocol complete.")
                    break
                scenario = sched
                if (phase, scenario) != announced:
                    announced = (phase, scenario)
                    if phase == "lead":
                        print(f"\n>>> next: {scenario} - {protocol.INSTRUCTIONS[scenario]}")
                    else:
                        print(f"    recording {scenario} for {args.seconds:.0f}s")

            # MediaPipe wants RGB - the exact opposite of Ultralytics, which
            # wants BGR. Same frame, two conventions. Getting this backwards
            # does not crash, it just quietly degrades both models.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode needs strictly increasing timestamps.
            ts = max(int(t_elapsed * 1000), last_ts + 1)
            last_ts = ts

            t_infer = time.perf_counter()
            pose_res = pose.detect_for_video(image, ts)
            face_res = face.detect_for_video(image, ts)
            infer_ms = (time.perf_counter() - t_infer) * 1000

            feats = extract(pose_res, face_res)

            if writer and phase == "record":
                row = {"ts_iso": datetime.now().isoformat(timespec="milliseconds"),
                       "t_elapsed": round(t_elapsed, 3), "frame_idx": frame_idx,
                       "scenario": scenario,
                       "seed": args.seed if args.shuffle else "",
                       "infer_ms": round(infer_ms, 2)}
                row.update({k: round(v, 5) for k, v in feats.items()})
                writer.writerow(row)
                handle.flush()

            draw_skeleton(frame, pose_res)
            draw_panel(frame, feats)
            if args.protocol:
                protocol.draw_overlay(frame, phase, scenario, remaining)

            frame_idx += 1
            frames_since += 1
            if now - last_report >= 1.0:
                fps = frames_since / (now - last_report)
                frames_since, last_report = 0, now

            h = frame.shape[0]
            # Raw normalised wrist y, unclamped. Anything above 1.00 is below
            # the bottom edge of the image - watch this while you drop a hand
            # under the desk and compare it against wrist_*_vis in the panel.
            if pose_res.pose_landmarks:
                plm = pose_res.pose_landmarks[0]
                cv2.putText(frame,
                            f"raw wrist y  L {plm[L_WRIST].y:+.2f}   R {plm[R_WRIST].y:+.2f}"
                            f"   (>1.00 = below frame)",
                            (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
            cv2.putText(frame, f"{fps:.0f} fps | branch B {infer_ms:.1f} ms",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("branch B: posture + gaze - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.duration and t_elapsed >= args.duration:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        face.close()
        if handle:
            handle.close()


if __name__ == "__main__":
    main()
