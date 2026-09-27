"""One feature vector per frame, from every source the fusion model will see.

Three sources, three prefixes - so ablations are a column filter:

  a_*   Branch A, YOLO phone detection      "is there a phone, and where?"
  b_*   Branch B, MediaPipe hands           "what are the hands doing, and
                                             how close are they to it?"
  os_*  keyboard / mouse activity           "are they typing?"

The camera is FIXED, pointing at the desk. That is a hard requirement, not a
preference: raw positions in frame are features now, and a model trained on
one angle has never seen any other.

Smartwatches: YOLO's COCO "cell phone" class fires on a watch face. A watch
sits on the forearm side of the wrist joint, a held phone on the finger side,
so box selection prefers finger-side boxes (a watch only becomes the "phone"
if nothing else is in frame) and the box's geometry relative to the hand is
logged so the model can learn to discount it. No box is ever discarded.

The recorder (record.py) and the live monitor (monitor.py) both go through
this one class, so the model sees identical features at train and test time.
Drift between those two paths is the most common way a trained model quietly
stops working in production.
"""

import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import torch
from mediapipe.tasks.python import BaseOptions, vision
from ultralytics import YOLO

import hands as hands_mod

PHONE_CLASS_ID = 67

# A box is "wrist-side" when it sits at or behind the wrist joint along the
# hand axis and close to it. That is where a watch lives and a phone does not.
WRIST_SIDE_ALONG = 0.30   # hand-scale units along wrist->knuckle axis
WRIST_SIDE_DIST = 0.80    # hand-scale units from the wrist joint

A_FEATURES = ["a_phone_conf", "a_phone_cx", "a_phone_cy", "a_phone_area",
              "a_phone_dist_wrist", "a_phone_along_hand",
              "a_n_boxes", "a_wrist_boxes", "a_ms"]
B_FEATURES = [f"b_{n}" for n in hands_mod.FEATURE_NAMES] + ["b_ms"]
OS_FEATURES = ["os_keys_1s", "os_keys_5s", "os_mouse_1s", "os_mouse_5s", "os_since_input"]
FEATURE_NAMES = A_FEATURES + B_FEATURES + OS_FEATURES


class InputMonitor:
    """Counts keyboard and mouse events in rolling windows.

    Timing only. The key pressed is never read, stored, or logged - the
    on_press callback discards its argument.

    Keystrokes are strong evidence AGAINST distraction; silence is not
    evidence FOR it. The feature is asymmetric and the model learns that.
    """

    def __init__(self):
        self._keys = deque()
        self._mouse = deque()
        self._lock = threading.Lock()
        self._last = None
        try:
            from pynput import keyboard, mouse
            keyboard.Listener(on_press=self._on_key, daemon=True).start()
            mouse.Listener(on_move=self._on_mouse, on_click=self._on_mouse,
                           on_scroll=self._on_mouse, daemon=True).start()
            self.available = True
        except Exception as e:  # pragma: no cover
            print(f"InputMonitor unavailable ({e}); os_* features will be zero.")
            self.available = False

    def _on_key(self, _key):  # the key itself is deliberately ignored
        self._stamp(self._keys)

    def _on_mouse(self, *_args):
        self._stamp(self._mouse)

    def _stamp(self, dq):
        now = time.monotonic()
        with self._lock:
            dq.append(now)
            self._last = now

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            for dq in (self._keys, self._mouse):
                while dq and now - dq[0] > 5.0:
                    dq.popleft()
            k1 = sum(1 for t in self._keys if now - t <= 1.0)
            m1 = sum(1 for t in self._mouse if now - t <= 1.0)
            since = (now - self._last) if self._last is not None else 60.0
            return {
                "os_keys_1s": k1,
                "os_keys_5s": len(self._keys),
                "os_mouse_1s": m1,
                "os_mouse_5s": len(self._mouse),
                "os_since_input": round(min(since, 60.0), 2),
            }


class FeatureExtractor:
    def __init__(self, yolo_model="yolo11x.pt", conf=0.15, use_os=True):
        self.conf = conf
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.device_name = (f"cuda:0 ({torch.cuda.get_device_name(0)})"
                            if torch.cuda.is_available()
                            else "CPU - CUDA not available, expect ~5x slower")
        self.yolo = YOLO(yolo_model)
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hands_mod.HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))
        self.inputs = InputMonitor() if use_os else None
        self._last_ts = -1
        self.last_boxes = []      # (x1, y1, x2, y2, conf, wrist_side)
        self.last_hands = None

    def __call__(self, frame_bgr, t_elapsed):
        h, w = frame_bgr.shape[:2]
        f = {}

        # ---- Branch A: YOLO expects BGR ------------------------------------
        r = self.yolo.predict(frame_bgr, classes=[PHONE_CLASS_ID], conf=self.conf,
                              device=self.device, verbose=False)[0]
        raw_boxes = [(*box.xyxy[0].tolist(), float(box.conf[0])) for box in r.boxes]
        f["a_ms"] = float(r.speed.get("inference", 0.0))

        # ---- Branch B: MediaPipe expects RGB --------------------------------
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(t_elapsed * 1000), self._last_ts + 1)
        self._last_ts = ts
        t0 = time.perf_counter()
        hand_res = self.hands.detect_for_video(image, ts)
        f["b_ms"] = (time.perf_counter() - t0) * 1000
        self.last_hands = hand_res
        landmarks = hand_res.hand_landmarks or []

        # ---- Box selection: prefer anything that is NOT on a wrist ----------
        scored = []
        for x1, y1, x2, y2, c in raw_boxes:
            centre = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)
            dist_wrist, along = hands_mod.box_hand_geometry(centre, landmarks)
            wrist_side = along < WRIST_SIDE_ALONG and dist_wrist < WRIST_SIDE_DIST
            scored.append((wrist_side, -c, x1, y1, x2, y2, c, centre, dist_wrist, along))
        scored.sort()  # non-wrist first, then highest confidence
        self.last_boxes = [(s[2], s[3], s[4], s[5], s[6], s[0]) for s in scored]

        phone_center = None
        if scored:
            wrist_side, _, x1, y1, x2, y2, c, centre, dist_wrist, along = scored[0]
            phone_center = centre
            f["a_phone_conf"] = c
            f["a_phone_cx"], f["a_phone_cy"] = centre
            f["a_phone_area"] = (x2 - x1) * (y2 - y1) / (w * h)
            f["a_phone_dist_wrist"] = dist_wrist
            f["a_phone_along_hand"] = along
        else:
            f["a_phone_conf"] = f["a_phone_cx"] = f["a_phone_cy"] = f["a_phone_area"] = 0.0
            f["a_phone_dist_wrist"], f["a_phone_along_hand"] = hands_mod.NO_HAND_GEOM
        f["a_n_boxes"] = len(raw_boxes)
        f["a_wrist_boxes"] = sum(1 for s in scored if s[0])

        for k, v in hands_mod.extract(hand_res, phone_center).items():
            f[f"b_{k}"] = v

        # ---- OS input ------------------------------------------------------
        if self.inputs:
            f.update(self.inputs.snapshot())
        else:
            f.update({k: 0.0 for k in OS_FEATURES})

        return f

    def draw(self, frame):
        for x1, y1, x2, y2, c, wrist_side in self.last_boxes:
            colour = (0, 165, 255) if wrist_side else (0, 0, 255)   # orange = watch?
            label = f"{c:.2f}" + (" wrist" if wrist_side else "")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
            cv2.putText(frame, label, (int(x1), max(int(y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        if self.last_hands is not None:
            hands_mod.draw_hands(frame, self.last_hands)

    def close(self):
        self.hands.close()
