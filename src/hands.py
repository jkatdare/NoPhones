"""Branch B v2: hand features from MediaPipe, for a fixed desk-facing camera.

v1 (pose.py) read shoulders and face. It was measuring camera tilt, because
the camera moved between scenarios. With the camera fixed pointing at the
desk there is no face and no shoulders - but there is the best possible view
of the hands, and hands are what actually touch phones.

Three features carry most of the weight:

  dist_phone   palm centre to YOLO's phone box centre. Phone in hand => ~0.
               Phone on the desk while you type => large. This is the
               feature that separates "phone next to keyboard while reading"
               from "scrolling at desk level" - YOLO and keystrokes alone
               cannot tell those apart.
  extension    tip-to-knuckle distance / hand scale. HIGH = fingers extended
               (over keys, flat on desk). LOW = wrapped round a phone.
  cx, cy       raw position in frame. Meaningful ONLY because the camera is
               fixed: keyboard height, lap height, rising out of shot.

Hands are ordered left-to-right IN FRAME (h0 = leftmost), not by MediaPipe's
handedness label, which is unreliable from this angle. With a fixed camera,
"the hand on the left of the image" is a stable identity.

box_hand_geometry() answers a different question: where does a YOLO box sit
RELATIVE TO the hand's own axis? A smartwatch lives on the forearm side of
the wrist joint; a phone lives on the finger side. That sign is what tells a
watch from a phone - not size, which also varies with distance from camera.
"""

import math

import cv2

HAND_MODEL = "models/hand_landmarker.task"

WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
FINGERS = [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
           (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

PER_HAND = ["present", "cx", "cy", "extension", "thumb_index", "dist_phone"]
FEATURE_NAMES = ([f"h0_{k}" for k in PER_HAND] + [f"h1_{k}" for k in PER_HAND]
                 + ["n_hands", "min_dist_phone"])

NO_PHONE_DIST = 2.0  # larger than any real normalised distance
NO_HAND_GEOM = (2.0, 0.0)  # (dist_wrist, along_hand) when no hand is in frame


def _d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def box_hand_geometry(box_center, hand_landmarks):
    """Where a box sits relative to the nearest hand, in hand-scale units.

    Returns (dist_wrist, along_hand):
      dist_wrist   box centre to the wrist joint, / hand scale.
      along_hand   signed projection of the box centre onto the wrist ->
                   middle-knuckle axis. ~0 or negative = at/behind the wrist
                   (a watch). ~+0.5..+1.5 = in the palm or fingers (a phone
                   being held). Large = not on this hand at all.

    Uses whichever hand the box is closest to. (2.0, 0.0) if no hands.
    """
    if not hand_landmarks or box_center is None:
        return NO_HAND_GEOM
    bx, by = box_center
    best = None
    for lm in hand_landmarks:
        w, m = lm[WRIST], lm[MIDDLE_MCP]
        ax, ay = m.x - w.x, m.y - w.y
        scale = max(math.hypot(ax, ay), 1e-3)
        dist_wrist = math.hypot(bx - w.x, by - w.y) / scale
        along = ((bx - w.x) * ax + (by - w.y) * ay) / (scale * scale)
        if best is None or dist_wrist < best[0]:
            best = (dist_wrist, along)
    return best


def _hand_features(lm, phone_center):
    """One hand's landmarks -> feature dict. All distances / hand scale."""
    palm_x = (lm[WRIST].x + lm[MIDDLE_MCP].x) / 2
    palm_y = (lm[WRIST].y + lm[MIDDLE_MCP].y) / 2
    scale = max(_d(lm[WRIST], lm[MIDDLE_MCP]), 1e-3)  # wrist -> middle knuckle

    # Extended finger: tip far from its knuckle. Curled: tip near it.
    extension = sum(_d(lm[t], lm[m]) for t, m in FINGERS) / (4 * scale)
    thumb_index = _d(lm[THUMB_TIP], lm[INDEX_TIP]) / scale

    if phone_center is None:
        dist_phone = NO_PHONE_DIST
    else:
        dist_phone = math.hypot(palm_x - phone_center[0], palm_y - phone_center[1])

    return {"present": 1.0, "cx": palm_x, "cy": palm_y, "extension": extension,
            "thumb_index": thumb_index, "dist_phone": dist_phone}


def extract(hand_res, phone_center=None):
    """HandLandmarkerResult (+ optional (cx, cy) of the phone) -> flat dict.

    phone_center is in normalised image coordinates, matching MediaPipe's.
    """
    empty = {"present": 0.0, "cx": 0.0, "cy": 0.0, "extension": 0.0,
             "thumb_index": 0.0, "dist_phone": NO_PHONE_DIST}
    hands = []
    for lm in (hand_res.hand_landmarks or []):
        hands.append(_hand_features(lm, phone_center))
    hands.sort(key=lambda h: h["cx"])  # leftmost in frame first
    hands = hands[:2]

    f = {}
    for i in range(2):
        src = hands[i] if i < len(hands) else empty
        for k in PER_HAND:
            f[f"h{i}_{k}"] = src[k]
    f["n_hands"] = float(len(hands))
    f["min_dist_phone"] = min((h["dist_phone"] for h in hands), default=NO_PHONE_DIST)
    return f


def draw_hands(frame, hand_res):
    if not hand_res.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for lm in hand_res.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 255, 0), -1)
