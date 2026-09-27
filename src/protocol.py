"""Shared guided-protocol machinery.

One scenario list, one label mapping, one timing scheme - shared by every
recorder so sessions are comparable and mergeable.

Scenario design (v3 - fixed camera pointing at the desk. Do NOT move it.):

  Four NOT-distracted scenarios, each a deliberate hard negative for one of
  the signals - so the fusion model is forced to learn that no single branch
  is sufficient:
    work_typing     keystrokes present; should be near-certain
    work_screen     no keystrokes, hands idle; vision alone must carry it
    work_desk       sustained gaze-down while working; defeats gaze-only
    phone_on_desk   phone visible while working; defeats YOLO-only

  Three DISTRACTED scenarios covering the ways Branch A failed:
    phone_hand      normal use - vary grip, angle, and position naturally
    phone_lap       phone in lap, looking down at it
    phone_raised    phone lifted toward the face - leaves the top of frame

The earlier diagnostic scenarios (occluded, dark-on-dark, face-down) were
for measuring where each branch breaks. For TRAINING, natural variety within
a label beats prescribed sub-cases - so they fold into phone_hand.
"""

import cv2

PROTOCOL = [
    "work_typing",
    "work_screen",
    "work_desk",
    "phone_on_desk",
    "phone_hand",
    "phone_lap",
    "phone_raised",
]

# 1 = distracted, 0 = not. This is the training target.
LABELS = {
    "work_typing": 0,
    "work_screen": 0,
    "work_desk": 0,
    "phone_on_desk": 0,
    "phone_hand": 1,
    "phone_lap": 1,
    "phone_raised": 1,
    "idle": -1,  # untagged; never used for training
}

INSTRUCTIONS = {
    "work_typing": "Type. Hands on the keyboard, eyes on the screen. Phone out of sight.",
    "work_screen": "Read or watch the screen. Hands idle - lap or desk. Phone out of sight.",
    "work_desk": "Read or write on something physical on the desk. Phone out of sight.",
    "phone_on_desk": "Phone on the desk, in view, wherever you'd normally leave it. Keep working.",
    "phone_hand": "Use the phone in your hand at desk level. Vary grip and angle.",
    "phone_lap": "Phone in your lap. Look down at it and use it.",
    "phone_raised": "Lift the phone up toward your face and use it - it will leave the frame.",
    "idle": "Untagged.",
}

# Manual-mode keys.
SCENARIOS = {str(i + 1): name for i, name in enumerate(PROTOCOL)}
SCENARIOS["0"] = "idle"


def make_order(shuffle=False, seed=0):
    """Scenario order for one session.

    Fixed order correlates elapsed time with the label, so any feature that
    merely drifts through a session looks discriminative. Always shuffle for
    training data, and record the seed - it identifies the session.
    """
    order = list(PROTOCOL)
    if shuffle:
        import random
        random.Random(seed).shuffle(order)
    return order


def total_seconds(seconds, lead, order=None):
    return len(order or PROTOCOL) * (seconds + lead)


def print_menu(protocol, seconds, lead, order=None):
    order = order or PROTOCOL
    if protocol:
        print(f"\nGuided protocol: {len(order)} scenarios x {seconds:.0f}s "
              f"(+{lead:.0f}s count-in) = {total_seconds(seconds, lead, order):.0f}s total\n")
        for i, name in enumerate(order, 1):
            print(f"  {i}. {name:<15} [{LABELS[name]}]  {INSTRUCTIONS[name]}")
        print("\n  Count-in frames are NOT logged. Press q to abort.\n")
    else:
        print("\nScenario keys (window must have focus):")
        for k, v in sorted(SCENARIOS.items()):
            print(f"  {k} -> {v}")
        print("  q -> quit\n")


def schedule(t, seconds, lead, order=None):
    """Map elapsed seconds onto the protocol.

    Returns (phase, scenario, remaining) with phase in {"lead","record","done"}.
    Only "record" frames get logged: during the count-in you are still moving
    into position, so those frames would carry the next label while showing
    the previous behaviour.
    """
    order = order or PROTOCOL
    block = seconds + lead
    idx = int(t // block)
    if idx >= len(order):
        return "done", None, 0.0
    within = t - idx * block
    if within < lead:
        return "lead", order[idx], lead - within
    return "record", order[idx], block - within


def draw_centered(frame, text, y, scale, colour, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def draw_overlay(frame, phase, scenario, remaining, show_protocol=True):
    """Count-in screen or recording header, drawn in place."""
    h, w = frame.shape[:2]
    if show_protocol and phase == "lead":
        dim = frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(dim, 0.6, frame, 0.4, 0, frame)
        draw_centered(frame, "GET READY", int(h * 0.26), 1.1, (255, 255, 255), 3)
        draw_centered(frame, scenario, int(h * 0.40), 0.9, (0, 255, 255), 2)
        draw_centered(frame, INSTRUCTIONS[scenario], int(h * 0.50), 0.55, (200, 200, 200), 2)
        draw_centered(frame, f"{int(remaining) + 1}", int(h * 0.75), 2.2, (255, 255, 255), 4)
        return

    tag = f"[{scenario}]"
    if show_protocol:
        tag += f"  REC {remaining:4.1f}s"
        cv2.circle(frame, (w - 26, 26), 9, (0, 0, 255), -1)
    cv2.putText(frame, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if show_protocol:
        cv2.putText(frame, INSTRUCTIONS[scenario], (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
