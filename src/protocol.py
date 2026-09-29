"""Shared guided-protocol machinery.

One scenario list, one label mapping, one timing scheme - shared by every
recorder so sessions are comparable and mergeable.

Scenario design (v4 - fixed camera pointing at the desk. Do NOT move it.)

  NOT distracted (label 0) - each a hard negative for one of the signals:
    work_typing     keystrokes present; should be near-certain
    work_screen     no keystrokes, hands idle
    work_desk       reading or writing on the desk, eyes down
    phone_on_desk   phone visible while working; defeats YOLO alone
    drink           a hand raised to your face holding a cup - which looks a
                    lot like a raised phone. Added after real use: training on
                    the raised-phone misses without it would teach "hand at
                    face = phone" and fire on every sip of coffee.

  DISTRACTED (label 1):
    phone_hand      normal use - vary grip, angle, and position naturally
    phone_lap       phone in lap, looking down at it
    phone_raised    phone lifted toward the face
    phone_high      phone held up by your face where the camera loses it -
                    two of the first real-use misses looked like this

  TEST ONLY (label -1, never trained on):
    glance          two cued looks at the phone, 24 s apart. During a glance
                    the phone really is in your hand, so any frame label would
                    be wrong one way or the other - "phone" contradicts
                    "glances are fine", "working" teaches that a phone in hand
                    is not phone use. It exists to test the ALERT rule instead:
                    a glance shorter than the trigger must not set it off.

  glance and drink are done with hands OFF the keyboard. A keystroke vetoes
  alerts for 5 s - long enough to cover a whole glance or sip - so typing
  would make both tests pass no matter what the model does.

Sessions recorded before v4 contain only the seven CORE scenarios and still load.
"""

import cv2

CORE = [
    "work_typing",
    "work_screen",
    "work_desk",
    "phone_on_desk",
    "phone_hand",
    "phone_lap",
    "phone_raised",
]
PROTOCOL = CORE + ["drink", "glance", "phone_high"]

# Frame label for training: 1 = distracted, 0 = not, -1 = never trained on.
LABELS = {
    "work_typing": 0,
    "work_screen": 0,
    "work_desk": 0,
    "phone_on_desk": 0,
    "drink": 0,
    "phone_hand": 1,
    "phone_lap": 1,
    "phone_raised": 1,
    "phone_high": 1,
    "glance": -1,
    "idle": -1,  # untagged
}

# What the ALERT should do during each recording - what tune_alerts scores.
# Only the distracted scenarios should alert; glance must not.
ALERT_EXPECTED = {s: LABELS[s] == 1 for s in PROTOCOL}

INSTRUCTIONS = {
    "work_typing": "Type. Hands on the keyboard, eyes on the screen. Phone out of sight.",
    "work_screen": "Read or watch the screen. Hands idle - lap or desk. Phone out of sight.",
    "work_desk": "Read or write on something physical on the desk. Phone out of sight.",
    "phone_on_desk": "Phone on the desk, in view, wherever you'd normally leave it. Keep working.",
    "drink": "Read the screen, hands off the keyboard. Sip a drink every 5-10 s. Phone away.",
    "phone_hand": "Use the phone in your hand at desk level. Vary grip and angle.",
    "phone_lap": "Phone in your lap. Look down at it and use it.",
    "phone_raised": "Lift the phone up toward your face and use it - it will leave the frame.",
    "phone_high": "Hold the phone up by your face and use it - it's fine if the red box vanishes.",
    "glance": "Read the screen, hands off the keyboard. At GLANCE: look at phone 2-3 s, put it back.",
    "idle": "Untagged.",
}

LABEL_WORDS = {1: "phone", 0: "work", -1: "test"}

GLANCE_FIRST = 3.0    # first cue, seconds into the glance recording
GLANCE_FOR = 3.0      # how long each cue stays on screen
# Cues must be further apart than the longest alert window tuning tries (20 s),
# or two harmless glances land in one window, their evidence adds up, and the
# test measures how OFTEN you glance instead of how LONG. 24 s apart leaves a
# 21 s gap: two isolated glances per 40 s recording.
GLANCE_EVERY = 24.0


def glance_cued(elapsed):
    """True while a glance is cued, `elapsed` seconds into the recording."""
    if elapsed < GLANCE_FIRST:
        return False
    return (elapsed - GLANCE_FIRST) % GLANCE_EVERY < GLANCE_FOR


# Manual-mode keys.
SCENARIOS = {str(i + 1): name for i, name in enumerate(PROTOCOL[:9])}
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
            print(f"  {i:>2}. {name:<15} [{LABEL_WORDS[LABELS[name]]}]  {INSTRUCTIONS[name]}")
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


def draw_overlay(frame, phase, scenario, remaining, show_protocol=True, seconds=None):
    """Count-in screen or recording header, drawn in place.

    Pass `seconds` (the recording length) to get the glance cues.
    """
    h, w = frame.shape[:2]
    if show_protocol and phase == "lead":
        dim = frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(dim, 0.6, frame, 0.4, 0, frame)
        draw_centered(frame, "GET READY", int(h * 0.26), 1.1, (255, 255, 255), 3)
        draw_centered(frame, scenario, int(h * 0.40), 0.9, (0, 255, 255), 2)
        draw_centered(frame, INSTRUCTIONS[scenario], int(h * 0.50), 0.5, (200, 200, 200), 2)
        draw_centered(frame, f"{int(remaining) + 1}", int(h * 0.75), 2.2, (255, 255, 255), 4)
        return

    tag = f"[{scenario}]"
    if show_protocol:
        tag += f"  REC {remaining:4.1f}s"
        cv2.circle(frame, (w - 26, 26), 9, (0, 0, 255), -1)
    cv2.putText(frame, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if show_protocol:
        cv2.putText(frame, INSTRUCTIONS[scenario], (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 2)
    if show_protocol and scenario == "glance" and seconds is not None \
            and glance_cued(seconds - remaining):
        band = frame.copy()
        cv2.rectangle(band, (0, int(h * 0.36)), (w, int(h * 0.56)), (0, 140, 255), -1)
        cv2.addWeighted(band, 0.75, frame, 0.25, 0, frame)
        draw_centered(frame, "GLANCE AT YOUR PHONE", int(h * 0.49), 1.1, (255, 255, 255), 3)
