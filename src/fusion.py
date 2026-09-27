"""Shared preprocessing for the fusion model - used by train.py AND monitor.py.

The model must see identical features at training time and at run time. Any
difference between the two paths - a window computed slightly differently, a
placeholder handled one way here and another way there - is "train/serve
skew", the most common way a model that scored well in evaluation quietly
misbehaves once it is running live. So the cleaning, the rolling window and
the exact column list live in this one file, and both sides call it.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

import protocol
from features import FEATURE_NAMES
from hands import NO_HAND_GEOM, NO_PHONE_DIST

# Pipeline timing is not behaviour. Worse, it differs by session - seed 3's
# first attempt ran YOLO at 87 ms, seed 1 at 15 ms - so a model allowed to see
# it can learn "which session is this?". Under leave-one-session-out that is a
# leak that flatters the score and then fails in real use.
EXCLUDE = {"a_ms", "b_ms"}
BASE_FEATURES = [f for f in FEATURE_NAMES if f not in EXCLUDE]

# A single frame is noisy: a hand landmark jitters, YOLO blinks on a background
# object for one frame at 0.2 confidence. The trailing one-second average is
# what the behaviour actually looks like.
ROLL_FEATURES = ["a_phone_conf", "a_wrist_boxes", "b_n_hands", "b_min_dist_phone",
                 "b_h0_extension", "b_h1_extension", "b_h0_cy", "b_h1_cy"]
ROLL_WINDOW = "1s"
ROLL_NAMES = [f"{f}_r1s" for f in ROLL_FEATURES]

MODEL_FEATURES = BASE_FEATURES + ROLL_NAMES

# Branch membership by prefix. A rolled feature belongs to the branch it was
# rolled from, so ablations stay clean: "A only" means A and nothing derived
# from anything else.
BRANCHES = {"A": "a_", "B": "b_", "OS": "os_"}

MIN_FRAMES_PER_SCENARIO = 300   # ~12 s at the slowest fps seen; below this a
                                # session is too thin to trust as a test fold


def columns_for(branches):
    """Model features belonging to any of the given branches, e.g. ("A", "OS")."""
    prefixes = tuple(BRANCHES[b] for b in branches)
    return [f for f in MODEL_FEATURES if f.startswith(prefixes)]


def clean(df):
    """Replace placeholder values with NaN.

    When something is absent the extractor writes a fixed placeholder: 0.0 for
    the position of a hand that is not there, 2.0 for the distance to a phone
    that does not exist, (2.0, 0.0) for wrist geometry with no hand. Those
    placeholders collide with real values - cx = 0.0 is also a genuine position
    at the left edge, and along_hand = 0.0 is exactly where a watch sits.

    NaN says "absent" without pretending to be a measurement. Histogram
    gradient boosting handles NaN natively: at every split it learns which way
    missing values should go, so the model discovers what absence means.
    """
    df = df.copy()
    for i in (0, 1):
        absent = df[f"b_h{i}_present"] <= 0
        for k in ("cx", "cy", "extension", "thumb_index", "dist_phone"):
            df.loc[absent, f"b_h{i}_{k}"] = np.nan
    for k in ("b_h0_dist_phone", "b_h1_dist_phone", "b_min_dist_phone"):
        df.loc[df[k] >= NO_PHONE_DIST, k] = np.nan

    no_phone = df["a_phone_conf"] <= 0
    df.loc[no_phone, ["a_phone_cx", "a_phone_cy", "a_phone_area"]] = np.nan
    no_geom = no_phone | (np.isclose(df["a_phone_dist_wrist"], NO_HAND_GEOM[0])
                          & np.isclose(df["a_phone_along_hand"], NO_HAND_GEOM[1]))
    df.loc[no_geom, ["a_phone_dist_wrist", "a_phone_along_hand"]] = np.nan
    return df


def add_rolling(df, window=ROLL_WINDOW):
    """Trailing time-window means, computed separately for each session.

    By TIME, not frame count: sessions ran anywhere from 19 to 29 fps, so a
    30-frame window would mean 1.0 s in one session and 1.6 s in another.
    Trailing only - at run time the future does not exist yet.
    The protocol's 8-second count-ins are never logged, so a 1-second window
    cannot bleed from one scenario into the next.
    """
    parts = []
    for _, g in df.groupby("session", sort=False):
        g = g.sort_values("t_elapsed").reset_index(drop=True)
        when = pd.Timestamp(0) + pd.to_timedelta(g["t_elapsed"], unit="s")
        rolled = (g[ROLL_FEATURES].set_axis(pd.DatetimeIndex(when))
                  .rolling(window, min_periods=1).mean().reset_index(drop=True))
        rolled.columns = ROLL_NAMES
        parts.append(pd.concat([g, rolled], axis=1))
    return pd.concat(parts, ignore_index=True)


def prepare(df):
    """clean -> add_rolling. The one entry point both train and monitor use."""
    return add_rolling(clean(df))


def load_sessions(folder="data/sessions", verbose=True):
    """Every complete training session, validated.

    Skips (loudly) anything with the wrong schema or a missing / thin
    scenario - a partial session as a test fold would quietly skew every
    number, which is exactly what the 241-row seed 3 would have done.
    """
    frames = []
    for p in sorted(Path(folder).glob("session_*.csv")):
        df = pd.read_csv(p)
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            if verbose:
                print(f"  skip {p.name}: old schema, {len(missing)} feature columns missing")
            continue
        counts = df["scenario"].value_counts()
        thin = [s for s in protocol.PROTOCOL if counts.get(s, 0) < MIN_FRAMES_PER_SCENARIO]
        if thin:
            if verbose:
                print(f"  skip {p.name}: incomplete - {', '.join(thin)} "
                      f"under {MIN_FRAMES_PER_SCENARIO} frames")
            continue
        frames.append(df)
    if not frames:
        raise SystemExit("No complete sessions in data/sessions.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    return df


class LiveFusion:
    """Frame-by-frame scoring, using EXACTLY the training preprocessing.

    Keeps the last ~1.5 s of raw feature rows and runs them through prepare() -
    the very function train.py used - then scores the newest row. An
    incremental rolling mean would be faster, but it would be a second
    implementation of the same maths, and a second implementation is where
    train/serve skew creeps in. test_live.py replays a recorded session through
    this class and through the batch path and checks they agree.
    """

    def __init__(self, bundle, keep_seconds=1.5):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.keep = keep_seconds
        self.buf = deque()

    def reset(self):
        self.buf.clear()

    def score(self, feats, t):
        row = {k: feats[k] for k in FEATURE_NAMES}
        row["t_elapsed"] = t
        row["session"] = "live"
        self.buf.append(row)
        while t - self.buf[0]["t_elapsed"] > self.keep:
            self.buf.popleft()
        X = prepare(pd.DataFrame(list(self.buf)))
        return float(self.model.predict_proba(X.iloc[[-1]][self.features])[0, 1])
