"""Analyse a Branch B feature log from pose.py.

Answers what you should not have to eyeball:

  1. Does each feature move between scenarios? (per-scenario means)
  2. How well does each feature ALONE separate distracted from not-distracted?
  3. Is that separation real, or an artefact of the protocol running scenarios
     in a fixed order? (drift column - read the warning below)

Usage:
    python src/analyze_pose.py                # newest pose_*.csv in data/
    python src/analyze_pose.py data/pose_x.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# THE LABEL DECISION. Yours to make - edit it deliberately.
#
# "phone_in_lap" is the debatable one. Phone resting on your leg while you keep
# working belongs in NOT_DISTRACTED; looking down at it belongs here. What you
# choose changes every metric below and every model trained afterwards.
# ---------------------------------------------------------------------------
DISTRACTED = {
    "phone_normal",
    "phone_occluded",
    "phone_arms_length",
    "phone_dark_on_dark",
    "phone_in_lap",
}
NOT_DISTRACTED = {"clutter_only", "phone_facedown_desk"}

SKIP_COLS = {"ts_iso", "t_elapsed", "frame_idx", "scenario", "infer_ms", "seed"}

# When no face is detected these come back as 0.0 - a default, not a
# measurement. Averaging those zeros in with real values drags the mean toward
# zero in exactly the scenarios where the face is hardest to see. So every
# statistic for these is computed only over frames with face_present == 1.
FACE_FEATURES = {"head_pitch", "head_yaw", "head_roll", "eye_look_down"}


def rankdata(a):
    """Average ranks, ties shared. Equivalent to scipy.stats.rankdata."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auc(pos, neg):
    """Probability a random positive scores above a random negative.

    0.5 = no information. 1.0 = perfect. 0.0 = perfect but inverted, so what
    matters is distance FROM 0.5. Computed from rank sums (the Mann-Whitney U
    identity), so it needs no threshold and handles ties.
    """
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def drift(values, times, scen, valid):
    """Mean |correlation with time| measured WITHIN each scenario.

    The protocol runs scenarios in a fixed order, so elapsed time is almost
    perfectly correlated with the label. Any feature that simply drifts as you
    settle into your chair will therefore look discriminative when it is only
    tracking the clock. Correlating within a single scenario - where the label
    is constant - isolates that drift.

    High drift does not prove a feature is worthless, but it does mean its AUC
    is not trustworthy until you re-run with the scenario order shuffled.
    """
    out = []
    for s in set(scen):
        m = (scen == s) & valid
        if m.sum() < 30:
            continue
        v, t = values[m], times[m]
        if v.std() < 1e-9 or t.std() < 1e-9:
            continue
        out.append(abs(float(np.corrcoef(v, t)[0, 1])))
    return float(np.mean(out)) if out else 0.0


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        files = sorted(Path("data").glob("pose_*.csv"), key=lambda p: p.stat().st_mtime)
        if not files:
            print("No pose_*.csv in data/ - run: python src/pose.py --protocol")
            return
        path = files[-1]

    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print(f"{path} is empty")
        return

    feats = [c for c in rows[0] if c not in SKIP_COLS]
    scen = np.array([r["scenario"] for r in rows])
    times = np.array([float(r["t_elapsed"]) for r in rows])
    data = {f: np.array([float(r[f]) for r in rows]) for f in feats}
    order = list(dict.fromkeys(scen))
    face_ok = data.get("face_present", np.ones(len(rows))) > 0.5

    def valid_mask(f):
        return face_ok if f in FACE_FEATURES else np.ones(len(rows), dtype=bool)

    print(f"\n{'=' * 102}\n  {path.name}   {len(rows)} frames, {len(feats)} features\n{'=' * 102}")

    print("\n  MEAN PER SCENARIO   (* = computed only over frames with a detected face)\n")
    head = f"  {'feature':<16}" + "".join(f"{s[:11]:>12}" for s in order)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for f in feats:
        star = "*" if f in FACE_FEATURES else " "
        line = f"  {f:<15}{star}"
        for s in order:
            m = (scen == s) & valid_mask(f)
            line += f"{data[f][m].mean():>12.2f}" if m.sum() else f"{'-':>12}"
        print(line)

    # Face availability per scenario - a low number means the starred rows
    # above rest on few frames, and that face_present itself carries signal.
    line = f"  {'(face frames)':<16}"
    for s in order:
        line += f"{(face_ok & (scen == s)).sum():>12}"
    print(line)

    mask_pos = np.isin(scen, list(DISTRACTED))
    mask_neg = np.isin(scen, list(NOT_DISTRACTED))
    print(f"\n  SEPARATION: distracted (n={mask_pos.sum()}) vs "
          f"not-distracted (n={mask_neg.sum()})")
    print(f"  distracted     = {sorted(DISTRACTED)}")
    print(f"  not distracted = {sorted(NOT_DISTRACTED)}\n")

    scored = []
    for f in feats:
        v = valid_mask(f)
        a = auc(data[f][mask_pos & v], data[f][mask_neg & v])
        d = drift(data[f], times, scen, v)
        scored.append((abs(a - 0.5), a, d, f))
    scored.sort(reverse=True)

    print(f"  {'feature':<16}{'AUC':>8}{'|AUC-.5|':>10}{'drift':>8}"
          f"{'mean D':>10}{'mean ND':>10}   verdict")
    print("  " + "-" * 92)
    for strength, a, d, f in scored:
        v = valid_mask(f)
        md = data[f][mask_pos & v].mean() if (mask_pos & v).sum() else float("nan")
        mn = data[f][mask_neg & v].mean() if (mask_neg & v).sum() else float("nan")
        if strength <= 0.04:
            verdict = "no signal"
        elif d > 0.45:
            verdict = "SUSPECT - tracks the clock"
        elif strength > 0.25:
            verdict = "strong"
        elif strength > 0.10:
            verdict = "useful"
        else:
            verdict = "weak"
        print(f"  {f:<16}{a:>8.3f}{strength:>10.3f}{d:>8.2f}"
              f"{md:>10.2f}{mn:>10.2f}   {verdict}")

    print("\n  AUC < 0.5 just means the feature runs the other way; only distance")
    print("  from 0.5 matters. 'drift' is mean |corr with time| measured inside a")
    print("  single scenario, where the label cannot change - so anything high is")
    print("  tracking session time, not behaviour, and its AUC is not credible.\n")


if __name__ == "__main__":
    main()
