"""Compare feature separation across multiple Branch B sessions.

A feature that separates the classes in ONE session may only be tracking
something incidental to that session: the order you ran the scenarios in,
what "working normally" happened to mean that day, where your other hand was.
A feature that separates in EVERY session, in the SAME direction, is measuring
behaviour. That is the difference between a feature you can build on and one
that silently breaks next week.

Usage:
    python src/compare_sessions.py                 # every pose_*.csv in data/
    python src/compare_sessions.py a.csv b.csv     # specific sessions
"""

import csv
import sys
from pathlib import Path

import numpy as np

from analyze_pose import DISTRACTED, FACE_FEATURES, NOT_DISTRACTED, SKIP_COLS, auc


def session_aucs(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    feats = [c for c in rows[0] if c not in SKIP_COLS]
    scen = np.array([r["scenario"] for r in rows])
    data = {f: np.array([float(r[f]) for r in rows]) for f in feats}
    face_ok = data.get("face_present", np.ones(len(rows))) > 0.5
    pos = np.isin(scen, list(DISTRACTED))
    neg = np.isin(scen, list(NOT_DISTRACTED))
    out = {}
    for f in feats:
        v = face_ok if f in FACE_FEATURES else np.ones(len(rows), dtype=bool)
        out[f] = auc(data[f][pos & v], data[f][neg & v])
    return out, int(pos.sum()), int(neg.sum())


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        Path("data").glob("pose_*.csv"), key=lambda p: p.stat().st_mtime)
    if len(paths) < 2:
        print("Need at least two pose_*.csv sessions to compare.")
        return

    sessions, counts = {}, {}
    for p in paths:
        name = p.stem.replace("pose_", "")
        sessions[name], *counts[name] = session_aucs(p)
    names = list(sessions)
    feats = [f for f in sessions[names[0]] if all(f in sessions[n] for n in names)]

    print(f"\n{'=' * 100}")
    print(f"  {len(names)} sessions")
    for n in names:
        print(f"    {n:<28} distracted n={counts[n][0]:<6} not-distracted n={counts[n][1]}")
    print(f"{'=' * 100}\n")

    rows = []
    for f in feats:
        aucs = [sessions[n][f] for n in names]
        strengths = [abs(a - 0.5) for a in aucs]
        signs = [a > 0.5 for a in aucs]
        same_dir = all(signs) or not any(signs)
        lo, hi = min(strengths), max(strengths)
        if not same_dir and hi > 0.08:
            verdict, rank = "UNSTABLE - flips direction", -1.0
        elif lo > 0.20:
            verdict, rank = "robust", lo
        elif lo > 0.10:
            verdict, rank = "moderate", lo
        elif hi > 0.15:
            verdict, rank = "one-session only", lo
        else:
            verdict, rank = "no signal", lo
        rows.append((rank, f, aucs, lo, verdict))

    rows.sort(key=lambda r: -r[0])

    head = f"  {'feature':<15}" + "".join(f"{n[-14:]:>16}" for n in names) + f"{'weakest':>10}   verdict"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for _, f, aucs, lo, verdict in rows:
        line = f"  {f:<15}" + "".join(f"{a:>16.3f}" for a in aucs) + f"{lo:>10.3f}   {verdict}"
        print(line)

    print("\n  'weakest' = smallest |AUC-0.5| across sessions: the feature is only as good")
    print("  as its worst day. 'robust' means it cleared 0.20 in EVERY session, same")
    print("  direction. Anything that flips direction between sessions was not measuring")
    print("  behaviour - it was measuring that session.\n")


if __name__ == "__main__":
    main()
