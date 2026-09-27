"""Sanity-check recorded training sessions before you record more of them.

Per session: frames and fps per scenario, whether each branch is alive, the
key feature means per scenario, and a per-feature AUC against the label.
Fails loudly on the things that would make a session worthless.

    python src/analyze_session.py                    # every session in data/sessions
    python src/analyze_session.py data/sessions/x.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_pose import auc
from protocol import LABELS, PROTOCOL

META = {"ts_iso", "t_elapsed", "frame_idx", "session", "seed", "scenario", "label"}

KEY = ["a_phone_conf", "a_phone_cy", "a_phone_area", "a_phone_along_hand",
       "a_phone_dist_wrist", "a_wrist_boxes", "b_n_hands", "b_min_dist_phone",
       "b_h0_extension", "b_h1_extension", "b_h0_cy", "b_h1_cy",
       "os_keys_5s", "os_mouse_5s", "os_since_input"]


def load(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    feats = [c for c in rows[0] if c not in META]
    scen = np.array([r["scenario"] for r in rows])
    label = np.array([int(r["label"]) for r in rows])
    t = np.array([float(r["t_elapsed"]) for r in rows])
    data = {f: np.array([float(r[f]) for r in rows]) for f in feats}
    return rows, feats, scen, label, t, data


def check(path):
    rows, feats, scen, label, t, d = load(path)
    if "b_n_hands" not in d or not rows:
        print()
        print(f"  SKIP {path.name}: old schema or empty ({len(rows)} rows) - move it to _discard/")
        return
    order = [s for s in PROTOCOL if s in set(scen)]
    warnings = []

    print()
    print("=" * 96)
    print(f"  {path.name}    {len(rows)} rows")
    print("=" * 96)

    print()
    print(f"  {'scenario':<15}{'label':>6}{'frames':>8}{'fps':>7}"
          f"{'phone%':>8}{'hands%':>8}{'keys%':>7}{'A ms':>7}{'B ms':>7}")
    print("  " + "-" * 73)
    for s in order:
        m = scen == s
        n = int(m.sum())
        span = t[m].max() - t[m].min() if n > 1 else 0
        fps = n / span if span > 0 else 0
        phone = (d["a_phone_conf"][m] > 0).mean()
        hands = (d["b_n_hands"][m] > 0).mean()
        keys = (d["os_keys_5s"][m] > 0).mean()
        print(f"  {s:<15}{LABELS[s]:>6}{n:>8}{fps:>7.1f}{phone:>8.0%}{hands:>8.0%}"
              f"{keys:>7.0%}{d['a_ms'][m].mean():>7.1f}{d['b_ms'][m].mean():>7.1f}")
        if fps and fps < 15:
            warnings.append(f"{s}: only {fps:.0f} fps - pipeline is too slow")
        if hands < 0.3:
            warnings.append(f"{s}: hands detected in only {hands:.0%} of frames")
        if s == "work_typing" and keys < 0.5:
            warnings.append(f"work_typing: keystrokes seen in only {keys:.0%} of frames - "
                            "is the listener hooking?")
        if s == "phone_hand" and phone < 0.3:
            warnings.append(f"phone_hand: YOLO saw the phone in only {phone:.0%} of frames")
        if s.startswith("work") and phone > 0.3:
            warnings.append(f"{s}: YOLO fired in {phone:.0%} of frames - was the phone out of sight?")

    print()
    print("  KEY FEATURE MEANS PER SCENARIO")
    print()
    head = f"  {'feature':<18}" + "".join(f"{s[:11]:>12}" for s in order)
    print(head)
    print("  " + "-" * (len(head) - 2))
    # Position / shape features are 0.0 when the hand or phone is absent. Averaging
    # those zeros in makes an absent hand look like one at the top-left corner.
    # So condition each on its own presence flag.
    def valid(f):
        if f.startswith("b_h0_") and f != "b_h0_present":
            return d["b_h0_present"] > 0
        if f.startswith("b_h1_") and f != "b_h1_present":
            return d["b_h1_present"] > 0
        if f.startswith("a_phone_") and f != "a_phone_conf":
            return d["a_phone_conf"] > 0
        return np.ones(len(d[f]), dtype=bool)

    for f in KEY:
        if f not in d:
            continue
        cells = []
        for s in order:
            m = (scen == s) & valid(f)
            cells.append(f"{d[f][m].mean():>12.2f}" if m.sum() else f"{'-':>12}")
        star = "*" if f.startswith(("b_h0_", "b_h1_", "a_phone_c", "a_phone_a", "a_phone_d")) else " "
        print(f"  {f:<17}{star}" + "".join(cells))
    print("  (* = mean over frames where that hand / the phone was actually detected)")

    pos, neg = label == 1, label == 0
    print()
    print(f"  SINGLE-FEATURE AUC vs label   (distracted n={pos.sum()}, not n={neg.sum()})")
    print()
    scored = sorted(((abs(auc(d[f][pos], d[f][neg]) - 0.5), auc(d[f][pos], d[f][neg]), f)
                     for f in feats if f not in ("a_ms", "b_ms")), reverse=True)
    print(f"  {'feature':<20}{'AUC':>8}{'|AUC-.5|':>10}   bar")
    print("  " + "-" * 60)
    for strength, a, f in scored[:14]:
        print(f"  {f:<20}{a:>8.3f}{strength:>10.3f}   " + "#" * int(strength * 60))

    print()
    if warnings:
        print("  WARNINGS")
        for w in warnings:
            print("   ! " + w)
    else:
        print("  no warnings - session looks healthy")
    print()


def confusability(path):
    """For every pair of scenarios, how separable are they at all?

    Score = the best |AUC-0.5| any single feature achieves between the pair.
    It is an optimistic bound: a model combining features can do better, but
    if NO single feature separates a pair, they are probably close to
    indistinguishable in this feature space.

    Only CROSS-LABEL pairs matter. The model never needs to tell work_typing
    from work_desk - both are "not distracted". It absolutely must tell
    work_screen from phone_lap.
    """
    rows, feats, scen, label, t, d = load(path)
    if "b_n_hands" not in d:
        return
    order = [s for s in PROTOCOL if s in set(scen)]
    feats = [f for f in feats if f not in ("a_ms", "b_ms")]

    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            ma, mb = scen == a, scen == b
            if ma.sum() < 30 or mb.sum() < 30:
                continue
            best_s, best_f = 0.0, ""
            for f in feats:
                s_ = abs(auc(d[f][mb], d[f][ma]) - 0.5)
                if s_ > best_s:
                    best_s, best_f = s_, f
            cross = LABELS[a] != LABELS[b]
            pairs.append((best_s, a, b, best_f, cross))

    cross_pairs = sorted([p for p in pairs if p[4]])
    print()
    print("  CROSS-LABEL PAIR SEPARABILITY  (worst first - these must all be separable)")
    print()
    print(f"  {'scenario A':<15}{'scenario B':<15}{'best':>7}   {'by feature':<20} verdict")
    print("  " + "-" * 78)
    for best_s, a, b, f, _ in cross_pairs:
        if best_s < 0.15:
            verdict = "INSEPARABLE"
        elif best_s < 0.25:
            verdict = "weak"
        else:
            verdict = "ok"
        print(f"  {a:<15}{b:<15}{best_s:>7.3f}   {f:<20} {verdict}")

    same = sorted([p for p in pairs if not p[4]])[:3]
    if same:
        print()
        print("  (most similar same-label pairs, for reference - similarity here is harmless)")
        for best_s, a, b, f, _ in same:
            print(f"  {a:<15}{b:<15}{best_s:>7.3f}   {f}")
    print()


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        Path("data/sessions").glob("session_*.csv"), key=lambda p: p.stat().st_mtime)
    if not paths:
        print("no sessions found in data/sessions")
        return
    for p in paths:
        check(p)
        confusability(p)


if __name__ == "__main__":
    main()
