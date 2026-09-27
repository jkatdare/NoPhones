"""Analyse a detection log from detect.py.

Usage:
    python src/analyze.py                    # newest CSV in data/
    python src/analyze.py data/some.csv

The interesting question this answers without any scenario tags: which
detections are furniture and which are the phone. A false positive on a
calculator sits in the same pixels all session; a phone moves. So cluster
detections by normalised position and look at how concentrated each cluster
is in space and how persistent it is in time.
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GRID = 16  # bins per axis for spatial clustering


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def scenario_breakdown(rows, grid=GRID, baseline_tag="clutter_only", min_occ=0.05):
    """Separate 'saw the phone' from 'saw anything'.

    A raw per-scenario detection rate is confounded: your calculator and wrist
    rest keep firing no matter which scenario you are acting out. So build a
    baseline from the clutter_only segment - the cells that fire when no phone
    is present - and count only detections in OTHER cells as phone-attributable.

    That turns detection rate into something closer to real recall. It is not
    perfect: if the phone happens to sit in a baseline cell it gets discounted,
    which makes these numbers a LOWER bound.
    """
    frames_by = defaultdict(dict)
    for r in rows:
        sc = r["scenario"]
        fi = int(r["frame_idx"])
        f = frames_by[sc].setdefault(fi, [])
        if r["conf"] not in ("", None):
            cx, cy = float(r["cx_norm"]), float(r["cy_norm"])
            key = (min(int(cx * grid), grid - 1), min(int(cy * grid), grid - 1))
            f.append((key, float(r["conf"])))

    base = frames_by.get(baseline_tag)
    if not base:
        print(f"\n  (no '{baseline_tag}' segment - skipping baseline correction)")
        return

    occ = Counter()
    for cells in base.values():
        for key in {k for k, _ in cells}:
            occ[key] += 1
    baseline_cells = {k for k, v in occ.items() if v / len(base) >= min_occ}

    print(f"\n  baseline correction using '{baseline_tag}' ({len(base)} frames)")
    print(f"  cells firing in >={min_occ:.0%} of baseline frames are treated as clutter: "
          f"{sorted(baseline_cells)}")
    print(f"\n  {'scenario':<22}{'frames':>8}{'raw det':>9}{'phone det':>11}"
          f"{'phone conf':>12}{'median':>9}")
    print("  " + "-" * 72)

    order = [s for s in frames_by if s != baseline_tag] + [baseline_tag]
    for sc in order:
        fr = frames_by[sc]
        n = len(fr)
        raw = sum(1 for c in fr.values() if c)
        novel_conf = []
        for cells in fr.values():
            outside = [c for k, c in cells if k not in baseline_cells]
            if outside:
                novel_conf.append(max(outside))
        marker = "  <- baseline" if sc == baseline_tag else ""
        if novel_conf:
            arr = np.array(novel_conf)
            print(f"  {sc:<22}{n:>8}{raw / n:>8.0%}{len(arr) / n:>11.0%}"
                  f"{arr.mean():>12.2f}{np.median(arr):>9.2f}{marker}")
        else:
            print(f"  {sc:<22}{n:>8}{raw / n:>8.0%}{0:>11.0%}{'-':>12}{'-':>9}{marker}")


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        files = sorted(Path("data").glob("detections_*.csv"), key=lambda p: p.stat().st_mtime)
        if not files:
            print("No detection CSVs in data/")
            return
        path = files[-1]

    rows = load(path)
    if not rows:
        print(f"{path} is empty")
        return

    # Per-frame view: a frame appears once per detection, or once with n_det=0.
    frames = {}
    dets = []
    for r in rows:
        fi = int(r["frame_idx"])
        t = float(r["t_elapsed"])
        n = int(r["n_det"])
        frames.setdefault(fi, {"t": t, "n": n, "best": 0.0, "ms": float(r["infer_ms"])})
        if r["conf"] not in ("", None):
            c = float(r["conf"])
            frames[fi]["best"] = max(frames[fi]["best"], c)
            dets.append({
                "t": t, "frame": fi, "conf": c,
                "cx": float(r["cx_norm"]), "cy": float(r["cy_norm"]),
                "area": float(r["area_frac"]),
            })

    n_frames = len(frames)
    duration = max(f["t"] for f in frames.values())
    best = np.array([f["best"] for f in frames.values()])
    hits = best[best > 0]
    ms = np.array([f["ms"] for f in frames.values()])
    ndet = np.array([f["n"] for f in frames.values()])

    print(f"\n{'=' * 78}")
    print(f"  {path.name}")
    print(f"{'=' * 78}")
    print(f"  frames            {n_frames}")
    print(f"  duration          {duration:.1f}s   ({n_frames / duration:.1f} fps)")
    print(f"  inference         {ms.mean():.1f} ms mean  ->  {1000 / ms.mean():.0f} fps ceiling")
    print(f"  detection rate    {len(hits) / n_frames:.0%} of frames had >=1 box")
    print(f"  total detections  {len(dets)}")

    print(f"\n  confidence of best box per frame (detections only)")
    for p in (10, 25, 50, 75, 90, 99):
        print(f"    p{p:<3} {pct(hits, p):.3f}")
    print(f"    max  {hits.max():.3f}")

    print(f"\n  boxes per frame")
    for k in range(0, int(ndet.max()) + 1):
        share = (ndet == k).sum() / n_frames
        print(f"    {k} box{'es' if k != 1 else '  '}  {share:>6.1%}  {(ndet == k).sum():>5}")

    # ---- spatial clustering -------------------------------------------------
    cells = defaultdict(list)
    for d in dets:
        key = (min(int(d["cx"] * GRID), GRID - 1), min(int(d["cy"] * GRID), GRID - 1))
        cells[key].append(d)

    print(f"\n  distinct {GRID}x{GRID} cells occupied: {len(cells)}")
    print(f"\n  {'cell':<10}{'n':>6}{'share':>8}{'conf':>7}{'area':>8}"
          f"{'t_first':>9}{'t_last':>8}{'span%':>7}  verdict")
    print("  " + "-" * 76)

    ranked = sorted(cells.items(), key=lambda kv: -len(kv[1]))
    for (gx, gy), ds in ranked[:10]:
        ts = np.array([d["t"] for d in ds])
        cf = np.array([d["conf"] for d in ds])
        ar = np.array([d["area"] for d in ds])
        span = (ts.max() - ts.min()) / duration
        share = len(ds) / len(dets)
        # Span is the honest signal. A box present across most of the session
        # never went away; one confined to a short window was an event.
        # Deliberately NOT calling these "furniture" - a phone left resting on
        # the desk all session is indistinguishable from a calculator here.
        # Separating those two needs scenario tags, not geometry.
        verdict = "persistent" if span > 0.70 else ("episodic" if span < 0.35 else "mixed")
        print(f"  ({gx:>2},{gy:>2}){len(ds):>8}{share:>8.1%}{cf.mean():>7.2f}"
              f"{ar.mean():>8.4f}{ts.min():>9.1f}{ts.max():>8.1f}{span:>7.0%}  {verdict}")

    # ---- does box size predict confidence? ----------------------------------
    if len(dets) > 40:
        areas = np.array([d["area"] for d in dets])
        confs = np.array([d["conf"] for d in dets])
        qs = np.percentile(areas, [25, 50, 75])
        print(f"\n  confidence vs apparent size (tests the small-object effect)")
        print(f"    {'area quartile':<18}{'n':>7}{'mean conf':>12}")
        bounds = [(0, qs[0], "smallest 25%"), (qs[0], qs[1], "25-50%"),
                  (qs[1], qs[2], "50-75%"), (qs[2], 1e9, "largest 25%")]
        for lo, hi, label in bounds:
            m = (areas >= lo) & (areas < hi)
            if m.sum():
                print(f"    {label:<18}{m.sum():>7}{confs[m].mean():>12.3f}")
        r = float(np.corrcoef(areas, confs)[0, 1])
        print(f"    pearson r = {r:+.3f}")

    scenario_breakdown(rows)
    print()


if __name__ == "__main__":
    main()
