"""Tests for the two things that must not silently break.

    python src/test_live.py

1. TRAIN/SERVE SKEW. The live monitor scores one frame at a time through
   fusion.LiveFusion; training scored whole sessions through fusion.prepare.
   If those two paths ever disagree, the evaluation numbers stop describing
   the thing actually running. This replays a recorded session through both
   and requires them to match.

2. ALERT LOGIC. Scripted score sequences with known right answers - including
   the exact situations that broke the simple "hold for N seconds" rule.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import fusion
from alerts import AlertStateMachine

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def test_skew():
    print("\ntrain/serve skew")
    bundle = joblib.load("models/fusion.joblib")
    path = sorted(Path("data/sessions").glob("session_*.csv"))[0]
    raw = pd.read_csv(path).sort_values("t_elapsed").reset_index(drop=True)

    # Batch path over the WHOLE session, exactly as train.py does it.
    batch = fusion.prepare(raw)
    p_batch = bundle["model"].predict_proba(batch[bundle["features"]])[:, 1]

    # Live path, one frame at a time. Start 2 s early so the 1 s rolling window
    # is warm, and span the first count-in gap between two scenarios.
    gap = int(np.argmax(raw["t_elapsed"].diff().to_numpy() > 5))
    lo, hi = max(gap - 400, 0), min(gap + 300, len(raw))
    warm = raw.index[raw["t_elapsed"] >= raw.loc[lo, "t_elapsed"] - 2.0][0]
    live = fusion.LiveFusion(bundle)
    p_live, times = {}, []
    for i in range(warm, hi):
        row = raw.loc[i]
        t0 = time.perf_counter()
        p = live.score({k: row[k] for k in fusion.FEATURE_NAMES}, float(row["t_elapsed"]))
        times.append(time.perf_counter() - t0)
        if i >= lo:
            p_live[i] = p

    idx = np.array(sorted(p_live))
    diff = np.abs(np.array([p_live[i] for i in idx]) - p_batch[idx])
    check("live == batch on every frame", diff.max() < 1e-9,
          f"{len(idx)} frames incl. a count-in gap, max |diff| = {diff.max():.2e}")
    ms = 1000 * np.median(times)
    check("live scoring fast enough", ms < 25, f"median {ms:.1f} ms per frame")


def feed(sm, seq, fps=25.0, keys=None):
    """seq: list of (seconds, score). Returns [(t, event)]."""
    events, t = [], 0.0
    for dur, score in seq:
        for _ in range(int(round(dur * fps))):
            t += 1.0 / fps
            ev = sm.update(t, score, keys(t) if keys else 0)
            if ev:
                events.append((round(t, 2), ev))
    return events


def test_alerts():
    print("\nalert logic  (window 10 s, on 60%, off 30%)")
    mk = lambda **kw: AlertStateMachine(window=10, on=0.6, off=0.3, realert=15, **kw)

    ev = feed(mk(), [(30, 0.1)])
    check("steady low score never alerts", ev == [], str(ev))

    # The trigger needs on x window = 6 s of evidence. A solid burst shorter
    # than that must not alert; anything longer will - the vote does not make
    # that go away. What it buys is dip tolerance (next tests), which is what
    # lets tuning push the trigger ABOVE the longest false burst without
    # losing real episodes.
    ev = feed(mk(), [(5, 0.9), (30, 0.1)])
    check("a burst shorter than the trigger does not alert", ev == [], str(ev))

    ev = feed(mk(), [(20, 0.9)])
    first = ev[0][0] if ev else None
    check("steady phone use alerts after ~6 s", bool(ev) and ev[0][1] == ("alert", 1)
          and abs(first - 6.0) < 0.2, f"first alert at {first}s")

    dips = [(3, 0.9), (1, 0.2)] * 10     # the case that broke the hold rule
    ev = feed(mk(), dips)
    check("real use with repeated dips still alerts", any(e[1][0] == "alert" for e in ev),
          f"first alert at {ev[0][0] if ev else None}s")

    # The same dippy episode under the old rule: never 5 s continuously high.
    longest = max(d for d, s in dips if s >= 0.5)
    check("...where a 5 s hold rule would never fire", longest < 5,
          f"longest continuous high stretch {longest}s")

    ev = feed(mk(), [(40, 0.9)])
    levels = [e[1][1] for e in ev if e[1][0] == "alert"]
    check("escalates while it continues", levels == [1, 2, 3], f"levels {levels}")

    ev = feed(mk(), [(20, 0.9), (20, 0.1)])
    clears = [e for e in ev if e[1][0] == "clear"]
    check("clears after you put the phone down", len(clears) == 1,
          f"episode {clears[0][1][1]:.1f}s" if clears else "no clear")

    ev = feed(mk(), [(30, 0.9)], keys=lambda t: 3)
    check("typing vetoes a high score", ev == [], str(ev))

    ev_slow = feed(mk(), [(20, 0.9)], fps=15)
    ev_fast = feed(mk(), [(20, 0.9)], fps=30)
    a, b = ev_slow[0][0], ev_fast[0][0]
    check("same timing at 15 fps and 30 fps", abs(a - b) < 0.15, f"{a}s vs {b}s")

    sm = mk()
    sm.update(0.0, 0.9)
    sm.update(3.0, 0.9)                     # a 3-second stall in the loop
    check("a stalled loop is not counted as evidence", sm.high_secs <= 0.5 + 1e-9,
          f"credited {sm.high_secs:.2f}s")


def test_monitor():
    """The whole per-frame path - extractor, live scoring, alert logic, log,
    feedback, report - driven by synthetic frames. No camera, no sound, no
    flash window, and nothing written to the real data/ folder."""
    print("\nmonitor, end to end (synthetic frames, no camera)")
    import csv
    import json
    import tempfile
    from features import FEATURE_NAMES
    from monitor import FocusLog, Monitor
    from record import META_COLUMNS
    from report import summarize

    bundle = joblib.load("models/fusion.joblib")
    cfg = json.loads(Path("models/alert_config.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        log_path, fb_dir = Path(tmp) / "log.csv", Path(tmp) / "feedback"
        # use_os=False: with the real keyboard listener on, whoever happens to be
        # typing while the test runs vetoes its alerts - which is exactly what
        # happened the first time this test ran.
        mon = Monitor(bundle, cfg, sound=False, flash=False, log_path=log_path,
                      feedback_dir=fb_dir, use_os=False)
        frame = np.zeros((480, 640, 3), np.uint8)

        t, dt, times = 0.0, 0.04, []
        for _ in range(25):                            # real pipeline, blank frames
            t0 = time.perf_counter()
            _, p, ev = mon.step(frame, t)
            times.append(time.perf_counter() - t0)
            t += dt
        ms = 1000 * np.median(times[5:])
        check("full pipeline runs on a frame", True, f"median {ms:.0f} ms per frame -> ~{1000 / ms:.0f} fps")
        check("blank desk scores low", p < 0.5, f"score {p:.2f}")

        real_score = mon.live.score

        def run(score, seconds):
            """Pretend the model scores every frame `score` for `seconds`."""
            nonlocal t
            mon.live.score = lambda feats, t_: score
            events = []
            for _ in range(int(seconds / dt)):
                _, _, ev = mon.step(frame, t)
                if ev:
                    events.append((t, ev))
                t += dt
            return events

        mon.log.write("session_start")
        ev = run(0.95, 20)                             # on the phone
        first = next((et for et, e in ev if e[0] == "alert"), None)
        check("alerts when on the phone", first is not None,
              f"after {first - 1.0:.1f}s of phone" if first else "no alert")
        ev = run(0.05, 15)                             # phone down
        check("clears when the phone goes down", any(e[0] == "clear" for _, e in ev))

        run(0.95, 10)                                  # a second alert...
        mon.mark_false(t)                              # ...that was wrong
        check("f resets the alert state", mon.sm.state == "focused")
        run(0.05, 3)
        mon.mark_missed(t)                             # and one it missed
        mon.live.score = real_score
        mon.close(t)
        mon.log.write("session_end", seconds=t)
        mon.log.close()

        events = [r["event"] for r in csv.DictReader(open(log_path, encoding="utf-8"))]
        check("log records alert, episode, false alarm and miss",
              {"alert", "episode", "false_alarm", "missed"} <= set(events), f"{events}")

        rows = [r for f in fb_dir.glob("feedback_*.csv")
                for r in csv.DictReader(open(f, encoding="utf-8"))]
        n_false = sum(r["scenario"] == "false_alarm" for r in rows)
        n_missed = sum(r["scenario"] == "missed" for r in rows)
        labels_ok = all(r["label"] == ("0" if r["scenario"] == "false_alarm" else "1") for r in rows)
        check("f and m save labelled frames in the session format",
              bool(rows) and list(rows[0].keys()) == META_COLUMNS + FEATURE_NAMES
              and labels_ok and n_false > 0 and n_missed > 0,
              f"{n_false} false-alarm frames (label 0), {n_missed} missed frames (label 1)")

        s = summarize(log_path)["total"]
        check("report counts it right", (s["alerts"], s["false"], s["missed"]) == (2, 1, 1),
              f"alerts {s['alerts']}, false {s['false']}, missed {s['missed']}")

        from monitor import FLASH_SECONDS, Flasher
        fl = Flasher(True)
        fl.fire(10.0, 1)                               # pattern only - no window is opened
        up = fl.pattern == [(10.0, 10.0 + FLASH_SECONDS)]
        fl.dismiss()
        check(f"alert screen stays up {FLASH_SECONDS:.0f} s, f dismisses it",
              up and fl.pattern == [])

        old = Path(tmp) / "old_log.csv"
        old.write_text("time,event,level,seconds,note\n"
                       "2026-09-27T09:49:54,session_start,,,x\n", encoding="utf-8")
        FocusLog(old).close()
        upgraded = list(csv.DictReader(open(old, encoding="utf-8")))
        check("an old log is upgraded without losing rows",
              list(upgraded[0].keys()) == FocusLog.FIELDS and upgraded[0]["event"] == "session_start")


if __name__ == "__main__":
    test_alerts()
    test_skew()
    test_monitor()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("all passed")
