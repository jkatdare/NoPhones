"""How is it doing in real use?

    python src/report.py

Reads data/focus_log.csv - everything the monitor has logged - and reports
per day and in total: hours watched, alerts, false alarms (your f presses),
misses (your m presses) and time on the phone.

The error bars are the point. The recorded sessions held 11 minutes of working
footage, and one false alarm in 11 minutes could mean anything from 0.1 to 30
an hour. Every hour of real use narrows it.
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scipy.stats import beta, chi2

LOG_PATH = Path("data/focus_log.csv")


def poisson_ci(k, conf=0.95):
    """Exact 95% interval for an observed count (Garwood)."""
    a = 1 - conf
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2
    hi = chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return lo, hi


def proportion_ci(k, n, conf=0.95):
    """Exact 95% interval for k successes out of n (Clopper-Pearson)."""
    if n == 0:
        return float("nan"), float("nan")
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def _new_bucket():
    return {"seconds": 0.0, "alerts": set(), "false": set(), "missed": 0, "episode_s": {}}


def summarize(path=LOG_PATH):
    """Per-day and total counts from the log.

    Episode ids restart every run of the monitor, so episodes are keyed by
    (run, id). Older logs without ids get one per first-level alert.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    days = defaultdict(_new_bucket)
    run, current, synthetic = 0, None, 0
    start = start_day = last = None

    def finish_open_run():
        # A run with no session_end crashed or was killed - count up to its last event.
        if start is not None and last is not None:
            days[start_day]["seconds"] += (last - start).total_seconds()

    for r in rows:
        when = datetime.fromisoformat(r["time"])
        day = when.date().isoformat()
        ev = r["event"]
        ep = r.get("episode") or ""
        if ev == "session_start":
            finish_open_run()
            run, start, start_day, current = run + 1, when, day, None
        elif ev == "session_end":
            days[start_day or day]["seconds"] += float(r["seconds"] or 0)
            start = None
        elif ev == "alert":
            first = r["level"] in ("", "1")
            if not ep:
                if first:
                    synthetic += 1
                ep = f"s{synthetic}" if first else current
            if first:
                current = ep
                days[day]["alerts"].add((run, ep))
        elif ev == "episode":
            days[day]["episode_s"][(run, ep or current)] = float(r["seconds"] or 0)
        elif ev == "false_alarm":
            days[day]["false"].add((run, ep or current))
        elif ev == "missed":
            days[day]["missed"] += 1
        last = when
    finish_open_run()

    total = _new_bucket()
    for d in days.values():
        total["seconds"] += d["seconds"]
        total["alerts"] |= d["alerts"]
        total["false"] |= d["false"]
        total["missed"] += d["missed"]
        total["episode_s"].update(d["episode_s"])
    return {"days": {k: _metrics(v) for k, v in sorted(days.items())}, "total": _metrics(total)}


def _metrics(d):
    hours = d["seconds"] / 3600
    alerts = len(d["alerts"])
    false = len(d["false"])
    real = max(alerts - false, 0)
    missed = d["missed"]
    phone_s = sum(s for ep, s in d["episode_s"].items() if ep not in d["false"])
    lo, hi = poisson_ci(false)
    return {"hours": hours, "alerts": alerts, "false": false, "real": real, "missed": missed,
            "phone_min": phone_s / 60,
            "false_per_hr": false / hours if hours else float("nan"),
            "false_per_hr_ci": (lo / hours, hi / hours) if hours else (float("nan"),) * 2,
            "precision": real / alerts if alerts else float("nan"),
            "precision_ci": proportion_ci(real, alerts),
            "recall": real / (real + missed) if real + missed else float("nan"),
            "recall_ci": proportion_ci(real, real + missed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(LOG_PATH))
    args = ap.parse_args()
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"{path} not found - run the monitor first")

    s = summarize(path)
    t = s["total"]
    watched = f"{t['hours']:.1f} h" if t["hours"] >= 1 else f"{t['hours'] * 60:.0f} min"
    print(f"\nreal use: {watched} monitored over {len(s['days'])} day(s)\n")
    print(f"  {'day':<12}{'hours':>7}{'alerts':>8}{'false':>7}{'missed':>8}{'phone min':>11}")
    for day, d in list(s["days"].items()) + [("total", t)]:
        print(f"  {day:<12}{d['hours']:>7.1f}{d['alerts']:>8}{d['false']:>7}"
              f"{d['missed']:>8}{d['phone_min']:>11.1f}")

    def pct(x):
        return "-" if x != x else f"{100 * x:.0f}%"

    lo, hi = t["false_per_hr_ci"]
    print()
    if t["hours"]:
        print(f"  false alarms per hour  {t['false_per_hr']:.1f}   (95% range {lo:.1f} to {hi:.1f})")
    print(f"  alerts that were real  {pct(t['precision'])}   "
          f"(95% range {pct(t['precision_ci'][0])} to {pct(t['precision_ci'][1])})")
    print(f"  phone use it caught    {pct(t['recall'])}   "
          f"(95% range {pct(t['recall_ci'][0])} to {pct(t['recall_ci'][1])})   of the ones you noticed")
    print("\nThe ranges narrow as you log more hours. Press f on every wrong alert and m on every miss.\n")


if __name__ == "__main__":
    main()
