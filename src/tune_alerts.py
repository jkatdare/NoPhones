"""Choose the alert settings, and measure the whole system honestly.

    python src/tune_alerts.py        (after train.py)

Input is results/oof.csv from train.py: every frame's score there came from a
model that never saw that frame's session. Each 40-second recording is then
replayed through the real AlertStateMachine, and counted the way you would
experience it:

  working recording  -> any alert is a FALSE ALERT
  phone recording    -> at least one alert means CAUGHT; the time it took is
                        the LATENCY

The settings are ALSO chosen by leave-one-session-out: for each session, pick
the best settings using only the other sessions, then score the held-out one
with them. Tune thresholds on the same data you report them on and you get
the hold rule's story again - perfect at 8 s, clearing the longest false
alarm by 0.2 s, fragile on anything new.

Two families compete under identical rules:
  hold  - score stays >= 0.5 continuously for N seconds (the original idea)
  vote  - AlertStateMachine: `on` share of the last `window` seconds were high

Writes models/alert_config.json (the settings monitor.py uses) and
results/alerts.md.
"""

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fusion
import protocol
from alerts import AlertStateMachine

SCORE = "p_A+B"            # the shipped model's held-out score
MARGIN_SECS = 1.5          # worst false burst must stay this far below the trigger
REALERT = 15.0

VOTE_GRID = [dict(window=w, on=on, off=on / 2, smooth=s)
             for w, on, s in itertools.product([6, 8, 10, 12, 15, 20],
                                               [0.5, 0.6, 0.7, 0.8], [0.0, 2.0])]
HOLD_GRID = [dict(hold=h) for h in (5, 6, 8, 10, 12, 15)]


def trigger_family(trigger):
    """Every vote setting that alerts after exactly `trigger` seconds of evidence.

    They differ only in window length - how big a gap they tolerate. How long a
    glance may last before it counts is a product decision the recordings
    cannot answer (none of them contains a quick glance), so with --trigger
    you fix it, and tuning only picks the most robust shape around it.
    Left free, tuning simply picks the fastest setting that happened to get
    away with it - on these sessions, a 2.5-second hair trigger.
    """
    return [dict(window=w, on=trigger / w, off=trigger / w / 2, smooth=s)
            for w in (6, 7, 8, 10, 12, 15) if trigger / w <= 0.9
            for s in (0.0, 2.0)]


def segments(df):
    """Yield (session, scenario, frame table) for every 40-second recording."""
    for (sess, scen), g in df.groupby(["session", "scenario"], sort=False):
        yield sess, scen, g.sort_values("t_elapsed")


def run_vote(g, cfg):
    sm = AlertStateMachine(threshold=0.5, veto=True, realert=REALERT, **cfg)
    t = g["t_elapsed"].to_numpy()
    p = g[SCORE].to_numpy()
    k = g["os_keys_5s"].to_numpy()
    alerts, first, peak = 0, None, 0.0
    for ti, pi, ki in zip(t, p, k):
        ev = sm.update(ti, pi, ki)
        peak = max(peak, sm.high_secs)
        if ev and ev[0] == "alert" and ev[1] == 1:
            alerts += 1
            first = ti - t[0] if first is None else first
    trigger = cfg["on"] * cfg["window"]
    return alerts, first, peak, trigger


def run_hold(g, cfg):
    """Continuous-hold rule on the 2 s smoothed score, with the same typing veto."""
    sm = AlertStateMachine(window=1, on=1, off=0, smooth=2.0)   # only used to smooth
    t = g["t_elapsed"].to_numpy()
    p = g[SCORE].to_numpy()
    k = g["os_keys_5s"].to_numpy()
    alerts, first, start, fired, longest = 0, None, None, False, 0.0
    for ti, pi, ki in zip(t, p, k):
        sm.update(ti, pi, 0)
        high = sm.score >= 0.5 and ki == 0
        if high:
            start = ti if start is None else start
            longest = max(longest, ti - start)
            if not fired and ti - start >= cfg["hold"]:
                alerts += 1
                fired = True
                first = ti - t[0] if first is None else first
        else:
            start, fired = None, False
    return alerts, first, longest, float(cfg["hold"])


def evaluate(df, family, grid, runner):
    """Per (config, session): false alerts, working seconds, margin, catches, latencies."""
    rows = []
    for ci, cfg in enumerate(grid):
        per = {}
        for sess, scen, g in segments(df):
            r = per.setdefault(sess, {"fa": 0, "work_s": 0.0, "margin": np.inf,
                                      "caught": 0, "n_phone": 0, "lat": []})
            alerts, first, peak, trigger = runner(g, cfg)
            if protocol.LABELS[scen] == 0:
                r["fa"] += alerts
                r["work_s"] += g["t_elapsed"].iloc[-1] - g["t_elapsed"].iloc[0]
                r["margin"] = min(r["margin"], trigger - peak)
            else:
                r["n_phone"] += 1
                if alerts:
                    r["caught"] += 1
                    r["lat"].append(first)
        for sess, r in per.items():
            rows.append({"family": family, "cfg": ci, "session": sess, **r})
    return pd.DataFrame(rows)


def aggregate(res, cfg, sessions):
    sub = res[(res["cfg"] == cfg) & (res["session"].isin(sessions))]
    lat = [x for l in sub["lat"] for x in l]
    return {"fa": int(sub["fa"].sum()), "work_s": float(sub["work_s"].sum()),
            "margin": float(sub["margin"].min()),
            "caught": int(sub["caught"].sum()), "n_phone": int(sub["n_phone"].sum()),
            "lat": lat, "median_lat": float(np.median(lat)) if lat else np.inf}


def choose(res, grid, sessions):
    """Fewest false alerts, then a safety margin, then fewest misses, then fastest."""
    def key(ci):
        a = aggregate(res, ci, sessions)
        return (a["fa"], int(a["margin"] < MARGIN_SECS), a["n_phone"] - a["caught"], a["median_lat"])
    return min(range(len(grid)), key=key)


def describe(family, cfg):
    if family == "hold":
        return f"hold {cfg['hold']:.0f} s"
    return (f"{cfg['on']:.0%} of {cfg['window']:.0f} s"
            + (f", {cfg['smooth']:.0f} s smoothing" if cfg["smooth"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", type=float, default=None,
                    help="alert after exactly this many seconds of evidence "
                         "(default: tuning chooses)")
    args = ap.parse_args()
    if args.trigger:
        vote_grid = trigger_family(args.trigger)
        hold_grid = [dict(hold=args.trigger)]
    else:
        vote_grid, hold_grid = VOTE_GRID, HOLD_GRID

    oof = pd.read_csv("results/oof.csv")
    raw = fusion.load_sessions(verbose=False)[["session", "t_elapsed", "os_keys_5s"]]
    df = oof.merge(raw, on=["session", "t_elapsed"], how="left")
    assert df["os_keys_5s"].notna().all(), "oof.csv and sessions are out of sync - rerun train.py"
    sessions = sorted(df["session"].unique())
    print(f"{len(df)} held-out frames, {len(sessions)} sessions")

    print(f"simulating {len(vote_grid)} vote settings and {len(hold_grid)} hold settings...")
    res = {"vote": evaluate(df, "vote", vote_grid, run_vote),
           "hold": evaluate(df, "hold", hold_grid, run_hold)}
    grids = {"vote": vote_grid, "hold": hold_grid}

    # ---- nested leave-one-session-out ---------------------------------------
    nested = {}
    for fam in ("hold", "vote"):
        rows = []
        for s in sessions:
            ci = choose(res[fam], grids[fam], [x for x in sessions if x != s])
            a = aggregate(res[fam], ci, [s])
            rows.append({"session": s, "cfg": ci, **a})
        nested[fam] = rows

    def totals(rows):
        lat = [x for r in rows for x in r["lat"]]
        fa, ws = sum(r["fa"] for r in rows), sum(r["work_s"] for r in rows)
        return {"fa": fa, "fa_per_hr": fa / ws * 3600, "work_min": ws / 60,
                "caught": sum(r["caught"] for r in rows), "n_phone": sum(r["n_phone"] for r in rows),
                "median_lat": float(np.median(lat)) if lat else np.inf,
                "max_lat": float(max(lat)) if lat else np.inf}

    tot = {fam: totals(rows) for fam, rows in nested.items()}

    # ---- final settings: chosen on ALL sessions -------------------------------
    best = choose(res["vote"], vote_grid, sessions)
    cfg = dict(vote_grid[best])
    final_all = aggregate(res["vote"], best, sessions)
    config = {**cfg, "threshold": 0.5, "veto": True, "realert": REALERT, "trigger": args.trigger,
              "chosen_on": sessions, "chosen_at": datetime.now().isoformat(timespec="seconds"),
              "heldout": {k: (float(v) if not isinstance(v, (list, int)) else v)
                          for k, v in tot["vote"].items()}}
    Path("models").mkdir(exist_ok=True)
    Path("models/alert_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---- report ---------------------------------------------------------------
    L = ["# Alerts\n",
         "Every number here is from sessions the model never trained on, with alert settings "
         "chosen **without** looking at the session being scored (nested leave-one-session-out). "
         "Each 40 s recording counts once: an alert during a working recording is a false alert; "
         "a phone recording is caught if it raises at least one alert.\n",
         "## Hold rule vs vote, held out\n",
         "| rule | false alerts | per hour of work | phone recordings caught | median latency | slowest |",
         "|---|---|---|---|---|---|"]
    for fam, label in (("hold", "hold for N seconds"), ("vote", "vote with hysteresis")):
        t = tot[fam]
        L.append(f"| {label} | {t['fa']} in {t['work_min']:.0f} min | {t['fa_per_hr']:.1f} | "
                 f"{t['caught']}/{t['n_phone']} | {t['median_lat']:.0f} s | {t['max_lat']:.0f} s |")
    L += ["", "## Per held-out session\n",
          "| held out | rule | settings chosen on the others | false alerts | caught | latencies |",
          "|---|---|---|---|---|---|"]
    for fam in ("hold", "vote"):
        for r in nested[fam]:
            lat = ", ".join(f"{x:.0f}s" for x in r["lat"]) or "-"
            L.append(f"| {r['session'].split('_')[-1]} | {fam} | {describe(fam, grids[fam][r['cfg']])} | "
                     f"{r['fa']} | {r['caught']}/{r['n_phone']} | {lat} |")
    trigger_note = (f"Trigger fixed at **{args.trigger:g} s** of evidence by choice - how long "
                    f"you may look at your phone before it counts is a product decision, not "
                    f"something these recordings can settle. Tuning chose only the window shape. "
                    f"(`python src/tune_alerts.py --trigger {args.trigger:g}`)\n"
                    if args.trigger else "")
    L += ["", "## Shipped settings\n", trigger_note,
          f"**{describe('vote', cfg)}** - alert when {cfg['on']:.0%} of the last {cfg['window']:.0f} s "
          f"scored as distracted, clear when that falls to {cfg['off']:.0%}. Typing vetoes. "
          f"Re-alert every {REALERT:.0f} s while it continues.\n",
          f"Chosen on all {len(sessions)} sessions: {final_all['fa']} false "
          f"alert{'' if final_all['fa'] == 1 else 's'}, "
          f"{final_all['caught']}/{final_all['n_phone']} caught, median latency "
          f"{final_all['median_lat']:.0f} s. The worst false burst while working got "
          + (f"to within {final_all['margin']:.1f} s of evidence of the trigger.\n"
             if final_all['margin'] >= 0 else
             f"{-final_all['margin']:.1f} s of evidence PAST the trigger - that is the false "
             f"alert above.\n"),
          "_Latency is from the start of the recorded 40 s - after an 8 s count-in, so you were already "
          "in position. Recordings are acted, and real episodes run for minutes, not 40 s: a "
          "recording that is 'missed' here is usually one that would have been caught a little later._"]
    Path("results").mkdir(exist_ok=True)
    Path("results/alerts.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    print("\nsaved models/alert_config.json, results/alerts.md")


if __name__ == "__main__":
    main()
