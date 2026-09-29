"""Train and evaluate the fusion model.

    python src/train.py

EVALUATION: leave-one-session-out (LOSO). Train on every session but one,
test on the one held out, rotate until each has been the test set once.

Why not a random 80/20 split of frames? Adjacent frames are near-duplicates -
at 25 fps, frame 1000 and frame 1001 are the same picture. A random split puts
copies of every test frame into the training set, and the model scores ~99%
by recognising frames it has effectively already seen. That number means
nothing. A whole held-out session - a different day, different light,
different clothes - is the only honest estimate of how the model does on a day
it has never seen. It is also why ~22,000 frames is much less data than it
sounds: the number of genuinely independent examples is closer to the number
of 40-second scenario recordings, a few dozen.

ABLATIONS: the same model trained on each branch alone and in combination,
plus a naive hand-written rule ("YOLO confidence >= 0.5 means phone") as the
before-picture. This is the table that shows whether fusion earned its keep.

Writes:
    models/fusion.joblib    final model trained on ALL sessions + its config
    results/metrics.md      tables, ready for the README
    results/metrics.json    the same numbers, machine-readable
    results/oof.csv         held-out predictions for every frame, for charts
"""

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import fusion
import protocol

THRESHOLD = 0.5
SMOOTH_WINDOW = "2s"       # evaluation-only preview of what temporal smoothing buys
RULE_CONF = 0.5            # the naive baseline's threshold on YOLO confidence
N_REPEATS = 5              # permutation-importance repeats per fold

SUBSETS = {
    "A (YOLO)": ["A"],
    "B (hands)": ["B"],
    "OS (keys/mouse)": ["OS"],
    "A+B (vision)": ["A", "B"],
    "A+OS": ["A", "OS"],
    "B+OS": ["B", "OS"],
    "A+B+OS (full)": ["A", "B", "OS"],
}
FULL = "A+B+OS (full)"
# The model that gets shipped. Keystrokes are left OUT of it on purpose: on
# the first three sessions, adding them lowered held-out AUC (0.966 -> 0.956),
# because silence is ambiguous - reading without touching anything looks
# exactly like phone use - while the one unambiguous case, typing, vision
# already gets 100% right. Keystrokes live on as a veto in the state machine.
SHIP = "A+B (vision)"


def make_model():
    """Histogram gradient boosting: an ensemble of shallow decision trees, each
    fitted to the errors of the ones before it. Strong on tabular data, handles
    the NaN "absent" values natively, and fast enough to retrain dozens of
    times for the ablations.

    min_samples_leaf=200 is the important regulariser here: a leaf must cover
    ~8 seconds of frames, so the trees cannot memorise a few frames of one
    session's quirk - the failure mode near-duplicate frames invite.
    """
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, class_weight="balanced", random_state=0)


def smooth(df, col, window=SMOOTH_WINDOW):
    """Trailing time-window mean of a prediction column, per session."""
    out = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("session", sort=False):
        g = g.sort_values("t_elapsed")
        when = pd.DatetimeIndex(pd.Timestamp(0) + pd.to_timedelta(g["t_elapsed"], unit="s"))
        out.loc[g.index] = (g[col].set_axis(when).rolling(window, min_periods=1)
                            .mean().to_numpy())
    return out


def loso(X, cols, sessions):
    """Out-of-fold P(distracted) for every frame, plus the per-fold models.

    Fits only on labelled frames (0/1) from the other sessions, but scores
    EVERY frame of the held-out session - test-only recordings like `glance`
    included - so tune_alerts can check the alert rule on them.
    """
    labelled = X["label"].isin([0, 1])
    p = np.full(len(X), np.nan)
    models = {}
    for s in sessions:
        test = X["session"] == s
        fit = ~test & labelled
        m = make_model().fit(X.loc[fit, cols], X.loc[fit, "label"])
        p[test.to_numpy()] = m.predict_proba(X.loc[test, cols])[:, 1]
        models[s] = m
    return p, models


def per_fold_auc(X, pcol, sessions):
    return {s: roc_auc_score(X.loc[X.session == s, "label"], X.loc[X.session == s, pcol])
            for s in sessions}


def scenario_accuracy(X, pred_col):
    """% of frames correct in each scenario - reads as recall for distracted
    scenarios and as 1 - false-alarm-rate for the working ones."""
    out = {}
    for s in protocol.PROTOCOL:
        m = X["scenario"] == s
        if protocol.LABELS[s] not in (0, 1) or not m.any():   # test-only, or not recorded yet
            continue
        out[s] = float((X.loc[m, pred_col] == X.loc[m, "label"]).mean())
    return out


def grouped_permutation_importance(X, cols, models, sessions, rng):
    """Held-out AUC drop when a feature is shuffled.

    A raw feature and its rolled twin are shuffled TOGETHER. Shuffle only the
    raw one and the model reads the same information off the rolled column,
    so the feature looks unimportant when it is not.
    """
    bases = [c for c in cols if not c.endswith("_r1s")]
    drops = {b: [] for b in bases}
    for s in sessions:
        test = (X["session"] == s) & X["label"].isin([0, 1])
        Xt, yt = X.loc[test, cols].copy(), X.loc[test, "label"].to_numpy()
        base_auc = roc_auc_score(yt, models[s].predict_proba(Xt)[:, 1])
        for b in bases:
            group = [b] + ([b + "_r1s"] if b + "_r1s" in cols else [])
            d = []
            for _ in range(N_REPEATS):
                Xp = Xt.copy()
                Xp[group] = Xt[group].to_numpy()[rng.permutation(len(Xt))]
                d.append(base_auc - roc_auc_score(yt, models[s].predict_proba(Xp)[:, 1]))
            drops[b].append(float(np.mean(d)))
    return {b: (float(np.mean(v)), float(np.min(v)), float(np.max(v))) for b, v in drops.items()}


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("loading sessions...")
    raw = fusion.load_sessions()
    X = fusion.prepare(raw)
    sessions = sorted(X["session"].unique())
    lab = X["label"].isin([0, 1])           # trained and scored on; the rest are test-only
    n_pos, n_neg = int((X.label == 1).sum()), int((X.label == 0).sum())
    n_test = int((~lab).sum())
    print(f"  {len(X)} frames, {len(sessions)} sessions, {n_pos} distracted / {n_neg} not"
          + (f" / {n_test} test-only (e.g. glance)" if n_test else ""))
    for s in sessions:
        print(f"    {s}  {int((X.session == s).sum())} frames")
    if len(sessions) < 3:
        raise SystemExit("Need at least 3 complete sessions for leave-one-session-out.")

    # ---- naive rule: the before-picture ------------------------------------
    X["pred_rule"] = (X["a_phone_conf"] >= RULE_CONF).astype(int)
    X["p_rule"] = X["a_phone_conf"]   # its "score", for AUC

    # ---- LOSO for every branch combination ---------------------------------
    summary = {}
    fold_models = {}
    print(f"\nleave-one-session-out, {len(SUBSETS)} feature sets x {len(sessions)} folds...")
    for name, branches in SUBSETS.items():
        cols = fusion.columns_for(branches)
        key = "p_" + name.split()[0]
        X[key], models = loso(X, cols, sessions)
        X[key + "_s"] = smooth(X, key)
        XL = X[lab]
        folds = per_fold_auc(XL, key, sessions)
        summary[name] = {
            "n_features": len(cols),
            "auc": float(roc_auc_score(XL.label, XL[key])),
            "auc_worst_session": float(min(folds.values())),
            "auc_by_session": {k: float(v) for k, v in folds.items()},
            "bal_acc": float(balanced_accuracy_score(XL.label, XL[key] >= THRESHOLD)),
            "bal_acc_smoothed": float(balanced_accuracy_score(XL.label, XL[key + "_s"] >= THRESHOLD)),
            "key": key,
        }
        if name == SHIP:
            fold_models = models
        print(f"  {name:<18} AUC {summary[name]['auc']:.3f}   "
              f"worst session {summary[name]['auc_worst_session']:.3f}")

    XL = X[lab]
    rule_folds = per_fold_auc(XL, "p_rule", sessions)
    rule = {
        "auc": float(roc_auc_score(XL.label, XL["p_rule"])),
        "auc_worst_session": float(min(rule_folds.values())),
        "bal_acc": float(balanced_accuracy_score(XL.label, XL["pred_rule"])),
    }

    # ---- per-scenario accuracy ---------------------------------------------
    table_cols = {"naive rule": "pred_rule"}
    for name in ("A (YOLO)", "B (hands)", "OS (keys/mouse)", FULL, SHIP):
        key = summary[name]["key"]
        X["pred_" + key] = (X[key] >= THRESHOLD).astype(int)
        table_cols[name.split()[0]] = "pred_" + key
    ship_key = summary[SHIP]["key"]
    X["pred_ship_smoothed"] = (X[ship_key + "_s"] >= THRESHOLD).astype(int)
    table_cols["A+B, smoothed"] = "pred_ship_smoothed"
    by_scenario = {label: scenario_accuracy(X, col) for label, col in table_cols.items()}

    # ---- feature importance --------------------------------------------------
    print("\npermutation importance (held-out, grouped raw+rolled)...")
    imp = grouped_permutation_importance(X, fusion.columns_for(SUBSETS[SHIP]),
                                         fold_models, sessions, rng)

    # ---- final model on ALL sessions -----------------------------------------
    ship_cols = fusion.columns_for(SUBSETS[SHIP])
    final = make_model().fit(X.loc[lab, ship_cols], X.loc[lab, "label"])
    bundle = {
        "model": final,
        "feature_set": SHIP,
        "features": ship_cols,
        "roll_features": fusion.ROLL_FEATURES,
        "roll_window": fusion.ROLL_WINDOW,
        "threshold": THRESHOLD,
        "sessions": sessions,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "loso": {k: {kk: vv for kk, vv in v.items() if kk != "key"} for k, v in summary.items()},
        "sklearn_version": sklearn.__version__,
    }
    Path("models").mkdir(exist_ok=True)
    joblib.dump(bundle, "models/fusion.joblib")

    # ---- report ---------------------------------------------------------------
    lines = []
    lines.append("# Results\n")
    lines.append(f"Leave-one-session-out across **{len(sessions)} sessions**, "
                 f"{n_pos + n_neg:,} labelled frames ({n_pos:,} distracted, {n_neg:,} not). "
                 f"Each number is measured on a session the model never saw in training.\n")
    lines.append("## Headline\n")
    lines.append("| model | features | ROC AUC | worst session | balanced acc | balanced acc, 2 s smoothed |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| naive rule (YOLO conf >= {RULE_CONF}) | 1 | {rule['auc']:.3f} | "
                 f"{rule['auc_worst_session']:.3f} | {pct(rule['bal_acc'])} | - |")
    for name, r in summary.items():
        lines.append(f"| {name} | {r['n_features']} | {r['auc']:.3f} | {r['auc_worst_session']:.3f} | "
                     f"{pct(r['bal_acc'])} | {pct(r['bal_acc_smoothed'])} |")
    lines.append("")
    lines.append("## Per scenario - % of frames classified correctly\n")
    lines.append("For distracted scenarios this is the detection rate; for working scenarios "
                 "it is 1 minus the false-alarm rate.\n")
    head = "| scenario | truth | " + " | ".join(by_scenario) + " |"
    lines.append(head)
    lines.append("|---|---|" + "---|" * len(by_scenario))
    for s in by_scenario["naive rule"]:      # labelled scenarios that were recorded
        truth = "phone" if protocol.LABELS[s] == 1 else "working"
        lines.append(f"| {s} | {truth} | " + " | ".join(pct(by_scenario[c][s]) for c in by_scenario) + " |")
    lines.append("")
    lines.append("## Feature importance - shipped model (A+B)\n")
    lines.append("Drop in held-out ROC AUC when the feature (and its 1 s rolling mean) is shuffled, "
                 "averaged over folds. Correlated features share credit, so a low score means "
                 "*replaceable*, not necessarily *useless*.\n")
    lines.append("| feature | AUC drop | min fold | max fold |")
    lines.append("|---|---|---|---|")
    for b, (mean, lo, hi) in sorted(imp.items(), key=lambda kv: -kv[1][0])[:15]:
        lines.append(f"| {b} | {mean:.4f} | {lo:.4f} | {hi:.4f} |")
    lines.append("")
    lines.append(f"_Sessions: {', '.join(sessions)}. Generated {bundle['trained_at']}._")
    (out_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")

    json_out = {"sessions": sessions, "frames": len(X), "rule": rule, "summary": bundle["loso"],
                "by_scenario": by_scenario,
                "importance": {b: {"mean": m, "min": lo, "max": hi} for b, (m, lo, hi) in imp.items()}}
    (out_dir / "metrics.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")

    keep = ["session", "t_elapsed", "scenario", "label", "p_rule", ship_key, ship_key + "_s"] + \
           [summary[n]["key"] for n in summary if summary[n]["key"] != ship_key]
    X[keep].to_csv(out_dir / "oof.csv", index=False)

    print("\n" + "\n".join(lines))
    print(f"\nsaved models/fusion.joblib, results/metrics.md, results/metrics.json, results/oof.csv"
          f"  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
