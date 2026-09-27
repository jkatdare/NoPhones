"""Branch A: phone detection with per-detection CSV logging.

Two ways to label what you are doing:

  --protocol   guided mode. The window tells you what to act out, counts you
               in, records for a fixed window, then advances on its own. Use
               this - you cannot reliably press keys while holding a phone.

  (default)    manual mode. Press number keys to set the tag yourself. The
               window must have focus for cv2.waitKey to see them.

On exit a summary prints the confidence separation between real phones and
false positives, which is the number that sets your threshold.
"""

import argparse
import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from capture import open_camera

# COCO class index for "cell phone".
PHONE_CLASS_ID = 67

# Keystroke -> scenario tag (manual mode).
SCENARIOS = {
    "0": "idle",
    "1": "phone_normal",
    "2": "phone_occluded",
    "3": "phone_facedown_desk",
    "4": "phone_in_lap",
    "5": "phone_dark_on_dark",
    "6": "phone_arms_length",
    "7": "clutter_only",
}

# Guided mode order. Starts with the phone away and works toward holding it,
# so you handle the phone as little as possible mid-session.
PROTOCOL = [
    "clutter_only",
    "phone_facedown_desk",
    "phone_in_lap",
    "phone_normal",
    "phone_occluded",
    "phone_arms_length",
    "phone_dark_on_dark",
]

INSTRUCTIONS = {
    "clutter_only": "Phone OUT OF SIGHT. Work normally.",
    "phone_facedown_desk": "Phone face-down on the desk. Keep typing.",
    "phone_in_lap": "Phone in your lap, below the desk edge.",
    "phone_normal": "Hold it up and scroll, as you normally would.",
    "phone_occluded": "Grip it so your hand covers most of it.",
    "phone_arms_length": "Hold it out at arm's length.",
    "phone_dark_on_dark": "Hold it against dark clothing.",
    "idle": "Untagged.",
}

CSV_COLUMNS = [
    "ts_iso", "t_elapsed", "frame_idx", "scenario", "model",
    "infer_ms", "n_det", "det_idx", "conf",
    "x1", "y1", "x2", "y2", "w", "h", "cx_norm", "cy_norm", "area_frac",
]


def print_menu(protocol, seconds, lead):
    if protocol:
        total = len(PROTOCOL) * (seconds + lead)
        print(f"\nGuided protocol: {len(PROTOCOL)} scenarios x {seconds}s "
              f"(+{lead}s count-in) = {total}s total\n")
        for i, name in enumerate(PROTOCOL, 1):
            print(f"  {i}. {name:<22} {INSTRUCTIONS[name]}")
        print("\n  Frames during the count-in are NOT logged. Press q to abort.\n")
    else:
        print("\nScenario keys (window must have focus):")
        for k, v in SCENARIOS.items():
            print(f"  {k} -> {v}")
        print("  s -> print running summary")
        print("  q -> quit\n")


def schedule(t, seconds, lead):
    """Map elapsed time onto the protocol.

    Returns (phase, scenario, remaining). phase is "lead", "record" or "done".
    Only "record" frames get logged - count-in frames would be mislabelled,
    since you are still moving into position.
    """
    block = seconds + lead
    idx = int(t // block)
    if idx >= len(PROTOCOL):
        return "done", None, 0.0
    within = t - idx * block
    if within < lead:
        return "lead", PROTOCOL[idx], lead - within
    return "record", PROTOCOL[idx], block - within


def draw_centered(frame, text, y, scale, color, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def summarise(stats):
    """Per-scenario detection rate and confidence distribution.

    Headline metric is max-confidence-per-frame. During 'clutter_only' that is
    your worst false positive; during 'phone_normal' it is your true positive.
    The gap between those two distributions is the whole answer.
    """
    print("\n" + "=" * 94)
    print(f"{'scenario':<22}{'frames':>8}{'det rate':>10}"
          f"{'mean':>8}{'median':>8}{'p90':>8}{'max':>8}{'ms':>8}")
    print("-" * 94)
    for name in SCENARIOS.values():
        s = stats.get(name)
        if not s or s["frames"] == 0:
            continue
        confs = np.array(s["frame_max_conf"], dtype=float)
        hits = confs[confs > 0]
        rate = len(hits) / s["frames"]
        ms = float(np.mean(s["infer_ms"])) if s["infer_ms"] else 0.0
        if len(hits):
            print(f"{name:<22}{s['frames']:>8}{rate:>9.0%}"
                  f"{hits.mean():>8.2f}{np.median(hits):>8.2f}"
                  f"{np.percentile(hits, 90):>8.2f}{hits.max():>8.2f}{ms:>8.1f}")
        else:
            print(f"{name:<22}{s['frames']:>8}{rate:>9.0%}"
                  f"{'-':>8}{'-':>8}{'-':>8}{'-':>8}{ms:>8.1f}")
    print("=" * 94)
    print("det rate = fraction of frames with >=1 detection above --conf")
    print("conf cols = distribution of the highest-confidence box per frame\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11x.pt", help="yolo11n/s/m/l/x .pt")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="confidence floor. Keep this LOW while logging so the "
                         "false-positive distribution is visible; pick the real "
                         "threshold afterwards from the summary.")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--duration", type=float, default=0,
                    help="auto-stop after N seconds (0 = until you press q)")
    ap.add_argument("--protocol", action="store_true",
                    help="guided mode: the window walks you through each scenario")
    ap.add_argument("--seconds", type=float, default=30,
                    help="recording seconds per scenario in guided mode")
    ap.add_argument("--lead", type=float, default=6,
                    help="count-in seconds before each scenario (not logged)")
    args = ap.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source

    model = YOLO(args.model)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()} | model: {args.model}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print_menu(args.protocol, args.seconds, args.lead)

    writer = handle = None
    if not args.no_log:
        Path("data").mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("data") / f"detections_{Path(args.model).stem}_{stamp}.csv"
        handle = open(out, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        print(f"logging to {out}")

    cap = open_camera(source)
    stats = defaultdict(lambda: {"frames": 0, "frame_max_conf": [], "infer_ms": []})

    scenario = "idle"
    phase = "record"
    remaining = 0.0
    t0 = time.perf_counter()
    last_report = t0
    frames_since_report = 0
    frame_idx = 0
    fps = 0.0
    announced = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed.")
                break

            h_img, w_img = frame.shape[:2]
            now = time.perf_counter()
            t_elapsed = now - t0

            if args.protocol:
                phase, sched_scenario, remaining = schedule(t_elapsed, args.seconds, args.lead)
                if phase == "done":
                    print("\nprotocol complete.")
                    break
                scenario = sched_scenario
                if (phase, scenario) != announced:
                    announced = (phase, scenario)
                    if phase == "lead":
                        print(f"\n>>> next: {scenario} - {INSTRUCTIONS[scenario]}")
                    else:
                        print(f"    recording {scenario} for {args.seconds:.0f}s")

            # Ultralytics expects BGR for NumPy input - no conversion here.
            results = model.predict(frame, classes=[PHONE_CLASS_ID], conf=args.conf,
                                    device=device, verbose=False)
            r = results[0]
            infer_ms = float(r.speed.get("inference", 0.0))
            boxes = r.boxes
            n_det = len(boxes)
            ts_iso = datetime.now().isoformat(timespec="milliseconds")

            best_conf = 0.0
            rows = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                c = float(box.conf[0])
                best_conf = max(best_conf, c)
                w, h = x2 - x1, y2 - y1
                rows.append({
                    "ts_iso": ts_iso, "t_elapsed": round(t_elapsed, 3),
                    "frame_idx": frame_idx, "scenario": scenario, "model": args.model,
                    "infer_ms": round(infer_ms, 2), "n_det": n_det, "det_idx": i,
                    "conf": round(c, 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "w": round(w, 1), "h": round(h, 1),
                    # Normalised centre and area survive a resolution change,
                    # and both become fusion features in Step 6.
                    "cx_norm": round((x1 + x2) / 2 / w_img, 4),
                    "cy_norm": round((y1 + y2) / 2 / h_img, 4),
                    "area_frac": round(w * h / (w_img * h_img), 5),
                })
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"{c:.2f}", (int(x1), max(int(y1) - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # A frame with zero detections is a data point too - it is a miss.
            if not rows:
                rows.append({
                    "ts_iso": ts_iso, "t_elapsed": round(t_elapsed, 3),
                    "frame_idx": frame_idx, "scenario": scenario, "model": args.model,
                    "infer_ms": round(infer_ms, 2), "n_det": 0, "det_idx": "",
                    "conf": "", "x1": "", "y1": "", "x2": "", "y2": "",
                    "w": "", "h": "", "cx_norm": "", "cy_norm": "", "area_frac": "",
                })

            logging_now = phase == "record"
            if writer and logging_now:
                writer.writerows(rows)
                handle.flush()  # survive a crash mid-session

            if logging_now:
                s = stats[scenario]
                s["frames"] += 1
                s["frame_max_conf"].append(best_conf)
                s["infer_ms"].append(infer_ms)

            frames_since_report += 1
            if now - last_report >= 1.0:
                fps = frames_since_report / (now - last_report)
                frames_since_report = 0
                last_report = now

            # ---- overlay --------------------------------------------------
            if args.protocol and phase == "lead":
                # Dim the frame so it is unmistakable that nothing is recording.
                dim = frame.copy()
                cv2.rectangle(dim, (0, 0), (w_img, h_img), (0, 0, 0), -1)
                cv2.addWeighted(dim, 0.6, frame, 0.4, 0, frame)
                draw_centered(frame, "GET READY", int(h_img * 0.28), 1.1, (255, 255, 255), 3)
                draw_centered(frame, scenario, int(h_img * 0.42), 0.9, (0, 255, 255), 2)
                draw_centered(frame, INSTRUCTIONS[scenario], int(h_img * 0.52), 0.6, (200, 200, 200), 2)
                draw_centered(frame, f"{int(remaining) + 1}", int(h_img * 0.75), 2.2, (255, 255, 255), 4)
            else:
                tag = f"[{scenario}]"
                if args.protocol:
                    tag += f"  REC {remaining:4.1f}s"
                    cv2.circle(frame, (w_img - 26, 26), 9, (0, 0, 255), -1)
                cv2.putText(frame, tag, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if args.protocol:
                    cv2.putText(frame, INSTRUCTIONS[scenario], (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            n_logged = stats[scenario]["frames"] if scenario else 0
            cv2.putText(frame,
                        f"{fps:.0f} fps | {infer_ms:.1f} ms | {n_det} det | n={n_logged}",
                        (10, h_img - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("branch A - phone detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                summarise(stats)
            if not args.protocol:
                ch = chr(key) if key != 255 else ""
                if ch in SCENARIOS:
                    scenario = SCENARIOS[ch]
                    print(f"scenario -> {scenario}")

            if args.duration and not args.protocol and t_elapsed >= args.duration:
                print(f"\nreached --duration {args.duration:.0f}s, stopping.")
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if handle:
            handle.close()
        summarise(stats)


if __name__ == "__main__":
    main()
