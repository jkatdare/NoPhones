"""Record a training session: every feature, every frame, with its label.

    python src/record.py --seed 3

Always shuffles. The seed is the session's identity - use a new one each
time, and vary the day, lighting, and clothing between sessions. One session
teaches the model what you looked like that afternoon; four sessions teach it
what you look like.

Output: data/sessions/session_<timestamp>_seed<N>.csv, one row per recorded
frame, with columns for the scenario, the 0/1 label, and every feature from
features.py. Count-in frames are not written.

The display is mirrored so it reads naturally, but the models always see the
raw camera frame - flipping what the model sees would swap MediaPipe's
left/right and silently change the meaning of every wrist feature.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

import protocol
from capture import open_camera
from features import FEATURE_NAMES, FeatureExtractor

META_COLUMNS = ["ts_iso", "t_elapsed", "frame_idx", "session", "seed", "scenario", "label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True,
                    help="shuffle seed; identifies the session - use a new one every time")
    ap.add_argument("--seconds", type=float, default=40,
                    help="recording seconds per scenario")
    ap.add_argument("--lead", type=float, default=8,
                    help="count-in seconds before each scenario (not logged)")
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--no-mirror", action="store_true", help="show the raw frame instead")
    args = ap.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    order = protocol.make_order(shuffle=True, seed=args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = f"{stamp}_seed{args.seed}"

    out_dir = Path("data") / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"session_{session}.csv"

    # Open the camera FIRST. It is the thing most likely to fail, and failing
    # after a ten-second model load is a worse experience than failing at once.
    cap = open_camera(source)
    ok, probe = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Camera opened but returned no frame - is another app using it?")
    print(f"camera {source}: {probe.shape[1]}x{probe.shape[0]}")

    # One clean frame at session start, plus one annotated frame from the middle
    # of each scenario. The CSV records what the models measured; these record
    # what they were looking at - so a strange number can be checked by eye
    # instead of reverse-engineered from statistics. Saved un-mirrored, so pixel
    # positions match the logged cx / cy. Compare _start.jpg across sessions to
    # confirm the camera has not moved.
    snap_dir = out_dir / "snapshots" / session
    snap_dir.mkdir(parents=True, exist_ok=True)
    snapped = set()

    print("loading models...")
    extractor = FeatureExtractor(args.model)
    print(f"inputs monitor: {'on' if extractor.inputs and extractor.inputs.available else 'OFF'}")
    print(f"YOLO device:    {extractor.device_name}")

    # The first frames after a webcam opens are near-black while auto-exposure
    # settles, and frames buffered during the model load are stale. Discard a
    # second's worth, THEN take the reference frame for comparing camera
    # position across sessions.
    for _ in range(30):
        cap.read()
    ok, ref = cap.read()
    if ok:
        cv2.imwrite(str(snap_dir / "_start.jpg"), ref)
    protocol.print_menu(True, args.seconds, args.lead, order)
    print(f"logging to {out}")

    handle = open(out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=META_COLUMNS + FEATURE_NAMES)
    writer.writeheader()
    t0 = time.perf_counter()
    last_report, frames_since, fps = t0, 0, 0.0
    frame_idx = 0
    written = 0
    announced = None
    warned_fps = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed.")
                break

            now = time.perf_counter()
            t_elapsed = now - t0
            phase, scenario, remaining = protocol.schedule(t_elapsed, args.seconds, args.lead, order)
            if phase == "done":
                print("\nprotocol complete.")
                break
            if (phase, scenario) != announced:
                announced = (phase, scenario)
                if phase == "lead":
                    print(f"\n>>> next: {scenario} [{protocol.LABEL_WORDS[protocol.LABELS[scenario]]}] - "
                          f"{protocol.INSTRUCTIONS[scenario]}")
                else:
                    print(f"    recording {scenario} for {args.seconds:.0f}s")

            feats = extractor(frame, t_elapsed)

            if phase == "record":
                row = {"ts_iso": datetime.now().isoformat(timespec="milliseconds"),
                       "t_elapsed": round(t_elapsed, 3), "frame_idx": frame_idx,
                       "session": session, "seed": args.seed,
                       "scenario": scenario, "label": protocol.LABELS[scenario]}
                row.update({k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in feats.items()})
                writer.writerow(row)
                written += 1
                if written % 30 == 0:
                    handle.flush()

            frames_since += 1
            if now - last_report >= 1.0:
                fps = frames_since / (now - last_report)
                frames_since, last_report = 0, now

            # ---- display --------------------------------------------------
            extractor.draw(frame)
            if phase == "record" and scenario not in snapped and remaining <= args.seconds / 2:
                cv2.imwrite(str(snap_dir / f"{scenario}.jpg"), frame)
                snapped.add(scenario)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            protocol.draw_overlay(frame, phase, scenario, remaining, seconds=args.seconds)
            h = frame.shape[0]
            cv2.putText(frame,
                        f"{fps:.0f} fps | A {feats['a_ms']:.0f}ms B {feats['b_ms']:.0f}ms | "
                        f"phone {feats['a_phone_conf']:.2f} | keys/5s {feats['os_keys_5s']} | "
                        f"rows {written}",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Seed 3's first attempt ran at 8 fps because YOLO was fighting for the
            # GPU, and nothing on screen said so. Say so.
            if 0 < fps < 15 and t_elapsed > 4:
                cv2.putText(frame, f"LOW FPS ({fps:.0f}) - press q, close other GPU apps, restart",
                            (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                if not warned_fps:
                    print(f"\n!!! fps dropped to {fps:.0f} - the pipeline is running slow. "
                          "This session will be thin; consider q and restart.")
                    warned_fps = True
            cv2.imshow("record - q aborts", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\naborted.")
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        handle.close()
        print(f"\nwrote {written} rows -> {out}")
        print(f"snapshots -> {snap_dir}")


if __name__ == "__main__":
    main()
