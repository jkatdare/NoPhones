"""Live preview of both vision branches. No recording, no protocol.

Use this to AIM THE CAMERA before committing to a recording session. Hold
each pose you plan to record and confirm the pipeline actually sees it -
five minutes of recording an angle that cannot see your lap is five minutes
wasted, and four of them is most of an afternoon.

    python src/preview.py

The coverage tracker at the bottom accumulates the vertical range over which
hands and phones were actually detected. Hold a pose, watch the bar fill.
If the lap band never lights up, the camera is still pointed too high.
"""

import argparse
import time

import cv2

from capture import open_camera
from features import FeatureExtractor

BANDS = [("top", 0.00, 0.25), ("upper", 0.25, 0.50),
         ("lower", 0.50, 0.75), ("bottom", 0.75, 1.00)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    cap = open_camera(source)
    ok, probe = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Camera opened but returned no frame.")
    print(f"camera {source}: {probe.shape[1]}x{probe.shape[0]}")

    print("loading models...")
    ex = FeatureExtractor(args.model)
    print(f"inputs monitor: {'on' if ex.inputs and ex.inputs.available else 'OFF'}")
    print("\nHold each pose and watch the readout:")
    print("  hands on keyboard   -> n_hands 2, keys climbing")
    print("  phone in your hand  -> phone conf high, dist near 0")
    print("  PHONE IN YOUR LAP   -> the one that matters. hands and/or phone")
    print("                         must be detected, in a band below 'lower'")
    print("  r resets coverage, q quits\n")

    hand_bands = {b[0]: 0 for b in BANDS}
    phone_bands = {b[0]: 0 for b in BANDS}
    t0 = time.perf_counter()
    last_report, frames_since, fps = t0, 0, 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed.")
                break
            now = time.perf_counter()
            f = ex(frame, now - t0)

            for i in (0, 1):
                if f[f"b_h{i}_present"] > 0:
                    cy = f[f"b_h{i}_cy"]
                    for name, lo, hi in BANDS:
                        if lo <= cy < hi:
                            hand_bands[name] += 1
            if f["a_phone_conf"] > 0:
                for name, lo, hi in BANDS:
                    if lo <= f["a_phone_cy"] < hi:
                        phone_bands[name] += 1

            frames_since += 1
            if now - last_report >= 1.0:
                fps = frames_since / (now - last_report)
                frames_since, last_report = 0, now

            ex.draw(frame)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Band guides, so you can see where the thresholds fall.
            for name, lo, hi in BANDS:
                y = int(lo * h)
                cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)
                cv2.putText(frame, name, (4, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (90, 90, 90), 1)

            panel = frame.copy()
            cv2.rectangle(panel, (0, h - 118), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

            hands_ok = f["b_n_hands"] > 0
            phone_ok = f["a_phone_conf"] > 0
            cv2.putText(frame,
                        f"hands {int(f['b_n_hands'])}   phone {f['a_phone_conf']:.2f}   "
                        f"dist {f['b_min_dist_phone']:.2f}   keys/5s {int(f['os_keys_5s'])}",
                        (10, h - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if (hands_ok or phone_ok) else (0, 140, 255), 2)

            cv2.putText(frame, "coverage (frames detected per band)", (10, h - 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            for i, (name, _, _) in enumerate(BANDS):
                x = 10 + i * 150
                hb, pb = hand_bands[name], phone_bands[name]
                colour = (0, 255, 0) if (hb + pb) > 30 else (80, 80, 200)
                cv2.putText(frame, f"{name:<7} h{hb:<5} p{pb}", (x, h - 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

            cv2.putText(frame,
                        f"{fps:.0f} fps | A {f['a_ms']:.0f}ms B {f['b_ms']:.0f}ms | r reset, q quit",
                        (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("preview - aim the camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                hand_bands = {b[0]: 0 for b in BANDS}
                phone_bands = {b[0]: 0 for b in BANDS}
                print("coverage reset")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ex.close()
        print("\nfinal coverage (frames with a detection in each band):")
        for name, _, _ in BANDS:
            print(f"  {name:<8} hands {hand_bands[name]:>6}   phone {phone_bands[name]:>6}")


if __name__ == "__main__":
    main()
