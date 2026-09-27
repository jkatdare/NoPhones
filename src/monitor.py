"""No Phones - the live monitor.

    python src/monitor.py              preview window + alerts
    python src/monitor.py --headless   alerts only, no window - leave it running
    python src/monitor.py --no-sound   flash only
    python src/monitor.py --no-flash   sound only

Webcam -> features (YOLO phone detection + MediaPipe hand landmarks) ->
fusion model -> alert logic -> a beep and a screen flash when you have been on
your phone. It escalates if you keep going: louder, and three flashes.

In the preview window:  q quits.  f marks the last alert as a FALSE ALARM -
that goes in the log, which is how you measure the real false-alarm rate, the
one number the recorded sessions could not pin down.

What is stored: data/focus_log.csv - times of alerts, episodes and false-alarm
marks. Nothing else. No images, no audio. The keyboard listener behind the
typing veto counts THAT a key was pressed, never which one.

The camera must be exactly where it was for the training sessions. The model
learned where your hands and phone sit in THAT frame.
"""

import argparse
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np

from alerts import AlertStateMachine
from capture import open_camera
from features import FeatureExtractor
from fusion import LiveFusion

FONT = cv2.FONT_HERSHEY_SIMPLEX
LOG_PATH = Path("data/focus_log.csv")


class Beeper:
    """Escalating beeps on a background thread, so the video loop never stalls."""

    PATTERNS = {1: [(880, 150), (0, 70), (880, 150)],
                2: [(988, 220), (0, 90)] * 3,
                3: [(1319, 260), (0, 80)] * 4}

    def __init__(self, enabled):
        self.enabled = enabled
        self.busy = False

    def fire(self, level):
        if not self.enabled or self.busy:
            return
        self.busy = True
        threading.Thread(target=self._play, args=(min(level, 3),), daemon=True).start()

    def _play(self, level):
        try:
            import winsound
            for freq, ms in self.PATTERNS[level]:
                if freq:
                    winsound.Beep(freq, ms)
                else:
                    time.sleep(ms / 1000)
        except Exception:
            print("\a", end="", flush=True)   # not Windows, or no audio device
        finally:
            self.busy = False


class Flasher:
    """A full-screen red flash, driven by the frame loop instead of sleeps."""

    NAME = "no phones - alert"

    def __init__(self, enabled):
        self.enabled = enabled
        self.pattern = []          # [(start, end), ...] in loop time
        self.shown = False
        img = np.zeros((360, 640, 3), np.uint8)
        img[:] = (20, 20, 210)     # BGR red
        text = "PUT THE PHONE DOWN"
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.4, 3)
        cv2.putText(img, text, ((640 - tw) // 2, (360 + th) // 2), FONT, 1.4, (255, 255, 255), 3)
        self.img = img

    def fire(self, now, level):
        if not self.enabled:
            return
        n = 1 if level == 1 else 3
        self.pattern = [(now + i * 0.5, now + i * 0.5 + 0.3) for i in range(n)]

    def tick(self, now):
        want = any(a <= now < b for a, b in self.pattern)
        if want and not self.shown:
            cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(self.NAME, cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow(self.NAME, self.img)
            self.shown = True
        elif not want and self.shown:
            cv2.destroyWindow(self.NAME)
            self.shown = False
        if self.pattern and now >= self.pattern[-1][1]:
            self.pattern = []

    def close(self):
        if self.shown:
            cv2.destroyWindow(self.NAME)
            self.shown = False


class FocusLog:
    """Append-only event log: data/focus_log.csv."""

    FIELDS = ["time", "event", "level", "seconds", "note"]

    def __init__(self, path=LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self.handle = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        if new:
            self.writer.writeheader()

    def write(self, event, level="", seconds="", note=""):
        self.writer.writerow({"time": datetime.now().isoformat(timespec="seconds"),
                              "event": event, "level": level,
                              "seconds": round(seconds, 1) if seconds != "" else "",
                              "note": note})
        self.handle.flush()

    def close(self):
        self.handle.close()


class Monitor:
    """Everything that happens per frame, separate from the camera and the UI,
    so it can be driven by synthetic frames in a test."""

    def __init__(self, bundle, cfg, yolo_model="yolo11x.pt", sound=True, flash=True,
                 log_path=LOG_PATH):
        self.extractor = FeatureExtractor(yolo_model)
        self.live = LiveFusion(bundle)
        keys = ("window", "on", "off", "threshold", "smooth", "veto", "realert")
        self.sm = AlertStateMachine(**{k: cfg[k] for k in keys})
        self.beeper = Beeper(sound)
        self.flasher = Flasher(flash)
        self.log = FocusLog(log_path)
        self.episodes, self.alerts, self.false_marks = [], 0, 0
        self.last_alert_t = None

    def step(self, frame, t):
        feats = self.extractor(frame, t)
        p = self.live.score(feats, t)
        event = self.sm.update(t, p, feats["os_keys_5s"])
        if event and event[0] == "alert":
            level = event[1]
            self.alerts += 1
            self.last_alert_t = t
            self.beeper.fire(level)
            self.flasher.fire(t, level)
            self.log.write("alert", level=level)
            print(f"  {datetime.now():%H:%M:%S}  ALERT level {level}")
        elif event and event[0] == "clear":
            self.episodes.append(event[1])
            self.log.write("episode", seconds=event[1])
            print(f"  {datetime.now():%H:%M:%S}  back to work after {event[1]:.0f}s")
        self.flasher.tick(t)
        return feats, p, event

    def mark_false(self, t):
        """The last alert was wrong. Log it, and stop nagging."""
        if self.last_alert_t is None:
            return
        self.false_marks += 1
        self.log.write("false_alarm", note=f"alert {t - self.last_alert_t:.0f}s ago")
        self.sm.reset()
        self.live.reset()
        print(f"  {datetime.now():%H:%M:%S}  marked as a false alarm")

    def close(self, t):
        if self.sm.state == "distracted":
            self.episodes.append(t - self.sm.since)
            self.log.write("episode", seconds=t - self.sm.since, note="open at exit")
        self.flasher.close()
        self.extractor.close()


def draw_hud(frame, mon, fps, elapsed, cfg):
    h, w = frame.shape[:2]
    sm = mon.sm
    if sm.state == "distracted":
        secs = int(sm.last_t - sm.since)
        label, colour = f"PHONE  {secs // 60}:{secs % 60:02d}", (30, 30, 220)
    else:
        label, colour = "FOCUSED", (40, 160, 40)
    cv2.rectangle(frame, (10, 10), (250, 54), colour, -1)
    cv2.putText(frame, label, (22, 43), FONT, 0.95, (255, 255, 255), 2)

    # Evidence meter: fills while you are on the phone, alerts at the red line.
    x0, x1, y0, y1 = 10, w - 10, h - 64, h - 44
    panel = frame.copy()
    cv2.rectangle(panel, (0, h - 92), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70), -1)
    cv2.rectangle(frame, (x0, y0), (int(x0 + (x1 - x0) * sm.density), y1), colour, -1)
    for mark, c in ((cfg["on"], (0, 0, 255)), (cfg["off"], (0, 200, 255))):
        xm = int(x0 + (x1 - x0) * mark)
        cv2.line(frame, (xm, y0 - 5), (xm, y1 + 5), c, 2)
    cv2.putText(frame, f"evidence {sm.density:.0%} of the last {cfg['window']:.0f}s"
                       f"   (alert at {cfg['on']:.0%})   score {sm.score:.2f}",
                (x0, y0 - 9), FONT, 0.45, (230, 230, 230), 1)
    m, s = divmod(int(elapsed), 60)
    cv2.putText(frame, f"{fps:.0f} fps | {m}:{s:02d} | episodes {len(mon.episodes)} | "
                       f"alerts {mon.alerts} | q quit   f false alarm",
                (10, h - 16), FONT, 0.45, (0, 255, 0), 1)


def main():
    ap = argparse.ArgumentParser(description="No Phones - live distraction monitor")
    ap.add_argument("--source", default="0", help="camera index")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--headless", action="store_true", help="no preview window")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--no-flash", action="store_true")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds")
    args = ap.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    try:
        bundle = joblib.load("models/fusion.joblib")
        cfg = json.loads(Path("models/alert_config.json").read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SystemExit(f"{e.filename} missing - run: python src/train.py, "
                         f"then python src/tune_alerts.py")

    cap = open_camera(source)
    print("loading models...")
    mon = Monitor(bundle, cfg, args.model, sound=not args.no_sound, flash=not args.no_flash)
    print(f"YOLO device:    {mon.extractor.device_name}")
    print(f"typing veto:    {'on' if mon.extractor.inputs and mon.extractor.inputs.available else 'OFF'}")
    print(f"model:          {bundle['feature_set']}, trained on {len(bundle['sessions'])} sessions")
    print(f"alert rule:     {cfg['on']:.0%} of the last {cfg['window']:.0f}s, "
          f"re-alert every {cfg['realert']:.0f}s")
    print(f"held-out test:  {cfg['heldout']['caught']}/{cfg['heldout']['n_phone']} caught, "
          f"median {cfg['heldout']['median_lat']:.0f}s, "
          f"{cfg['heldout']['fa_per_hr']:.1f} false alerts/hour")
    print(f"logging to:     {LOG_PATH}  (events only - no images, no key content)")
    print("\nwatching. " + ("Ctrl+C to stop." if args.headless else "q quits, f marks a false alarm.") + "\n")

    t0 = time.perf_counter()
    last_report, frames_since, fps = t0, 0, 0.0
    mon.log.write("session_start", note=f"{cfg['on']:.0%} of {cfg['window']:.0f}s")
    t = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed - camera disconnected?")
                break
            now = time.perf_counter()
            t = now - t0
            mon.step(frame, t)

            frames_since += 1
            if now - last_report >= 1.0:
                fps = frames_since / (now - last_report)
                frames_since, last_report = 0, now

            key = -1
            if not args.headless:
                mon.extractor.draw(frame)
                frame = cv2.flip(frame, 1)
                draw_hud(frame, mon, fps, t, cfg)
                cv2.imshow("no phones", frame)
            key = cv2.waitKey(1) & 0xFF   # also services the flash window
            if key == ord("q"):
                break
            if key == ord("f"):
                mon.mark_false(t)
            if args.duration and t >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        mon.close(t)
        mon.log.write("session_end", seconds=t)
        mon.log.close()
        cap.release()
        cv2.destroyAllWindows()

        distracted = sum(mon.episodes)
        m, s = divmod(int(t), 60)
        print(f"\nsession {m}:{s:02d}   on the phone {distracted:.0f}s "
              f"({distracted / t:.0%})   episodes {len(mon.episodes)}   alerts {mon.alerts}"
              + (f"   longest {max(mon.episodes):.0f}s" if mon.episodes else "")
              + (f"   marked false {mon.false_marks}" if mon.false_marks else "")
              if t > 0 else "")


if __name__ == "__main__":
    main()
