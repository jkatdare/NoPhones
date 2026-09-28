"""No Phones - the live monitor.

    python src/monitor.py              preview window + alerts
    python src/monitor.py --headless   alerts only, no window
    python src/monitor.py --no-sound   flash only
    python src/monitor.py --no-flash   sound only

Or double-click "No Phones.bat" in the project folder.

Webcam -> features (YOLO phone detection + MediaPipe hand landmarks) ->
fusion model -> alert logic -> a beep and a red screen for 3 seconds when you
have been on your phone. Keep going and it alerts again every 15 s, louder.

In the preview window:
  q  quit
  f  the last alert was WRONG - you were working
  m  it's MISSING one - press it with your other hand while still on the phone

Both keys log the event and save the frames around it, labelled, to
data/feedback/. That is the point of running it day to day: the recorded
sessions held 11 minutes of working footage, far too little to pin down the
false-alarm rate, and every f and m is a training example from exactly where
the model gets it wrong. python src/report.py summarises the log.

Stored: data/focus_log.csv (event times) and data/feedback/ (feature numbers
around each f / m - no images). The keyboard listener behind the typing veto
counts THAT a key was pressed, never which one.

The camera must be exactly where it was for the training sessions.
"""

import argparse
import csv
import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np

from alerts import AlertStateMachine
from capture import open_camera
from features import FEATURE_NAMES, FeatureExtractor
from fusion import LiveFusion
from record import META_COLUMNS

FONT = cv2.FONT_HERSHEY_SIMPLEX
LOG_PATH = Path("data/focus_log.csv")
FEEDBACK_DIR = Path("data/feedback")

BUFFER_SECONDS = 60.0   # recent frames kept in memory, so f / m can save them
FALSE_BEFORE = 10.0     # f saves from 10 s before the alert up to the key press
MISS_WINDOW = 12.0      # m saves the 12 s before the key press - pressed mid-use, those are
MISS_GAP = 2.0          # certain phone frames - minus the last 2, while your other hand reached for the key
FLASH_SECONDS = 3.0     # how long the red alert screen stays up


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
    """A full-screen red alert that stays up for FLASH_SECONDS.

    Driven by the frame loop rather than a sleep, so the camera and the model
    keep running underneath it. f dismisses it early.
    """

    NAME = "no phones - alert"

    def __init__(self, enabled):
        self.enabled = enabled
        self.pattern = []          # [(start, end), ...] in loop time
        self.shown = False
        self.imgs = {1: self._image("PUT THE PHONE DOWN"),
                     2: self._image("STILL ON YOUR PHONE")}
        self.img = self.imgs[1]

    @staticmethod
    def _image(text):
        img = np.zeros((360, 640, 3), np.uint8)
        img[:] = (20, 20, 210)     # BGR red
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.4, 3)
        cv2.putText(img, text, ((640 - tw) // 2, (360 + th) // 2 - 20), FONT, 1.4, (255, 255, 255), 3)
        hint = "press f if this is wrong"
        (hw, _), _ = cv2.getTextSize(hint, FONT, 0.6, 1)
        cv2.putText(img, hint, ((640 - hw) // 2, (360 + th) // 2 + 30), FONT, 0.6, (255, 255, 255), 1)
        return img

    def fire(self, now, level):
        if not self.enabled:
            return
        self.img = self.imgs[1 if level == 1 else 2]
        self.pattern = [(now, now + FLASH_SECONDS)]

    def dismiss(self):
        self.pattern = []          # the next tick() takes the window down

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
    """Append-only event log: data/focus_log.csv.

    `episode` ties an alert, its escalations, its end and any false-alarm mark
    together, so report.py can tell which phone time was real.
    """

    FIELDS = ["time", "event", "episode", "level", "seconds", "note"]

    def __init__(self, path=LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                header = reader.fieldnames or []
            if header != self.FIELDS:   # an older log without the episode column: upgrade it, keep every row
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=self.FIELDS)
                    w.writeheader()
                    w.writerows({k: r.get(k, "") for k in self.FIELDS} for r in rows)
        new = not path.exists()
        self.handle = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        if new:
            self.writer.writeheader()

    def write(self, event, episode="", level="", seconds="", note=""):
        self.writer.writerow({"time": datetime.now().isoformat(timespec="seconds"),
                              "event": event, "episode": episode, "level": level,
                              "seconds": round(seconds, 1) if seconds != "" else "",
                              "note": note})
        self.handle.flush()

    def close(self):
        self.handle.close()


class FeedbackStore:
    """Labelled feature windows from f and m.

    Same columns as a recorded session, so retraining can load them like any
    other data. Numbers only - no images.
    """

    def __init__(self, folder=FEEDBACK_DIR):
        self.folder = folder
        self.run = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saved_until = float("-inf")    # never save the same frame twice

    def save(self, frames, label, kind):
        frames = [f for f in frames if f["t"] > self.saved_until]
        if not frames:
            return 0
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"feedback_{self.run[:8]}.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=META_COLUMNS + FEATURE_NAMES)
            if new:
                w.writeheader()
            for f in frames:
                row = {"ts_iso": f["ts_iso"], "t_elapsed": round(f["t"], 3),
                       "frame_idx": f["frame_idx"],
                       # One group per monitor run: t restarts at 0 every run, and
                       # the rolling features must never mix two runs together.
                       "session": f"feedback_{self.run}", "seed": "",
                       "scenario": kind, "label": label}
                row.update({k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in f["feats"].items()})
                w.writerow(row)
        self.saved_until = frames[-1]["t"]
        return len(frames)


class Monitor:
    """Everything that happens per frame, separate from the camera and the UI,
    so it can be driven by synthetic frames in a test."""

    def __init__(self, bundle, cfg, yolo_model="yolo11x.pt", sound=True, flash=True,
                 log_path=LOG_PATH, feedback_dir=FEEDBACK_DIR, use_os=True):
        self.extractor = FeatureExtractor(yolo_model, use_os=use_os)
        self.live = LiveFusion(bundle)
        keys = ("window", "on", "off", "threshold", "smooth", "veto", "realert")
        self.sm = AlertStateMachine(**{k: cfg[k] for k in keys})
        self.beeper = Beeper(sound)
        self.flasher = Flasher(flash)
        self.log = FocusLog(log_path)
        self.feedback = FeedbackStore(feedback_dir)
        self.recent = deque()           # the last BUFFER_SECONDS of frames
        self.frame_idx = 0
        self.episode = 0                # id of the latest episode, counted per run
        self.episode_alert_t = None     # when it began
        self.false_marked = None        # episode already marked false
        self.episodes, self.alerts, self.false_marks, self.misses = [], 0, 0, 0
        self.toast, self.toast_until = "", 0.0

    def step(self, frame, t):
        feats = self.extractor(frame, t)
        self.recent.append({"t": t, "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
                            "frame_idx": self.frame_idx, "feats": feats})
        self.frame_idx += 1
        while self.recent and t - self.recent[0]["t"] > BUFFER_SECONDS:
            self.recent.popleft()

        p = self.live.score(feats, t)
        event = self.sm.update(t, p, feats["os_keys_5s"])
        if event and event[0] == "alert":
            level = event[1]
            if level == 1:
                self.episode += 1
                self.episode_alert_t = t
                self.alerts += 1
            self.beeper.fire(level)
            self.flasher.fire(t, level)
            self.log.write("alert", episode=self.episode, level=level)
            print(f"  {datetime.now():%H:%M:%S}  ALERT level {level}")
        elif event and event[0] == "clear":
            self.episodes.append(event[1])
            self.log.write("episode", episode=self.episode, seconds=event[1])
            print(f"  {datetime.now():%H:%M:%S}  back to work after {event[1]:.0f}s")
        self.flasher.tick(t)
        return feats, p, event

    def mark_false(self, t):
        """f: the last alert was wrong. Save the frames that fooled the model as
        'working', log it, and stop nagging."""
        if self.episode_alert_t is None:
            self._say(t, "no alert to mark")
            return
        if self.false_marked == self.episode:
            self._say(t, "already marked")
            return
        self.false_marked = self.episode
        self.false_marks += 1
        in_memory = bool(self.recent) and self.recent[0]["t"] <= self.episode_alert_t
        saved = 0
        if in_memory:
            start = self.episode_alert_t - FALSE_BEFORE
            saved = self.feedback.save([f for f in self.recent if f["t"] >= start], 0, "false_alarm")
        self.log.write("false_alarm", episode=self.episode,
                       note=f"{saved} frames saved" if in_memory else "too late to save frames")
        if self.sm.state == "distracted":
            self.sm.reset()
            self.live.reset()
        self.flasher.dismiss()
        self._say(t, f"false alarm noted, {saved} frames saved")

    def mark_missed(self, t):
        """m: you are on your phone and the bar isn't moving. Press it with your
        other hand while still on the phone - the seconds just before the key
        press are then certain phone frames, saved as 'on the phone'."""
        self.misses += 1
        frames = [f for f in self.recent if t - MISS_WINDOW <= f["t"] <= t - MISS_GAP]
        saved = self.feedback.save(frames, 1, "missed")
        self.log.write("missed", note=f"{saved} frames saved")
        self._say(t, f"miss noted, {saved} frames saved")

    def _say(self, t, text):
        self.toast, self.toast_until = text, t + 2.5
        print(f"  {datetime.now():%H:%M:%S}  {text}")

    def close(self, t):
        if self.sm.state == "distracted":
            self.episodes.append(t - self.sm.since)
            self.log.write("episode", episode=self.episode, seconds=t - self.sm.since,
                           note="open at exit")
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

    if elapsed < mon.toast_until:
        (tw, _), _ = cv2.getTextSize(mon.toast, FONT, 0.55, 2)
        cv2.rectangle(frame, (w - tw - 28, 10), (w - 10, 44), (50, 50, 50), -1)
        cv2.putText(frame, mon.toast, (w - tw - 19, 33), FONT, 0.55, (255, 255, 255), 2)

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
                       f"q quit  f false alarm  m missed",
                (10, h - 16), FONT, 0.45, (0, 255, 0), 1)


def main():
    ap = argparse.ArgumentParser(description="No Phones - live distraction monitor")
    ap.add_argument("--source", default="0", help="camera index")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--headless", action="store_true", help="no preview window (no f / m)")
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
                         f"then python src/tune_alerts.py --trigger 5")

    cap = open_camera(source)
    print("loading models...")
    mon = Monitor(bundle, cfg, args.model, sound=not args.no_sound, flash=not args.no_flash)
    trigger = cfg["on"] * cfg["window"]
    print(f"YOLO device:    {mon.extractor.device_name}")
    print(f"typing veto:    {'on' if mon.extractor.inputs and mon.extractor.inputs.available else 'OFF'}")
    print(f"model:          {bundle['feature_set']}, trained on {len(bundle['sessions'])} sessions")
    print(f"alert rule:     {trigger:.0f}s on the phone within the last {cfg['window']:.0f}s, "
          f"re-alert every {cfg['realert']:.0f}s")
    print(f"logging to:     {LOG_PATH}, feedback to {FEEDBACK_DIR}/  (no images, no key content)")
    if args.headless:
        print("\nwatching. Ctrl+C to stop. (f and m need the preview window.)\n")
    else:
        print("\nwatching.  q quit   f false alarm   m missed (press while still on the phone)\n")

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
            if key == ord("m"):
                mon.mark_missed(t)
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

        if t > 0:
            m, s = divmod(int(t), 60)
            phone = sum(mon.episodes)
            line = (f"\nsession {m}:{s:02d}   on the phone {phone:.0f}s ({phone / t:.0%})   "
                    f"episodes {len(mon.episodes)}")
            if mon.false_marks or mon.misses:
                line += f"   marked false {mon.false_marks}   missed {mon.misses}"
            print(line)
            print("run python src/report.py for totals across days")


if __name__ == "__main__":
    main()
