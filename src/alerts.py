"""Alert logic: turn a noisy per-frame score into "you have been on your phone".

The obvious rule - alert when the score stays above 0.5 for N seconds - was
tested on held-out predictions and is fragile. During real phone use the score
DIPS (YOLO loses a phone in your lap about half the time), and every dip resets
a continuous timer. Meanwhile false-alarm bursts during reading run up to
7-8 seconds. On three sessions the only N that worked cleared the longest
false alarm by 0.2 s - tuned, not robust.

So this is a sliding-window VOTE with HYSTERESIS:

  * Count how many of the last `window` seconds had a high score.
  * Switch to DISTRACTED when that reaches `on` (say 60% of the window).
  * Switch back only when it falls to `off` (lower, say 30%).

A dip in the middle of scrolling only lowers the count a little, so real
episodes survive it. A burst of false alarms has to fill most of the window
before anything happens. And the gap between `on` and `off` stops a score
hovering near the line from flapping between states every few frames.

Two extra rules:
  * Typing vetoes. A keystroke in the last 5 s counts as a focused frame, full
    stop - your hands are on the keyboard, so they are not on a phone.
  * Escalation. Still distracted `realert` seconds after an alert? Alert
    again, one level louder.

The count is in SECONDS, not frames, so the logic behaves the same at 19 fps
and at 29. Both the tuning simulation (tune_alerts.py) and the live monitor
(monitor.py) use this one class, so they cannot drift apart.
"""

from collections import deque

MAX_DT = 0.5   # a stalled loop must not credit a 3-second gap as 3 s of evidence


class AlertStateMachine:
    def __init__(self, window=10.0, on=0.6, off=0.3, threshold=0.5,
                 smooth=0.0, veto=True, realert=15.0):
        if not 0 <= off < on <= 1:
            raise ValueError("need 0 <= off < on <= 1")
        self.window, self.on, self.off = window, on, off
        self.threshold, self.smooth, self.veto, self.realert = threshold, smooth, veto, realert
        self.reset()

    def reset(self):
        self.hist = deque()        # (t, dt, high) for frames inside the window
        self.high_secs = 0.0
        self.scores = deque()      # (t, score) for optional score smoothing
        self.score_sum = 0.0
        self.last_t = None
        self.state = "focused"
        self.since = None          # start of the current episode
        self.level = 0
        self.next_realert = None
        self.density = 0.0
        self.score = 0.0

    def update(self, t, score, keys_5s=0):
        """Feed one frame. Returns None, ("alert", level) or ("clear", seconds)."""
        dt = 0.0 if self.last_t is None else min(max(t - self.last_t, 0.0), MAX_DT)
        self.last_t = t

        # optional trailing-mean smoothing of the raw score
        if self.smooth > 0:
            self.scores.append((t, score))
            self.score_sum += score
            while self.scores and self.scores[0][0] <= t - self.smooth:
                self.score_sum -= self.scores.popleft()[1]
            score = self.score_sum / len(self.scores)
        self.score = score

        typing = self.veto and keys_5s > 0
        high = score >= self.threshold and not typing
        self.hist.append((t, dt, high))
        if high:
            self.high_secs += dt
        while self.hist and self.hist[0][0] <= t - self.window:
            _, old_dt, old_high = self.hist.popleft()
            if old_high:
                self.high_secs -= old_dt
        self.high_secs = min(max(self.high_secs, 0.0), self.window)
        self.density = self.high_secs / self.window

        if self.state == "focused":
            if self.density >= self.on:
                self.state, self.since, self.level = "distracted", t, 1
                self.next_realert = t + self.realert
                return ("alert", 1)
        else:
            if self.density <= self.off:
                episode = t - self.since
                self.state, self.since, self.level = "focused", None, 0
                return ("clear", episode)
            if t >= self.next_realert:
                self.level += 1
                self.next_realert = t + self.realert
                return ("alert", self.level)
        return None

    def config(self):
        return {"window": self.window, "on": self.on, "off": self.off,
                "threshold": self.threshold, "smooth": self.smooth,
                "veto": self.veto, "realert": self.realert}
