# No Phones

A webcam app that beeps and flashes the screen when you pick up your phone while working.

## How it works

- YOLO detects the phone.
- MediaPipe tracks your hands.
- A gradient-boosted model combines both into an "on the phone" score for each frame.
- An alert fires after 5 seconds on the phone. Typing cancels it.

## Setup

Windows, Python 3.12, NVIDIA GPU.

```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
curl.exe -L --create-dirs -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

The YOLO weights download automatically on first run.

## Train it on your desk

The model learns one person, one desk and one camera angle. Fix the camera in place first and don't move it.

1. Record at least 4 sessions on different days, with a new seed each time (about 8 minutes each, 10 scenarios including drinking and glancing at your phone):
   ```
   python src/record.py --seed 1
   ```
2. Train and evaluate the model:
   ```
   python src/train.py
   ```
3. Tune the alerts:
   ```
   python src/tune_alerts.py --trigger 5
   ```
   `--trigger` is how many seconds on your phone before it alerts. Lower is faster but gives more false alarms.

Also useful:
- `python src/preview.py` shows what the camera sees while you aim it
- `python src/analyze_session.py` checks a recorded session for problems
- `python src/test_live.py` runs the tests

## Run

Double-click the No Phones icon on your desktop, or `No Phones.bat`, or:

```
python src/monitor.py
```

- `q` quits
- `f` marks the last alert as a false alarm
- `m` marks a miss: press it with your other hand while you're on your phone and the bar isn't moving
- `--headless` runs without a window (no `f` or `m`)
- `--no-sound` / `--no-flash` turn off either alert

To create the desktop icon (once):

```
powershell -ExecutionPolicy Bypass -File make_shortcut.ps1
```

To see how it's doing in real use, from your `f` and `m` presses:

```
python src/report.py
```

## Results

Measured on sessions the model never trained on (8 sessions, about 62,000 frames):

- YOLO confidence alone: AUC 0.76
- YOLO + hands model: AUC 0.94
- Alerts at 5 seconds: 25 of 28 phone episodes caught, median 5 s
- False alarms: 10 in 27 minutes of test recordings, mostly while drinking. The tests sip every few seconds, far more than real life. Real use so far: 0 in 25 minutes.
- A 7 second trigger halves the false alarms and catches one fewer episode

Known limits: a phone held fully out of view can't be seen, and a hand raised with a drink looks like a hand raised with a phone.

Full tables are in `results/metrics.md` and `results/alerts.md`.

## Privacy

- Monitoring saves `data/focus_log.csv` (alert times). Pressing `f` or `m` also saves the last few seconds of features (numbers, no images) to `data/feedback/`.
- Recording saves features and a few snapshot images to `data/sessions/`. Keep that folder private.
- The keyboard listener counts key presses and never records which key.
