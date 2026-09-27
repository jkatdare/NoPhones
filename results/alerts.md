# Alerts

Every number here is from sessions the model never trained on, with alert settings chosen **without** looking at the session being scored (nested leave-one-session-out). Each 40 s recording counts once: an alert during a working recording is a false alert; a phone recording is caught if it raises at least one alert.

## Hold rule vs vote, held out

| rule | false alerts | per hour of work | phone recordings caught | median latency | slowest |
|---|---|---|---|---|---|
| hold for N seconds | 1 in 11 min | 5.6 | 12/12 | 5 s | 26 s |
| vote with hysteresis | 1 in 11 min | 5.6 | 12/12 | 5 s | 26 s |

## Per held-out session

| held out | rule | settings chosen on the others | false alerts | caught | latencies |
|---|---|---|---|---|---|
| seed1 | hold | hold 5 s | 0 | 3/3 | 5s, 26s, 5s |
| seed2 | hold | hold 5 s | 1 | 3/3 | 5s, 6s, 7s |
| seed4 | hold | hold 5 s | 0 | 3/3 | 5s, 5s, 5s |
| seed3 | hold | hold 5 s | 0 | 3/3 | 5s, 17s, 5s |
| seed1 | vote | 83% of 6 s, 2 s smoothing | 0 | 3/3 | 5s, 26s, 5s |
| seed2 | vote | 83% of 6 s, 2 s smoothing | 1 | 3/3 | 5s, 6s, 7s |
| seed4 | vote | 83% of 6 s, 2 s smoothing | 0 | 3/3 | 5s, 5s, 5s |
| seed3 | vote | 83% of 6 s, 2 s smoothing | 0 | 3/3 | 5s, 17s, 5s |

## Shipped settings

Trigger fixed at **5 s** of evidence by choice - how long you may look at your phone before it counts is a product decision, not something these recordings can settle. Tuning chose only the window shape. (`python src/tune_alerts.py --trigger 5`)

**83% of 6 s, 2 s smoothing** - alert when 83% of the last 6 s scored as distracted, clear when that falls to 42%. Typing vetoes. Re-alert every 15 s while it continues.

Chosen on all 4 sessions: 1 false alert, 12/12 caught, median latency 5 s. The worst false burst while working got 0.1 s of evidence PAST the trigger - that is the false alert above.

_Latency is from the start of the recorded 40 s - after an 8 s count-in, so you were already in position. Recordings are acted, and real episodes run for minutes, not 40 s: a recording that is 'missed' here is usually one that would have been caught a little later._