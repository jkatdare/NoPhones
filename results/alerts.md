# Alerts

Every number here is from sessions the model never trained on, with alert settings chosen **without** looking at the session being scored (nested leave-one-session-out). Each 40 s recording counts once: an alert during a recording that should not alert (working, drinking, glancing) is a false alert; a phone recording is caught if it raises at least one alert.

## Hold rule vs vote, held out

| rule | false alerts | per hour (working, drinking, glancing) | phone recordings caught | median latency | slowest |
|---|---|---|---|---|---|
| hold for N seconds | 8 in 27 min | 18.0 | 26/28 | 5 s | 34 s |
| vote with hysteresis | 10 in 27 min | 22.5 | 25/28 | 5 s | 33 s |

## By recording type, held out (vote)

| recording | should alert | recordings | result |
|---|---|---|---|
| work_typing | no | 8 | 0 false alerts |
| work_screen | no | 8 | 2 false alerts |
| work_desk | no | 8 | 2 false alerts |
| phone_on_desk | no | 8 | 0 false alerts |
| phone_hand | yes | 8 | 8/8 caught, median 5 s |
| phone_lap | yes | 8 | 5/8 caught, median 16 s |
| phone_raised | yes | 8 | 8/8 caught, median 5 s |
| drink | no | 4 | 4 false alerts |
| glance | no | 4 | 2 false alerts |
| phone_high | yes | 4 | 4/4 caught, median 5 s |

## Per held-out session

| held out | rule | settings chosen on the others | false alerts | caught | latencies |
|---|---|---|---|---|---|
| seed1 | hold | hold 5 s | 0 | 3/3 | 5s, 26s, 5s |
| seed2 | hold | hold 5 s | 1 | 3/3 | 5s, 5s, 15s |
| seed4 | hold | hold 5 s | 0 | 3/3 | 5s, 5s, 5s |
| seed3 | hold | hold 5 s | 0 | 3/3 | 5s, 16s, 5s |
| seed5 | hold | hold 5 s | 2 | 3/4 | 6s, 5s, 5s |
| seed6 | hold | hold 5 s | 3 | 3/4 | 5s, 6s, 5s |
| seed7 | hold | hold 5 s | 0 | 4/4 | 5s, 9s, 5s, 34s |
| seed8 | hold | hold 5 s | 2 | 4/4 | 5s, 5s, 5s, 5s |
| seed1 | vote | 83% of 6 s, 2 s smoothing | 0 | 3/3 | 5s, 26s, 5s |
| seed2 | vote | 83% of 6 s | 1 | 2/3 | 5s, 14s |
| seed4 | vote | 83% of 6 s, 2 s smoothing | 0 | 3/3 | 5s, 5s, 5s |
| seed3 | vote | 83% of 6 s, 2 s smoothing | 0 | 3/3 | 5s, 16s, 5s |
| seed5 | vote | 83% of 6 s, 2 s smoothing | 2 | 3/4 | 6s, 5s, 5s |
| seed6 | vote | 83% of 6 s | 4 | 3/4 | 5s, 6s, 5s |
| seed7 | vote | 83% of 6 s, 2 s smoothing | 1 | 4/4 | 5s, 8s, 5s, 33s |
| seed8 | vote | 83% of 6 s, 2 s smoothing | 2 | 4/4 | 5s, 5s, 5s, 5s |

## Shipped settings

Trigger fixed at **5 s** of evidence by choice - how long you may look at your phone before it counts is a product decision, not something these recordings can settle. Tuning chose only the window shape. (`python src/tune_alerts.py --trigger 5`)

**83% of 6 s, 2 s smoothing** - alert when 83% of the last 6 s scored as distracted, clear when that falls to 42%. Typing vetoes. Re-alert every 15 s while it continues.

Chosen on all 8 sessions: 9 false alerts, 26/28 caught, median latency 5 s. The worst false burst while working got 1.0 s of evidence PAST the trigger - that is the false alert above.

_Latency is from the start of the recorded 40 s - after an 8 s count-in, so you were already in position. Recordings are acted, and real episodes run for minutes, not 40 s: a recording that is 'missed' here is usually one that would have been caught a little later._