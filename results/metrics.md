# Results

Leave-one-session-out across **4 sessions**, 29,887 frames (12,556 distracted, 17,331 not). Each number is measured on a session the model never saw in training.

## Headline

| model | features | ROC AUC | worst session | balanced acc | balanced acc, 2 s smoothed |
|---|---|---|---|---|---|
| naive rule (YOLO conf >= 0.5) | 1 | 0.792 | 0.737 | 75% | - |
| A (YOLO) | 10 | 0.893 | 0.835 | 86% | 89% |
| B (hands) | 20 | 0.919 | 0.893 | 82% | 84% |
| OS (keys/mouse) | 5 | 0.713 | 0.653 | 76% | 75% |
| A+B (vision) | 30 | 0.971 | 0.945 | 90% | 92% |
| A+OS | 15 | 0.937 | 0.887 | 84% | 84% |
| B+OS | 25 | 0.905 | 0.874 | 78% | 81% |
| A+B+OS (full) | 35 | 0.969 | 0.940 | 91% | 91% |

## Per scenario - % of frames classified correctly

For distracted scenarios this is the detection rate; for working scenarios it is 1 minus the false-alarm rate.

| scenario | truth | naive rule | A | B | OS | A+B+OS | A+B | A+B, smoothed |
|---|---|---|---|---|---|---|---|---|
| work_typing | working | 98% | 99% | 100% | 97% | 100% | 100% | 100% |
| work_screen | working | 100% | 97% | 45% | 1% | 88% | 95% | 95% |
| work_desk | working | 98% | 95% | 82% | 45% | 89% | 92% | 96% |
| phone_on_desk | working | 65% | 93% | 99% | 87% | 97% | 99% | 100% |
| phone_hand | phone | 89% | 98% | 90% | 98% | 99% | 99% | 100% |
| phone_lap | phone | 39% | 51% | 78% | 97% | 72% | 66% | 65% |
| phone_raised | phone | 52% | 80% | 80% | 90% | 91% | 88% | 91% |

## Feature importance - shipped model (A+B)

Drop in held-out ROC AUC when the feature (and its 1 s rolling mean) is shuffled, averaged over folds. Correlated features share credit, so a low score means *replaceable*, not necessarily *useless*.

| feature | AUC drop | min fold | max fold |
|---|---|---|---|
| a_phone_conf | 0.1056 | 0.0641 | 0.1420 |
| b_h0_extension | 0.0417 | 0.0043 | 0.0637 |
| a_phone_cy | 0.0286 | 0.0207 | 0.0464 |
| b_h1_cy | 0.0151 | 0.0003 | 0.0519 |
| b_h0_cy | 0.0106 | 0.0051 | 0.0170 |
| b_min_dist_phone | 0.0072 | 0.0042 | 0.0101 |
| b_h0_cx | 0.0055 | -0.0051 | 0.0185 |
| b_n_hands | 0.0022 | -0.0025 | 0.0137 |
| a_phone_cx | 0.0016 | 0.0001 | 0.0050 |
| b_h0_thumb_index | 0.0006 | -0.0006 | 0.0013 |
| b_h1_cx | 0.0002 | -0.0004 | 0.0007 |
| a_phone_area | 0.0002 | -0.0004 | 0.0018 |
| b_h1_thumb_index | 0.0001 | 0.0000 | 0.0003 |
| a_phone_along_hand | 0.0001 | -0.0001 | 0.0003 |
| a_phone_dist_wrist | 0.0000 | -0.0002 | 0.0002 |

_Sessions: 20260922_161134_seed1, 20260924_121549_seed2, 20260926_093707_seed4, 20260926_122925_seed3. Generated 2026-09-26T12:35:57._