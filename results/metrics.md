# Results

Leave-one-session-out across **8 sessions**, 62,127 labelled frames (26,997 distracted, 35,130 not). Each number is measured on a session the model never saw in training.

## Headline

| model | features | ROC AUC | worst session | balanced acc | balanced acc, 2 s smoothed |
|---|---|---|---|---|---|
| naive rule (YOLO conf >= 0.5) | 1 | 0.755 | 0.685 | 70% | - |
| A (YOLO) | 10 | 0.872 | 0.786 | 84% | 86% |
| B (hands) | 20 | 0.900 | 0.763 | 82% | 83% |
| OS (keys/mouse) | 5 | 0.733 | 0.688 | 71% | 72% |
| A+B (vision) | 30 | 0.939 | 0.833 | 84% | 85% |
| A+OS | 15 | 0.925 | 0.841 | 84% | 85% |
| B+OS | 25 | 0.895 | 0.781 | 79% | 80% |
| A+B+OS (full) | 35 | 0.940 | 0.825 | 84% | 84% |

## Per scenario - % of frames classified correctly

For distracted scenarios this is the detection rate; for working scenarios it is 1 minus the false-alarm rate.

| scenario | truth | naive rule | A | B | OS | A+B+OS | A+B | A+B, smoothed |
|---|---|---|---|---|---|---|---|---|
| work_typing | working | 99% | 99% | 100% | 98% | 100% | 100% | 100% |
| work_screen | working | 100% | 100% | 60% | 8% | 72% | 86% | 82% |
| work_desk | working | 99% | 92% | 79% | 31% | 82% | 82% | 88% |
| phone_on_desk | working | 60% | 94% | 97% | 86% | 100% | 99% | 100% |
| phone_hand | phone | 84% | 95% | 86% | 93% | 95% | 95% | 96% |
| phone_lap | phone | 21% | 32% | 66% | 95% | 53% | 45% | 44% |
| phone_raised | phone | 44% | 80% | 91% | 89% | 93% | 94% | 96% |
| drink | working | 99% | 96% | 56% | 15% | 76% | 64% | 69% |
| phone_high | phone | 49% | 80% | 92% | 92% | 91% | 89% | 90% |

## Feature importance - shipped model (A+B)

Drop in held-out ROC AUC when the feature (and its 1 s rolling mean) is shuffled, averaged over folds. Correlated features share credit, so a low score means *replaceable*, not necessarily *useless*.

| feature | AUC drop | min fold | max fold |
|---|---|---|---|
| a_phone_cy | 0.0793 | 0.0483 | 0.1203 |
| a_phone_conf | 0.0683 | 0.0559 | 0.0903 |
| b_h0_cy | 0.0184 | -0.0094 | 0.0351 |
| b_h0_extension | 0.0162 | -0.0092 | 0.0361 |
| b_h1_cy | 0.0139 | 0.0024 | 0.0428 |
| b_h0_cx | 0.0108 | -0.0072 | 0.0315 |
| b_h0_thumb_index | 0.0044 | -0.0004 | 0.0149 |
| b_n_hands | 0.0030 | -0.0012 | 0.0084 |
| a_phone_area | 0.0016 | -0.0006 | 0.0036 |
| b_h1_extension | 0.0011 | -0.0060 | 0.0075 |
| a_phone_cx | 0.0011 | -0.0026 | 0.0055 |
| a_phone_dist_wrist | 0.0002 | -0.0006 | 0.0014 |
| b_h0_dist_phone | 0.0001 | -0.0003 | 0.0004 |
| b_h1_thumb_index | 0.0000 | -0.0014 | 0.0012 |
| a_phone_along_hand | 0.0000 | -0.0001 | 0.0002 |

_Sessions: 20260922_161134_seed1, 20260924_121549_seed2, 20260926_093707_seed4, 20260926_122925_seed3, 20260928_183434_seed5, 20260929_100808_seed6, 20260929_141418_seed7, 20260929_184815_seed8. Generated 2026-09-29T19:14:48._