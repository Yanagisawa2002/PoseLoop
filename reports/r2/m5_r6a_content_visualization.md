# PoseLoop M5-R6A content visualization

Status: **RU-APC DEVELOPMENT REPLAY VISUALIZED**

These figures and the replay video show actual recorded RU-APC RGB-D frames with official CAD silhouettes. Green is GT; magenta is the estimator output. The examples were selected only after the development result was frozen and were not used to alter any method, threshold, metric, or gate.

| Role | Object | Track improvement | Replay frame | Nearest error | Measurement-first error |
|---|---:|---:|---:|---:|---:|
| weakest | 10 | +8.13% | 85 | 43.300 | 3.486 |
| typical | 8 | +20.62% | 38 | 41.824 | 17.696 |
| strongest | 9 | +54.66% | 66 | 45.033 | 0.711 |

The missing-support storyboard uses object 13 and its longest/highest-uncertainty natural gap (36 frames). On every missing frame, the nearest and measurement-first pose matrices are exactly equal. Confidence falls from 0.783 to 0.000, while uncertainty rises from 8.53 to 2537.38; the first available frame after the gap reacquires a measurement.

This is recorded cluttered-shelf development replay with nominal timing, oracle target association, and GT visible masks. It is not a simulator, hardware deployment, deployable association result, or unseen-sensor validation. The separate YCB-V sealed protocol stopped at its frozen input-feasibility gate before inference.
