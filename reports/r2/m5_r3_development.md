# PoseLoop M5-R3 development and candidate freeze

**FROZEN_M5_R3_CANDIDATE**

Only the already-consumed M5-R2 RealSense replay was used here. No M5-R3 Photoneo prediction, evaluator label, pose error, or temporal outcome was read.

| Candidate | All-frame improvement | Missing-frame improvement |
| --- | ---: | ---: |
| last_5_measurements | +1.31% | -7.05% |
| last_10_measurements | +2.08% | -2.57% |
| last_20_measurements | +5.52% | +4.45% |
| all_history | +5.69% | +4.27% |

Leave-one-object-out selection chose the same candidate in every fold:

| Held-out object | Selected on other four | Held-out all | Held-out missing |
| ---: | --- | ---: | ---: |
| 01 | all_history | +21.08% | +20.40% |
| 09 | all_history | -13.92% | -0.60% |
| 12 | all_history | -3.35% | -30.80% |
| 14 | all_history | +38.57% | +33.87% |
| 17 | all_history | +12.08% | +36.37% |

Frozen candidate: **all_history**, all frames +5.69%, natural missing +4.27%.

The development hierarchical-bootstrap 10th percentile is -10.76%. It is a recorded uncertainty warning, not positive evidence; the candidate must pass all gates on the untouched Photoneo split before M5-R3 can be called successful.
