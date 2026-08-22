# PoseLoop M6-R1 development result

**PASS_FREEZE_M6_R1**

M6-R1 predicts failure risk for the exact frozen M3-R1/M4-R1 output. Every reported learned prediction is outer-fold OOF by physical-instance track. The risk model is not allowed to reselect a pose.

| Metric | Learned risk | Frozen raw-score risk | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.7783 | 0.3767 | +0.4016 |
| AURC (lower is better) | 0.1983 | 0.5184 | +61.7% relative |
| Risk at 80% coverage | 0.2875 | 0.4167 | — |

Positive-direction outer folds: 5/5. Selected feature families by outer fold: `{'output_plus_prefix_policy': 3, 'full_multiview_cad': 1, 'selected_output_only': 1}`.

| Fold | N | Learned AUROC | Raw AUROC | Learned AURC | Raw AURC | Both improve |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 40 | 0.649 | 0.402 | 0.337 | 0.533 | yes |
| 1 | 79 | 0.905 | 0.288 | 0.155 | 0.595 | yes |
| 2 | 49 | 0.672 | 0.451 | 0.208 | 0.431 | yes |
| 3 | 69 | 0.820 | 0.418 | 0.215 | 0.541 | yes |
| 4 | 63 | 0.786 | 0.321 | 0.126 | 0.412 | yes |

This is a grouped development result, not sealed evidence. The final model and feature order are frozen for the single fresh-sensor validation; opening that split before data are available is forbidden by the R1 protocol.
