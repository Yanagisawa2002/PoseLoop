# PoseLoop R2 track-disjoint Photoneo validation

**PASS_R2_SEALED_VALIDATION**

The evaluator was opened once after all predictions, sequential M3 budgets, M4 rankings, final pose selections, and deployable M6 risks were frozen and hashed.

| Stage | Frozen method | Baseline | Delta | Gate |
| --- | ---: | ---: | ---: | :---: |
| M3 macro combined | 83.87% | 82.62% | +1.25 pp | pass |
| M4 continue vs fixed slot 2 | 72.74% | 71.22% | +1.52 pp | pass |
| M4 continue vs uniform random | 72.74% | 70.56% | +2.18 pp | pass |
| M6 AUROC | 0.7667 | 0.2408 | +0.5258 | pass |
| M6 AURC | 0.1213 | 0.4208 | +71.2% relative | pass |

M3 evaluation mean views: 2.187. Final deployed-path mean views: 1.480. M4 continue targets: 72. Final failures: 38/150.

Cross-view GT audit: normalized MSSD max 6.05e-05 / 0.0005; MSPD max 0.01474 / 0.01613 px.

Claim boundary: zero-label Photoneo score calibration followed by evaluation on 139 calibration-track-disjoint Photoneo tracks. This is not zero-shot unseen-sensor evidence, and the 150 targets are not 150 independent instances.
