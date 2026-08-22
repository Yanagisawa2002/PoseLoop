# PoseLoop M5-R6A RU-APC measurement-first development

Status: **PASS_M5_R6A_DEVELOPMENT**

The frozen nearest estimator owns persistent pose state. Object-local confidence may apply only a bounded measurement-first current-frame correction; missing frames are exact anchor passthrough. YCB-V remained unavailable and unopened throughout this development evaluation.

| Development condition | Observed | Required |
|---|---:|---:|
| Cross-fitted all-frame improvement | +21.33% | >= 5% |
| Cross-fitted missing-frame improvement | +0.00% | >= 0% |
| Bootstrap 10th percentile | +17.66% | >= 0% |
| Missing uncertainty/error Spearman | +0.416 | >= 0.20 |
| Objects represented in missing-frame metric | 12 | >= 12 |
| Held-out objects with positive all-frame improvement | 14 / 14 | 14 / 14 |
| Missing-anchor mismatches | 0 | <= 0 |
| Unsafe forced reacquisitions | 0 | <= 0 |
| Nonfinite outputs | 0 | <= 0 |

| Candidate | All-frame improvement | Missing improvement | Corrections |
|---|---:|---:|---:|
| anchor_calibrated | +0.00% | +0.00% | 0 |
| measurement_cautious | +9.59% | +0.00% | 841 |
| measurement_balanced | +15.52% | +0.00% | 884 |
| measurement_first | +21.33% | +0.00% | 947 |

Modal fold winner for any future sealed freeze: `measurement_first`.

Held-out object improvements ranged from +8.13% to +54.66%, with a median of +19.40%. All 14 leave-one-object-out folds selected `measurement_first`.

The raw result JSON retains one stale human-readable `stage` string inherited from the generalized LM-O evaluator. This has no numerical effect: protocol ID `poseloop-m5-r6a-v1`, stage ID `M5-R6A`, RU-APC contract/source hashes, and all `ruapc-test-*` track IDs are correct. The independent validator recomputed the numerical result from the frozen track artifact.

This is development evidence only. The full pass permits a separately committed sealed-candidate/YCB-V protocol; it is not itself sealed evidence.
