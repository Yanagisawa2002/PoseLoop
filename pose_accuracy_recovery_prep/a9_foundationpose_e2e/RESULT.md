# A-R9 to FoundationPose end-to-end closeout

Status: **PASS_PACKAGE_AND_CLOSE**

This result is a custom evaluation on already-consumed XYZ-IBD RealSense
development frames. It is not official BOP leaderboard AR/AP, it is not sealed,
and scene 9 was not read.

## Frozen scope

- 25 frames from scenes 10, 25, 30, 40, and 65
- 820 frozen A-R9 instance predictions
- 770 development ground-truth instances
- FoundationPose commit `a1b694b83e633c2cb6115b9063d940a687759392`
- BOP Toolkit commit `cea62d651c7e395b2e1962b9749e4e89693c6ac4`
- Protocol SHA-256
  `589bae96278f7d43c81d04cb422fdda7568a9b4f2adc06e10a2df04b54cf6c0e`

## Result

| Metric | Result | Frozen gate |
| --- | ---: | ---: |
| Runtime completion | 820/820 (1.0000) | >= 0.98 |
| Mask-IoU50 matched poses | 577/770 | diagnostic |
| Joint pose precision | 0.5878 | diagnostic |
| Joint pose recall | 0.6260 | >= 0.25 |
| Joint pose F1 | 0.6063 | diagnostic |
| Joint pose AP | 0.5319 | >= 0.20 |
| AR MSSD | 0.6305 | diagnostic |
| AR MSPD | 0.6457 | diagnostic |
| Combined AR MSSD/MSPD | 0.6381 | >= 0.25 |
| Positive scenes | 5/5 | >= 4/5 |

Joint pose success requires detector-mask IoU >= 0.50, normalized MSSD < 0.10,
and MSPD < 10 px. All promotion conditions are AND gates frozen before the run.

| Scene | GT | Predictions | Joint successes | Joint recall |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 185 | 182 | 148 | 0.8000 |
| 25 | 295 | 234 | 129 | 0.4373 |
| 30 | 105 | 137 | 83 | 0.7905 |
| 40 | 135 | 198 | 83 | 0.6148 |
| 65 | 50 | 69 | 39 | 0.7800 |

The primary run completed in 575.473 seconds. Its label, GT, evaluator,
official-scorer, and scene-9 access counters were all zero. Development labels
were opened only after the primary completion receipt was frozen. The oracle
diagnostic and rescue budgets were unused (`0/1` each).

The 25 triptychs show A-R9 masks, FoundationPose CAD projections, and
development-GT CAD projections. Scene 10 is visually strong; scene 25 remains
the weakest and most clutter-sensitive scene despite passing the aggregate
gate.

## Evidence

The complete local evidence archive is intentionally under the ignored
`artifacts/` tree:

`artifacts/a9_foundationpose_e2e_20260822/a9-foundationpose-e2e-evidence.tar.gz`

- Size: 6,664,024 bytes
- SHA-256: `331476865053d5aeb771cded173163ef9a81cf02cf39de38f6a718a2a7857504`
- Members: 35
- Independent internal SHA verification: 34/34 payload members passed
- Primary prediction SHA-256:
  `a029289f61afe1222b4b45562784076287c36ed952b85820e78fba1187ed39c06`
- Evaluation result SHA-256:
  `b58231994492be504b8bbb1636ea76fac9e479059f883434bcdd862ebf663254`

Primary inference ran from implementation commit
`bc2a62e390b5cd7e10896b83a651a193ed94ee07`. The evaluation import-path fix ran
from `8cfa74109dddee7b68f71ff321de6e9bde9450b9`; it changed only evaluator module
discovery and did not rerun or alter the 820 FoundationPose predictions.

Before any metric was produced, environment-only blockers were corrected:
Python 3.12-compatible pybind11/Boost dependencies, complete official
FoundationPose requirements, regular-file checkpoint materialization, the
exact BOP Toolkit checkout, and the evaluator script path. No model formula,
checkpoint, threshold, candidate count, refine count, input prediction, or
promotion gate was changed.
