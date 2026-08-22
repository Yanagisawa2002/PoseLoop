# PoseLoop M5-R1 phase 1 development result

**PASS_CONTINUE_TO_REAL_REPLAY**

The proposed method is a causal 12-state quotient constant-velocity Kalman filter with pose/twist cross-covariance. It is compared with nearest-representative CT under equal four-configuration budgets; every reported pair is selected without its outer replicate fold.

| Metric | Nearest | Quotient CV Kalman |
| --- | ---: | ---: |
| Cross-fitted mean trajectory loss | 0.2603 | 0.2307 |
| Relative improvement | — | +11.3% |
| One-sided 90% paired-bootstrap lower | — | +11.1% |

The benchmark contains 768 fresh deterministic trajectories: four symmetry classes, four motions, six stress families, and eight replicates. Evaluator truth is never passed to either estimator.

This phase establishes synthetic mechanism value only. Real sequence replay is a separate phase and cannot inherit a positive result unless its timestamp and observed-dropout input gate is satisfied.
