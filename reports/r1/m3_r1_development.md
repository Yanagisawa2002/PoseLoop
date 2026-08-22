# PoseLoop M3-R1 development result

**Decision: `PASS_CONTINUE_TO_M4_R1`.**

M3-R1 predicts the marginal value of acquiring more views, not the success probability of the current pose. Every score below is nested, physical-instance-grouped out-of-fold development evidence from M2 only; the frozen legacy M3 holdout was not read.

## Primary result

| Metric | Value |
| --- | ---: |
| Macro-object combined score | 72.33% |
| Exact-budget matched-random expectation | 69.78% |
| Gain over matched random | +2.55 pp |
| One-sided 90% grouped-bootstrap lower gain | +1.50 pp |
| Mean acquired views | 2.180 / cap 3.0 |
| k=1 / k=3 / k=5 counts | 156 / 111 / 33 |

## Gate checks

| Check | Pass |
| --- | ---: |
| `gain` | yes |
| `bootstrap_lower` | yes |
| `mean_view_cap` | yes |
| `difficulty_guard` | yes |

## Difficulty strata

| Visibility | Rows | Mean views | Active | Matched random | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| low | 34 | 1.824 | 55.71% | 41.76% | +13.95 pp |
| mid | 124 | 2.419 | 60.97% | 57.82% | +3.15 pp |
| high | 142 | 2.056 | 88.14% | 86.99% | +1.16 pp |

## Interpretation boundary

A development PASS authorizes M4-R1 work; it is not a holdout or sealed claim. A development FAIL stops before M4-R1 and leaves the legacy M3-M6 baseline unchanged.
