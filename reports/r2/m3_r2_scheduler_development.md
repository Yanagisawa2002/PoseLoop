# PoseLoop M3-R2 scheduler development

**Decision: `PASS_FREEZE_FOR_R2`.**

The R1 marginal-value model is unchanged. R2 replaces sensor-sensitive absolute thresholds with sequential batch quantiles whose quotas were copied from the already frozen R1 primary OOF operating point. Stage two is ranked only after stage-one views are acquired.

## RealSense nested-OOF validation

| Metric | Value |
| --- | ---: |
| Active macro combined | 71.78% |
| Exact-budget matched random | 69.78% |
| Gain | +2.00 pp |
| One-sided 90% grouped-bootstrap lower gain | +0.35 pp |
| Mean views | 2.180 |
| k=1 / k=3 / k=5 | 156 / 111 / 33 |

## Consumed Photoneo label-free dry run

The scheduler selected k=1/3/5 counts 78/55/17 with mean views 2.187. No evaluator artifact or pose outcome was read.

## Gate

- `gain`: pass
- `bootstrap_lower`: pass
- `mean_view_cap`: pass
- `difficulty_guard`: pass
