# PoseLoop M6-R2 deployable-feature development

**PASS_FREEZE_M6_R2**

All inner and outer models are restricted to `selected_output_only`. No M3 k3 score, unacquired-view prediction, pair feature, CAD feature, or object identity is available to this risk model.

| Metric | Learned | Raw-score baseline | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.7561 | 0.3767 | +0.3794 |
| AURC | 0.2130 | 0.5184 | +58.9% relative |
| Positive outer folds | 5/5 | — | — |

| Fold | N | AUROC gain | AURC reduction | Both improve |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 40 | +0.247 | +30.0% | yes |
| 1 | 79 | +0.616 | +73.7% | yes |
| 2 | 49 | +0.222 | +51.6% | yes |
| 3 | 69 | +0.362 | +54.9% | yes |
| 4 | 63 | +0.432 | +65.4% | yes |

This replaces the broader R1 OOF claim for deployable M6 evidence.
