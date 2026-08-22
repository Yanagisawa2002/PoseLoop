# Archived negative results

These routes are frozen as negative or unresolved evidence. They are retained
for auditability and interview discussion, not as active release work. No
further threshold tuning, blocker repair, or evaluation-split reuse is planned.

## Generic geometry, SAM2, and CAD proposals

On the fixed 25-frame real-development comparison:

| Proposal source | F1 at IoU 0.50 | AP50 | PQ |
| --- | ---: | ---: | ---: |
| Raw FastSAM diagnostic | 0.038121 | 0.002314 | 0.021011 |
| Geometry instances | 0.002024 | 0.000019 | 0.001455 |
| Geometry + SAM2 | 0.008097 | 0.001257 | 0.005365 |
| Geometry + SAM2 + CAD | 0 | 0 | 0 |

CAD candidate ranking had AUROC 0.078704 with a frame-bootstrap interval of
`[0.0, 0.5]` and removed all four true positives produced by the SAM2 variant.
SAM2 improved only 3/25 frames and 1/5 scenes, with a confidence interval that
touched zero. The causal conclusion is that mask refinement and post-filtering
cannot recover instances absent from the upstream proposal topology.

Frozen source: [`real_causal_ablation_v1/DEVELOPMENT_RESULT.md`](../pose_accuracy_recovery_prep/real_causal_ablation_v1/DEVELOPMENT_RESULT.md).

## Missing-frame pose recovery

The nearest-anchor method produced zero pose improvement over 481 naturally
missing development frames. A later covariance-based SE(3) propagation
candidate changed every missing pose but worsened all-frame normalized loss
from 13.363220 to 22.520539 and missing-frame loss from 20.117781 to 23.634636.
The failure was caused by pose-mode jumps contaminating propagated state.

This route remains a negative research result. Its protocol machinery is not a
positive temporal-recovery claim and is excluded from the release narrative.

## Confidence and risk modeling

The confidence-validation chain is structurally auditable, but it did not pass
the scientific promotion boundary and never established an independent sealed
result. It is archived as infrastructure evidence only.

## Final disposition

The released project contains one supported story: a true instance detector
fixed the upstream proposal bottleneck and enabled a positive detector-to-pose
pipeline on real development data. The archived routes cannot be used to claim
additional successful stages, sealed generalization, or official benchmark
performance.
