# PoseLoop M4-R1 CAD visibility/ranking development result

**Decision: `PASS_FREEZE_M4_R1`.**

M4-R1 ranks candidate views only for targets where the cross-fitted M3-R1 primary policy requests more than one view. The candidate ranker predicts utility differences from frozen static slot 2 and falls back to that slot unless another candidate exceeds the frozen 0.01 predicted-gain margin.

## Primary staged result

| Metric | Value |
| --- | ---: |
| M3-continue targets | 144 |
| CAD-ranker macro combined | 79.17% |
| Fixed slot 2 | 76.48% |
| Uniform-random expectation | 74.74% |
| Gain vs fixed slot 2 | +2.69 pp |
| Gain vs random | +4.43 pp |
| 90% lower gain vs fixed | +0.17 pp |
| 90% lower gain vs random | +1.16 pp |

## Ablation on the same M3-continue population

| Feature family | Macro combined | vs fixed slot 2 | vs random |
| --- | ---: | ---: | ---: |
| `camera_geometry_only` | 78.42% | +1.95 pp | +3.68 pp |
| `camera_geometry_plus_front_facing_cad_visibility` | 78.10% | +1.62 pp | +3.36 pp |
| `camera_geometry_plus_occlusion_aware_cad_visibility` | 79.17% | +2.69 pp | +4.43 pp |

## Gate checks

| Check | Pass |
| --- | ---: |
| `gain_vs_fixed_slot` | yes |
| `gain_vs_uniform_random` | yes |
| `bootstrap_vs_fixed_slot` | yes |
| `bootstrap_vs_uniform_random` | yes |
| `each_seed_member_gain` | yes |

## Boundary

This is grouped-CV development evidence, not sealed validation. Candidate RGB, depth, masks, predictions, scores, and outcomes were not used by the pre-acquisition feature builders.
