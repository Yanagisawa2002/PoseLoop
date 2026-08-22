# PoseLoop M5-R4A HB Primesense development

Status: **FAIL_M5_R4A_DEVELOPMENT**

M5-R4A was frozen before inference after M5-R4 exposed only an input-feasibility shortfall. It uses new HB Primesense scenes 1-8 with object-grouped cross-fitting. The estimator uses only causal object-local mask/depth support and innovation consistency; raw FoundationPose scores are excluded from confidence. Kinect 2 scenes 9-13 remain untouched.

| Development condition | Observed | Required |
|---|---:|---:|
| Cross-fitted all-frame improvement | -10.49% | >= 5% |
| Cross-fitted missing-frame improvement | -197.67% | >= 0% |
| Bootstrap 10th percentile | -31.75% | >= 0% |
| Missing uncertainty/error Spearman | +0.721 | >= 0.20 |
| Nonfinite outputs | 0 | <= 0 |

| Candidate | All-frame improvement | Missing improvement |
|---|---:|---:|
| adaptive_hold | -16.28% | -238.89% |
| adaptive_short | -12.41% | -206.13% |
| adaptive_medium | -11.10% | -199.21% |
| adaptive_agile | -10.49% | -197.67% |

Modal fold winner for any future sealed freeze: `adaptive_agile`.

This is development evidence only. A failed gate forbids downloading or inspecting the Kinect 2 sealed split.
