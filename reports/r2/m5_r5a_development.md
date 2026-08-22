# PoseLoop M5-R5A LM-O anchored-confidence development

Status: **FAIL_M5_R5A_DEVELOPMENT**

The frozen nearest estimator owns persistent pose state. Object-local confidence may apply only a bounded current-frame correction; missing frames are exact anchor passthrough. YCB-V sealed data remains unavailable until every development condition passes.

| Development condition | Observed | Required |
|---|---:|---:|
| Cross-fitted all-frame improvement | +0.14% | >= 5% |
| Cross-fitted missing-frame improvement | +0.00% | >= 0% |
| Bootstrap 10th percentile | +0.11% | >= 0% |
| Missing uncertainty/error Spearman | +0.236 | >= 0.20 |
| Objects represented in missing-frame metric | 8 | >= 8 |
| Missing-anchor mismatches | 0 | <= 0 |
| Unsafe forced reacquisitions | 0 | <= 0 |
| Nonfinite outputs | 0 | <= 0 |

| Candidate | All-frame improvement | Missing improvement | Corrections |
|---|---:|---:|---:|
| anchor_calibrated | +0.00% | +0.00% | 0 |
| anchored_conservative | +0.05% | +0.00% | 402 |
| anchored_balanced | +0.11% | +0.00% | 404 |
| anchored_responsive | +0.14% | +0.00% | 404 |

Modal fold winner for any future sealed freeze: `anchored_responsive`.

This is development evidence only. A failed gate forbids downloading or inspecting YCB-V sealed data.

Result integrity:

- development result SHA-256: `93054a0c7ef8aa7fbae9ad2b56fd5df90708dd38c01f4f703b52f7772a7767dd`;
- cross-fitted track results SHA-256: `f3df3a3026340167f091ffc6fe0349417d9d667ca5951dcb3a0f6ebcb9c81025`;
- frozen predictions SHA-256: `e323478190013d91323c7274b8d459c2906d7776ad412c53b4a121b93dd9de26`;
- YCB-V remained unavailable and unopened.
