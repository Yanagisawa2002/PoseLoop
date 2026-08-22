# PoseLoop M5-R5A LM-O development input audit

Status: **DEVELOPMENT_BUNDLE_READY**

This bundle uses only GT-present LM-O frames. Natural missing frames are caused by the frozen visible-mask/depth support rule; frames without an evaluable GT pose are excluded rather than counted as dropout. No artificial deletion was used, and no YCB-V sealed archive was downloaded or read.

| Quantity | Value |
|---|---:|
| Selected tracks/objects | 8 |
| Replay frames | 1536 |
| Inference frames | 1423 |
| Natural missing frames | 113 |
| long_gap_8_plus frames | 9 |
| medium_gap_3_7 frames | 61 |
| short_gap_1_2 frames | 43 |

## Legacy LM-O rigid-pose adapter

The initially frozen builder stopped before window selection because legacy LM-O `cam_R_m2c` values are not all valid SO(3) rotations. The source-only adapter was added after that stop and before any inference or pose-error outcome. It projects every bounded, orientation-preserving matrix to the closest proper rotation by SVD; translation, selection, candidates, metrics, and gates are unchanged.

| Adapter audit | Value |
|---|---:|
| Archived GT entries checked/projected | 9209 |
| Entries beyond the shared parser tolerance | 5958 |
| Maximum raw orthogonality error | 0.00938687613882 |
| Raw determinant range | 0.999989814436 to 1.01368877691 |
| Raw singular-value range | 0.999987465082 to 1.00479347572 |
| Maximum elementwise projection correction | 0.00467401076753 |
| Maximum Frobenius projection correction | 0.0078798217122 |
| Visible-mask metadata exact matches | 9131 / 9209 |
| Visible-mask metadata nonzero deltas | 78 |
| Maximum visible-mask metadata delta | 2 pixels |

Availability is always computed from the decoded PNG mask; archived `px_count_visib` is only a bounded source-integrity cross-check.

| Object | Window start | Available | Missing |
|---:|---:|---:|---:|
| 1 | 259 | 183 | 9 |
| 5 | 192 | 189 | 3 |
| 6 | 446 | 150 | 42 |
| 8 | 165 | 191 | 1 |
| 9 | 987 | 168 | 24 |
| 10 | 436 | 181 | 11 |
| 11 | 196 | 171 | 21 |
| 12 | 800 | 190 | 2 |

Source archives:

- `lmo_base.zip`: `d0e99cf6ab1acef7040a56ddf1f5b3d486449ef7b9927ce7898a62b07dee08e2`
- `lmo_models.zip`: `48e4cbe0aac6c2d2da03339d5110496b2fb1201a16043f9e04ae85a5022a7842`
- `lmo_test_all.zip`: `33c9de88c15d0fb21a08d686c7eb45c2fa335a4a96e10384756f946be71c73e2`

This audit is not a method outcome.

Freeze and integrity boundary:

- corrected pre-inference protocol/code commit: `92d0a8b`;
- protocol SHA-256: `74a5648425deec9ef452e196db348e640483bf831ac1379964e85416f964ddff`;
- bundle contract SHA-256: `93d6a3a355d1132b3d8172d51517ca633fcf6feaeffb0cf8bf30de7bf4123512`;
- every contract-recorded code and bundle-file hash matched on audit;
- all eight tracks start available and all eight contain natural support loss;
- the 1,423-row inference manifest contains no evaluator pose/error field;
- no prediction file existed at input-freeze audit time.
