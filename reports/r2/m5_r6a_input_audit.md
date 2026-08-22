# PoseLoop M5-R6A RU-APC development input audit

Status: **DEVELOPMENT_BUNDLE_READY**

This bundle uses only GT-present target-object frames. Natural missing frames are caused by the frozen visible-mask/depth support rule. Frames without an evaluable target pose are excluded rather than counted as dropout; no artificial deletion was used. YCB-V remained unavailable and unopened.

| Quantity | Value |
|---|---:|
| Selected tracks/objects | 14 |
| Replay frames | 1792 |
| Inference frames | 1311 |
| Natural missing frames | 481 |
| long_gap_8_plus frames | 361 |
| medium_gap_3_7 frames | 102 |
| short_gap_1_2 frames | 18 |

| Object | Scene | Window start | Available | Missing |
|---:|---:|---:|---:|---:|
| 1 | 1 | 228 | 92 | 36 |
| 2 | 2 | 0 | 128 | 0 |
| 3 | 3 | 0 | 117 | 11 |
| 4 | 4 | 157 | 64 | 64 |
| 5 | 5 | 0 | 106 | 22 |
| 6 | 6 | 248 | 118 | 10 |
| 7 | 7 | 196 | 74 | 54 |
| 8 | 8 | 0 | 128 | 0 |
| 9 | 9 | 44 | 112 | 16 |
| 10 | 10 | 120 | 64 | 64 |
| 11 | 11 | 142 | 64 | 64 |
| 12 | 12 | 12 | 116 | 12 |
| 13 | 13 | 189 | 64 | 64 |
| 14 | 14 | 246 | 64 | 64 |

Source archives:

- `ruapc_base.zip`: `245cabf78a52f24eef7ce4c187015dfecf753c1d85b556a2591e97175a08e247`
- `ruapc_models.zip`: `00e713c5782dd8d3d92e8df5744aa5d30595c2cac8f88fdfbabd8d193c9603eb`
- `ruapc_test_all.zip`: `d2b9c4528ba8e89ffffcf1f298101855b4aad48c6e12e04c06675a449a6981e7`

Source-encoding audit:

- The shared strict BOP rigid-pose parser accepted all 5964 target poses; maximum elementwise orthogonality error was `1.2034300001e-06`.
- Visible-mask counts: 5950 exact and 14 within the frozen one-pixel archive tolerance; maximum absolute delta was 1 pixel.
- Availability always uses the decoded PNG count; archived `px_count_visib` is only a bounded source-integrity check.

This audit is not a method outcome.
