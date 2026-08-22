# PoseLoop M5-R6 RU-APC source-encoding audit

Status: **PASS_SOURCE_ENCODING_WITH_BOUNDED_MASK_METADATA_ADAPTER**

The first post-freeze M5-R6 build stopped before window selection because one decoded `mask_visib` PNG differed from archived `px_count_visib` by one pixel. A full scan then covered all 5,964 GT-present mapped target-object entries. It did not select replay windows, run FoundationPose, or compute pose error.

| Check | Full-scan result |
|---|---:|
| Target entries | 5,964 |
| Visible-mask exact matches | 5,950 |
| Visible-mask `actual - declared = -1` | 8 |
| Visible-mask `actual - declared = +1` | 6 |
| Maximum absolute mask-count delta | 1 pixel |
| Strict BOP rotation-parser failures | 0 |
| Maximum rotation orthogonality element error | 0.000001203430000096 |
| Rotation determinant range | 0.999998851045 to 1.000001212288 |
| Rotation singular-value range | 0.999999180443 to 1.000000726174 |

Frozen RU-APC-specific handling:

- retain the shared strict BOP pose parser without rotation projection;
- accept only a `px_count_visib` difference of at most one pixel;
- always compute availability and inference masks from the decoded PNG, never from the archived count;
- reject any larger mask-count discrepancy before replay-window selection;
- leave the method, candidates, cross-fitting, windows, metrics, and numerical gates unchanged.

This is a source-integrity adapter, not a method result. YCB-V remained unavailable and unopened.
