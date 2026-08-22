# PoseLoop M5-R2 input freeze

**FROZEN_LABELS_UNOPENED**

The old M5-R1 result remains blocked. M5-R2 uses every audited physical track that contains a natural missing observation after causal initialization; selection reads mask/depth availability but no FoundationPose or temporal-method outcome.

| Quantity | Frozen value |
| --- | ---: |
| Physical tracks | 21 |
| Represented objects | 5 |
| Replay frames | 1048 |
| FoundationPose inference inputs | 973 |
| Natural missing replay frames | 75 |
| Initial unavailable frames excluded | 2 |

| Object | Tracks | Replay frames | Natural missing |
| ---: | ---: | ---: | ---: |
| 01 | 2 | 98 | 32 |
| 09 | 3 | 150 | 17 |
| 12 | 3 | 150 | 7 |
| 14 | 5 | 250 | 9 |
| 17 | 8 | 400 | 10 |

The replay clock is frame ordinal at nominal 20 Hz; the source has no hardware timestamps. Objects are static in the dataset world frame while the recorded camera moves. Oracle association is used only to construct the non-deployable sequences.

Contract: `poseloop-m5-r2-v1`; evaluator invocation count is 0.
