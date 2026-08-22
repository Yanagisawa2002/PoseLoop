# PoseLoop M5-R3 Photoneo input freeze

**SEALED_LABELS_UNOPENED**

The static all-history quotient medoid was frozen using only the consumed M5-R2 RealSense replay. These Photoneo tracks have zero overlap with the union of every R1 and R2 Photoneo group track.

| Quantity | Frozen value |
| --- | ---: |
| Prior Photoneo tracks excluded | 259 |
| Sealed physical tracks | 23 |
| Represented objects | 8 |
| Replay frames | 1146 |
| FoundationPose inference inputs | 837 |
| Natural missing frames | 309 |
| Pre-initialization missing excluded | 4 |

| Object | Tracks | Replay frames | Natural missing |
| ---: | ---: | ---: | ---: |
| 01 | 2 | 100 | 38 |
| 04 | 1 | 50 | 15 |
| 08 | 7 | 350 | 82 |
| 09 | 1 | 50 | 17 |
| 11 | 1 | 50 | 2 |
| 14 | 6 | 298 | 49 |
| 16 | 2 | 100 | 18 |
| 17 | 3 | 148 | 88 |

Selection used only natural input availability, frame order, object ID, and oracle association for split isolation. It did not read predictions, temporal outputs, or pose-error outcomes from the retained tracks.

Contract: `poseloop-m5-r3-v1`; evaluator invocation count is 0.
