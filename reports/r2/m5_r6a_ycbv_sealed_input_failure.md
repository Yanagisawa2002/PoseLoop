# PoseLoop M5-R6A-S1 YCB-V sealed input failure

Status: **FAIL_M5_R6A_SEALED_INPUT**

The one permitted frozen builder invocation stopped before FoundationPose inference with `YCB-V sealed input has too few missing-represented objects`. The sealed labels were opened, so this protocol is final and was not relaxed or rerun.

| Frozen input gate | Required | Observed | Result |
|---|---:|---:|---|
| Represented objects | 14 | 21 | PASS |
| Objects with natural missing frames | 12 | 7 | FAIL |
| Natural missing frames | 96 | 357 | PASS |
| Short-gap frames | 8 | 105 | PASS |
| Medium-gap frames | 16 | 131 | PASS |
| Long-gap frames | 8 | 121 | PASS |

Objects with zero natural missing frames in their frozen best windows: 1, 2, 3, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 21.

| Object | Scene | Start frame | Available | Missing | Short | Medium | Long |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 1 | 128 | 0 | 0 | 0 | 0 |
| 2 | 50 | 1 | 128 | 0 | 0 | 0 | 0 |
| 3 | 49 | 1 | 128 | 0 | 0 | 0 | 0 |
| 4 | 50 | 450 | 64 | 64 | 0 | 0 | 64 |
| 5 | 50 | 1 | 128 | 0 | 0 | 0 | 0 |
| 6 | 59 | 909 | 67 | 61 | 29 | 32 | 0 |
| 7 | 58 | 1 | 128 | 0 | 0 | 0 | 0 |
| 8 | 58 | 1 | 128 | 0 | 0 | 0 | 0 |
| 9 | 49 | 704 | 64 | 64 | 20 | 35 | 9 |
| 10 | 50 | 1 | 128 | 0 | 0 | 0 | 0 |
| 11 | 52 | 1 | 128 | 0 | 0 | 0 | 0 |
| 12 | 51 | 1 | 128 | 0 | 0 | 0 | 0 |
| 13 | 53 | 709 | 105 | 23 | 2 | 0 | 21 |
| 14 | 48 | 1 | 128 | 0 | 0 | 0 | 0 |
| 15 | 50 | 1 | 128 | 0 | 0 | 0 | 0 |
| 16 | 55 | 1 | 128 | 0 | 0 | 0 | 0 |
| 17 | 51 | 1 | 128 | 0 | 0 | 0 | 0 |
| 18 | 57 | 1735 | 105 | 23 | 18 | 5 | 0 |
| 19 | 54 | 1782 | 70 | 58 | 15 | 33 | 10 |
| 20 | 57 | 1559 | 64 | 64 | 21 | 26 | 17 |
| 21 | 57 | 1 | 128 | 0 | 0 | 0 | 0 |

No prediction, pose error, candidate comparison, GPU inference, or sealed numerical evaluation was run. This audit only replays the already-frozen source-support selection to preserve the input failure evidence.
