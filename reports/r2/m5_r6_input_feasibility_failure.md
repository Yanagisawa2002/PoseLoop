# PoseLoop M5-R6 RU-APC input feasibility failure

Status: **FAIL_INPUT_FEASIBILITY**

The frozen `poseloop-m5-r6-v1` builder stopped before writing a bundle because only 12 of 14 RU-APC objects satisfied its requirement that every selected 128-frame window contain at least one naturally unavailable frame. No FoundationPose prediction or pose-error outcome was read.

| Frozen R6 input condition | Required | Observed |
|---|---:|---:|
| Objects with an eligible 128-frame window | 14 | 12 |
| Objects with any valid 128-frame window starting available | 14 | 14 |
| Objects represented in natural-missing evaluation if all windows are retained | 14 | 12 |
| Natural missing frames if all windows are retained | 96 | 481 |
| Short-gap frames if all windows are retained | 8 | 18 |
| Medium-gap frames if all windows are retained | 16 | 102 |
| Long-gap frames if all windows are retained | 8 | 361 |

Objects 2 and 8 have zero naturally unavailable frames under the frozen mask/depth support rule in their best valid 128-frame windows. Excluding them would violate the 14-object object-adaptivity scope; inventing dropout or changing the support threshold after seeing input distributions would weaken provenance.

The next protocol must therefore use a new ID. The source-only correction is to retain all 14 objects while defining missing-frame metrics over the 12 objects that genuinely contain natural support loss. The numerical all-frame improvement, bootstrap, uncertainty-correlation, exact missing fallback, and structural gates must not be relaxed.

YCB-V remained unavailable and unopened.
