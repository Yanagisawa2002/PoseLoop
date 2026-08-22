# PoseLoop M5-R1 real replay input audit

**BLOCKED_REAL_REPLAY_INPUT_GATE**

The local XYZ-IBD RealSense data provide ordered recorded frames but no hardware timestamps, so the frozen protocol maps common-frame ordinal to a nominal 20 Hz clock. Observation availability uses only input mask area and valid-depth support; predictor outcomes and pose errors are not read.

| Input-gate quantity | Observed |
| --- | ---: |
| Selected object sequences | 15 |
| Sequences with at least five observations | 15 |
| Sequences with observed missing frames | 0 |
| Total observed missing frames | 0 |

FoundationPose replay was not launched because the input gate is evaluated before pose inference. Artificially deleting valid frames would create a synthetic-dropout replay and would not satisfy the frozen real-dropout claim.
