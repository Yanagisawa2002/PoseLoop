# PoseLoop M5-R2 recorded RealSense replay

**FAIL_M5_R2_REAL_REPLAY**

The two phase-1-frozen temporal methods were replayed on 21 recorded physical tracks selected only because their input streams contain natural post-initialization dropouts. FoundationPose was rerun independently on every available frame before the evaluator stream was opened once.

| Macro-object metric | Nearest | Quotient CV Kalman | Relative delta |
| --- | ---: | ---: | ---: |
| All replay frames | 7.7087 | 7.4437 | +3.4% |
| Natural missing frames | 7.8086 | 7.8096 | -0.0% |
| One-sided 90% hierarchical-bootstrap lower | — | — | -0.3% |

| Object | Tracks | Missing frames | Nearest loss | Proposed loss | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 2 | 32 | 16.5541 | 16.5802 | -0.2% |
| 09 | 3 | 17 | 11.7316 | 10.6772 | +9.0% |
| 12 | 3 | 7 | 8.1031 | 7.7490 | +4.4% |
| 14 | 5 | 9 | 1.3134 | 1.3450 | -2.4% |
| 17 | 8 | 10 | 0.8414 | 0.8673 | -3.1% |

Inference statuses: {'success': 973}. Non-finite temporal outputs: 0.

Claim boundary: this is observed-dropout-enriched development replay on recorded XYZ-IBD RealSense multi-view data with a nominal 20 Hz frame clock and oracle association. The objects are static in the world frame; it is not sealed, unseen-sensor, hardware-timestamp, deployable-association, or moving-object evidence. The old M5-R1 real-replay result remains blocked.
