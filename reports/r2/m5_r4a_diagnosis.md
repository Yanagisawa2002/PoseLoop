# PoseLoop M5-R4A development diagnosis

Status: **FAIL_M5_R4A_DEVELOPMENT — SEALED ACCESS FORBIDDEN**

M5-R4A ran once on the frozen HB Primesense development bundle. All 1,279 available-frame FoundationPose registrations succeeded, but the cross-fitted method failed three required pose-improvement conditions. The Kinect 2 sealed archive was not downloaded or read.

## Gate result

| Condition | Observed | Required | Result |
|---|---:|---:|---|
| Cross-fitted all-frame improvement | -10.49% | >= 5% | FAIL |
| Cross-fitted natural-missing improvement | -197.67% | >= 0% | FAIL |
| Hierarchical-bootstrap 10th percentile | -31.75% | >= 0% | FAIL |
| Missing uncertainty/error Spearman | +0.721 | >= 0.20 | PASS |
| Nonfinite outputs | 0 | <= 0 | PASS |

The confidence objective is therefore a real positive development result: uncertainty ranks missing-frame error well. It is not enough to advance because the pose estimator itself is worse than the frozen nearest baseline.

## Root cause

Across available frames, the selected adaptive estimator reduced the micro-average normalized error by about 18.9%. Across the 129 naturally missing frames it increased the micro-average error from 1.61 to 6.61. Every frozen candidate failed; even `adaptive_hold` reached only -16.28% all-frame and -238.89% missing-frame improvement. This rules out a simple choice of prediction horizon as the repair.

Two mechanisms dominate:

1. **Unanchored dropout motion.** Error grows with gap length even after extrapolation stops because the held pose is already displaced while the object keeps moving. For object 15 in scene 5, a 20-frame gap starts from baseline/proposed errors 0.37/0.57 and ends at 1.28/13.79. Confidence falls and uncertainty rises correctly, but neither protects the pose state.
2. **Unsafe forced reacquisition.** The estimator reacquires after three rejected measurements without requiring the reacquisition measurement itself to be trustworthy. For object 17 in scene 7, raw measurement errors are 33.78 and 34.84 before a four-frame gap; the next raw error is 23.82. The frozen nearest baseline rejects it and remains at 0.59, while M5-R4A force-reacquires to 23.82 with confidence 0.05. The following 11-frame gap averages 25.97 proposed error versus 0.93 baseline.

The strong uncertainty correlation and the failed pose update point to a specific next design: decouple confidence estimation from pose ownership. Keep the frozen nearest estimator as a safety anchor, use object-local confidence only to permit bounded corrections or report abstention, fall back to the anchor at low confidence, and never force reacquisition solely because a rejection counter reached a threshold.

This diagnosis may guide a new protocol, but no revised method may be selected or validated on the consumed HB Primesense development outcomes.

## Immutable artifact hashes

- `contract.json`: `4176ddaf5489f54f71422445737e5d163281fc1a214d79412adb0da64c78ed14`
- `predictions.jsonl`: `f5c99c7adfa835b24e3e7389bfec8e368edafabe43d0b4502f8ccf52fa0b2ffb`
- `inference_receipt.json`: `01adcea902d928370695351fb57138438d878d1729d99bd47c809ddec0b8ada8`
- `development_result.json`: `68c00173ff75affcb626bc0323f56f68cc8dde30e64b3c0738113863cc20d33f`
- `development_track_results.jsonl`: `cb7efd52489f3ea339b55efe0219ec4f75b4b956b63db3993c7cf4eefccecebe`
