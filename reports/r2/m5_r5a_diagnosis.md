# PoseLoop M5-R5A post-outcome diagnosis

Status: **DIAGNOSIS OF FAILED DEVELOPMENT GATE — NOT A NEW CANDIDATE RESULT**

M5-R5A passed 10 of 11 frozen development conditions. The only failure was the required all-frame macro-object relative improvement: observed `+0.1408%` versus required `+5%`.

What did work:

- all eight leave-one-object-out folds independently selected `anchored_responsive`;
- all eight held-out objects had a positive all-frame delta;
- the 10th-percentile paired hierarchical bootstrap bound was positive at `+0.1070%`;
- missing-frame output matched the nearest anchor exactly on all 113 frames;
- missing-frame improvement was exactly `0%`, with zero unsafe reacquisitions and zero non-finite outputs;
- missing uncertainty/error Spearman was `+0.2363`, above the frozen `+0.20` gate.

The candidate family was directionally ordered but far too weak:

| Candidate | All-frame improvement | Corrected available frames |
|---|---:|---:|
| anchor_calibrated | 0.0000% | 0 |
| anchored_conservative | 0.0542% | 402 |
| anchored_balanced | 0.1059% | 404 |
| anchored_responsive | 0.1408% | 404 |

## Decisive failure mechanism

A post-outcome diagnostic compared the frozen nearest anchor with the already-recorded raw FoundationPose measurement on the 1,423 available frames. These pooled values are diagnostic only; they were not frozen evaluation metrics and cannot support a method claim.

| Available-frame subset | Frames | Anchor loss | Measurement loss | Proposed loss | Measurement better |
|---|---:|---:|---:|---:|---:|
| All available | 1,423 | 10.4024 | 1.4196 | 10.3855 | 76.25% |
| Correction accepted | 404 | 1.3642 | 1.2964 | 1.3046 | 19.55% |
| Correction rejected | 1,019 | 13.9858 | 1.4684 | 13.9858 | 98.72% |

An unavailable diagnostic oracle that selected the lower-loss pose independently on every available frame would reduce pooled anchor loss by 87.67%. This is not an achievable result; it only shows that enough signal exists and that the frozen gate rejects it in the wrong region.

The rejection pattern is explained by anchor/measurement disagreement:

- disagreement below 1: measurement was better on only 13.4% of 373 frames;
- disagreement from 1 to 2: measurement was better on 93.5% of 31 frames;
- disagreement from 3 to 5: measurement was better on 100% of 87 frames;
- disagreement from 5 to 10: measurement was better on 100% of 350 frames;
- disagreement from 10 to 30: measurement was better on 98.6% of 504 frames;
- disagreement above 30: measurement was better on 92.3% of 78 frames.

M5-R5A assumed that large anchor/measurement innovation primarily signals a bad current measurement. In this LM-O replay it primarily signals a stale persistent nearest anchor. The hard maximum-disagreement gate therefore preserved the wrong pose exactly when a fresh measurement was most useful. Mask/depth support was generally high and did not resolve which pose was stale.

## Consequence for the next protocol

LM-O is now consumed development data and must not be used for another claimed tuning or validation result. The next stage requires another genuinely unused development dataset and a new pre-frozen protocol.

The next candidate family should preserve the successful safety boundary—nearest remains the persistent owner, missing frames are exact anchor passthrough, no forced reacquisition—but reverse the available-frame logic:

1. assess current-measurement reliability from causal object-local support and measurement-to-measurement consistency;
2. do not reject a reliable measurement merely because it disagrees with the stale anchor;
3. permit bounded measurement-first current-frame output without feeding it back into persistent state;
4. retain causal gap-decayed confidence and explicitly include anchor/measurement divergence in missing-frame uncertainty;
5. freeze the method, candidates, metrics, and gates before opening any pose/error outcome on the new data.

YCB-V remains the unopened sealed dataset and cannot be repurposed as development after this failed gate.
