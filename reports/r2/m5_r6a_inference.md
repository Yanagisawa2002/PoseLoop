# PoseLoop M5-R6A RU-APC label-blind inference

Status: **PASS_INFERENCE_COMPLETENESS_AND_INTEGRITY**

The frozen M5-R6A inference runner completed all available frames in one uninterrupted local RTX 4090 process. Evaluator labels and pose-error outcomes were not read during inference.

| Quantity | Result |
|---|---:|
| Requested samples | 1,311 |
| Successful samples | 1,311 |
| Failed/non-finite samples | 0 |
| Pose hypotheses per sample | 252 |
| Mean registration time | 0.969624 s |
| Median registration time | 0.949642 s |
| P95 registration time | 1.259576 s |
| Maximum registration time | 1.452838 s |
| Maximum recorded CUDA allocated memory | 3,736,312,320 bytes |
| Evaluator-field leaks | 0 |

Integrity:

- prediction SHA-256: `b0e7bd42e5209817c515e091ce12b02a127fe1dd54a4c031e15aee8001c6d826`;
- inference receipt SHA-256: `0f142277cb3a0c75f62752b66eaf394178e939685a53e88e0773f95a3da04d77`;
- inference log SHA-256: `a772eaf5934656a55dd04bbdadc71b443c228c2df9b14ce98e0c4ceefe12321d`;
- resource-bounded scorer input batch: 8 candidates;
- scorer/refiner feature batches: 32 candidates;
- candidate attention scope: unchanged full 252-candidate set;
- raw FoundationPose scores were recorded for audit but were not used for filtering or confidence;
- YCB-V remained unavailable and unopened.

This report establishes inference completeness, not method effectiveness. Effect gates are applied only by the frozen evaluator.
