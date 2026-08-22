# A-R8 real causal ablation development result

Date: 2026-08-21

Status: **ABANDON_ROUTE_SWITCH_INSTANCE_DETECTOR_OR_3D_PROPOSALS**.

This is a 25-frame result on already-consumed XYZ-IBD Realsense development
data. Inference was label-blind, development labels were opened only by the
evaluator, and scene 9 was not read. This result is not sealed or formal.

## Frozen comparison

| Variant | Predictions | TP / FP / FN at IoU 0.50 | Precision | Recall | F1 | AP50 | AP75 | PQ | Merge / split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-R6 baseline | 19 | 0 / 19 / 770 | 0 | 0 | 0 | 0 | 0 | 0 | 6 / 0 |
| Raw FastSAM eligible (diagnostic) | 699 | 28 / 671 / 742 | 0.040057 | 0.036364 | 0.038121 | 0.002314 | 0 | 0.021011 | 17 / 70 |
| Geometry instances | 218 | 1 / 217 / 769 | 0.004587 | 0.001299 | 0.002024 | 0.000019 | 0 | 0.001455 | 26 / 0 |
| Geometry + SAM2 | 218 | 4 / 214 / 766 | 0.018349 | 0.005195 | 0.008097 | 0.001257 | 0.000022 | 0.005365 | 23 / 1 |
| Geometry + SAM2 + causal CAD | 186 | 0 / 186 / 770 | 0 | 0 | 0 | 0 | 0 | 0 | 19 / 1 |

Merge and split are diagnostics only; they are not absolute stop gates.

## Causal decisions

- **Drop CAD.** CAD AUROC is 0.078704 with frame-bootstrap 95% interval
  `[0.0, 0.5]` (2 positive and 216 negative candidates). CAD filtering removes
  all four SAM2 true positives. Its paired mean frame-F1 delta is -0.0112,
  interval `[-0.025067, 0.0]`, with no positive frame or scene.
- **Drop SAM2.** Versus geometry, the paired mean frame-F1 delta is +0.008533,
  but its interval is `[0.0, 0.018667]`; only 3/25 frames and 1/5 scenes improve.
- **Abandon this geometry route.** The selected geometry successor improves
  over the single-output A-R6 baseline on only 1/25 frames and has interval
  `[0.0, 0.008]`. More importantly, it is stably worse than raw FastSAM:
  mean frame-F1 delta -0.032870, interval `[-0.050675, -0.017059]`, with 0/25
  positive frames.

The visual failure is upstream instance formation: dense, repetitive objects
are largely missed, while geometry follows basket holes, background edges, and
partial object fragments. SAM2 can refine a supplied mask but does not recover
the missing instance topology. CAD post-filtering cannot create missed
instances and does not rank the rare correct candidates above the wrong ones.

Do not build a new synthetic set and do not replay scene 9 from this branch.
The next route must start from a true instance detector/segmenter or a 3D
instance proposal method, and earn a stable real-development gain before either
action is reconsidered.

## Evidence identity

- implementation commit: `ef2828238b01daf2e24893ad7c6367a149b3bdbe`
- protocol SHA-256: `d54a9981400cc4049291e4214e8f297a98a872565e87ff65593242630eed320a`
- prediction manifest SHA-256: `9680fdad88bc7d95c840337d9a9f086033e80ef5a8c37281c30e6a92f64b5079`
- result SHA-256: `2adec52736b3010f6b7d0ebbb58fb7da0d0f9d32d7a82765f31f3f2b489b49cc`
- local evidence archive SHA-256: `84f0ae083adaa8a2ae42e73f9dd52f90b9deb3c11a43d8b4d4303bb078eb2c1a`
- remote run root: `/root/autodl-tmp/poseloop_a_r8_run_ef28282`

The evidence archive contains the result, prediction manifest, protocol,
producer/evaluator logs and exit receipts, implementation source, A-R7 freeze
record, and three downsampled qualitative previews. The full-resolution 25-frame
visual set remains in the remote run root.
