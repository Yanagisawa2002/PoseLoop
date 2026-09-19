# A measured batching hypothesis that failed equivalence

## Problem

Test whether a memory-bounded FoundationPose refiner chunk size of 32 unnecessarily limited throughput on an RTX 5090, without changing the frozen scientific workload.

## Baseline

Original A9: 820 registrations / 25 frames, 551.107429 s, median 0.655114 s, p95 0.681723 s, peak allocated 7.789287 GiB. Evaluation: 577 mask matches, 482 joint successes. Detector predictions, weights, candidates, seed and evaluation thresholds were frozen.

## Profiler evidence

A bounded 60 s Nsight Systems trace showed convolution/GEMM/attention at 57.37% of kernel time; one convolution accounted for 38.93%. Refine intervals contained 26.268 s of GPU kernels across 80 complete registrations. Outer device synchronization cost only 1.601 ms, and transfer execution was small. The trace sampled one scene/object and had disclosed edge-loss limitations.

## Hypothesis

Increasing only the refiner chunk size from 32 to 64 might reduce repeated forward dispatch and improve CNN throughput using available memory headroom. This was a testable hypothesis, not an assumed optimization.

## Change

Created a separate performance entry point that reuses A9 and binds the explicit override, base protocol SHA, source SHA and Git commit in its run evidence. The original CLI remains refine=32. No synchronization, precision, scorer, warp, model or evaluator changes were made.

## A/B measurement

A1 took 551.107429 s; clean B1 took 546.915344 s: an observed saving of 4.192086 s (0.760666%, 1.007665×). Median improved to 0.646965 s, but p95 worsened to 0.700397 s. Peak allocated memory was 7.618559 GiB. These are single-run observations; repeat controls were not entered after the accuracy gate failed.

## Correctness/equivalence validation

18 contract tests passed, and all 820 registrations succeeded. A five-scene safety test exposed large raw rotations on symmetric objects; a preserved label-free geometry diagnosis explained those sampled rotations, permitting full evaluation without changing inference or acceptance thresholds. Full outputs had only 5 exact pose matches, 40 exact top scores and translation outliers up to 0.128982 m. The unchanged evaluator produced **479 instead of 482 joint successes**; F1 fell from 0.606289 to 0.602516, and four scene results changed.

## Tradeoff

Rejected the candidate as an equivalence-preserving optimization. The tiny observed wall-time reduction is not established beyond noise, and the accuracy loss violates the acceptance contract. I did not run the conditional optimized trace or claim a confirmed kernel mechanism. Original artifacts and the original default path were preserved.

## What I would investigate next

Stop this performance experiment. If separately authorized, investigate candidate ranking/ties and numerical propagation in the changed-pose items before considering any further performance change. No second optimization or broad Nsight Compute run was performed.
