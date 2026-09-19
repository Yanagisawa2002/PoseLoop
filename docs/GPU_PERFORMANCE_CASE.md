# ML/CV GPU profiling with a correctness gate

**Decision: reject refine=64.** The experiment completed the profiling,
hypothesis, A/B, correctness-validation and decision workflow; issue #6 is
completed. The code in PR #13 was closed without merging.

## Measurement and decision

The clean baseline registered 820 instances in **551.107429348 s**, with median
0.655113700 s, p95 0.681723405 s and peak allocated VRAM 7.789287 GiB.
A delayed 60-second Nsight Systems capture covered about 81 registrations
(80 complete), on scene 10/object 2. It showed 1.605 million kernels, median
2.752 microseconds, with 69.77% below 10 microseconds. Conv/GEMM/attention accounted
for 57.37% of summed kernel time; one convolution accounted for 38.93%.
These are kernel-time shares, not end-to-end wall-time shares.

The hypothesis was that doubling only the refiner chunk from 32 to 64 could
reduce repeated forward/chunk work with available memory headroom. Candidate
count 252, iterations 5, seed 0, inputs, precision, warp and scorer batching,
weights and evaluation thresholds were held fixed. A bounded safety check
preceded the full run. Large sampled raw rotations were inspected with the
declared object symmetries; that diagnostic did not establish full equivalence.

| Metric | Baseline (32) | Candidate (64) |
| --- | ---: | ---: |
| Completed registrations | 820/820 | 820/820 |
| Unprofiled wall time | 551.107429348 s | 546.915343799 s |
| Median latency | 0.655113700 s | 0.646964629 s |
| p95 latency | 0.681723405 s | 0.700396545 s |
| Peak allocated VRAM | 7.789287 GiB | 7.618559 GiB |
| Mask matches | 577 | 577 |
| Joint successes | 482 | 479 |
| Joint F1 | 0.6062893081761006 | 0.6025157232704402 |
| Combined AR | 0.6381168831168831 | 0.6335064935064936 |

Observed saving was 4.192085549 s (0.760665766%; ratio 1.007664962×), from one
baseline/candidate pair. Only 5/820 poses and 40/820 top scores were exactly equal.
Maximum pose-element absolute difference was 1.998599887; a pose matrix mixes
rotation and translation, so this is not a physical pose-error metric. Maximum
translation-element difference was 0.128981538 m. All five scene results and
aggregate metrics were checked; evaluation equivalence failed.

> Rejected: +0.76% wall-time improvement but failed numerical/evaluation equivalence.

The conditional repeat controls and optimized Nsight capture were not run after
that failure. No reproducible speedup, noise bound or confirmed mechanism is
claimed. The original route remains refine=32. No Candidate #2 is planned.

## What the trace does and does not prove

Transfer time was small; the outer device synchronizations were about 1.6 ms
across the capture. Large StreamSynchronize API time mostly overlapped GPU work,
so it was not automatically removable overhead. Most work used one stream.
The trace motivates a batching experiment; it does not establish achieved
occupancy, memory-bandwidth saturation, or the benefit of kernel fusion.
Capture-edge incomplete events were documented, with interior checks used to
bound their impact. No broad Nsight Compute pass was performed.

See [machine-readable A/B, semantic and evaluation comparisons](../reports/reproduction_5090/summary.json),
[full profiling evidence](https://github.com/Yanagisawa2002/PoseLoop/tree/4a96a7c102510abc43eee4b35e40f02cd932bee2/reports/gpu_profile_5090),
and [full experiment evidence](https://github.com/Yanagisawa2002/PoseLoop/tree/1ce0d7a94d1fc08b21875b6dc7f18085d86eb36f/reports/gpu_perf_case).

## Resume bullets

- Recovered a missing Mask R-CNN checkpoint through deterministic RTX 5090
  retraining; verified bit-exact checkpoint and detector-output identities and
  reproduced recorded 820-instance FoundationPose evaluation metrics.
- Reconstructed a 770-instance failure taxonomy, separating 193 mask-handoff
  failures from 95 pose failures with symmetry-aware evaluation and hashed provenance.
- Profiled an ML/CV GPU pipeline with Nsight Systems, tested a single-variable
  batching hypothesis, and rejected a 0.76% single-pair wall-time reduction after
  numerical checks and evaluation exposed a drop from 482 to 479 joint successes.

## 90-second interview story

“PoseLoop is my RGB-D instance-detection and 6D-pose pipeline for crowded
industrial scenes. I built the integration, evaluation and evidence tooling
around Mask R-CNN and FoundationPose.

The first challenge was reproducibility: the frozen detector checkpoint was
missing. I recovered it through deterministic retraining on an RTX 5090 and
verified an exact SHA-256 match. The detector outputs also matched, and the
original pose pipeline reproduced all recorded aggregate and per-scene metrics
across 820 registrations. I then reconstructed failures for all 770 ground-truth
instances, separating detection handoff problems from pose errors.

For performance, I established an unprofiled baseline of about 551 seconds and
used Nsight Systems to identify convolution-heavy work and many short kernels.
I tested one hypothesis: doubling the refiner batch from 32 to 64 while keeping
everything else fixed.

The candidate finished about four seconds faster, but correctness checks showed
different poses and three fewer successful joint detections. I rejected it rather
than claim a speedup. The timing was only one pair, so it also did not establish
a repeatable gain. The key lesson was to make output correctness a hard gate in
GPU optimization, and to distinguish byte identity, numerical differences and
task-level equivalence.”

## GPU Performance Engineer deep-dive questions

**Why Nsight Systems before Nsight Compute?**
I needed the end-to-end execution pattern, CPU/CUDA API interactions and dominant
kernel families first. Compute would answer a bounded kernel question after that;
I did not measure occupancy or infer a memory bottleneck from short kernels alone.

**Why batch 64?**
The refiner dominated measured compute and the baseline used under 8 GiB of about
32 GiB. Increasing its chunk was a small, testable hypothesis. Memory headroom
alone does not predict throughput or correctness.

**Why can a batch-size change affect results?**
Kernel selection and floating-point execution can change with tensor shape.
Near-tied candidates can then select different poses, including symmetric poses.
That is a plausible explanation, not a proven root cause in this experiment.
I measured numerical deltas and ran the unchanged task evaluator instead of
assuming those differences were harmless.

**What was the correctness gate?**
First real-sample safety checks: finite outputs, no CUDA/OOM errors and unchanged
candidate count. Then all 820 IDs, poses, top scores and margins were compared,
followed by the same mask/joint-pose metrics and five-scene results. Completion
alone was insufficient: joint successes fell from 482 to 479.

**Did the candidate improve performance?**
One pair was 0.7607% faster in total, while p95 became about 18.7 ms worse. That
does not establish a repeatable speedup. Repeated controls and a comparable
candidate trace were conditional on correctness and were skipped after failure.

**Would removing synchronization help?**
Not based on this evidence. Stream waits often expose outstanding GPU work rather
than cause it; outer device-sync time was negligible. Removing necessary waits
could change timing semantics or correctness without improving real throughput.

**How did you avoid benchmark leakage?**
Inputs and thresholds were frozen, primary outputs were hashed before evaluation,
and runtime access counters were recorded. I explicitly label this as consumed
development data, not external generalization evidence or an official BOP result.

**What did you personally implement?**
Detector preparation/training/inference integration, mask-to-pose orchestration,
evaluation and provenance tooling, failure analysis, and the controlled profiling
experiment. Mask R-CNN and FoundationPose are upstream models; I do not claim
their architectures or CUDA kernels as my own.

**Why stop here?**
The hypothesis produced a useful rejected result and the engineering workflow is
complete. Further tiny gains on this consumed split are lower priority than new
external evidence or a real bug. The repository is frozen for experiments.
