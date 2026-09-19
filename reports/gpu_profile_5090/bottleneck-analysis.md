# Measured bottleneck and proposed first A/B

## One primary mechanism

**B: refiner neural compute, dominated by a repeated cuDNN implicit-GEMM convolution.** This is the largest specifically attributable compute phase. We do not select outer per-instance synchronization: its total measured CPU cost is 1.601 ms, and the leading half is only 0.721 ms. We also do not treat the unassigned 23.485 s GPU-activity complement as a single proven CPU mechanism.

The 60.000 s trace contains 35.859 s summed GPU kernel time. Convolution/GEMM/attention contributes 20.573 s (57.37%); one convolution contributes 13.959 s (38.93%). In the 80 complete registrations, exactly six `RasterizeCudaFwdShaderKernel` markers occur per registration. The original FoundationPose source executes five refine iterations and one score phase, so markers 1–5 delimit inferred refinement intervals and marker 6 begins the scoring interval. This source-supported inference yields:

| Inferred interval | All kernel time | Conv/GEMM/attention | Dominant convolution |
| --- | ---: | ---: | ---: |
| Five refinement iterations | 26.268 s | 16.252 s | 11.602 s, 41,600 calls |
| Scoring | 9.080 s | 4.145 s | 2.319 s, 8,320 calls |

The phase boundaries are GPU rendering shader starts, not exact Python function boundaries. Some preparation can fall on the adjacent side. Attribution of the CNN bursts is stronger than attribution of all miscellaneous kernels. The dominant convolution repeats 520 times per full registration in refinement (104 per iteration), consistent with the eight original 32-candidate chunks for 252 candidates. Overall kernel median is 2.752 us; 69.77% are shorter than 10 us, and 99.99% of kernels use one stream. This supports investigating batching, but does not prove low occupancy or predict a speedup.

The clean peak allocation of 7.789 GiB leaves headroom on the 32607 MiB GPU. Peak allocated memory is not total device memory, and larger-batch workspace requirements need measurement.

## Optimization candidate #1 — proposal only

**Change only the refiner feature chunk size from 32 to 64 in a separately identified performance variant.** Keep warp=32, score_data=8 and score_feature=32. For 252 candidates, refiner forward chunks become 64/64/64/60 instead of seven 32s and one 28; five refinement iterations then issue 20 chunk-forward calls rather than 40 per registration. Keep all candidates and output order.

Exact implementation sites at executed PoseLoop commit `3d5a33d64d9fbcdaecf97172fdea6f20ab647c41`:

- `scripts/run_r1_sealed_inference.py:171` — `install_memory_bounded_refine_forward`; the unchanged chunked calls/slicing are at lines 178–188.
- `pose_accuracy_recovery_prep/a9_foundationpose_e2e/runtime.py:818` — passes `batches["refine"]` to that installer. A future performance-only, recorded override should affect this argument, not silently rewrite the frozen historical protocol or modify a default used by other experiments.
- FoundationPose `learning/models/refine_network.py:73–91` at `a1b694b83e633c2cb6115b9063d940a687759392` — candidate-independent `forward`, convolution encoders and heads. No model-source change is proposed.

Expected mechanism: fewer Python/chunk-forward dispatches and larger convolution/GEMM problems may improve throughput in the measured dominant phase. It does not reduce the required mathematical work, and the outcome could be neutral or worse. No percentage speedup is claimed. Preserve source/protocol/run-lock identity accounting: label the B run as a performance variant with the one resource-batch difference, never as unchanged original A9.

Correctness risks: batch shape can change cuDNN algorithm selection, FP16 accumulation/reduction order, attention kernels, score ordering and iterative pose refinement; larger workspaces can increase peak VRAM or cause OOM. Candidate independence is a structural argument, not a numerical-equivalence guarantee. Verify eval mode remains set and no cross-candidate semantics/order changes occur.

## Future A/B acceptance

1. Preserve the clean 551.107429 s campaign as the historical timing authority. Run contemporaneous, unprofiled A (32) and B (64) on the same idle GPU/environment, preferably alternating A/B/A/B with new create-only outputs. Record warm-up policy, full 820-item wall time, registration median/p95, throughput and peak allocated/reserved/device memory. No profiler wall time is used as the performance denominator.
2. Freeze all 820 ordered inputs, detector manifest/checkpoint, FoundationPose weights/configuration, dataset, seed=0, iterations=5, candidates=252 and evaluation thresholds. Only the explicitly recorded refiner resource batch changes. Use the same source except for the isolated override; preserve every run receipt.
3. Compare each item by stable item ID: population, order, status, predicted 4x4 pose, top score, score margin and candidate count. First require exact semantic equality to the saved clean predictions; timing/memory/provenance fields are excluded. If exact comparison fails, report full per-item deltas and reject an equivalence claim until a numerical boundary has been explicitly agreed. Do not quietly introduce permissive tolerances or tune on this consumed split.
4. Require 820/820 success, 577 mask-IoU matches and 482 joint successes. Rerun the unchanged evaluator and require every aggregate value and five per-scene values to match the frozen reference. These include precision 0.5878048780487805, recall 0.625974025974026, F1 0.6062893081761006, AP 0.5318658235280908, AR_MSSD 0.6305194805194805, AR_MSPD 0.6457142857142858, combined AR 0.6381168831168831; per-scene successes: 10=148, 25=129, 30=83, 40=83, 65=39. Aggregate F1 alone is insufficient.
5. If semantics pass and unprofiled timing improves beyond A-run variability, use the same bounded Systems window to test whether dominant convolution/launch behavior changed. Check all-scene full-workload timing, since this baseline trace samples scene 10/object 2 only.

## Synchronization, gaps and limitations

`runtime.py:988` is the leading device synchronization; `runtime.py:1001` is the trailing one. The trace contains 81 inferred calls of each, totaling 721.163 us and 879.713 us respectively. Attribution uses alternating short post/pre gaps, subsequent depth-erosion kernels and the source loop, not stack samples. Both occur in GPU-idle gaps, but surrounding CPU-side intervals are much longer than the calls themselves. Removing the leading call is not proposed as a second candidate because its measured contribution is negligible.

102,017 stream synchronization calls total 18.635 s. Across all synchronization APIs, 18.236 s overlaps recorded GPU work and 0.401 s overlaps recorded GPU inactivity. This is dependency waiting, not an additive 18.6 s optimization opportunity. GPU idle outside CUDA API intervals is 16.835 s; without CPU sampling or application NVTX, it cannot be split into preparation, Python dispatch, serialization, scheduling and profiling effects. OS-runtime wait totals span multiple background threads and are not a wall-time attribution.

Recorded gaps >=1 ms sum to 7.312 s (775 gaps); >=10 ms sum to 5.806 s (186 gaps). The aggregate includes intra-registration and inter-registration gaps and capture boundaries. We have not established that these are specifically between feature chunks. Achieved SM occupancy, memory bandwidth saturation and low-level instruction stalls are unavailable from this capture.

The profiler reports incomplete events at stop. All 602 launch calls lacking correlated kernels fall in the final second; none fall in the interior 1–59 s. Interior dominant-convolution share is 38.98%, consistent with 38.93% over the full capture. Treat activity-union values as recorded activity and acknowledge potential missing events; do not equate them with hardware utilization counters.

## Possible later Nsight Compute target

Nominate **one family only**: the dominant `sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize128x32x32_stage4_warpsize4x1x1_g1_tensor16x8x16_execute_kernel__5x_cudnn` inside a representative refiner chunk. If a later Compute pass is justified, filter to a few matching launches and inspect achieved occupancy, tensor-pipe activity, memory throughput and launch dimensions for A versus B. Do not profile the complete 820-item workload with Compute.

No optional second optimization is proposed. No optimization, NVTX insertion, model/configuration change or Nsight Compute collection was performed in this task.
