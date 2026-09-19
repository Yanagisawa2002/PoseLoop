# RTX 5090 original A9 Nsight Systems baseline

**PASS — bounded steady-state profiling, with disclosed capture-edge loss. Ready to design/run optimization A/B; no optimization was implemented.** Issue #6 remains open. PR #12 remains unmerged.

The authoritative unprofiled baseline remains **551.107429 s**, median registration **0.655114 s**, p95 **0.681723 s**, maximum allocated VRAM **7.789287 GiB**. The profiler-instrumented run completed **820/820**, zero failures, in **604.573805 s**; its wall time is not a performance baseline. All 820 per-item semantic outputs, including pose, top score and margin, exactly equal the clean run. Evaluation was not rerun: the prior 577 matches / 482 joint successes and all aggregate/five-scene results remain the reference, with unchanged prediction semantics. All five primary access counters are zero.

## Capture

| Field | Result |
| --- | --- |
| Nsight Systems | CLI-only 2026.5.1.161-265138896106v0 |
| Hardware / runtime | RTX 5090, 32607 MiB; driver 595.71.05; torch 2.8.0+cu128; CUDA runtime 12.8 |
| Executed PoseLoop commit | `3d5a33d64d9fbcdaecf97172fdea6f20ab647c41`, clean |
| FoundationPose | `a1b694b83e633c2cb6115b9063d940a687759392`, tracked source unchanged |
| Design | Original 820-item command; 60 s delay, 60 s capture, `--kill=none` |
| Actual captured duration | 60.000211702 s |
| Registrations represented | Approximately 81 registration equivalents; 80 complete pre/post-bounded registrations plus two edge fragments |
| Sampling limitation | Conservative ordinal envelope 73–160 lies in scene 10 / object 2. This is an early steady-state workload window, not an all-scene performance sample. |
| Trace | 98,570,303 bytes (94.004 MiB) |
| Trace SHA-256 | `20a72e4e34740a15a41af95fcaf4a52f0af88869ab1e70f5b99b9a73ac0deb2a` |
| Disk remaining after completion | 38,759,555,072 bytes (36.098 GiB) |

The CLI was selected from [NVIDIA's CLI-only distribution](https://developer.nvidia.com/nsight-systems/get-started). Installed help was inspected before choosing flags; its exact text is backed up outside Git. The command traces CUDA, existing/library NVTX, OS runtime, cuBLAS and cuDNN; sampling and CPU context-switch tracing are disabled. No application NVTX annotations, source changes, configuration changes, detector reruns or Nsight Compute runs were introduced.

`nsys profile` returned after report generation at about 145 s; its child continued normally to 820/820. `controller-receipt.json` describes capture-process completion, not application completion. The separate primary receipt proves workload completion. The 2 GiB trace/temp and 10 GiB free-space guardrail did not trigger; the sampled trace/temp peak was 1,264,021,931 bytes. Original clean predictions/receipt/run-lock hashes are unchanged.

## CLI measurements

| CUDA API | Calls | Summed CPU API time |
| --- | ---: | ---: |
| cudaStreamSynchronize | 102,017 | 18.634760 s |
| cudaLaunchKernel | 1,499,730 | 5.381758 s |
| cudaFree | 8,093 | 4.207852 s |
| cudaMemcpyAsync | 198,894 | 1.541339 s |
| cudaMalloc | 8,023 | 1.166143 s |
| cudaDeviceSynchronize | 162 | 0.001601 s |

Allocation/free CPU time is material at 5.374 s, but is not all independent overhead: allocations overlap GPU work; their overlap with recorded GPU-idle intervals is 2.445 s. Stream synchronization mostly waits on useful GPU work. Detailed counts include driver launch variants and asynchronous allocation in `cuda-api-summary.csv`.

| Kernel family (symbol grouping) | GPU time | Share of summed kernel time |
| --- | ---: | ---: |
| Convolution / GEMM / attention | 20.573 s | 57.37% |
| Tensor elementwise / copy / concat | 9.582 s | 26.72% |
| cuDNN layout / batchnorm | 2.932 s | 8.18% |
| nvdiffrast rasterization / interpolation | 1.248 s | 3.48% |
| Other | 1.525 s | 4.25% |

The top individual cuDNN implicit-GEMM convolution is **13.959 s / 38.93%**, **50,058 calls**, mean **278.864 us**. Next are CUTLASS SIMT SGEMM (**1.943 s / 5.42%**) and tensor-op GEMM+ReLU (**1.269 s / 3.54%**). Complete names, calls and durations are in `kernel-summary.csv`. No recognizable PyTorch3D KNN/ball-query/farthest-point kernel appeared in this steady-state window; startup geometry is not represented.

There are **1,605,437 kernels**, median **2.752 us**; **1,120,114 (69.77%)** are under 10 us. Almost all run on one stream (1,605,275 on stream 14). This proves many short, serially submitted kernels, but does not measure SM occupancy or establish that every gap is caused by historical chunking.

| Transfers | Calls | Bytes | GPU time |
| --- | ---: | ---: | ---: |
| H2D | 31,268 | 3,876,721,244 | 0.411511 s |
| D2H | 71,796 | 605,100,924 | 0.054308 s |
| D2D | 95,842 | 86,912,535,912 | 0.166325 s |

Copy engine time is not the primary bottleneck: all copies sum to 0.632 s, about 1.05% of the window. This does not dismiss CPU submission/dependency costs of frequent small copies. Tensor copy kernels are separately counted in kernel time, not these memcpy totals.

## CPU and GPU relationship

Unioning recorded kernels, memcpy and memset intervals gives **36.515 s active / 23.485 s without recorded GPU activity** (60.86% / 39.14%). This is timeline activity, not SM utilization. Of **18.636 s** in all synchronization APIs, **18.236 s overlaps GPU activity**, and only **0.401 s overlaps recorded GPU-idle time**. The 162 device synchronizations total only 1.601 ms; inferred pre-register calls total **0.721 ms / 81**, and post-register calls **0.880 ms / 81**. Their entire API intervals lie in GPU gaps, but they do not explain the much larger surrounding gaps.

There are **16.835 s of recorded GPU-idle time outside traced CUDA APIs**. CPU preparation, Python dispatch, serialization, scheduling and profiler overhead are not separable without additional ranges/sampling; do not label all of this time as preprocessing. Summed OS-runtime waits across background threads are not wall time or main-thread blockage.

## Decision and evidence limits

**Primary bottleneck: B — refiner neural compute, especially the repeated cuDNN convolution in the 32-candidate refiner chunks.** In 80 complete registrations, the five refine intervals contain **26.268 s** of GPU kernels, including **16.252 s** of convolution/GEMM/attention and **11.602 s / 41,600 calls** of the dominant convolution. The score intervals contain 9.080 s of kernels. Stage attribution is inferred from the six rendering markers and unchanged source, not application NVTX labels. See [bottleneck-analysis.md](bottleneck-analysis.md) for the single proposed change and strict A/B gates.

Nsight reports 49/50 CUPTI buffers, 384 incomplete events dropped and generic CUDA/NVTX/OSRT incompleteness warnings at capture stop. The launch-to-kernel check finds **602 unmatched launch calls, all in the final second; zero in seconds 1–59**. This check is not proof that every event type is complete. The interior 58 s still has **61.01% recorded GPU activity** and **38.98% dominant-convolution kernel share**, supporting the same decision. Raw 60 s totals are observed totals; GPU idle is a complement of recorded activity and may be overstated by missing events. CPU scheduling, exact Python attribution, achieved occupancy and bandwidth saturation remain unavailable.

## Artifacts and reproducibility

- Server trace: `/root/autodl-tmp/poseloop-a9-perf/capture/baseline.nsys-rep`.
- Verified local trace: `C:/Users/cgliu/Documents/Codex/poseloop-v12-5090/artifacts/gpu-profile-5090/server/capture/baseline.nsys-rep`.
- SQLite, full log, primary predictions/receipt/lock, help text and scripts are also backed up outside Git. Paths, sizes and hashes are in `profiler-file-inventory.json`.
- `environment.json` and `capture-command.json`: exact run identities and command.
- `baseline.json`, `primary-validation.json`: authoritative timing, captured measurements and exact semantic comparison.
- `nsys-summary.txt`, `cuda-api-summary.csv`, `kernel-summary.csv`, `kernel-family-summary.csv`, `transfer-summary.csv`: CLI-readable statistics.
- `timeline-analysis.json`, `stage-analysis.json`, `device-sync-events.csv`, `trace-edges.json`: interval arithmetic, stage inference and capture-quality evidence.
- Analysis can be rerun with `python analysis/analyze_trace.py <evidence-root>` and `python analysis/analyze_stages.py <evidence-root>` where the root contains `capture/baseline.sqlite` and a `stats/` directory. `validate_primary.py` is the recorded remote comparison script and uses the original server paths.

**Stop condition reached: profiler evidence and one bottleneck selected. No performance-critical code changed.**
