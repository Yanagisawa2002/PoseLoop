# GPU profiling plan

This document defines the measurement protocol for turning PoseLoop into a
credible GPU-performance case. It intentionally contains no fabricated timing or
utilization result.

## Baseline first

Profile an unchanged supported pipeline before optimizing anything. Record:

- GPU model and memory capacity;
- driver, CUDA and PyTorch versions;
- FoundationPose commit and checkpoint identities;
- PoseLoop commit and detector checkpoint SHA-256;
- frame/prediction population;
- total wall time and completed registrations;
- peak VRAM where available.

The frozen v1.1.0 historical primary run completed 820/820 registrations in
575.473 seconds, but new profiling must report its own environment and timings.

## Nsight Systems pass

Add or use explicit ranges around:

1. detector preprocessing / inference / postprocessing;
2. per-instance input preparation;
3. FoundationPose rendering;
4. refinement;
5. scoring;
6. output materialization / serialization.

Inspect:

- CPU gaps while the GPU is idle;
- repeated synchronization points;
- H2D/D2H copies and whether they serialize work;
- kernel-launch density and small-kernel overhead;
- repeated allocation/free behavior;
- whether independent instance work can overlap safely.

Do not start with Nsight Compute across the entire application. Use Systems to
identify the expensive ranges first.

## Nsight Compute pass

For material kernels only, capture the metrics needed to distinguish:

- memory-bandwidth pressure;
- low occupancy;
- instruction/compute saturation;
- launch-size inefficiency;
- divergent or otherwise inefficient execution where applicable.

The exact metric set can vary by GPU architecture. Save the profiler reports and
a text/CSV summary so the conclusion is reviewable without the GUI.

## Optimization A/B protocol

For each candidate optimization:

1. state the profiler evidence that motivates it;
2. freeze the baseline command and input population;
3. change one major mechanism at a time;
4. rerun the same workload;
5. report median/percentile latency where repetitions are meaningful, total
   throughput, peak memory and profiler evidence;
6. verify that prediction/evaluation identity is unchanged, or declare a new
   evaluation boundary if the optimization changes numerical/model semantics.

High-value candidates to investigate only if the trace supports them:

- remove avoidable device synchronization;
- reuse allocations and materialized GPU buffers;
- batch compatible per-instance work;
- overlap CPU preparation with GPU execution;
- overlap transfers with compute when data dependencies permit;
- avoid repeated model/context setup.

## Interview-ready output

The final case should fit on one page:

**Problem → profiler evidence → bottleneck → change → A/B measurement → tradeoff →
accuracy/equivalence check.**

Do not present an optimization as successful without before/after measurements.
