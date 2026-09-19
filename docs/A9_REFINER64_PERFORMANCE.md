# A9 refiner64 performance experiment

The historical entry point `python -m pose_accuracy_recovery_prep.a9_foundationpose_e2e run-primary` still uses the frozen protocol's refiner batch of 32. Neither the protocol nor `release/v1.1.0/` is changed.

The separate create-only entry point is:

```bash
python -m pose_accuracy_recovery_prep.a9_foundationpose_e2e.performance \
  --manifest /path/to/frozen/input-manifest.json \
  --foundationpose-root /path/to/pinned/FoundationPose \
  --output-root /new/performance/output \
  --implementation-commit <full-clean-deployed-Git-SHA>
```

This entry point allows exactly one inference override: refiner chunk size 64. Warp=32, score_data=8, score_feature=32, precision, synchronization, candidates, seed and iteration count remain unchanged. It reuses `runtime.run_primary`; it does not copy the runtime or mutate the protocol in memory.

Its run lock and completion receipt include `experiment_kind=PERFORMANCE_EXPERIMENT_NOT_FROZEN_V1_1`, `performance_variant`, the base protocol/manifest SHA, entry-point source SHA and Git commit. `frozen_inference` identifies the historical base configuration; `effective_resource_batches` identifies the actually executed batch sizes. The run-lock hash binds both. An original-route resume rejects performance output before model allocation. Performance output is create-only and cannot resume through the experiment CLI.

For a bounded real-data safety run, add `--safety-check`. It selects the first frozen item from each scene, preserves item order, records the IDs and marks the completion stage `performance-safety-check`. The full evaluator rejects this partial completion.

Compare to an original clean primary root with:

```bash
python scripts/compare_a9_performance.py \
  --baseline-root /path/to/clean/refine32 \
  --candidate-root /path/to/refine64 \
  --output /new/comparison.json
```

Add `--safety` only for the five-item safety population. Its conservative raw-element screen returns exit code 2 if rotation elements differ by more than .01, translations by more than .001 m, top scores or margins by more than 1, or allocated VRAM reaches 24 GiB. These are diagnostic safety-screen limits, not new evaluation thresholds. Symmetric objects can fail raw matrix comparison while having small symmetry-aware geometry changes. Preserve that failure and investigate it before making a documented decision; never silently substitute symmetry-aware equivalence for raw numerical equality.

The comparator reports exact counts and numerical deltas for all poses/scores/margins. Its pose-element median is over all 16 matrix elements of all compared poses, including exact zeros and homogeneous rows. It does not infer accuracy equivalence. After full primary inference freezes, use the unchanged A9 evaluator and compare all aggregate and five-scene metrics. A changed metric is a tradeoff, not a free optimization.

The measured campaign and its disposition are recorded in `reports/gpu_perf_case/`. Large primary artifacts and profiler traces are external to Git, with an inventory and SHA-256 verification.
