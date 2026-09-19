# Refiner 32 → 64: TRADEOFF, not an accepted optimization

The single-variable refiner64 experiment completed **820/820 with zero runtime failures**, but the unchanged A9 evaluator produced **479 joint successes instead of 482**. Four of five scene results and all requested aggregate pose metrics changed. **Do not promote refine=64 or describe this as equivalence-preserving.** The original v1.1 protocol, release artifacts and default refine=32 CLI remain unchanged.

Implementation ran at `af0cc13c89e6038c91385ca9a25b54ffa1ab9198`, on a fresh performance branch based on main `12c3e49`. The new entry point was added in `aae08325e79e0f5aa2ba7763ea4eeee62726cd09`. PR #12 was neither extended nor merged for this task. See [entry-point documentation](../../docs/A9_REFINER64_PERFORMANCE.md).

## One inference variable and safety evidence

Only refiner feature chunk size changed from 32 to 64. The 820 ordered inputs / 25 frames, detector masks, manifest, FoundationPose commit and weights, seed=0, iterations=5, candidates=252, warp=32, score_data=8, score_feature=32, precision, synchronization and evaluator thresholds were held fixed. `invariant-validation.json` verifies every shared run-lock field and the unchanged original artifact hashes. The run lock explicitly distinguishes the historical `frozen_inference` from `effective_resource_batches` and marks this as a performance experiment.

Environment: RTX 5090 (32607 MiB), driver 595.71.05, torch 2.8.0+cu128, CUDA runtime 12.8, pinned FoundationPose `a1b694b83e633c2cb6115b9063d940a687759392`, tracked source unchanged.

Before B1, five real items (first item from each scene) completed without OOM/CUDA errors, with 252 candidates, finite 4x4 poses and scores, and peak allocated memory 6,437,664,768 bytes. The predeclared raw-element safety screen **failed**: two rotations changed substantially (maximum element difference 1.36435). This failure was preserved, not relabeled as exact equivalence.

Execution paused for a bounded, label-free geometry diagnosis. Both large rotations involved declared symmetric objects with zero score margins: object 2 has 24 symmetry transforms; object 1 has 315. Original BOP symmetry-aware comparison between B and A predictions, not GT poses, gave maximum pairwise MSSD **0.612017 mm** and MSPD **0.704194 px** over the five items. A documented decision then advanced to full evaluation because the large raw rotations had a symmetry explanation and did not represent similarly large geometric displacement in these samples. No safety thresholds or inference settings were changed. `safety-decision.json` records this adjudication; it is not an accuracy-equivalence acceptance. Full-population results subsequently failed that separate acceptance gate.

## Clean timing observations

| Run | Refiner chunk | Wall seconds | Status |
| --- | ---: | ---: | --- |
| A1 | 32 | 551.107429348 | Existing authoritative clean baseline |
| B1 | 64 | 546.915343799 | Clean, unprofiled; evaluation changed |
| B2 | 64 | unavailable | Conditional repeat not entered |
| A2 | 32 | unavailable | Conditional control not entered |

Observed A1−B1 saving: **4.192085549 s**, **0.760665766%**, ratio **1.007664962×**. These are a **single-pair observation**, not a reproducible speedup. Each group's descriptive mean/median equals its lone observation; repeated-run central tendency, noise bounds and environmental-variance control are unavailable. B2/A2 were conditional on unchanged B1 evaluation in the requested plan; that condition failed. No fastest-run selection was performed.

| Registration latency | A1 seconds | B1 seconds | B1−A1 seconds |
| --- | ---: | ---: | ---: |
| Median | 0.655113700 | 0.646964629 | −0.008149071 |
| p95 | 0.681723405 | 0.700396545 | **+0.018673140** |
| Mean | 0.656999169 | 0.651959394 | −0.005039775 |
| Maximum | 1.872796806 | 1.295782618 | −0.577014187 |

Peak allocated VRAM: **7.789286613 → 7.618559361 GiB**, signed delta **−0.170727253 GiB (−2.191821426%)**. This measured peak decreased despite the larger chunk; its cause was not profiled. Allocated peak is not total device usage or reserved memory. The five-item safety peak is not substituted for the full-run peak.

## All-item semantic comparison

| Check | B1 vs A1 |
| --- | ---: |
| Ordered unique items / successful status / candidate count | 820 / 820 / 252 throughout |
| Exact pose matches | **5 / 820** |
| Non-exact poses | **815 / 820** |
| Maximum pose-element absolute difference | 1.998599886894226 |
| Median pose-element absolute difference | 0.0000649280846118927 |
| Maximum translation-element difference | **0.12898153811693192 m** |
| Exact top scores | **40 / 820** |
| Maximum top-score absolute difference | **4.626953125** |
| Exact score margins | **396 / 820** |
| Maximum margin absolute difference | **0.3046875** |

The pose-element median includes all 16 matrix entries, including exact homogeneous rows. Raw rotation differences can include symmetry-related orientation choices, but the full-population translation/score outliers and changed evaluation must not be dismissed as harmless rounding. Exact semantics did not hold. All poses/scores were finite. `per-item-deltas.csv` contains every item; complete original and candidate predictions are preserved externally.

## Unchanged evaluator: DIFFERENT results

Primary predictions/receipt/run-lock were hashed before labels opened, and reverified afterward. All five primary access counters remained zero. No evaluator or threshold was changed.

| Metric | A1 | B1 |
| --- | ---: | ---: |
| Runtime success | 820 | 820 |
| Mask IoU50 matches | 577 | 577 |
| Joint successes | **482** | **479** |
| Joint precision | 0.5878048780487805 | 0.5841463414634146 |
| Joint recall | 0.625974025974026 | 0.6220779220779221 |
| Joint F1 | 0.6062893081761006 | 0.6025157232704402 |
| Joint AP | 0.5318658235280908 | 0.5292226099228429 |
| AR MSSD | 0.6305194805194805 | 0.6258441558441559 |
| AR MSPD | 0.6457142857142858 | 0.6411688311688312 |
| Combined AR | 0.6381168831168831 | 0.6335064935064936 |

| Scene | GT | A1 joint successes | B1 joint successes |
| --- | ---: | ---: | ---: |
| 10 | 185 | 148 | **146** |
| 25 | 295 | 129 | **128** |
| 30 | 105 | 83 | **82** |
| 40 | 135 | 83 | **84** |
| 65 | 50 | 39 | 39 |

Threshold-recall arrays and all per-scene fields are compared in `evaluation-equivalence.json`; per-frame outputs are also different. These are custom already-consumed development metrics, not an official BOP leaderboard claim.

## Profiling and disposition

No optimized Nsight trace was collected: unchanged evaluation and reproducible improvement were prerequisites, and neither was established. Therefore refine-kernel, convolution/GEMM/attention, dominant-convolution, launch-count and GPU-activity deltas are **unavailable**. The source changes 40 refiner chunk-forward calls into 20 for five iterations over 252 candidates; that structural count does not prove reduced measured GPU overhead. See [profiler-comparison.md](profiler-comparison.md).

**Decision: reject promotion of refine=64 as a free optimization. Stop after this one mechanism.** Keep issue #6 open. No candidate #2, NVTX insertion, broad Nsight Compute pass or PR #12 merge was performed.

## Validation and artifacts

- Existing A9 and new performance-contract tests: **18 passed**.
- Five-scene real safety path and complete 820-item primary path ran on the target GPU.
- Independent stdlib recomputation agrees with the semantic summary; original baseline hashes and frozen invariants pass.
- External evidence archive: `/root/autodl-tmp/poseloop-a9-perf/refine64-ab-evidence.tar.gz`.
- Archive SHA-256: `776785fd1586e467ed2e1ee8d937337be36a601ff4d3279e980eb3e43fcc3afe` (7,453,376 bytes; **57 members verified** off-server).
- Local archive: `C:/Users/cgliu/Documents/Codex/poseloop-perf-batch64/artifacts/gpu-perf-case/refine64-ab-evidence.tar.gz`.
- Predictions, source/run locks, full logs, evaluation CSV/visualizations, safety diagnosis and execution scripts are preserved. Paths and hashes are in `artifact-inventory.json`.
- Required compact summaries: `baseline-summary.json`, `refine64-summary.json`, `ab-comparison.json`, `semantic-equivalence.json`, [interview-case.md](interview-case.md).
