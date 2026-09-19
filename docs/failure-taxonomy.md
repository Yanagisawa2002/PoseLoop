# PoseLoop v1.1.0 failure taxonomy - recovery status

This page describes the original frozen aggregate bundle and its limitations.
The subsequent [5090 reconstruction result](FAILURE_TAXONOMY_RESULT.md) completes
the 770-instance taxonomy; use that page for the current result.

## What is proven now

| Stage | GT-level count | Share of all 288 GT failures |
| --- | ---: | ---: |
| Does not reach the IoU50 pose handoff | 193 | 67.0% |
| Reaches IoU50 handoff but misses the joint pose gate | 95 | 33.0% |
| Joint pose success | 482 | - |

The frozen aggregate chain is **770 GT -> 577 mask-IoU50 matches ->
482 joint pose successes**. The first 193 cases cannot be
honestly split into detector miss, merge, split, match competition, or boundary
failure from the tracked aggregate bundle alone. Likewise, the 95
matched pose failures cannot be split into MSSD-only, MSPD-only, both-metric,
or runtime/input failures without the original per-instance evaluation evidence.

## Where the failures concentrate

| Scene | GT | Joint successes | GT-level failures | Share of all failures |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 185 | 148 | 37 | 12.8% |
| 25 | 295 | 129 | 166 | 57.6% |
| 30 | 105 | 83 | 22 | 7.6% |
| 40 | 135 | 83 | 52 | 18.1% |
| 65 | 50 | 39 | 11 | 3.8% |

Scene 25 contributes the majority of the frozen GT-level failures, but this is
a localization fact, not a causal label. It must not be used to retune the
already-consumed split.

## Evidence needed for the full 770-row taxonomy

The evaluator code shows that the original run produced enough information to
finish the taxonomy without rerunning detector training or FoundationPose
inference, provided the frozen runtime artifacts still exist:

- frozen A9 input manifest, including detector mask bindings;
- primary predictions.jsonl and matching completion-receipt.json;
- XYZ-IBD development data referenced by the manifest;
- pinned BOP Toolkit checkout for the same symmetry-aware MSSD/MSPD calculation.

The original evaluation also wrote matched-pose-errors.csv with frame_id,
gt_index, mask_iou, normalized_mssd, and mspd_px. That CSV is sufficient to
split the matched pose failures, while detector-mask bindings are needed for
the unmatched geometric taxonomy.

Those per-instance artifacts were intentionally kept under the ignored
artifacts tree and were not included in the public v1.1.0 Release. Therefore
the current tracked repository is **BLOCKED_PER_INSTANCE_EVIDENCE**, not a
completed fine-grained taxonomy.

## Recovery command

If the frozen manifest and primary root are recovered, run:

    python -B scripts/build_failure_taxonomy.py extract \
      --protocol protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json \
      --manifest /path/to/input-manifest.json \
      --primary-root /path/to/foundationpose-primary \
      --dataset-root /path/to/xyzibd \
      --toolkit-root /path/to/bop_toolkit \
      --output-root /path/to/failure-taxonomy

The extractor is create-only and verifies the frozen manifest, primary
completion receipt, primary prediction hash, exact aggregate reconciliation,
and v1.1.0 stage counts before writing any final summary.

## Interpretation

The next measured question remains: **what fraction of the 193 upstream losses
are true detector misses versus instance-formation errors, and what fraction of
the 95 matched pose losses fail MSSD, MSPD, or both?**

Until the original per-instance evidence is recovered, any more specific answer
would be invented.
