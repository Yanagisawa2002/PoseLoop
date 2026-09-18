# Original A9 v1.1 FoundationPose replay

**A9 V1.1 SEMANTIC REPLAY: EXACT AT RECORDED EVALUATION GRANULARITY**

The original frozen A9 protocol completed 820/820 primary registrations and reproduced every specified historical aggregate and per-scene metric exactly. The recovered detector checkpoint and prediction manifest remain bit-exact. This does not establish per-instance pose/joint-assignment identity or FoundationPose artifact byte identity: comparable historical raw outputs are unavailable, and current provenance, timing, environment and run-lock fields change the artifact bytes.

## Frozen environment and execution

- Actual deployed PoseLoop commit: `3d5a33d64d9fbcdaecf97172fdea6f20ab647c41`.
- A9 protocol SHA-256: `589bae96278f7d43c81d04cb422fdda7568a9b4f2adc06e10a2df04b54cf6c0e`.
- FoundationPose: `a1b694b83e633c2cb6115b9063d940a687759392`.
- nvdiffrast: `253ac4fcea7de5f396371124af597e6cc957bfae`.
- PyTorch3D: `3143b3baf8ef8b1023ed76f225af59e2e8a71e06`.
- BOP Toolkit: `cea62d651c7e395b2e1962b9749e4e89693c6ac4`.
- Python 3.12.3; torch 2.8.0+cu128; torchvision 0.23.0+cu128; CUDA 12.8; RTX 5090, capability (12, 0).
- Pinned source builds/imports, model construction, 252-candidate grid, CUDA rasterization and PyTorch3D CUDA KNN: PASS. Synthetic preflight opened no dataset/GT frames.
- FoundationPose and pinned GPU dependency tracked source: **unmodified**.
- Iterations 5, seed 0, candidate count 252, resource batches, score and pose thresholds unchanged. No training or detector replay, A10 switch, Nsight profiling, or scene-9 access occurred in this phase.

The repository provisioning script created a new venv over the existing working torch base. Only environment fixes were needed: system `libegl1`/`libgl1`, `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` for the Open3D C++ ABI, PyPNG for BOP, and the pinned BOP checkout in PYTHONPATH for taxonomy. Initial dependency failures were preserved in the supplemental archive. No primary items were rerun, and no evaluation/taxonomy algorithm was edited.

## Primary evidence

- Coverage / runtime success: **820/820**, failures **0**.
- Wall time: **551.107429348 s**. Historical 575.473 s is a reference, not a cross-GPU reproduction target.
- Registration seconds: min 0.625978, median 0.655114, p95 0.681723, p99 0.709339, max 1.872797, mean 0.656999.
- Per-registration peak allocated VRAM GiB: min 4.227972, median 7.145885, p95 7.786967, p99 7.789287, max 7.789287.
- Primary access counters: label / GT-path / evaluator-path / official-scorer / scene9 = **0 / 0 / 0 / 0 / 0**, as recorded by the original runtime.
- Predictions SHA-256: `300deaca726983a5e15513e6ed49814527a752581c11bb957d0bde8135927b77`.
- Completion receipt file SHA-256: `5a0344b9098b0ed54222ead6f884c92451f62762a769569ce85fa85a49959092`.
- Run-lock file SHA-256: `c40a67fab64971165181a3c4803c696cf156e5875a9609daba74c9e70c239783`.
- Canonical run-lock identity SHA-256: `26a24359c08f90ba03156967d45c0b1d1c56295df5b7974958c64526437688e4`.

The completion receipt and primary hashes were frozen before evaluation opened labels. The full predictions file differs from historical `a029289f61afe1222b4b45562784076287c36ed952b85820e78fba1187ed39c06`; that is not evidence of a semantic mismatch.

## Evaluation comparison

All numerical deltas below are exactly zero against the supplied frozen reference. Results concern already-consumed development data, not a sealed evaluation or official BOP leaderboard result.

| Metric | Replay | Delta |
| --- | ---: | ---: |
| runtime_success_count | 820 | 0 |
| mask_iou50_match_count | 577 | 0 |
| joint_pose_success_count | 482 | 0 |
| joint_pose_precision | 0.5878048780487805 | 0.0 |
| joint_pose_recall | 0.625974025974026 | 0.0 |
| joint_pose_f1 | 0.6062893081761006 | 0.0 |
| joint_pose_ap | 0.5318658235280908 | 0.0 |
| ar_mssd | 0.6305194805194805 | 0.0 |
| ar_mspd | 0.6457142857142858 | 0.0 |
| combined_ar_mssd_mspd | 0.6381168831168831 | 0.0 |

| Scene | Joint successes / GT | Joint recall | Recall delta |
| --- | --- | ---: | ---: |
| 10 | 148 / 185 | 0.800000000000 | 0.0 |
| 25 | 129 / 295 | 0.437288135593 | 0.0 |
| 30 | 83 / 105 | 0.790476190476 | 0.0 |
| 40 | 83 / 135 | 0.614814814815 | 0.0 |
| 65 | 39 / 50 | 0.780000000000 | 0.0 |

## Reconstructed replay taxonomy

This is a **reconstructed replay taxonomy from the exact detector and original frozen A9 protocol**. Historical taxonomy byte identity is not claimed. Independent CSV validation confirms 770 unique rows, 577 IoU50 matches, 482 successes, and 577 matched-pose-error rows.

| Bucket | Count |
| --- | ---: |
| DETECTOR_MISS | 1 |
| MASK_BOUNDARY_IOU_FAILURE | 114 |
| OVER_SEGMENTATION | 24 |
| POSE_MSPD_ONLY_FAILURE | 2 |
| POSE_MSSD_AND_MSPD_FAILURE | 73 |
| POSE_MSSD_ONLY_FAILURE | 20 |
| SUCCESS | 482 |
| UNDER_SEGMENTATION_OR_MERGE | 54 |

There are 288 unsuccessful GT instances: 193 unmatched at IoU50 and 95 matched masks failing the joint pose thresholds. The largest individual bucket is MASK_BOUNDARY_IOU_FAILURE (114/288, 39.6%). Scene 25 contains 166/288 failures (57.6%). These are descriptive diagnostics, not authorization to retune this consumed split.

Taxonomy files are preserved in `failure-taxonomy/per-instance.csv`, `failure-taxonomy/summary.json`, and `failure-taxonomy/REPORT.md`; exact hashes appear in `artifact-inventory.json` and `SHA256SUMS`.

## Evidence and backup

- Server archive: `/root/autodl-tmp/poseloop-v1.1-replay/a9-v1.1-replay-evidence.tar.gz`.
- Evidence archive SHA-256: `06f04ae887786f798d5303981e8881ede648f9f1a9f36b8e2ab824fd49743bfe` (6678963 bytes).
- Local evidence archive: `C:/Users/cgliu/Documents/Codex/poseloop-v12-5090/artifacts/a9-v11-replay-20260919/a9-v1.1-replay-evidence.tar.gz`.
- Local taxonomy/log/receipt supplement: `C:/Users/cgliu/Documents/Codex/poseloop-v12-5090/artifacts/a9-v11-replay-20260919/a9-v1.1-replay-supplement.tar.gz`.
- Supplement SHA-256: `bc45a759c5d6216e1caa58558356b5014b80ba2091ba82389ade40fe745c8fb4` (842520 bytes).
- Original package member hashes, all supplement member hashes, and all 12 critical artifact inventory entries verified locally.
- Exact detector checkpoint and complete detector prediction artifacts remain verified off-server copies from the preceding phases.
- Primary predictions, completion receipt, run-lock, A9 freeze, matched pose errors, evaluation, taxonomy, archives and SHA256SUMS are all backed up. **No required large files remain to be downloaded.**
- Post-package free disk: 39381594112 bytes (36.676968 GiB); supplement packaging uses an additional 842520 bytes.

**READY FOR NSIGHT BASELINE: YES.** No Nsight run was started. PR #12 remains unmerged and issue #11 stays open for the next profiling/evidence phase. The server remains running.
