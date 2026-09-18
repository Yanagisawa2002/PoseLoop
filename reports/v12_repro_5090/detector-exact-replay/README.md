# V1.1 detector replay: BIT-EXACT

Subsequent phase: the [original A9 v1.1 FoundationPose replay](../a9-v11-replay/README.md) completed with exact historical aggregate and per-scene metrics, reconstructed taxonomy, and verified off-server backups. The boundary described below records this earlier detector-only task.

One unchanged historical detector prediction run reproduced the frozen v1.1 prediction manifest exactly. Its bound frame metadata and packed masks were independently hash-validated. The original A9 input freeze and full asset validation passed. No FoundationPose registrations were started.

## Identity correction

The recovered checkpoint is the historical v1.1 detector, not a different v1.2 detector. This supersedes the interpretation in the original formal-training receipt and issue #11 training update. Those original receipts and archives remain unchanged for provenance; no checkpoint bytes were renamed or overwritten. No retraining, threshold tuning, repeated prediction attempts, successor protocol generation, or weakening of A9 identity checks occurred.

- Training and replay source: `3d5a33d64d9fbcdaecf97172fdea6f20ab647c41`.
- Checkpoint SHA-256: `a90d4134cb36cb242e98481440cfdc782c15b65f8d1068c2964243d7152fec2b`; historical exact match **YES**.
- Protocol SHA-256: `c200f80cd47ef338341ba9e1ba941b0cd81ba9fc28f8b1dca4c05dfb33bcc5f1`.
- Dataset manifest SHA-256: `b958091b602a3d9a62bc1b60ea28174d6f79ffcda5523c4e68bb33600f56810e`.
- Prediction manifest SHA-256: `dded85d8d9bec68f3c6d80343183635bd80e738a28e9a93a87531d93fbe7441b`.
- Historical prediction manifest SHA-256: `dded85d8d9bec68f3c6d80343183635bd80e738a28e9a93a87531d93fbe7441b`.
- Prediction manifest exact match: **YES**.

## Counts and metrics

- Ranked at score floor 0.01: **997**.
- Operating at score >= 0.25: **820**.
- GT access during inference / scene 9 reads: **0 / 0**, as recorded by the unchanged label-blind inference runtime. Source review confirms inference opens RGB and checkpoint inputs, not evaluation GT. No external filesystem tracing is claimed.
- TP / FP / FN at IoU50: **577 / 243 / 193**.
- Precision / recall / F1: **0.703658536585 / 0.749350649351 / 0.725786163522**.
- AP50 / AP75 / PQ: **0.702393231967 / 0.101145682956 / 0.510410241937**.
- Merge / split: **40 / 84**.
- Exact numeric match for all compared historical values: **True**; see `metrics-comparison.json`.

The original prediction CLI wrote and hashed the complete prediction manifest before the standalone evaluator opened labels. PR #12's standalone evaluator was used without A-R8 artifacts; its schema name does not create a successor detector identity. Results concern the already-consumed 25-frame development set, not a sealed evaluation or official BOP leaderboard.

## Original A9 handoff

The first freeze attempt stopped because OpenCV was absent; no freeze directory was created. After sourcing `/etc/network_turbo`, only `opencv-python-headless==4.10.0.84` was installed with `--no-deps`. The freeze then succeeded. The original error and installation logs are preserved; detector prediction and evaluation were not rerun.

- Original frozen A9 protocol: `protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json`.
- contract-check: **PASS**.
- freeze-inputs: **PASS**, 25 frames / 820 operating items.
- validate-inputs --verify-assets: **PASS**.
- Input manifest SHA-256: `e2d3bba5b743b214d2491ae6b6f3619d2849d861655a13b474fa170f4ed30a66`.
- Remote freeze: `/root/autodl-tmp/poseloop-v1.2-5090/a9-exact-replay-freeze/`.

The input manifest binds current absolute data paths; no historical input-manifest byte identity is claimed. Checkpoint and detector outputs are bit-exact. Next route: **ORIGINAL A9 V1.1 FOUNDATIONPOSE RUN**, pending separate execution; this task stops before that run.

## Persistence

The historical checkpoint has both a verified local original and verified entry in `artifacts/v12-formal-20260918/poseloop-v1.2-detector-formal-export.zip`. Replay predictions, packed masks, evaluation, A9 freeze, logs and receipts are independently backed up in `C:/Users/cgliu/Documents/Codex/poseloop-v12-5090/artifacts/detector-exact-replay-20260919/detector-exact-replay-evidence.tar.gz`.

- Replay archive SHA-256: `a0b60a846c21065712797d7a0bd5fcb33f0bf743599ae1ed5d776dfc82a239b2`.
- Archive size: 474453 bytes; every archived file was verified against SHA256SUMS.
- Post-replay free disk: 42304069632 bytes (39.398735 GiB); see `disk.txt` for df/du.
- Server originals preserved. PR #12 unmerged; issues #4 and #11 remain open. No server shutdown.

## Per-frame counts

| Frame | Ranked | Operating |
| --- | ---: | ---: |
| s000010-i000000 | 46 | 38 |
| s000010-i000010 | 42 | 37 |
| s000010-i000020 | 38 | 37 |
| s000010-i000030 | 41 | 35 |
| s000010-i000040 | 40 | 35 |
| s000025-i000000 | 57 | 50 |
| s000025-i000010 | 57 | 49 |
| s000025-i000020 | 54 | 44 |
| s000025-i000030 | 54 | 43 |
| s000025-i000040 | 55 | 48 |
| s000030-i000000 | 34 | 27 |
| s000030-i000010 | 35 | 30 |
| s000030-i000020 | 33 | 26 |
| s000030-i000030 | 35 | 28 |
| s000030-i000040 | 31 | 26 |
| s000040-i000000 | 53 | 41 |
| s000040-i000010 | 45 | 38 |
| s000040-i000020 | 49 | 41 |
| s000040-i000030 | 48 | 40 |
| s000040-i000040 | 45 | 38 |
| s000065-i000000 | 19 | 13 |
| s000065-i000010 | 23 | 14 |
| s000065-i000020 | 24 | 14 |
| s000065-i000030 | 18 | 14 |
| s000065-i000040 | 21 | 14 |

## Per-scene metrics

| Scene | Ranked | Operating | TP / FP / FN | Precision | Recall | F1 | AP50 | AP75 | PQ | Merge / split |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 207 | 182 | 153 / 29 / 32 | 0.840659 | 0.827027 | 0.833787 | 0.792724 | 0.242756 | 0.625056 | 11 / 11 |
| 25 | 277 | 234 | 192 / 42 / 103 | 0.820513 | 0.650847 | 0.725898 | 0.638815 | 0.108456 | 0.514365 | 28 / 6 |
| 30 | 168 | 137 | 99 / 38 / 6 | 0.722628 | 0.942857 | 0.818182 | 0.911072 | 0.238785 | 0.593491 | 0 / 20 |
| 40 | 240 | 198 | 87 / 111 / 48 | 0.439394 | 0.644444 | 0.522523 | 0.558356 | 0.000475 | 0.321276 | 0 / 34 |
| 65 | 105 | 69 | 46 / 23 / 4 | 0.666667 | 0.920000 | 0.773109 | 0.885893 | 0.138901 | 0.499561 | 1 / 13 |
