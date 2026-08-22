# GPU-A XYZ-IBD supported-metric official invocation v3

Protocol: `poseloop.r3.bop-industrial.xyzibd-supported.official.v3`

Outcome: **the public XYZ-IBD validation labels passed the v3 readiness gate,
the only authorized v3 batch attempted all four frozen commands once, the two
BOP24 pose commands succeeded, and the two BOP22 COCO commands failed inside
the pinned evaluator. The batch is sealed and cannot be rerun.**

| Metric | Single view | Multi view | Delta | Status |
| --- | ---: | ---: | ---: | --- |
| Official-toolkit BOP24 pose mAP on public XYZ-IBD val | 0.0000 | 0.0000 | 0.0000 | Available; exits 0/0 |
| BOP22 bbox AP | N/A | N/A | N/A | Unavailable; exit 1 |
| BOP22 segmentation AP | N/A | N/A | N/A | Unavailable; exit 1 |
| BOP19 AR | N/A | N/A | N/A | Unavailable because XYZ-IBD VSD is undefined; no command constructed |

These values are a fixed validation-set comparison on identical public targets,
not a BOP challenge test-server or leaderboard result.

## Label-asset readiness

The official XYZ-IBD distribution was pinned to Hugging Face revision
`4fe4671783172622313ac0c7182012cee618f217`. The required public validation
archive was `xyzibd_val.zip` at:

`https://huggingface.co/datasets/bop-benchmark/xyzibd/resolve/4fe4671783172622313ac0c7182012cee618f217/xyzibd_val.zip`

The published LFS object/content SHA-256 and the cached archive SHA-256 both
equal
`09c5639c6e55b8c9a0708a344037e98918c5935ddbd91b5db5a9b06f5484c659`;
the byte count is `7,710,607,370`.

The archive audit found 15 validation scenes
`[0, 5, 10, ..., 70]`, 15 `scene_gt_xyz.json` files, 15
`scene_gt_info_xyz.json` files, 15 `scene_camera_xyz.json` files, 750 images,
and 19,950 full/visible mask pairs. All JSON files parsed. The original archive
contains no `scene_gt_coco_xyz.json`; the official pinned toolkit documents
`scripts/calc_gt_coco.py` as the derivation path. Exactly 15 COCO GT files were
therefore generated from the public GT and masks only inside the isolated
evaluator namespace, yielding 750 images and 19,950 annotations. The source
dataset was not modified.

The official base assets retained their original test target files:

- `test_targets_bop19.json`: 60 rows, SHA-256
  `6ebafd20964d09f649a2e1acc98e5b8b1c39d89e746a0b8865c887f6d74fa011`.
- `test_targets_bop24.json`: 60 rows, SHA-256
  `7f620e5bdf8d6f11201d90464c5a93963ea37718ff662e5a392ca41eb10d6ebe`.

Those files address held-out test scenes and were not relabeled as validation
targets. The v3 evaluator-only validation targets instead cover all 750 public
validation images:

- `val_targets_bop19.json`: 750 rows, SHA-256
  `6256aac91ddee43294cbe452f284c4bd7b17039fc2199b7744e13767edfdf6f3`.
- `val_targets_bop24.json`: 750 rows, SHA-256
  `0c0780b0aaabb55f20a8f5419fa01fdd6014ff1e482f6cbc6dc8f859076ae3a1`.

Target coverage had zero missing keys for both target schemas, and the pinned
toolkit loader smoke passed. Prediction/selection label access remained 0.
No scorer ran and no score file existed during readiness.

Readiness was frozen before the scoring protocol:

- Readiness implementation commit:
  `6bd973a908cc8717487270455cc848f93004812f`.
- Readiness protocol SHA-256:
  `5f69bd278ed1d5d594efec6348fa7babc581485bd8cde526cd5e6adbdf0798f5`.
- Readiness lock SHA-256:
  `d7d33bb2c780385a7ebedea3985348084e1d21d9540f8f338c94d411412fbf12`.
- Readiness fingerprint:
  `a64d38bc5f46025f4e10d9cfcbaeef73b0c1e13981139e302b25e4b97bb47e0a`.
- Preparation receipt SHA-256:
  `d3c669058c9c269449d5f60e368ac06798520f6e9774682fe60b0ac2c2a68b62`.

## Frozen scoring boundary

- Branch: `codex/poseloop/r3-xyzibd-supported-official-run-v3`.
- Implementation commit:
  `3d535d1d9e5761b86e7416a306d460cb41d9f419`.
- Protocol SHA-256:
  `f3d0182fb0fe788d38263ebaf63e4eaddf89534202c06b10553b4e3ef35e4e3a`.
- Input-lock SHA-256:
  `c1a8f35882b4fd7895c9593f34f75a56c58a9a1115a3b1e1925f026fa999e5ce`.
- Input-lock fingerprint:
  `cad836a730d5f7d34c8c34a5e0d904c16e16b07ffc9d2d7455d9387f1ddeebe7`.
- Evaluator-only authorization SHA-256:
  `bc4ee8c870405a181a7142036c3eac718022f7c5f4daccd78a882b0bed414d4d`.
- Authorization reference:
  `codex_delegation:01a003cf-9081-7cc1-b89d-9842143a1ec3:r3-v3-label-assets-complete`.

The final no-evaluator pre-score gate verified the task venv, pinned toolkit
imports, renderer smoke, label readiness lock, clean repository, absent output,
zero scorer processes, zero prediction/selection label access, and zero BOP19
commands. The five frozen inputs remained bit-identical to v2:

- COCO detections/segmentations:
  `5ef98c1ea24a9ecc141d3f1af6a319dba1ce00f4cb21d522b99b4f1d3ab1a719`.
- Predicted association:
  `6883a6ff9ea8fffaa835eb92e2f717a1fb6ff4eb6e5194924bd76becbb26d7b2`.
- Single-view pose:
  `459ca38d8f7d6d9d7ebafad6c7a65a3147b8776e79e03517f3e93c4d412eb017`.
- Multi-view pose:
  `145f19e1d545472a49e16a0954a291d734e3109a59e71f2ba533fcf576244066`.
- Provenance:
  `ecc8dd05ae38bbd739a1330701746ef1ba7b5c2f4898282bd010efdf7f4759b1`.

No model, candidate, association, or prediction was changed. All prior v1/v2
protocols, failures, logs, and sealed receipts remained read-only.

## Unique four-command batch

Screen: `poseloop_r3_v3_official_once` (finished).

The invocation marker SHA-256 is
`65dfba800113162f0e76f9b61b65c2e9bfd70bd9c040bb7b5af1fcce7dfba51fd`.
The wrapper attempted every frozen command once and continued after failures:

1. BOP24 pose mAP, single view: exit 0, `bop24_mAP=0.0`.
2. BOP24 pose mAP, multi view: exit 0, `bop24_mAP=0.0`.
3. BOP22 bbox AP on the shared predicted input: exit 1, no score file.
4. BOP22 segmentation AP on the shared predicted input: exit 1, no score file.

The wrapper exit code is 3, representing a sealed partial result. Both BOP22
commands reached pinned `eval_bop22_coco.py:166`, which passes the already
loaded COCO annotation dictionary to `COCO(dataset_coco_ann)`. Installed
`pycocotools==2.0.10` then treated that dictionary as `annotation_file` and
attempted `open(annotation_file, "r")`, ending with:

`TypeError: expected str, bytes or os.PathLike object, not dict`

This is an evaluator/dependency API incompatibility, not missing labels. Since
the only authorized batch had begun, the environment was not modified and the
two failed commands were not retried. BOP22 bbox AP and segmentation AP remain
unavailable; no numeric AP is claimed.

The raw single and multi BOP24 score files have SHA-256 values
`127508886987dc7fd79f7ef707bede39fde8b8868de48676419753b4be3a5429`
and
`24ffdca2654e0209e92655b686a461cccfcf81c0de72ff63333981e20165661f`.
The official score-bundle receipt SHA-256 is
`684368bb008b0c7700e11de952fa32062be0338b9569be1bfd28d7b2b73d8acb`.
The sealed receipt SHA-256 is
`99ccc1974136a2520b766557e38daa9b13e193642f4730079e27631b26824a2e`;
it records exit codes `[0, 0, 1, 1]`, per-command invocation counts of one,
`evaluate_invocation_count=1`, `continued_after_failures=true`, and
`rerun_permitted=false`.

## Post-run verification and evidence

The post-run audit SHA-256 is
`2a7a86240045686ed6c6fdd9407e4dc3c45ef4c1cc85e0da55bcb6d2ccafc429`.
It verified all 188 sealed output-manifest entries with no hash errors, found no
running official scorer, and reconfirmed `rerun_permitted=false`. Remote tests
passed: v3 scoring 6/6, v3 readiness 6/6, and v2 regression 8/8. The post-run
GPU snapshot was `NVIDIA GeForce RTX 5090, 0 MiB, 0 %`.

Local evidence:

- Complete remote evidence archive:
  `artifacts/r3_bop_industrial_supported_v3/remote_gpu_a/r3_xyzibd_supported_v3_once_3d535d1.tar.gz`
- Archive bytes / SHA-256: `14,785,607` /
  `4f1ba7f2bba690297fa06792790f3b4d2c06f330e9a612033e2592ccf1f66e9e`.
- Curated readiness evidence:
  `artifacts/r3_bop_industrial_supported_v3/remote_gpu_a/extracted/evidence/r3_v3_label_readiness/`
- Curated pre-score locks and receipts:
  `artifacts/r3_bop_industrial_supported_v3/remote_gpu_a/extracted/workspace/artifacts/r3_bop_industrial_supported_v3/pre_score/`
- Invocation marker, official score receipt, sealed receipt, raw command logs,
  and raw BOP24 scores:
  `artifacts/r3_bop_industrial_supported_v3/remote_gpu_a/extracted/workspace/artifacts/r3_bop_industrial_supported_v3/official_once/`
- Post-run hash/process/Git/GPU/test evidence:
  `artifacts/r3_bop_industrial_supported_v3/remote_gpu_a/extracted/evidence/r3_v3_official_once/`

Remote canonical archive:

`/root/autodl-tmp/poseloop_r3_20260816_01a003cf/evidence/r3_v3_handoff/r3_xyzibd_supported_v3_once_3d535d1.tar.gz`

GPU-A was kept running. No shutdown, poweroff, release, disk deletion, push,
merge, or tag was performed.
