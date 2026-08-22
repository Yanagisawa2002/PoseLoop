# R3 remote GPU-A receipt

Remote task directory: `/root/autodl-tmp/poseloop_r3_20260816_01a003cf`

GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; preflight memory use/utilization `0/0`

Evidence archive SHA-256: `6dc49fecd942075531c12d3dd85b86be79361f10b9da7066850406777b810987`

## Environment and transfer outcome

- Official GitHub BOP Toolkit commit tarball: 9,437,786 bytes in 45.226 s
  (about 0.20 MiB/s), SHA-256
  `d7cb1397e21cf4622a11bde64fc0cdc8c0e855d8d6a91b2815ad8df7a8ad5526`.
- The required `ghproxy.link` fallback returned only a 1,739-byte HTML response,
  so it was rejected as an archive.
- HF mirror: `xyzibd_models.zip` downloaded at 922,988 B/s and both downloaded
  XYZ-IBD archives matched the frozen SHA-256 values. The 7,710,607,370-byte
  validation archive was not downloaded because it exceeded the bounded setup
  window.
- Aliyun PyPI downloads ranged from 3.4 MB/s to 51.0 MB/s in the retained final
  install logs.
  BOP Toolkit, NumPy/SciPy/OpenCV, and pycocotools imports succeeded; official
  BOP24 and BOP22 evaluator `--help` probes exited 0.

## Final remote result

The updated R3 contract suite passed 12/12. The final label-blind dry-run built
all six official evaluator commands and accepted the pinned 93-file source tree
from the GitHub commit archive. It accessed zero evaluator-only label paths and
did not run official scoring.

The final remote dry-run is correctly `blocked` by:

1. missing remote `test_targets_bop19.json`, `test_targets_bop24.json`, and the
   full `val` split;
2. missing R3 prediction roles: COCO detections/segmentations, predicted
   association, single-view poses, multi-view poses, and provenance;
3. unavailable baseline-ancestry verification in the code-only remote copy.

No AR/AP result is claimed. The imported evaluator/CLI probe is structural
runtime evidence, not a completed official metric run.

## Primary evidence

- `evidence/r3_remote_dry_run_final_v2.json`
- `logs/r3_remote_tests_final_v2.log`
- `evidence/evaluator_smoke_receipt.txt`
- `evidence/hf_mirror_xyzibd_speed.txt`
- `logs/pip_bop_toolkit_pinned.log`
- `evidence/network_diagnosis.txt`
