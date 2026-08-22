# GPU-A supported-metric official invocation v2

Protocol: `poseloop.r3.bop-industrial.xyzibd-supported.official.v2`

Outcome: **the only authorized v2 evaluate invocation attempted all four frozen
commands exactly once, all four exited 1, no supported metric score was
produced, and rerun is forbidden**.

## Frozen boundary before scoring

- Branch: `codex/poseloop/r3-xyzibd-supported-official-run-v2`.
- Implementation commit: `1278f690d9714e213ff5dd7ab23a0ae67c16c67e`.
- Protocol SHA-256:
  `b23860399ed43e268921070e7ccee15e82375c5514850b96b1507417c18740dc`.
- Final pre-score fingerprint:
  `d2e6f761f8f8b7d99179839b23e6be0c37a70ce4c9f9fcf9f99269c5ba5abdf4`.
- Official four-command manifest:
  `1585570ad6e77516951c4edafdc40ab99afebb270eb0ef9e28f3ee73f42534f5`.
- Environment smoke receipt:
  `75fc75961060a680642cf3cdc4115db1237be4a0d5ece21ecf047792089788fa`.
- Input lock:
  `c17885a10b410a742c0ad0ac44a5c9010b955b4046bda0e4008cbb3606d9f7e1`.
- Evaluator-only authorization:
  `842d3cb0f283afe5af7b255b40fbc3d9411a61588320b7113b5af862d5bdb710`.
- Final pre-score receipt:
  `4c6c270865d36e60a711d146e103170a9d6d3a3b553819a2ab79f446b52f901d`.
- Before invocation: repository clean; official output absent; official score
  file count 0; official evaluator process count 0; prediction/selection label
  access count 0; BOP19 entrypoint command count 0.

The exact task venv was used:
`/root/autodl-tmp/poseloop_r3_20260816_01a003cf/workspace/.venv/bin/python`.
The final no-evaluator smoke verified an active venv, all required imports, one
editable `bop_toolkit_lib` distribution bound to the pinned toolkit root, and a
VisPy/EGL public-model depth render with shape `48x64`, finite depth, and 500
positive pixels. The earlier four blocked smoke attempts are retained in the
evidence bundle; none called an evaluator, opened evaluator labels, or created a
score.

The five frozen input SHA-256 values remained unchanged:

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

The original failed protocol and supported-v1 failure remained immutable and
non-rerunnable. Their v1 protocol, invocation marker, sealed receipt, and raw
log hashes were respectively
`99afcaa801de2aa088776e3a2efdb3189802f92139fe8cba44ca2faaa3d160bb`,
`254228e4a1ca105846750aa414bf0760630dced428944e343656cbdc2735c087`,
`ca4b040d3f7639d9dc52e4b0327d838b2ccad7ee9cafc2d32283526adaa92cb9`,
and `abeb31400727099b6d37e674ffdd9328e22d6c420ee1d3df18560f0040019f1e`.

## Unique invocation and exact blockers

Screen: `poseloop_r3_v2_official_once`.

The invocation marker SHA-256 is
`2ac08339d5951915dc2affdf24ff4f772d969be8911e1826e04603a59ff785c6`.
The wrapper continued after failures as frozen and recorded these one-time
attempts:

1. BOP24 single-view pose mAP: exit 1. The pinned evaluator reached scene 1
   after loading estimates, then failed because
   `xyzibd/val/000001/scene_gt_xyz.json` was absent.
2. BOP24 multi-view pose mAP: exit 1 at the same required
   `scene_gt_xyz.json` path.
3. BOP22 bbox AP: exit 1 because
   `xyzibd/val/000001/scene_gt_coco_xyz.json` was absent.
4. BOP22 segmentation AP: exit 1 at the same required
   `scene_gt_coco_xyz.json` path.

Thus the precise blocker is that the locally frozen XYZ-IBD val tree does not
contain the GT filenames required by the pinned toolkit's XYZ-IBD path
templates. This was discovered only after the authorized official evaluator
opened its label boundary. It is not repaired here and does not authorize a
rerun.

The sealed receipt records `evaluate_invocation_count=1`,
`attempted_command_count=4`, exit codes `[1, 1, 1, 1]`, a per-command invocation
count of one, `continued_after_failures=true`, and `rerun_permitted=false`.
Its SHA-256 is
`0ce1482fa4febdd5134d266b9be4b098f12439435a669e7924020d85bbac7e64`.
The raw official score-bundle receipt SHA-256 is
`53a48bf7066fd3737025bc9e4744dea26781252926db27c8e9210ba4cf49a718`.

No expected official `scores*.json` was produced. Consequently:

- official BOP24 single-view mAP: unavailable;
- official BOP24 multi-view mAP: unavailable;
- multi-minus-single delta: unavailable;
- official BOP22 bbox AP: unavailable;
- official BOP22 segmentation AP: unavailable;
- official BOP19 AR: unavailable by the already-frozen VSD support boundary,
  and no BOP19 command was constructed or run.

No numeric AR/AP/mAP is claimed. Prediction/selection label access stayed 0,
and no model, candidate, association, or prediction was changed after scoring.
Post-run v2 tests (7) and v1 regression tests (7) passed. GPU-A remained running
(RTX 5090, 0 MiB used at the post-run snapshot). No shutdown, poweroff,
release, delete, push, merge, or tag was performed.

## Evidence

- Imported archive:
  `artifacts/r3_bop_industrial_supported_v2/remote_gpu_a/r3_xyzibd_supported_v2_once_1278f69.tar.gz`
- Archive SHA-256:
  `f50e19c01c239c2fd5c9095a55f901a5dbfad3623f5663d84e25ebc90fa8b79a`
- Extracted pre-score evidence:
  `artifacts/r3_bop_industrial_supported_v2/remote_gpu_a/extracted/workspace/artifacts/r3_bop_industrial_supported_v2/pre_score/`
- Extracted invocation marker, four raw logs, score-bundle receipt, and sealed
  receipt:
  `artifacts/r3_bop_industrial_supported_v2/remote_gpu_a/extracted/workspace/artifacts/r3_bop_industrial_supported_v2/official_once/`
- Artifact and dependency hash manifests:
  `artifacts/r3_bop_industrial_supported_v2/remote_gpu_a/extracted/evidence/`
- Remote canonical archive:
  `/root/autodl-tmp/poseloop_r3_20260816_01a003cf/evidence/r3_xyzibd_supported_v2_once_1278f69.tar.gz`
