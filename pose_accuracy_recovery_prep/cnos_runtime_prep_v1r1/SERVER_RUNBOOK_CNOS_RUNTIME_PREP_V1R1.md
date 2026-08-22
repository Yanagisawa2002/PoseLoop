# Official CNOS Runtime Prep A-R1 — Future GPU-A Runbook

Status: `AUTO_DEPLOY=false`. This file authorizes no server connection, download,
model import, inference, evaluator call, GPU-C trigger, or lifecycle action.

## Immutable predecessor and sole repair

R1 never edits or reinterprets `cnos_runtime_prep_v1`. It binds predecessor
implementation `0389cb3ac3f0b7854e9ca7c57eedda345aaa2fd6`, tree
`e40a12ee4cbfc41052f4547ea1c2d90b9d46e021`, and the immutable GPU-A
`PRE_MODEL_SCHEMA_BLOCKER` at:

```text
/root/autodl-tmp/poseloop_ga_cnos_0389cb3_20260818_blocked_v1
receipts/pre-model-schema-blocker-closeout.json
SHA-256 68b112abe9c654440a1306749d93aa4e3bd27951b76f541ee81d45b85d856a67
```

The only repair is workload interpretation. The frozen workload remains exactly
10,864 bytes, ten LF-terminated JSON objects, and SHA-256
`dd9f9c4ce9b9ca380614064f58e332661aee9ce63c9602918d52ae391267dfb3`.
It is parsed one line at a time without rewriting.

The historical `depth_*`, segmentation, COCO, camera-pose, and sensor values are
declared provenance only. R1 may read their JSON values to validate and hash the
manifest. It must never resolve, open, copy, or pass the referenced assets to a
runtime. The public audit stores field names and inventory hashes, not raw paths.
Every runtime item remains exactly `rgb`, `camera`, `cad`, and
`cad_render_descriptors`. Source manifest, camera JSON, request paths, runtime
paths, and model paths retain the original recursive/path hard rejection of depth,
legacy-mask, GT, oracle, evaluator, sealed, old-SAM, and SAM-6D tokens.

## New immutable identity

```text
route: protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r1.json
package: pose_accuracy_recovery_prep/cnos_runtime_prep_v1r1
deployment namespace: poseloop_ga_cnos_v1r1
screen prefix: poseloop_ga_cnos_v1r1
output namespace: output-v1r1
inner compatibility payload: output-v1r1/compat-v1-payload
```

The reviewed A-R1 implementation commit must be a clean strict descendant of
`0389cb3...`. All thirteen frozen v1 dependency bindings must match disk before
freeze or preflight. A dependency change requires another protocol; it cannot be
absorbed by relocking R1.

Validate locally before any future deployment:

```powershell
C:\ProgramData\anaconda3\python.exe -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1 validate-route `
  --route protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r1.json `
  --repository-root .
```

Expected: `status=valid`, `execution_ready=false`, `weights_downloaded=false`.

## Future receipt-first order

Create a new root; never reuse the blocked v1 root. Copy the exact JSONL unchanged
to `inputs/manifests/workload.json`. Do not copy any path it declares. Populate
only the v1-approved source/checkpoints, ten RGB/camera files, five CAD files, and
five level-0 descriptor files. The R1 request uses schema
`poseloop.pose-accuracy-recovery.cnos-deployment-request.v1r1`.

```bash
python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1 freeze-deployment \
  --route "$GA_ROOT/protocol/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r1.json" \
  --protocol "$GA_ROOT/protocol/poseloop_pose_accuracy_recovery_instance_proposal_v1.json" \
  --request "$GA_ROOT/deployment-request-v1r1.json" \
  --deployment-root "$GA_ROOT" \
  --deployment-output "$GA_ROOT/locks/deployment-v1r1.json" \
  --runtime-lock-output "$GA_ROOT/locks/runtime-lock-v1r1.json" \
  --receipt-output "$GA_ROOT/receipts/deployment-freeze-v1r1.json"

python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1 preflight \
  --route "$GA_ROOT/protocol/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r1.json" \
  --protocol "$GA_ROOT/protocol/poseloop_pose_accuracy_recovery_instance_proposal_v1.json" \
  --deployment "$GA_ROOT/locks/deployment-v1r1.json" \
  --runtime-lock "$GA_ROOT/locks/runtime-lock-v1r1.json" \
  --deployment-root "$GA_ROOT" \
  --output "$GA_ROOT/receipts/preflight-v1r1.json"
```

Stop unless the audit says `DECLARED_BUT_NOT_OPENED`, every provenance asset
counter is zero, the workload and dependency hashes match, and runtime roles are
exactly the four allowed roles.

The only future producer screen name starts with `poseloop_ga_cnos_v1r1`; its
output path must end in `output-v1r1`. Use the same planned-crash/resume policy as
v1, but point the command at this R1 module, route, deployment, and runtime lock.
The v1-compatible strict proposal bundle remains below `compat-v1-payload`; the
new `run-receipt-v1r1.json` binds it to the R1 route/deployment/workload audit.

No R1 action may invoke FoundationPose, GPU-C, an evaluator, a scorer, or shutdown.
