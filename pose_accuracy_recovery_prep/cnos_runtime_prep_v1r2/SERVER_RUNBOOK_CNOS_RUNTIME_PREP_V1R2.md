# Official CNOS Runtime Prep A-R2 — Future GPU-A Runbook

Status: `AUTO_DEPLOY=false`. This runbook authorizes no server connection,
download, model import, descriptor generation, deployment freeze, producer,
evaluator, GPU-C trigger, or lifecycle action.

## Immutable R1 blocker and sole repair

A-R2 is a new protocol/namespace above clean implementation
`7e15f06bc97c3466aa482b98b807e8f1cc211e2b`, tree
`f57d944cc00acc3c72a13dc58f72d90a28aad3df`. It binds the immutable local R1
closeout SHA-256
`998436f319d56db3e3e22df587bd7b79251353f1c7d6d9057e10fdecc610aceb`
and camera audit SHA-256
`b89744505c5a89c1e003139b1a59e5d3721e88759b1a4eec6f3773eca9d02867`.
R1 remains unchanged and must not be rerun.

The sole repair separates two roles:

1. `inputs/provenance/camera/<item_id>.json` is public, hash-only provenance.
   Each exact 418-byte asset must match SHA-256
   `bf93b9cb8c2a94515b8cec410b0aa6da60ab637d5dd79354ea80083f63fb8430`.
   Its JSON is never parsed and no field is read.
2. `inputs/runtime-camera/<item_id>.json` is deterministically generated from
   the exact workload row's `camera_intrinsics` only. It is canonical
   UTF-8/indent-2/sorted-key/LF JSON with an exact schema. No historical camera
   path or hash is substituted as its runtime bytes/SHA.

No external pose is required by the frozen CNOS adapter, so it is not copied.
The universal depth, legacy-mask, oracle, GT, evaluator, sealed, old-SAM, and
SAM-6D token/path rejection remains unchanged.

## Future reviewed deployment order

The future A-R2 implementation must be a clean reviewed commit strictly after
`7e15f06...`. First validate the inert route:

```bash
export GA_ROOT=/absolute/path/to/immutable/poseloop_ga_cnos_v1r2
python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r2 validate-route \
  --route protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r2.json \
  --protocol protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json \
  --repository-root sources/poseloop
```

Create the exact ten-item camera provenance manifest at
`inputs/manifests/camera-provenance-v1r2.json`, then derive cameras before any
model import:

```bash
python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r2 derive-cameras \
  --route "$GA_ROOT/protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r2.json" \
  --protocol "$GA_ROOT/protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json" \
  --deployment-root "$GA_ROOT" \
  --workload "$GA_ROOT/inputs/manifests/workload.json" \
  --camera-provenance-manifest "$GA_ROOT/inputs/manifests/camera-provenance-v1r2.json" \
  --manifest-output "$GA_ROOT/inputs/manifests/derived-camera-manifest-v1r2.json" \
  --receipt-output "$GA_ROOT/receipts/camera-derivation-v1r2.json"
```

Stop unless all ten raw provenance assets pass bytes/SHA, all ten canonical
runtime cameras pass exact-byte validation, and the receipt reports source JSON
parse/field/runtime-inclusion counts of zero.

The A-R2 deployment request is the v1 request plus the two exact manifest paths.
Every `input_items[].camera_path` must name the matching derived runtime camera.
RGB, CAD, descriptors, CNOS/DINO sources, checkpoints, and all other settings
remain under the unchanged v1 locks. Freeze and preflight with the reviewed
A-R2 request:

```bash
python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r2 freeze-deployment \
  --route "$GA_ROOT/protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r2.json" \
  --protocol "$GA_ROOT/protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json" \
  --request "$GA_ROOT/contracts/deployment-request-v1r2.json" \
  --deployment-root "$GA_ROOT" \
  --deployment-output "$GA_ROOT/contracts/deployment-v1r2.json" \
  --runtime-lock-output "$GA_ROOT/contracts/runtime-lock-v1r2.json" \
  --receipt-output "$GA_ROOT/receipts/freeze-v1r2.json"

python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r2 preflight \
  --route "$GA_ROOT/protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r2.json" \
  --protocol "$GA_ROOT/protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json" \
  --deployment "$GA_ROOT/contracts/deployment-v1r2.json" \
  --runtime-lock "$GA_ROOT/contracts/runtime-lock-v1r2.json" \
  --deployment-root "$GA_ROOT" \
  --output "$GA_ROOT/receipts/preflight-v1r2.json"
```

Only after both receipts pass, launch the planned-crash/resume producer sequence
under a screen name beginning `poseloop_ga_cnos_v1r2`, with the immutable output
root named exactly `output-v1r2`:

```bash
screen -dmS poseloop_ga_cnos_v1r2_producer bash -lc \
  'set -o pipefail; python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r2 run-producer \
  --route "$GA_ROOT/protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r2.json" \
  --protocol "$GA_ROOT/protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json" \
  --deployment "$GA_ROOT/contracts/deployment-v1r2.json" \
  --runtime-lock "$GA_ROOT/contracts/runtime-lock-v1r2.json" \
  --deployment-root "$GA_ROOT" \
  --output-root "$GA_ROOT/output-v1r2" 2>&1 | \
  tee "$GA_ROOT/logs/run-producer-v1r2.log"'
```

The R1 JSONL compatibility context remains restricted to the exact workload
path and the two frozen labels. It is serialized by its lock and restores both
hooks in `finally`, including concurrent and exceptional exits. No other read is
intercepted.

Do not proceed on any camera provenance drift, noncanonical derived bytes,
forbidden token, source-camera runtime inclusion, dependency byte change, dirty
implementation checkout, or nonzero label/GT/evaluator/sealed counter. No
accuracy claim is produced by derivation, freeze, preflight, or fixture tests.
