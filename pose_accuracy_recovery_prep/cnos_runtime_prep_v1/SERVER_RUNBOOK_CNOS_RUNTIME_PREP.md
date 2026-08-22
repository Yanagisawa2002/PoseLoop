# Official CNOS Runtime Prep — Future GPU-A Runbook

Status: `AUTO_DEPLOY=false`. This file prepares a future run; it does not authorize a
server connection, weight download, deployment, evaluator call, GPU-C trigger, or
shutdown.

## Frozen route

- Parent implementation base: `b5eea0522321721e08d3bef067f0f53e16b52b24`.
- Route: official CNOS `FastSAM-x + DINOv2 ViT-L/14 + PyRender CAD templates`.
- Official CNOS source: `https://github.com/nv-nguyen/cnos` at commit
  `298d1f3366171464ca271659f0e2f7a6eb8e39b4`, tree
  `595ba390ad1fdcd2141c8004e505b5da1eb403c9`.
- This route never imports SAM-6D pose, the old SAM ViT-B checkpoint, a depth prompt,
  a bbox/depth-derived proposal, a legacy mask, GT, evaluator assets, or a sealed split.
- The old SAM content-gate `NO_GO` remains immutable and is not retried or reinterpreted.
- Proposal ordering is target-CAD DINOv2 similarity descending, FastSAM box confidence
  descending, mask stability descending, then proposal index ascending.
- Raw target-CAD score is the mean of the official top-five normalized DINOv2 template
  cosines. The strict parent proposal schema stores `(raw_cosine + 1) / 2`. This affine
  transform is not clamped and preserves every raw-cosine ordering, including negative
  values. A separate self-locked score trace retains all five cosine values/indices,
  raw and normalized values, and the complete sorting key for every candidate.

The committed inert route is:

```text
protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json
```

Validate it locally before creating a deployment archive:

```powershell
C:\ProgramData\anaconda3\python.exe -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1 validate-route `
  --route protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json `
  --repository-root .
```

The expected result says `execution_ready=false` and `weights_downloaded=false`.

## Required immutable GPU-A layout

Create a new task-owned deployment root. Never reuse an old SAM/depth-prompt output
root. The request must describe this exact relative layout:

```text
config/cnos_adapter_config_v1.json
sources/cnos/                              # clean pinned checkout
sources/cnos-298d1f...-source.tar.gz       # pinned source archive
sources/poseloop/                          # clean master-approved implementation
sources/poseloop-cnos-runtime-prep-source.tar.gz
sources/dinov2/                            # clean local torch.hub checkout
sources/dinov2-source.tar.gz
models/FastSAM-x.pt
models/dinov2_vitl14_pretrain.pth
inputs/manifests/source.json
inputs/manifests/workload.json
inputs/rgb/<10 item files>
inputs/camera/<10 item files>
inputs/cad/<5 object files>
inputs/render/descriptors/<5 object files>
inputs/render/render-manifest.json
deployment-request.json
```

Before model import, all assets must exist and be non-empty. The deployment freeze
recomputes from disk:

- CNOS archive bytes/SHA, checkout commit/tree/clean status, and 11 source-interface
  files;
- the master-approved PoseLoop implementation checkout commit/tree/clean status,
  ancestry from `b5eea052...`, and its minimal source archive bytes/SHA;
- DINOv2 source archive bytes/SHA, checkout commit/tree/clean status, and canonical
  official origin `https://github.com/facebookresearch/dinov2`;
- both checkpoint paths, bytes, and SHA-256;
- adapter/PyRender config bytes and SHA-256;
- five exact CAD and descriptor path/bytes/SHA bindings plus the render-manifest lock;
- exact source/workload manifest SHA-256;
- ten RGB/camera path/bytes/SHA bindings and exact scene-image-object identity;
- the parent input-manifest lock, runtime lock, and additive deployment lock.

The render descriptor inventory must be produced from only the five frozen CAD models
with official CNOS PyRender level-0 views (42 views/object). Record the actual renderer
source/config and descriptor checkpoint identity in the render manifest. Do not infer
missing identities or substitute arbitrary 64-hex values.

The workload is exactly 10 rows, 5 scenes, and 5 objects. Any missing row, object swap,
CAD swap, descriptor swap, source/checkpoint change, forbidden token, nonzero access
counter, or dirty source checkout is a hard preflight failure.

Use sanitized camera JSON containing only the RGB intrinsics/frame convention needed by
this producer. Source/workload/camera JSON is recursively scanned; depth, old-mask,
GT, evaluator, oracle, scorer/sealed, old-SAM, and SAM-6D keys or string values are
rejected even when the containing file itself has a harmless name.

## Freeze before execution

Only after the future approved implementation commit is deployed, write a request with
schema `poseloop.pose-accuracy-recovery.cnos-deployment-request.v1`. Its
`implementation.approved_commit` and `approved_tree` must come from the master review
handoff; the checkout must be a clean strict descendant of `b5eea052...`. Place the
exact approved source archive beside it. Then run:

```bash
python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1 freeze-deployment \
  --route protocol/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json \
  --protocol protocol/poseloop_pose_accuracy_recovery_instance_proposal_v1.json \
  --request "$GA_ROOT/deployment-request.json" \
  --deployment-root "$GA_ROOT" \
  --deployment-output "$GA_ROOT/locks/deployment.json" \
  --runtime-lock-output "$GA_ROOT/locks/runtime-lock.json" \
  --receipt-output "$GA_ROOT/receipts/deployment-freeze.json" \
  2>&1 | tee "$GA_ROOT/logs/freeze-deployment.log"
```

`freeze-deployment` writes no model result. Its three outputs are immutable and the
receipt declares `execution_started=false` and `model_imported=false`. Copying a lock
to a new path requires a new deployment root and receipt; do not edit an existing lock.

Run the model-free disk preflight with the same interpreter that will execute CNOS:

```bash
python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1 preflight \
  --route protocol/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json \
  --protocol protocol/poseloop_pose_accuracy_recovery_instance_proposal_v1.json \
  --deployment "$GA_ROOT/locks/deployment.json" \
  --runtime-lock "$GA_ROOT/locks/runtime-lock.json" \
  --deployment-root "$GA_ROOT" \
  --output "$GA_ROOT/receipts/preflight.json" \
  2>&1 | tee "$GA_ROOT/logs/preflight.log"
```

Do not continue unless it reports 10 items, 5 objects, all actual assets ready, and
`label_access_count=0`. Hash the command, implementation archive, environment lock,
GPU identity, output root, freeze receipt, and preflight receipt before starting.

## Receipt-first screen execution and resume

Use a task-owned detached screen whose name contains `poseloop_ga`. The wrapper must
write the command, PID, GPU snapshot, disk snapshot, all input/implementation/model
hashes, and start timestamp before Python starts. Launch the bounded planned-crash
attempt first:

```bash
screen -dmS poseloop_ga_cnos_v1_attempt1 bash -lc '
  set -o pipefail
  python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1 run-producer \
    --route protocol/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json \
    --protocol protocol/poseloop_pose_accuracy_recovery_instance_proposal_v1.json \
    --deployment "$GA_ROOT/locks/deployment.json" \
    --runtime-lock "$GA_ROOT/locks/runtime-lock.json" \
    --deployment-root "$GA_ROOT" \
    --output-root "$GA_ROOT/output" \
    --planned-crash-after-items 2 \
    2>&1 | tee "$GA_ROOT/logs/attempt-0001.log"
  first_exit="${PIPESTATUS[0]}"
  test "$first_exit" -ne 0 || exit 41
  exit "$first_exit"
'
```

After that screen exits, independently inspect `run-state.json`, the first attempt
receipt, the two item receipts, their masks/visualizations/score traces, and the screen
exit. Continue only when the state is `PLANNED_CRASH`, `attempt_count=1`, exactly two
items are complete, and every recorded bytes/SHA still matches disk. Then launch the
resume as a separate receipt-visible screen:

```bash
screen -dmS poseloop_ga_cnos_v1_attempt2 bash -lc '
  set -o pipefail
  python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1 run-producer \
    --route protocol/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json \
    --protocol protocol/poseloop_pose_accuracy_recovery_instance_proposal_v1.json \
    --deployment "$GA_ROOT/locks/deployment.json" \
    --runtime-lock "$GA_ROOT/locks/runtime-lock.json" \
    --deployment-root "$GA_ROOT" \
    --output-root "$GA_ROOT/output" \
    2>&1 | tee "$GA_ROOT/logs/attempt-0002.log"
  exit "${PIPESTATUS[0]}"
'
```

Resume itself verifies every completed receipt, mask, visualization, and score trace
before the backend is called for any remaining row. It never overwrites a differing
output.

CUDA OOM handling retries only the failed descriptor slice with integer-halved proposal
chunks, from 16 down to 1, with at most four reductions. It does not change proposal
generation, thresholds, descriptors, scores, ranking, or already completed items. If
the bounded policy still OOMs, preserve `FAILED` and stop; do not tune or rerun under
the same protocol.

## Required output and stop boundary

A complete producer root contains:

```text
proposal-bundle.json
run-state.json
run-receipt.json
attempts/*.json
receipts/<10 item receipts>.json
score-traces/<10 self-locked candidate traces>.json
masks/<item>/<all target candidate masks>.png
visualizations/{rgb,proposal_overview,selected_mask,contours}/<10 files>.png
```

The proposal bundle remains the strict existing
`poseloop.pose-accuracy-recovery.instance-proposal-output.v1`; score evidence is not
added to that schema. The final run receipt binds the proposal bundle, all item
receipts, all score traces, the approved PoseLoop commit/tree/archive,
CNOS/DINO/model/runtime identities, zero-access boundary, OOM events, and attempt count.

After structural validation, GPU-A must retain RGB + all proposals + selected mask +
contours for the required 10/10 human single-target content gate. A machine similarity
score cannot replace that gate. Only a 10/10 human PASS permits construction of the
later five-variant A-to-C handoff. Do not trigger GPU-C automatically. Evaluation and
GT remain GPU-A evaluator-only work under separate authorization.

On any missing runtime asset or failed gate, preserve the exact receipt and report
`RUNTIME_NOT_READY`, `FAILED`, or `NO_GO`. Do not use synthetic inputs as a real result,
do not access a sealed split, and do not shut down or release any server without a new,
explicit lifecycle authorization.
