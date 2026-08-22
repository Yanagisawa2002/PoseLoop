# Official CNOS Descriptor Renderer A-R3 — Review-Only Runbook

Status: `AUTO_DEPLOY=false`. This file authorizes no SSH connection, download,
model import, renderer execution, producer, evaluator, GPU-C action, or server
lifecycle action. A reviewed commit strictly after
`ff9bd13248d4d707c7dc9e41efa4773168db7ba3` is required before deployment.

## Frozen predecessor and sole repair

A-R2 remains immutable with status `BLOCKED_BEFORE_DESCRIPTOR_FREEZE`.
The new route binds:

- blocker receipt SHA-256
  `b45fa38c1928c9d72717eff570c1bba3a03b4b6a5a065a0f54a478c7ad5b848e`;
- safe archive SHA-256
  `464ea2e2f091167d812f8d5e91bfbd4baf473aeb4e15f3a02363a22c9e02f946`;
- official CNOS commit
  `298d1f3366171464ca271659f0e2f7a6eb8e39b4`, tree
  `595ba390ad1fdcd2141c8004e505b5da1eb403c9`;
- exact archive-member and clean-checkout byte identities for
  `src/poses/pyrender.py` and `src/utils/trimesh_utils.py`.

The Git archive contains CRLF source bytes while the pinned clean checkout has
LF source bytes. Both identities are explicit and mandatory. The wrapper
imports only the verified clean checkout. It calls the unchanged official
`render()` and `as_mesh()` functions, uses the unchanged level-0 42-view pose
file, intrinsics, lighting, scaling, recentering, and output names, but passes
the already hash-bound CAD file path to the official path-only
`get_obj_diameter()` helper. A loaded `Trimesh` is never passed to that helper.
The CNOS checkout must remain clean and unmodified.

## Admitted inputs and boundary

Only these inputs may be opened by this wrapper:

1. the pinned CNOS source archive and checkout interface files;
2. the two frozen predecessor evidence files;
3. `obj_poses_level0.npy`;
4. the exact five CAD files for objects `1, 2, 4, 5, 6`;
5. the reviewed PoseLoop checkout and hash-bound venv interpreter.

Depth, any mask or `mask_visib`, GT, evaluator, sealed data, scorer,
FoundationPose, FastSAM, DINOv2 model matching, and GPU-C are outside this
route. All corresponding counters remain zero. Rendering is development-only
infrastructure and creates no accuracy claim.

## Future immutable deployment layout

Use a new root; never reuse the A-R2 failure root:

```text
<GA_ROOT>/
  evidence/predecessor/producer-blocker-closeout-v1r2.json
  evidence/predecessor/poseloop-a-r2-ff9bd132-safe-closeout.tar.gz
  sources/poseloop/                     # reviewed A-R3 commit, clean
  sources/cnos/                         # exact pinned CNOS checkout, clean
  sources/cnos-298d1f...-source.tar.gz
  inputs/cad/obj_000001.ply ... obj_000006.ply
  contracts/render-request-v1r3.json
  attempts/<attempt_id>/                # must not exist before render-all
```

The request self-locks the reviewed implementation commit/tree, exact admitted
paths, absolute existing `venv/bin`, its `python` bytes/SHA, GPU override string
`"0"`, five CAD identities, and all-zero boundary. The interpreter is invoked
by absolute path. Child `PATH` begins with that exact `venv/bin`; `PYTHONPATH`
begins with the reviewed PoseLoop checkout. A bare `python` is forbidden.

## Future preflight and one create-only renderer attempt

After a main-controller-reviewed A-R3 commit is deployed, validate the inert
route before creating a runtime request:

```bash
export GA_ROOT=/absolute/new/immutable/poseloop_ga_cnos_v1r3
export POSELOOP_ROOT="$GA_ROOT/sources/poseloop"

"$GA_ROOT/runtime/venv/bin/python" -m \
  pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3 validate-route \
  --route "$POSELOOP_ROOT/protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r3.json" \
  --repository-root "$POSELOOP_ROOT"
```

Create `contracts/render-request-v1r3.json` with schema
`poseloop.pose-accuracy-recovery.cnos-render-request.v1r3`, exact route lock,
attempt ID prefix `poseloop_ga_cnos_v1r3_`, reviewed implementation identity,
the five fixed paths from the protocol, the hash-bound absolute venv
interpreter, exact five object/path rows, all-zero boundary, and canonical
`request_lock_sha256`. Then run the no-output preflight:

```bash
"$GA_ROOT/runtime/venv/bin/python" -m \
  pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3 preflight \
  --route "$POSELOOP_ROOT/protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r3.json" \
  --request "$GA_ROOT/contracts/render-request-v1r3.json" \
  --deployment-root "$GA_ROOT" \
  --repository-root "$POSELOOP_ROOT" \
  --output "$GA_ROOT/receipts/preflight-v1r3.json"
```

Stop on any source/archive/checkout/evidence/CAD/interpreter/request drift.
After an authorized preflight PASS, launch the parent once under `screen+tee`:

```bash
screen -dmS poseloop_ga_cnos_v1r3_renderer bash -lc \
  'set -o pipefail; "$GA_ROOT/runtime/venv/bin/python" -m \
  pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3 render-all \
  --route "$POSELOOP_ROOT/protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r3.json" \
  --request "$GA_ROOT/contracts/render-request-v1r3.json" \
  --deployment-root "$GA_ROOT" \
  --repository-root "$POSELOOP_ROOT" \
  --output-root "$GA_ROOT/attempts/<attempt_id>" 2>&1 | \
  tee "$GA_ROOT/logs/render-all-v1r3.log"; \
  rc=${PIPESTATUS[0]}; printf "%s\n" "$rc" > \
  "$GA_ROOT/receipts/render-all-v1r3.exit"; exit "$rc"'
```

The parent writes `attempt-start.json` before launching children. Each of the
five children uses the same explicit venv interpreter and string GPU override.
Any nonzero child exit, missing child output, corrupt image, wrong frame/mode,
empty/full foreground, fewer than 42 canonical PNGs, or insufficient view
distinctness creates `attempt-failure.json` and fails the parent. A parent
exit zero without all 210 validated PNGs is impossible. Attempt/output paths
are create-only; a failed attempt is evidence and must not be overwritten.

This wrapper does not create DINOv2 descriptors or run CNOS proposals. Those
remain separate future steps requiring their own reviewed authorization after
the renderer content gate passes.
