# CNOS Descriptor Renderer A-R4 — Review-Only Runbook

Status: `AUTO_DEPLOY=false`. This namespace authorizes no SSH connection,
download, environment mutation, renderer execution, descriptor generation,
FastSAM/DINOv2 producer, evaluator, GPU-C action, or server lifecycle action.
A main-controller-reviewed commit strictly after
`1aef708bcdc678b3734f25fd35b1381090c7f55d` is required before any future
deployment.

## Frozen A-R3 evidence

A-R3 and both create-only attempts are immutable:

- implementation commit `1aef708bcdc678b3734f25fd35b1381090c7f55d`,
  tree `1311a5fede91ecd26cbc7c272ec9c5867ef72eca`;
- route lock
  `cb83957708fe25eca809c36ff8c11e85811ce18e214fc4e9b2e40da48f348c77`;
- attempt 1 safe archive: 125,100 bytes, SHA-256
  `2d082f486320e214daeea6895075ea5afd4458556db91bcae72677a78be84ac0`;
- attempt 2 geometry blocker: 4,831 bytes, SHA-256
  `729f0f2a84a3e9b10c455b249ccb3d09fafae9de594e9797544625cfc9e7056e`;
- attempt 2 content-audit addendum SHA-256
  `bb4e5f8566f1d747118da6351c54af5f27ddcc3e1fc83a123e69b1a407b7bdb3`;
- final safe archive: 274,172 bytes, SHA-256
  `5dffd275caa703531c1a3db420c1f48c0ccd3efcccc9263a11dcf1691550ee03`.

Attempt 2 produced 42 object-1 PNGs: 40 were fully transparent, two had four
visible pixels each, and only three PNG hashes were unique. No descriptor,
FastSAM, producer, FoundationPose, evaluator, scorer, or GPU-C call ran. A-R4
does not reinterpret that `NO-GO` and never reuses either failed output root.

## Sole geometry repair

The official CNOS repository remains fixed at commit
`298d1f3366171464ca271659f0e2f7a6eb8e39b4`, tree
`595ba390ad1fdcd2141c8004e505b5da1eb403c9`. Its checkout is imported read-only
and is never patched.

A-R3 computed the recenter transform from the unscaled CAD centroid, then
scaled meshes with diameter greater than 100 by `0.001`. The transform retained
the source-unit translation and displaced off-origin XYZ-IBD objects out of
frame. A-R4 changes only that ordering:

1. load the already hash-bound CAD;
2. call official `get_obj_diameter()` with the CAD file path, never a mesh;
3. if and only if `diameter > 100`, apply scale `0.001`;
4. read `mesh.bounding_box.centroid` after the optional scale;
5. construct the negative-centroid transform;
6. invoke the unchanged official `render()`.

The unscaled branch is unchanged. Camera poses, intrinsics, 42 level-0 views,
lighting, filenames, image size, five CAD identities, and all boundary counters
remain fixed.

## Venv entry repair

The request separately binds:

- the lexical `venv/bin/python` entry, including whether it is a symlink and
  the exact symlink target text hash; and
- the resolved target's absolute path, bytes, and SHA-256.

Validation may resolve the target to verify it, but child argv retains and
executes the lexical entry. This preserves normal venv `sys.prefix` behavior
and prevents the A-R3 failure in which resolving the symlink selected base
Python without `pyrender`. `PATH` begins with the same bound `venv/bin`, and
`PYOPENGL_PLATFORM=egl` is explicit. An unbound overlay is neither required nor
admitted.

## Future immutable layout and request

Use a new root after a reviewed A-R4 commit; never reuse either A-R3 root:

```text
<GA_ROOT>/
  evidence/predecessor/render-attempt2-geometry-blocker.json
  evidence/predecessor/poseloop-a-r3-1aef708b-final-safe-evidence.tar.gz
  sources/poseloop/                  # reviewed A-R4 commit, clean
  sources/cnos/                      # pinned official checkout, clean
  sources/cnos-298d1f...-source.tar.gz
  inputs/cad/obj_000001.ply ... obj_000006.ply
  contracts/render-request-v1r4.json
  attempts/<new-v1r4-attempt-id>/    # absent before render-all
```

The final archive is inspected in place and must contain the exact A-R3
geometry blocker, content-audit addendum, and attempt-1 archive hashes. The
request schema is
`poseloop.pose-accuracy-recovery.cnos-render-request.v1r4`; it self-locks the
new reviewed commit/tree, admitted paths, Python entry and target, GPU string
`"0"`, exact five objects, and the all-zero boundary.

## Future review and execution order

The current local-only phase ends after tests and review handoff. A future
authorization must perform, in order:

1. deploy a clean reviewed A-R4 commit to a new immutable root;
2. verify source/archive/CAD/predecessor/interpreter hashes;
3. run `validate-route` and a create-only `preflight` without importing a
   model or creating an attempt root;
4. freeze a screen name, job script, request, command, and launch receipt;
5. run one authorized `render-all` under `screen + tee`;
6. require 5 objects × 42 decodable PNGs with nonempty/nonfull RGB and RGBA
   alpha foreground, then preserve the attempt regardless of outcome.

CLI shape for later review (not authorization to run):

```bash
python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4 validate-route \
  --route "$POSELOOP_ROOT/protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r4.json" \
  --repository-root "$POSELOOP_ROOT"

python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4 preflight \
  --route "$ROUTE" --request "$REQUEST" \
  --deployment-root "$GA_ROOT" --repository-root "$POSELOOP_ROOT" \
  --output "$GA_ROOT/receipts/preflight-v1r4.json"
```

Even a renderer PASS is infrastructure evidence only. Descriptor generation,
proposal execution, instance correctness, pose accuracy, AR, and AP remain
unavailable until separately authorized and measured in their proper
namespaces.
