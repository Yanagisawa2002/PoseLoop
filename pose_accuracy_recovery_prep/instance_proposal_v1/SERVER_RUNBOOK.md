# Independent instance-proposal v1 server runbook

Status: `AUTO_DEPLOY=false`, `RUNTIME_NOT_READY`. This document is preparation,
not an execution authorization or an accuracy result.

## Frozen boundary

- Protocol: `poseloop.pose-accuracy-recovery.development.instance-proposal.v1`.
- Data role: `DEVELOPMENT_ONLY`; exact 10 rows, five scenes, five objects.
- Immutable prior result: SAM depth-component-bbox content gate receipt
  `ba744c3e62ede30356ad79dc4ccee710d2585a0cfcf1e65b3f50675ba8eebda5`
  remains `NO_GO`; it cannot be rerun, tuned, rewritten, or used as a proposal.
- Independent inputs are exactly RGB, camera, CAD, and CAD-render descriptors.
  `depth_component_mask`, its bbox, `bbox_mask`, legacy predicted masks, GT,
  evaluator assets, and sealed assets are forbidden proposal cues.
- GPU-C is never triggered automatically. GPU-A must remain powered on unless a
  later user instruction explicitly authorizes shutdown.
- The frame is fixed at `1440x1080`. Every execution-ready runtime must consume
  a self-locked per-sample input manifest that binds the source/workload
  manifests and all four real input assets for each exact item. Proposal files
  cannot supply or substitute their own input hashes.

## Current GPU-A readiness blocker

The bounded read-only audit at `2026-08-18T00:25:06+08:00` found no CNOS,
SAM-6D, GroundingDINO, Detectron2/MMDetection, CAD renderer, descriptor runtime,
or associated checkpoint/cache. The existing SAM ViT-B is excluded by the
immutable prior NO-GO. Five no-GT CAD models are present and hash-locked in the
protocol. Therefore no proposal process may be started from the current audit.

## Future controller-reviewed sequence

1. Select one official CAD-conditioned instance-proposal implementation. Lock
   its repository URL, full commit/tree, source archive SHA-256, model config,
   checkpoint SHA-256/bytes, renderer source/config, view sampling, CAD-render
   manifest, and descriptor-model SHA-256. Do not maintain alternative model
   routes inside one protocol.
2. If assets are absent, use `source /etc/network_turbo` only for the selected
   official GitHub/Hugging Face downloads. Do not download before the local
   implementation commit and protocol have passed controller review.
3. Build a fresh runtime lock with
   `poseloop.pose-accuracy-recovery.instance-proposal-runtime-lock.v1`. Its
   `data` member must use
   `poseloop.pose-accuracy-recovery.instance-proposal-input-manifest.v1`, bind
   exact item/sample identities, and match every target CAD SHA/byte count in
   the protocol. Then run:

   ```text
   python -m pose_accuracy_recovery_prep.instance_proposal_v1 validate-runtime-lock --protocol protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json --runtime-lock <runtime-lock.json> --input-root <immutable-input-root> --output <preflight>/runtime-validation.json
   ```

4. Freeze command, source/model/render/data hashes, fresh output roots, GPU
   snapshot, all zero access counters, and a pre-execution receipt. Launch the
   one fixed proposal command in a `poseloop_ga_*` detached `screen` with `tee`.
5. Emit exact 10-row `instance-proposal-output.v1`. Validate real proposal masks,
   frozen CAD-first ranking, rank-1 selection, four visualization roles, exact
   coverage, exact runtime-manifest input hashes, and zero
   legacy-depth/label/evaluator/sealed access. Every mask must be a decodable
   `1440x1080` single-channel binary PNG with non-empty foreground, recomputable
   pixel count/coverage/8-connected components, and a tight half-open bbox.
   Every visualization must be a decodable full-frame image:

   ```text
   python -m pose_accuracy_recovery_prep.instance_proposal_v1 validate-proposals --protocol protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json --runtime-lock <runtime-lock.json> --input-root <immutable-input-root> --proposal-bundle <proposal-output.json> --asset-root <proposal-root> --output <preflight>/proposal-validation.json
   ```

6. Generate one hash-bound panel per row plus a contact sheet showing RGB,
   proposal overview, selected mask, and contours. A human must explicitly mark
   whether it is one target instance and whether tray, border, or multiple
   objects are included. Machine scores cannot substitute for this review.

   ```text
   python -m pose_accuracy_recovery_prep.instance_proposal_v1 validate-content-gate --protocol protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json --runtime-lock <runtime-lock.json> --input-root <immutable-input-root> --proposal-bundle <proposal-output.json> --proposal-root <proposal-root> --review <content-gate.json> --review-root <review-root> --output <preflight>/content-validation.json
   ```

7. Any item failure yields `NO_GO`; freeze evidence and stop without tuning or a
   second run. Only an exact 10/10 human PASS allows GPU-A to prepare a new
   five-variant A-to-C bundle, after which it waits for the controller. It must
   not contact or trigger GPU-C itself.
