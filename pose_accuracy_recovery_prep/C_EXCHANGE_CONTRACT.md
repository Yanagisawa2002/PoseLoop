# GPU-A to GPU-C PREP exchange contract

Normative implementations:

- handoff exporter/validator: `pose_accuracy_recovery_prep/c_handoff.py`;
- producer result validator: `pose_accuracy_recovery_prep/c_results.py`;
- handoff protocol: `protocols/poseloop_pose_accuracy_recovery_c_handoff_v2.json`.

This is an inert DEVELOPMENT_ONLY preparation contract. `AUTO_DEPLOY=false`;
fixture execution is not an accuracy result.

## A-to-C manifest v2

Schema is exactly `poseloop.r4c.prep.runtime-isolated-manifest.v2`. Each unique
`(scene_id,image_id,object_id)` must have one ordered item for each of:

1. `official_known_sample_sanity`;
2. `oracle_mask_control`;
3. `predicted_mask`;
4. `depth_component_mask`;
5. `bbox_mask`.

Every item requires `item_id`, `sample_key`, `mask_variant_id`, `frame_size`,
five `inputs`, `camera_intrinsics`, `depth_scale`, and `mask_provenance`. Every
input has exactly `role`, `relative_path`, `sha256`, and `bytes`. Coverage is
100% over the identical base-key set for all five variants and is bound by the
ordered execution-key SHA-256 and `manifest_lock_sha256`.

RGB and depth sources may be the committed JSON fixture format or decoded
PNG/JPEG files; their frame sizes must match. Mask sources may be the JSON
fixture format or single-channel binary PNG/JPEG files with an auditable
development input score stored inside the hash-bound image metadata. Source
bytes must be non-empty and both source and copied bundle SHA-256 values are
verified.

GPU-A may read the two development control masks, but exports only copied bytes
under opaque `assets/input_masks/control-a|control-b/` paths. Original paths are
not serialized. The global contract must state:

```text
contains_gt_derived_control_inputs=true
contains_raw_gt_paths=false
gpu_c_resolves_derivation=false
label_access_count_on_gpu_c=0
evaluation_permitted_on_gpu_c=false
accuracy_claim_permitted=false
```

Thus the bundle is not globally label-free. GPU-C is label-isolated: it sees
only runtime input roles and may not read labels, GT poses, evaluator assets,
thresholds, or score files.

## C-to-A producer output v1

Every manifest item must produce exactly one ordered JSONL row with schema
`poseloop.pose-accuracy-recovery.producer-output.v1`. No field is optional and
the validator never fills defaults. Required groups are:

- manifest lock, item/sample/variant identity, and all five input hashes;
- full implementation commit plus implementation/model SHA-256 and a canonical
  SHA-256 binding the frozen refiner+scorer checkpoint hash map;
- initial/final model-to-camera pose in metres, exactly five ranked poses, and
  exactly six refiner states, each with an explicit finite objective;
- explicit status, one-based attempt, latency milliseconds, failure and OOM
  booleans, and failure reason (`null` only for success);
- a structured four-field object containing explicit zero GPU-C
  label/GT/evaluator/scorer path-access counters;
- RGB, input-mask, initial-pose, top-k, and final-pose visualization assets,
  each isolated by item and bound by path/bytes/SHA-256.

Failed/OOM rows still contain every field: pose fields are explicit `null`,
top-k/trace/visualizations are explicit empty lists, and `failure_reason` is a
non-empty string. Successful rows require legal SE(3), exactly five top-k poses, a
trace whose endpoints equal the initial/final poses, and all five producer-only
visualization roles. Those roles are exactly `rgb`, `input_mask`,
`initial_pose_overlay`, `top_k_overlay`, and `final_pose_overlay`; every member
must be a real file with verified positive bytes and SHA-256.

Only GPU-A may join accepted results with evaluator-only GT and run diagnostics
or official evaluation. C output alone supports provenance/coverage/runtime
claims, never accuracy claims.
