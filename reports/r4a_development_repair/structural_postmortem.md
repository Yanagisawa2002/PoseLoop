# R4-A label-free structural postmortem

Protocol: `poseloop.r4a.xyzibd-train-pbr.development.coverage-geometry.v1`

This postmortem uses only the five immutable R3-v3 prediction files, their
frozen hashes, the public model filename domain, the already-frozen aggregate
counts (750 images and 15 scenes), and public BOP/camera conventions. It does
not open XYZ-IBD validation GT, any threshold-level score JSON, or an evaluator.
The v3 official outcome and sealed receipt remain immutable and are not
reinterpreted here.

## Structural findings

| Surface | Frozen observation | Structural conclusion |
| --- | ---: | --- |
| COCO detections | 8 rows; 8 unique images; scene `0`; object `1` | A smoke workload, not a complete target enumerator |
| Aggregate image coverage | 8 / 750 (1.0667%) | Primary coverage defect |
| Public scene-domain coverage | 1 / 15 (6.6667%) | Global early-stop after the first scene |
| Public model-domain coverage | 1 / 15 (6.6667%) | Fixed `category_id=1`, not object enumeration |
| Association | 8 / 8 detections joined; 2 predicted tracks | No orphaned emitted detection; does not repair missing targets |
| Single-view pose | 8 rows; every row joins a detection | Emitted-row completeness only, not target completeness |
| Multi-view pose | 2 rows, one target view per predicted track | Track medoids operate only on the eight-row smoke domain |
| Rotation validity | 0 invalid; determinant about 1; max orthogonality error below `5e-7` | No non-finite or grossly invalid rotation defect in emitted rows |
| Translation representation | norm 645.36--669.29 mm; z 639.68--667.55 mm | Values are structurally consistent with BOP millimeters; no obvious 1000x error |

The producer source confirms the mechanism: it iterates sorted images, assigns
one fixed category to every depth component, stops globally at `max_images`,
and builds the FoundationPose workload only from those emitted COCO rows. A
target that was never enumerated creates neither a work item nor an explicit
failure record. Assembly checks that every *existing* work item succeeded, but
has no completeness contract against a scene/image/object target catalogue.

## Coordinate and unit chain

- XYZ-IBD CAD vertices are millimeters and are multiplied by `0.001` before
  FoundationPose.
- Raw depth is multiplied by `depth_scale * 0.001`, producing meters.
- `cam_t_w2c` is multiplied by `0.001`; `cam_R_w2c` and this translation form
  the public world-to-camera transform.
- FoundationPose produces model-to-camera poses in meters.
- Single-view output preserves that rotation and multiplies translation by
  `1000` for BOP millimeters.
- Multi-view conversion computes `T_w_m = inverse(T_c_w) @ T_c_m`, chooses a
  predicted world-pose medoid, then emits
  `T_c_target_m = T_c_target_w @ T_w_m_medoid`.

This chain is internally consistent. R4-A will lock it in tests and overlays;
it will not invent a coordinate flip from the consumed v3 score.

## Repair boundary

R4-A replaces the smoke enumerator with a complete, deterministic target
catalogue on a new `DEVELOPMENT_ONLY` `xyzibd/train_pbr` slice, validates the
identity BOP object-to-CAD mapping, makes missing inference explicit, and
enforces 100% declared-target coverage plus legal SE(3). The FoundationPose
source commit and checkpoints remain frozen. Development GT object IDs and
visible masks are disclosed oracle diagnostic inputs used identically by
before/after geometry runs; they cannot support a sealed end-to-end claim.
