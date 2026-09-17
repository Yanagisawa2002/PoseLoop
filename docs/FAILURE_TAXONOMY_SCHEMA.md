# Per-instance failure taxonomy schema

The next accuracy investigation should produce a machine-readable table with one
row per ground-truth instance. The table is diagnostic only; it must not be used
to retune the already-consumed frozen evaluation split.

## Minimum columns

| Field | Meaning |
| --- | --- |
| `scene_id`, `image_id`, `gt_index` | Stable ground-truth identity |
| `object_id` | Object identity where permitted by the development protocol |
| `matched_prediction_index` | IoU-matched detector prediction, or null |
| `mask_iou` | Matched mask IoU when a match exists |
| `detector_score` | Frozen detector score for the matched prediction |
| `mask_match_iou50` | Whether the GT reaches the current pose handoff |
| `pose_completed` | Whether FoundationPose execution completed for the matched input |
| `normalized_mssd` | Frozen symmetry-aware surface-distance diagnostic |
| `mspd_px` | Frozen projection-distance diagnostic |
| `joint_pose_success` | Existing joint gate result |
| `failure_bucket` | One primary diagnostic category from the list below |
| `notes` | Optional concise, non-tuning qualitative observation |

## Primary diagnostic buckets

Use mutually exclusive primary buckets where evidence permits:

- `DETECTOR_MISS`
- `DUPLICATE_OR_MATCH_COMPETITION`
- `OVER_SEGMENTATION`
- `UNDER_SEGMENTATION_OR_MERGE`
- `MASK_BOUNDARY_IOU_FAILURE`
- `POSE_INPUT_INVALID_OR_WEAK_DEPTH`
- `POSE_REGISTRATION_FAILURE`
- `POSE_MSSD_ONLY_FAILURE`
- `POSE_MSPD_ONLY_FAILURE`
- `POSE_MSSD_AND_MSPD_FAILURE`
- `SUCCESS`
- `UNRESOLVED`

Do not infer a geometric cause from score/IoU alone. If the tracked artifacts do
not contain enough evidence to distinguish two causes, use `UNRESOLVED` and add a
secondary diagnostic field rather than manufacturing certainty.

## Aggregate output

Produce counts and percentages by:

- primary failure bucket;
- scene;
- object identity if allowed;
- IoU band (`<0.50`, `0.50–0.75`, `>=0.75`);
- joint pose outcome.

The result should reconcile exactly to the frozen waterfall: 770 GT instances,
577 IoU50 matches, and 482 joint pose successes.
