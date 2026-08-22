# PoseLoop R3 BOP-Industrial read-only boundary audit

Protocol: `poseloop.r3.bop-industrial.e2e.v1`

Baseline: `e5e14ab6cf5f0b4adf97ca161ec04f188f6ea7e5`

Audit date: 2026-08-16

## Outcome

The legacy M1-M4 path is not an end-to-end BOP path. Ground-truth visible masks,
known object identities, ground-truth poses, and an oracle physical-instance
association enter before inference-side allocation and selection. Replacing only
the metric call would leave those oracle inputs intact.

R3 therefore starts at a new prediction boundary and does not import or rewrite
the frozen M0-M6, R1, R2, or M5-R6A evidence. The R3 adapter accepts only a
BOP-format predicted COCO file, a label-blind predicted-association file, two
fixed pose-result files, and a provenance receipt linking all four by SHA-256.
Ground truth remains behind the official evaluator process.

## Confirmed legacy dependencies

| Stage | Current dependency | Code evidence | R3 boundary |
|---|---|---|---|
| M1 adapter | Reads `scene_gt_realsense.json` and `scene_gt_info_realsense.json`; selects known `obj_id` | `scripts/build_m1_manifest.py:117-118`, `:161` | Detector/segmenter supplies predicted `category_id`; GT object identity is evaluator-only |
| M1 mask | Opens `mask_visib_realsense` and persists `visible_mask_pixel_count` | `scripts/build_m1_manifest.py:300`, `:374` | Predicted COCO RLE, bbox, and score are the only mask-side inputs |
| M1 pose label | Persists `gt_model_to_camera_pose_m` in the inference manifest | `scripts/build_m1_manifest.py:385` | No GT pose field is allowed in any R3 prediction artifact |
| M2 grouping | Builds tracks from `gt_instance_index` and writes `oracle_association.track_id` | `scripts/build_m2_groups.py:338-352`, `:857-864` | Association JSONL links prediction row indices and is declared label-blind before scoring |
| M2 selection | Uses each view's `visible_mask_pixel_count` | `scripts/evaluate_m2.py:1013` | Single/multi outputs may use only predicted mask geometry and prediction scores |
| M3 | Computes mask-area features from `visible_mask_pixel_count` and groups by `oracle_association.track_id` | `scripts/fit_m3_policy.py:235`, `:386`; the code itself records the known-oracle-mask scope at `:946-960` | Any future R3 M3 migration must source these features from predicted COCO RLE/bbox and predicted tracks |
| M4 | Decodes `target_manifest["mask_path"]`, checks the persisted GT visible-pixel count, and groups on the oracle track | `scripts/fit_m4_voi.py:331-354`, `:917-945` | Any future R3 M4 migration must consume predicted target masks and predicted associations only |
| M6-R1 | Builds mask fractions from upstream `visible_mask_pixel_count`; evaluator labels are later separated | `scripts/build_m6_r1_features.py:205-208`, `:297`, `:399-407` | M6 separation is useful, but its upstream inference stream must first be replaced; legacy M6 artifacts are not reusable as R3 inputs |

## Frozen replacement contract

1. `poseloop-r3-input_xyzibd-val.json` is an exact BOP extended-COCO prediction
   file with bbox, RLE segmentation, category score, and measured image time.
2. `poseloop-r3-predicted-association_xyzibd-val.jsonl` references the COCO row
   by `detection_index`. It cannot contain GT instance indices, visibility
   metadata, evaluator errors, or oracle track fields.
3. `poseloop-r3-single_xyzibd-val.csv` and
   `poseloop-r3-multiview_xyzibd-val.csv` use the official seven-column BOP pose
   result schema. The single path depends only on the COCO input. The multi path
   additionally depends on the frozen predicted association.
4. `poseloop-r3-provenance_xyzibd-val.json` binds those files, producer identity,
   lineage, label-blind declarations, and producer-code/model (or BOP-default
   source-package) artifacts by SHA-256. Input freezing must occur before any
   official score is opened.
5. The official BOP Toolkit at commit
   `cea62d651c7e395b2e1962b9749e4e89693c6ac4` is the only label consumer.
   A GitHub commit archive is permitted only through its frozen archive and
   normalized evaluator source-tree hashes.
   BOP19 produces localization AR, BOP24 produces 6D detection mAP, and BOP22
   produces bbox/segmentation AP.

## Local availability and exact blocker

Read-only machine inspection found:

- XYZ-IBD at `/home/cgliu/datasets/xyzibd` in WSL, with `val`, evaluation
  models, `test_targets_bop19.json`, and `test_targets_bop24.json`.
- The pinned clean BOP Toolkit checkout at
  `/mnt/c/Users/cgliu/OneDrive/Documents/PoseLoop/third_party/bop_toolkit`.
- No BOP default detection/segmentation file and no R3-compliant prediction
  bundle in the local dataset or R3 namespace.

The exact blocker is therefore the missing prediction-side bundle, not the
official evaluator or public dataset metadata. Existing M1-M4 predictions are
ineligible because they were produced with GT visible masks and oracle
association. No numerical R3 result is claimed.

## Claim boundary

The single/multi comparison evaluates two fixed PoseLoop acquisition variants
against the same official target files. It is not a BOP-2025 multi-view
leaderboard claim. A dry-run or passing contract test is structural evidence;
only an authorized official evaluator run can create AR/AP evidence.
