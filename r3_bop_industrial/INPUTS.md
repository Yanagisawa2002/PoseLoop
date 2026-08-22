# R3 BOP-Industrial input and execution checklist

The protocol is `protocols/poseloop_r3_bop_industrial_protocol.json`. All R3
prediction files must live outside the BOP dataset root and outside frozen
PoseLoop namespaces. `artifacts/r3_bop_industrial/inputs/` is the default local
location.

## Required prediction bundle

| Role | Required filename | Contract |
|---|---|---|
| Predicted detections/segmentations | `poseloop-r3-input_xyzibd-val.json` | Non-empty BOP extended-COCO list; exact fields `scene_id,image_id,category_id,score,bbox,segmentation,time`; predicted RLE only |
| Predicted association | `poseloop-r3-predicted-association_xyzibd-val.jsonl` | One row per associated COCO `detection_index`; exactly one target view per predicted track; no GT/oracle field |
| Single-view poses | `poseloop-r3-single_xyzibd-val.csv` | Official BOP `scene_id,im_id,obj_id,score,R,t,time` schema; non-negative measured time |
| Multi-view poses | `poseloop-r3-multiview_xyzibd-val.csv` | Same BOP schema; every output must link to a predicted target-view association |
| Provenance | `poseloop-r3-provenance_xyzibd-val.json` | Exact producer, origin, lineage, leakage declarations, prediction hashes, and source code/model/default-package hashes |

`input_origin` in the provenance file is either `bop_default` or
`audited_prediction`. A BOP-default COCO file still needs a label-blind predicted
association before it can support the multi-view variant. An audited prediction
must list `producer_code` and `model` source artifacts; a BOP-default input lists
the downloaded release/package as `default_prediction_source`. Every listed
artifact has a URI plus a 64-hex SHA-256 value.

The provenance schema is:

```json
{
  "schema_version": "poseloop.r3.prediction-provenance.v1",
  "protocol_id": "poseloop.r3.bop-industrial.e2e.v1",
  "input_origin": "audited_prediction",
  "producer": "REPLACE_ME",
  "producer_version": "REPLACE_ME",
  "source_uri_or_run_id": "REPLACE_ME",
  "created_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "label_blind": true,
  "uses_gt_visible_masks": false,
  "uses_oracle_association": false,
  "result_selection_uses_evaluator_metrics": false,
  "source_artifacts": [
    {"role": "producer_code", "uri": "REPLACE_ME", "sha256": "REPLACE_WITH_64_HEX"},
    {"role": "model", "uri": "REPLACE_ME", "sha256": "REPLACE_WITH_64_HEX"}
  ],
  "files": {
    "coco_predictions": {"filename": "poseloop-r3-input_xyzibd-val.json", "sha256": "REPLACE_ME"},
    "predicted_association": {"filename": "poseloop-r3-predicted-association_xyzibd-val.jsonl", "sha256": "REPLACE_ME"},
    "single_view_pose": {"filename": "poseloop-r3-single_xyzibd-val.csv", "sha256": "REPLACE_ME"},
    "multi_view_pose": {"filename": "poseloop-r3-multiview_xyzibd-val.csv", "sha256": "REPLACE_ME"}
  },
  "lineage": {
    "single_view_pose": ["coco_predictions"],
    "predicted_association": ["coco_predictions"],
    "multi_view_pose": ["coco_predictions", "predicted_association"]
  }
}
```

## Label-blind dry-run

From the R3 worktree in WSL:

```bash
python3 -B -m r3_bop_industrial dry-run \
  --dataset-root /home/cgliu/datasets/xyzibd \
  --toolkit-root /mnt/c/Users/cgliu/OneDrive/Documents/PoseLoop/third_party/bop_toolkit \
  --input-root artifacts/r3_bop_industrial/inputs \
  --eval-root artifacts/r3_bop_industrial/eval \
  --receipt reports/r3_bop_industrial/dry_run_receipt.json
```

This checks only public target/model metadata, the pinned toolkit, expected
prediction files, hashes, and command construction. It deliberately does not
enumerate or parse `scene_gt*`, `scene_gt_info*`, `scene_gt_coco*`, or mask
directories.

The toolkit may be either a clean Git checkout at the pinned commit or the
official GitHub commit tarball. The latter is accepted only when its normalized
93-file evaluator source-tree digest matches the frozen protocol; the download
archive SHA-256 is also frozen in the protocol.

## Freeze before evaluation

After the complete bundle exists and its provenance hashes are correct:

```bash
python3 -B -m r3_bop_industrial freeze \
  --dataset-root /home/cgliu/datasets/xyzibd \
  --toolkit-root /mnt/c/Users/cgliu/OneDrive/Documents/PoseLoop/third_party/bop_toolkit \
  --input-root artifacts/r3_bop_industrial/inputs \
  --eval-root artifacts/r3_bop_industrial/eval \
  --input-lock artifacts/r3_bop_industrial/input_lock.json
```

The freeze command fails if a file is missing, a hash/lineage differs, a pose is
not a valid rigid transform, an association does not link to the COCO row, or a
legacy GT/oracle field appears.

## Authorized official evaluation

Formal scoring is intentionally unavailable until a separate label-access
receipt explicitly approves the scope `official_bop_toolkit_evaluator_only` and
binds the exact input-lock SHA-256. Do not create that receipt merely to make the
command pass; it records authorization from the responsible reviewer.

Once authorized, `evaluate` invokes the pinned scripts without reimplementing
metrics and writes `official_scores.json` under the R3 artifact namespace. It
reports BOP19 AR and BOP24 mAP separately for single and multi variants, BOP22
bbox/segmentation AP for their shared prediction input, and fixed multi-minus-
single deltas.
