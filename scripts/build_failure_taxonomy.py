#!/usr/bin/env python3
"""Build a conservative PoseLoop v1.1.0 per-instance failure taxonomy.

The status mode uses only tracked frozen aggregate evidence. The extract mode
consumes the original frozen input manifest, primary predictions, XYZ-IBD data,
and pinned BOP Toolkit. It does not rerun detector or FoundationPose inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RELEASE_RESULT = ROOT / "release" / "v1.1.0" / "results.json"
A9_RESULT_MD = ROOT / "pose_accuracy_recovery_prep" / "a9_foundationpose_e2e" / "RESULT.md"

BUCKET_ORDER = (
    "DETECTOR_MISS",
    "DUPLICATE_OR_MATCH_COMPETITION",
    "OVER_SEGMENTATION",
    "UNDER_SEGMENTATION_OR_MERGE",
    "MASK_BOUNDARY_IOU_FAILURE",
    "POSE_INPUT_INVALID_OR_WEAK_DEPTH",
    "POSE_REGISTRATION_FAILURE",
    "POSE_MSSD_ONLY_FAILURE",
    "POSE_MSPD_ONLY_FAILURE",
    "POSE_MSSD_AND_MSPD_FAILURE",
    "SUCCESS",
    "UNRESOLVED",
)

CSV_FIELDS = (
    "scene_id", "image_id", "gt_index", "object_id",
    "matched_prediction_index", "item_id", "detector_score",
    "mask_iou", "max_mask_iou", "iou_band", "mask_match_iou50",
    "merge_evidence", "split_evidence", "pose_completed",
    "primary_failure_type", "normalized_mssd", "mspd_px",
    "joint_pose_success", "failure_bucket", "notes",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_scene_table() -> list[dict[str, int | float]]:
    text = A9_RESULT_MD.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\|\s*(10|25|30|40|65)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        r"\s*(\d+)\s*\|\s*([0-9.]+)\s*\|$",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(text):
        rows.append({
            "scene_id": int(match.group(1)),
            "gt": int(match.group(2)),
            "predictions": int(match.group(3)),
            "joint_successes": int(match.group(4)),
            "joint_recall": float(match.group(5)),
        })
    if [row["scene_id"] for row in rows] != [10, 25, 30, 40, 65]:
        raise ValueError("Could not recover the frozen five-scene result table")
    return rows


def status_markdown() -> str:
    release = _read_json(RELEASE_RESULT)
    pose = release["pose_pipeline"]
    total = int(release["dataset"]["ground_truth_instance_count"])
    matched = int(pose["mask_iou50_matches"])
    success = int(pose["joint_pose_successes"])
    upstream_loss = total - matched
    pose_loss = matched - success
    total_failure = total - success

    if (total, matched, success) != (770, 577, 482):
        raise ValueError("Frozen v1.1.0 waterfall changed")

    scene_rows = _parse_scene_table()
    if sum(int(row["gt"]) for row in scene_rows) != total:
        raise ValueError("Scene GT counts do not reconcile")
    if sum(int(row["joint_successes"]) for row in scene_rows) != success:
        raise ValueError("Scene joint successes do not reconcile")

    scene_lines = []
    for row in scene_rows:
        failures = int(row["gt"]) - int(row["joint_successes"])
        scene_lines.append(
            f"| {row['scene_id']} | {row['gt']} | {row['joint_successes']} | "
            f"{failures} | {failures / total_failure:.1%} |"
        )

    return f"""# PoseLoop v1.1.0 failure taxonomy - recovery status

This page is generated from tracked frozen evidence. It deliberately stops
where the repository no longer contains enough per-instance evidence.

## What is proven now

| Stage | GT-level count | Share of all 288 GT failures |
| --- | ---: | ---: |
| Does not reach the IoU50 pose handoff | {upstream_loss} | {upstream_loss / total_failure:.1%} |
| Reaches IoU50 handoff but misses the joint pose gate | {pose_loss} | {pose_loss / total_failure:.1%} |
| Joint pose success | {success} | - |

The frozen aggregate chain is **{total} GT -> {matched} mask-IoU50 matches ->
{success} joint pose successes**. The first {upstream_loss} cases cannot be
honestly split into detector miss, merge, split, match competition, or boundary
failure from the tracked aggregate bundle alone. Likewise, the {pose_loss}
matched pose failures cannot be split into MSSD-only, MSPD-only, both-metric,
or runtime/input failures without the original per-instance evaluation evidence.

## Where the failures concentrate

| Scene | GT | Joint successes | GT-level failures | Share of all failures |
| ---: | ---: | ---: | ---: | ---: |
{chr(10).join(scene_lines)}

Scene 25 contributes the majority of the frozen GT-level failures, but this is
a localization fact, not a causal label. It must not be used to retune the
already-consumed split.

## Evidence needed for the full 770-row taxonomy

The evaluator code shows that the original run produced enough information to
finish the taxonomy without rerunning detector training or FoundationPose
inference, provided the frozen runtime artifacts still exist:

- frozen A9 input manifest, including detector mask bindings;
- primary predictions.jsonl and matching completion-receipt.json;
- XYZ-IBD development data referenced by the manifest;
- pinned BOP Toolkit checkout for the same symmetry-aware MSSD/MSPD calculation.

The original evaluation also wrote matched-pose-errors.csv with frame_id,
gt_index, mask_iou, normalized_mssd, and mspd_px. That CSV is sufficient to
split the matched pose failures, while detector-mask bindings are needed for
the unmatched geometric taxonomy.

Those per-instance artifacts were intentionally kept under the ignored
artifacts tree and were not included in the public v1.1.0 Release. Therefore
the current tracked repository is **BLOCKED_PER_INSTANCE_EVIDENCE**, not a
completed fine-grained taxonomy.

## Recovery command

If the frozen manifest and primary root are recovered, run:

    python -B scripts/build_failure_taxonomy.py extract \\
      --protocol protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json \\
      --manifest /path/to/input-manifest.json \\
      --primary-root /path/to/foundationpose-primary \\
      --dataset-root /path/to/xyzibd \\
      --toolkit-root /path/to/bop_toolkit \\
      --output-root /path/to/failure-taxonomy

The extractor is create-only and verifies the frozen manifest, primary
completion receipt, primary prediction hash, exact aggregate reconciliation,
and v1.1.0 stage counts before writing any final summary.

## Interpretation

The next measured question remains: **what fraction of the 193 upstream losses
are true detector misses versus instance-formation errors, and what fraction of
the 95 matched pose losses fail MSSD, MSPD, or both?**

Until the original per-instance evidence is recovered, any more specific answer
would be invented.
"""


def _iou_band(value: float) -> str:
    if value < 0.50:
        return "<0.50"
    if value < 0.75:
        return "0.50-0.75"
    return ">=0.75"


def classify_pose(
    normalized_mssd: float,
    mspd_px: float,
    *,
    mssd_threshold: float,
    mspd_threshold: float,
) -> tuple[str, bool]:
    mssd_ok = normalized_mssd < mssd_threshold
    mspd_ok = mspd_px < mspd_threshold
    if mssd_ok and mspd_ok:
        return "SUCCESS", True
    if not mssd_ok and mspd_ok:
        return "POSE_MSSD_ONLY_FAILURE", False
    if mssd_ok and not mspd_ok:
        return "POSE_MSPD_ONLY_FAILURE", False
    return "POSE_MSSD_AND_MSPD_FAILURE", False


def classify_unmatched(
    *,
    has_iou50_candidate: bool,
    merge_evidence: bool,
    split_evidence: bool,
    max_iou: float,
) -> str:
    if has_iou50_candidate:
        return "DUPLICATE_OR_MATCH_COMPETITION"
    if merge_evidence:
        return "UNDER_SEGMENTATION_OR_MERGE"
    if split_evidence:
        return "OVER_SEGMENTATION"
    if max_iou > 0.0:
        return "MASK_BOUNDARY_IOU_FAILURE"
    return "DETECTOR_MISS"


def _looks_like_depth_input_failure(row: dict[str, Any]) -> bool:
    if row.get("status") == "success":
        return False
    message = " ".join(
        str(row.get(key, "")).lower()
        for key in ("failure_type", "failure_message")
    )
    return any(token in message for token in ("depth", "point cloud", "pointcloud"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def _summary_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    total = len(rows)
    bucket_counts = Counter(str(row["failure_bucket"]) for row in rows)
    bucket_lines = [
        f"| {bucket} | {bucket_counts.get(bucket, 0)} | "
        f"{bucket_counts.get(bucket, 0) / total:.1%} |"
        for bucket in BUCKET_ORDER
        if bucket_counts.get(bucket, 0)
    ]
    scene_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        scene_counts[int(row["scene_id"])][str(row["failure_bucket"])] += 1
    scene_lines = []
    for scene_id in sorted(scene_counts):
        failures = sum(
            count for bucket, count in scene_counts[scene_id].items()
            if bucket != "SUCCESS"
        )
        successes = scene_counts[scene_id].get("SUCCESS", 0)
        scene_lines.append(
            f"| {scene_id} | {successes + failures} | {successes} | {failures} |"
        )
    iou_counts = Counter(str(row["iou_band"]) for row in rows)
    iou_lines = [
        f"| {band} | {iou_counts.get(band, 0)} |"
        for band in ("<0.50", "0.50-0.75", ">=0.75")
    ]
    version = "v1.2" if summary["schema_version"].startswith("poseloop.v1.2.") else "v1.1.0"
    return f"""# PoseLoop {version} per-instance failure taxonomy

Status: **COMPLETE_DIAGNOSTIC_RECONSTRUCTION**

This is a post-hoc diagnostic of the already-consumed {version} development split.
It is not a new untouched evaluation and must not be used to retune {version} and
then presented as if the split were unseen.

## Reconciliation

- GT instances: **{summary['ground_truth_instance_count']}**
- IoU50 mask matches: **{summary['mask_iou50_match_count']}**
- joint pose successes: **{summary['joint_pose_success_count']}**
- primary prediction SHA-256: {summary['primary_predictions_sha256']}
- input manifest SHA-256: {summary['manifest_sha256']}

## Primary buckets

| Bucket | Count | Share of GT |
| --- | ---: | ---: |
{chr(10).join(bucket_lines)}

## By scene

| Scene | GT | Success | Failure |
| ---: | ---: | ---: | ---: |
{chr(10).join(scene_lines)}

## Diagnostic IoU bands

For unmatched GT, the band uses maximum overlap with any frozen detector
prediction. For matched GT, it uses the assigned greedy-match IoU.

| IoU band | GT rows |
| --- | ---: |
{chr(10).join(iou_lines)}

## Interpretation rule

Merge and split buckets use the same coverage definitions as the frozen detector
diagnostics: a merge prediction covers at least 50% of two or more GT masks; a
split GT contains at least two prediction fragments for which at least 50% of
the prediction lies inside that GT. IoU50 competition is checked before those
geometric buckets. Remaining nonzero sub-IoU50 overlap is conservatively labeled
MASK_BOUNDARY_IOU_FAILURE; zero overlap is DETECTOR_MISS.

Pose buckets use the frozen joint thresholds from the A9 protocol. A failed
primary registration is POSE_INPUT_INVALID_OR_WEAK_DEPTH only when its frozen
failure message explicitly references depth or point-cloud input; otherwise it
is POSE_REGISTRATION_FAILURE.
"""


def reconcile_counts(anchor, total, matched, joint):
    for key, actual in (("ground_truth_instance_count", total),
                        ("mask_iou50_match_count", matched), ("joint_pose_success_count", joint)):
        expected = anchor.get(key)
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0 or actual != expected:
            raise SystemExit(f"{key} reconciliation failed: {actual} != {expected}")
    if not 0 <= joint <= matched <= total:
        raise SystemExit("Impossible taxonomy population")


def extract_taxonomy(args: argparse.Namespace) -> None:
    import numpy as np
    import cv2

    if getattr(args, "result_anchor", None) is not None:
        from pose_accuracy_recovery_prep.a10_foundationpose_e2e_v2 import runtime as a9
    else:
        from pose_accuracy_recovery_prep.a9_foundationpose_e2e import runtime as a9
    from pose_accuracy_recovery_prep.real_causal_ablation_v1.runtime import (
        _greedy_matches,
        _iou_matrix,
    )
    from scripts.evaluate_m1 import (
        BOP_TOOLKIT_COMMIT,
        load_object_evaluation_data,
        load_official_models,
        official_errors,
        toolkit_commit,
    )

    protocol_path = args.protocol.resolve()
    manifest_path = args.manifest.resolve()
    primary_root = args.primary_root.resolve()
    dataset_root = args.dataset_root.resolve()
    toolkit_root = args.toolkit_root.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists():
        raise SystemExit(f"Output root is create-only: {output_root}")

    protocol = a9.load_protocol(protocol_path)
    manifest = a9.validate_input_manifest(
        manifest_path, protocol_path, verify_assets=True
    )
    if getattr(args, "result_anchor", None) is not None:
        a9.validate_primary_evidence(primary_root, manifest_path, protocol_path)
    completion_path = primary_root / "completion-receipt.json"
    predictions_path = primary_root / "predictions.jsonl"
    completion = a9._read_json(completion_path)
    if (
        completion.get("stage") != "primary-inference"
        or completion.get("status") != "COMPLETE"
        or completion.get("predictions_sha256") != a9._sha256_file(predictions_path)
        or any(
            int(completion.get(key, -1)) != 0
            for key in (
                "label_access_count", "gt_path_open_count",
                "evaluator_path_open_count", "official_scorer_run_count",
                "scene9_read_count",
            )
        )
    ):
        raise SystemExit("Primary completion receipt is absent, changed, or unsafe")

    primary_rows = a9._read_jsonl(predictions_path)
    expected_count = int(manifest["item_count"])
    if len(primary_rows) != expected_count + 1 or primary_rows[0].get("record_type") != "metadata":
        raise SystemExit("Primary prediction coverage changed")
    result_by_id = {str(row["item_id"]): row for row in primary_rows[1:]}
    if len(result_by_id) != expected_count or set(result_by_id) != {str(item["item_id"]) for item in manifest["items"]}:
        raise SystemExit("Primary prediction IDs are duplicated")

    if toolkit_commit(toolkit_root) != BOP_TOOLKIT_COMMIT:
        raise SystemExit("Pinned BOP Toolkit changed")
    model_params, model_info = load_official_models(dataset_root)
    object_data = {
        object_id: load_object_evaluation_data(object_id, model_params, model_info)
        for object_id in sorted(
            set(int(frame["object_id"]) for frame in manifest["frames"])
        )
    }

    items_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in manifest["items"]:
        items_by_frame[str(item["frame_id"])].append(item)

    threshold = float(protocol["evaluation"]["mask_iou_threshold"])
    pose_threshold = protocol["evaluation"]["joint_pose_threshold"]
    mssd_threshold = float(pose_threshold["mssd_fraction_of_diameter"])
    mspd_threshold = float(pose_threshold["mspd_pixels"])

    taxonomy_rows: list[dict[str, Any]] = []
    mask_match_count = 0
    joint_count = 0

    for frame in manifest["frames"]:
        frame_id = str(frame["frame_id"])
        scene_id = int(frame["scene_id"])
        image_id = int(frame["image_id"])
        object_id = int(frame["object_id"])
        scene_root = dataset_root / "val" / f"{scene_id:06d}"
        gt_rows = a9._read_json(
            scene_root / "scene_gt_realsense.json"
        ).get(str(image_id))
        if not isinstance(gt_rows, list) or not gt_rows:
            raise SystemExit(f"Development GT is missing: {frame_id}")

        gt_masks = []
        gt_poses = []
        for gt_index, gt_row in enumerate(gt_rows):
            if int(gt_row.get("obj_id", -1)) != object_id:
                raise SystemExit(f"Task-target object mapping changed: {frame_id}")
            mask_path = (
                scene_root / "mask_visib_realsense"
                / f"{image_id:06d}_{gt_index:06d}.png"
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or not np.any(mask):
                raise SystemExit(f"Development visible mask missing: {frame_id}/{gt_index}")
            gt_masks.append(mask > 0)
            rotation = np.asarray(gt_row["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            translation_m = np.asarray(
                gt_row["cam_t_m2c"], dtype=np.float64
            ).reshape(3) * 0.001
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation_m
            gt_poses.append(pose)

        pred_items = sorted(
            items_by_frame[frame_id],
            key=lambda item: int(item["prediction_index"]),
        )
        masks_path = a9._resolve_bound(
            Path(str(manifest["predictions_root"])), frame["prediction_masks"]
        )
        all_masks = a9._unpack_masks(masks_path)
        pred_masks = [
            all_masks[int(item["prediction_index"])] for item in pred_items
        ]

        matrix = _iou_matrix(gt_masks, pred_masks)
        matches = _greedy_matches(matrix, threshold)
        match_by_gt = {
            int(gt_index): (int(pred_index), float(iou))
            for gt_index, pred_index, iou in matches
        }
        mask_match_count += len(matches)

        merge_pred_indices = set()
        for pred_index, pred_mask in enumerate(pred_masks):
            pred_bool = np.asarray(pred_mask, dtype=bool)
            covered_gt = sum(
                int(np.logical_and(pred_bool, gt_mask).sum())
                / max(1, int(gt_mask.sum())) >= 0.5
                for gt_mask in gt_masks
            )
            if covered_gt >= 2:
                merge_pred_indices.add(pred_index)

        split_gt_indices = set()
        for gt_index, gt_mask in enumerate(gt_masks):
            fragments = sum(
                int(np.logical_and(np.asarray(pred_mask, dtype=bool), gt_mask).sum())
                / max(1, int(np.asarray(pred_mask, dtype=bool).sum())) >= 0.5
                for pred_mask in pred_masks
            )
            if fragments >= 2:
                split_gt_indices.add(gt_index)

        K = np.asarray(frame["camera_intrinsics_row_major"], dtype=np.float64)

        for gt_index, gt_mask in enumerate(gt_masks):
            matched = match_by_gt.get(gt_index)
            max_iou = float(np.max(matrix[gt_index])) if matrix.shape[1] else 0.0
            candidate_indices = (
                [int(i) for i in np.where(matrix[gt_index] >= threshold)[0]]
                if matrix.shape[1] else []
            )
            best_pred_index = (
                int(np.argmax(matrix[gt_index])) if matrix.shape[1] else None
            )

            merge_evidence = False
            for pred_index in merge_pred_indices:
                pred_bool = np.asarray(pred_masks[pred_index], dtype=bool)
                intersection = int(np.logical_and(pred_bool, gt_mask).sum())
                if intersection / max(1, int(gt_mask.sum())) >= 0.5:
                    merge_evidence = True
                    break
            split_evidence = gt_index in split_gt_indices

            base = {
                "scene_id": scene_id, "image_id": image_id, "gt_index": gt_index,
                "object_id": object_id, "matched_prediction_index": "",
                "item_id": "", "detector_score": "", "mask_iou": "",
                "max_mask_iou": max_iou, "iou_band": _iou_band(max_iou),
                "mask_match_iou50": False, "merge_evidence": merge_evidence,
                "split_evidence": split_evidence, "pose_completed": False,
                "primary_failure_type": "", "normalized_mssd": "", "mspd_px": "",
                "joint_pose_success": False, "failure_bucket": "UNRESOLVED",
                "notes": "",
            }

            if matched is None:
                base["failure_bucket"] = classify_unmatched(
                    has_iou50_candidate=bool(candidate_indices),
                    merge_evidence=merge_evidence,
                    split_evidence=split_evidence,
                    max_iou=max_iou,
                )
                if best_pred_index is not None:
                    base["notes"] = f"best_overlap_prediction_index={best_pred_index}"
                taxonomy_rows.append(base)
                continue

            pred_index, mask_iou = matched
            item = pred_items[pred_index]
            result = result_by_id[str(item["item_id"])]
            base.update({
                "matched_prediction_index": int(item["prediction_index"]),
                "item_id": str(item["item_id"]),
                "detector_score": float(item["detector_score"]),
                "mask_iou": mask_iou, "max_mask_iou": mask_iou,
                "iou_band": _iou_band(mask_iou), "mask_match_iou50": True,
            })

            if result.get("status") != "success":
                base["primary_failure_type"] = str(
                    result.get("failure_type", "UNKNOWN")
                )
                base["failure_bucket"] = (
                    "POSE_INPUT_INVALID_OR_WEAK_DEPTH"
                    if _looks_like_depth_input_failure(result)
                    else "POSE_REGISTRATION_FAILURE"
                )
                base["notes"] = str(result.get("failure_message", ""))[:300]
                taxonomy_rows.append(base)
                continue

            base["pose_completed"] = True
            mssd_mm, mspd_px = official_errors(
                np.asarray(
                    result["predicted_model_to_camera_pose_m"], dtype=np.float64
                ),
                gt_poses[gt_index], K, object_data[object_id],
            )
            normalized_mssd = (
                mssd_mm / float(object_data[object_id]["diameter_mm"])
            )
            bucket, joint = classify_pose(
                normalized_mssd, mspd_px,
                mssd_threshold=mssd_threshold, mspd_threshold=mspd_threshold,
            )
            base["normalized_mssd"] = normalized_mssd
            base["mspd_px"] = mspd_px
            base["joint_pose_success"] = joint
            base["failure_bucket"] = bucket
            joint_count += int(joint)
            taxonomy_rows.append(base)

    anchor_path = getattr(args, "result_anchor", None)
    if anchor_path is not None:
        anchor = _read_json(anchor_path)
        for key, path in (("protocol_sha256", protocol_path), ("manifest_sha256", manifest_path),
                          ("predictions_sha256", predictions_path),
                          ("primary_completion_receipt_sha256", completion_path)):
            if anchor.get(key) != _sha256_file(path):
                raise SystemExit(f"Result anchor does not bind current artifacts: {key}")
        if anchor.get("schema_version") != a9.EVALUATION_SCHEMA:
            raise SystemExit("Result anchor is not a v1.2 evaluation")
        reconcile_counts(anchor, len(taxonomy_rows), mask_match_count, joint_count)
    else:
        release = _read_json(RELEASE_RESULT)
        reconcile_counts({"ground_truth_instance_count": release["dataset"]["ground_truth_instance_count"],
                          "mask_iou50_match_count": release["pose_pipeline"]["mask_iou50_matches"],
                          "joint_pose_success_count": release["pose_pipeline"]["joint_pose_successes"]},
                         len(taxonomy_rows), mask_match_count, joint_count)

    bucket_counts = Counter(str(row["failure_bucket"]) for row in taxonomy_rows)
    iou_counts = Counter(str(row["iou_band"]) for row in taxonomy_rows)
    scene_summary = {}
    for scene_id in sorted({int(row["scene_id"]) for row in taxonomy_rows}):
        scene_rows = [
            row for row in taxonomy_rows if int(row["scene_id"]) == scene_id
        ]
        scene_summary[str(scene_id)] = {
            "gt": len(scene_rows),
            "joint_successes": sum(
                bool(row["joint_pose_success"]) for row in scene_rows
            ),
            "failures": sum(
                row["failure_bucket"] != "SUCCESS" for row in scene_rows
            ),
        }

    summary = {
        "schema_version": "poseloop.v1.2.failure-taxonomy.v1" if anchor_path else "poseloop.v1.1.0.failure-taxonomy.v1",
        "status": "COMPLETE_DIAGNOSTIC_RECONSTRUCTION",
        "claim_scope": (
            "post-hoc diagnostic of already-consumed XYZ-IBD development split; "
            "not an untouched evaluation and not a retuning authorization"
        ),
        "ground_truth_instance_count": len(taxonomy_rows),
        "mask_iou50_match_count": mask_match_count,
        "joint_pose_success_count": joint_count,
        "bucket_counts": dict(bucket_counts),
        "iou_band_counts": dict(iou_counts),
        "per_scene": scene_summary,
        "manifest_sha256": _sha256_file(manifest_path),
        "primary_predictions_sha256": _sha256_file(predictions_path),
        "primary_completion_receipt_sha256": _sha256_file(completion_path),
        "protocol_sha256": _sha256_file(protocol_path),
    }

    if anchor_path is not None:
        summary["result_anchor_sha256"] = _sha256_file(anchor_path)
    output_root.mkdir(parents=True)
    _write_csv(output_root / "per-instance.csv", taxonomy_rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "REPORT.md").write_text(
        _summary_markdown(taxonomy_rows, summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def self_test() -> None:
    pose_cases = [
        (0.01, 1.0, ("SUCCESS", True)),
        (0.20, 1.0, ("POSE_MSSD_ONLY_FAILURE", False)),
        (0.01, 20.0, ("POSE_MSPD_ONLY_FAILURE", False)),
        (0.20, 20.0, ("POSE_MSSD_AND_MSPD_FAILURE", False)),
    ]
    for mssd, mspd, expected in pose_cases:
        actual = classify_pose(
            mssd, mspd, mssd_threshold=0.10, mspd_threshold=10.0
        )
        if actual != expected:
            raise AssertionError((actual, expected))

    unmatched_cases = [
        (
            dict(has_iou50_candidate=True, merge_evidence=True,
                 split_evidence=True, max_iou=0.6),
            "DUPLICATE_OR_MATCH_COMPETITION",
        ),
        (
            dict(has_iou50_candidate=False, merge_evidence=True,
                 split_evidence=False, max_iou=0.4),
            "UNDER_SEGMENTATION_OR_MERGE",
        ),
        (
            dict(has_iou50_candidate=False, merge_evidence=False,
                 split_evidence=True, max_iou=0.3),
            "OVER_SEGMENTATION",
        ),
        (
            dict(has_iou50_candidate=False, merge_evidence=False,
                 split_evidence=False, max_iou=0.2),
            "MASK_BOUNDARY_IOU_FAILURE",
        ),
        (
            dict(has_iou50_candidate=False, merge_evidence=False,
                 split_evidence=False, max_iou=0.0),
            "DETECTOR_MISS",
        ),
    ]
    for kwargs, expected in unmatched_cases:
        actual = classify_unmatched(**kwargs)
        if actual != expected:
            raise AssertionError((actual, expected))
    print("failure-taxonomy self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status")
    status.add_argument("--output", type=Path)
    status.add_argument("--check", type=Path)

    commands.add_parser("self-test")

    extract = commands.add_parser("extract")
    extract.add_argument("--protocol", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--primary-root", type=Path, required=True)
    extract.add_argument("--dataset-root", type=Path, required=True)
    extract.add_argument("--toolkit-root", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--result-anchor", type=Path)

    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return
    if args.command == "status":
        rendered = status_markdown()
        if args.check is not None:
            if args.check.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"Generated status is stale: {args.check}")
        if args.output is not None:
            args.output.write_text(rendered, encoding="utf-8")
        if args.output is None and args.check is None:
            print(rendered, end="")
        return
    extract_taxonomy(args)


if __name__ == "__main__":
    main()
