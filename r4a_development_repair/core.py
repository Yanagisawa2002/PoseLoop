"""Contracts and label-free structural audits for PoseLoop R4-A.

This module deliberately has no evaluator entrypoint.  In particular, it never
opens XYZ-IBD validation ground truth or score JSON files.  Development labels
are handled by :mod:`r4a_development_repair.development` only after a pre-freeze
receipt has been created.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import PROTOCOL_ID


PROTOCOL_SCHEMA = "poseloop.r4a.development.protocol.v1"
PREFREEZE_SCHEMA = "poseloop.r4a.development.prefreeze.v1"
STRUCTURAL_AUDIT_SCHEMA = "poseloop.r4a.structural-postmortem.v1"


class ContractError(RuntimeError):
    """Raised when an immutable R4-A contract is violated."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict):
        raise ContractError("R4-A protocol must be a JSON object")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("R4-A protocol schema mismatch")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A protocol ID mismatch")
    if protocol.get("state") != "frozen_before_any_development_label_access":
        raise ContractError("R4-A protocol is not frozen before label access")
    role = protocol.get("role_declaration", {})
    if role.get("role") != "DEVELOPMENT_ONLY":
        raise ContractError("R4-A split role must be DEVELOPMENT_ONLY")
    if role.get("sealed_split_eligible") is not False:
        raise ContractError("Development data must never be eligible as sealed data")
    boundaries = protocol.get("immutable_boundaries", {})
    required_false = (
        "xyzibd_val_evaluator_permitted",
        "xyzibd_val_gt_access_permitted",
        "v3_result_reinterpretation_permitted",
        "checkpoint_change_permitted",
        "score_guided_pose_selection_permitted",
    )
    for key in required_false:
        if boundaries.get(key) is not False:
            raise ContractError(f"Immutable boundary changed: {key}")
    selection = protocol.get("development_split", {}).get("selection", {})
    if selection.get("scene_ids") != [0, 1, 2]:
        raise ContractError("Development scene selection changed")
    if selection.get("image_ids_per_scene") != [0, 1, 2, 3]:
        raise ContractError("Development image selection changed")
    if selection.get("target_policy") != "all_gt_instances_in_source_order":
        raise ContractError("Development target selection changed")
    archives = protocol.get("development_split", {}).get("archives", [])
    if len(archives) != 2:
        raise ContractError("Expected the two official split ZIP parts")
    for archive in archives:
        if not _valid_sha256(archive.get("sha256")) or int(archive.get("bytes", 0)) <= 0:
            raise ContractError("Invalid official archive hash contract")
    network = protocol.get("development_split", {}).get("network_contract", {})
    if network.get("whole_archive_download_permitted") is not False:
        raise ContractError("Whole train_pbr archive download must remain forbidden")
    if network.get("http_range_required") is not True:
        raise ContractError("Development slice must use auditable HTTP ranges")
    gates = protocol.get("development_gates", {})
    if gates.get("target_coverage") != 1.0:
        raise ContractError("Target coverage gate must remain 100%")
    if gates.get("stop_after_failed_coverage_or_geometry_gate") is not True:
        raise ContractError("Failed development gate must stop further tuning")
    return protocol


def _rotation_quality(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 9:
        raise ContractError("BOP rotation must contain nine values")
    r = [[float(values[row * 3 + col]) for col in range(3)] for row in range(3)]
    if not all(math.isfinite(value) for row in r for value in row):
        return math.nan, math.inf
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    error_squared = 0.0
    for row in range(3):
        for col in range(3):
            dot = sum(r[k][row] * r[k][col] for k in range(3))
            target = 1.0 if row == col else 0.0
            error_squared += (dot - target) ** 2
    return determinant, math.sqrt(error_squared)


def _parse_numbers(value: str, expected: int) -> list[float]:
    result = [float(item) for item in value.split()]
    if len(result) != expected:
        raise ContractError(f"Expected {expected} numeric values, got {len(result)}")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pose_csv_audit(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    determinants: list[float] = []
    orthogonality: list[float] = []
    translation_norms: list[float] = []
    translation_z: list[float] = []
    invalid = 0
    keys: list[tuple[int, int, int]] = []
    for row in rows:
        rotation = _parse_numbers(str(row["R"]), 9)
        translation = _parse_numbers(str(row["t"]), 3)
        determinant, error = _rotation_quality(rotation)
        finite = all(math.isfinite(value) for value in (*rotation, *translation))
        legal = finite and abs(determinant - 1.0) <= 1e-3 and error <= 1e-3
        if not legal:
            invalid += 1
        determinants.append(determinant)
        orthogonality.append(error)
        translation_norms.append(math.sqrt(sum(value * value for value in translation)))
        translation_z.append(translation[2])
        keys.append((int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"])))
    return {
        "row_count": len(rows),
        "unique_scene_ids": sorted({key[0] for key in keys}),
        "unique_image_count": len({(key[0], key[1]) for key in keys}),
        "unique_object_ids": sorted({key[2] for key in keys}),
        "unique_key_count": len(set(keys)),
        "duplicate_key_count": len(keys) - len(set(keys)),
        "invalid_se3_count": invalid,
        "determinant_min": min(determinants) if determinants else None,
        "determinant_max": max(determinants) if determinants else None,
        "orthogonality_error_min": min(orthogonality) if orthogonality else None,
        "orthogonality_error_max": max(orthogonality) if orthogonality else None,
        "translation_norm_mm_min": min(translation_norms) if translation_norms else None,
        "translation_norm_mm_max": max(translation_norms) if translation_norms else None,
        "translation_z_mm_min": min(translation_z) if translation_z else None,
        "translation_z_mm_max": max(translation_z) if translation_z else None,
    }


def audit_frozen_predictions(
    *,
    coco_path: Path,
    association_path: Path,
    single_path: Path,
    multi_path: Path,
    provenance_path: Path,
    expected_sha256: Mapping[str, str],
    frozen_public_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit only frozen predictions and public aggregate domains.

    No dataset root is accepted by this API, making accidental validation-label
    access structurally impossible.
    """

    paths = [coco_path, association_path, single_path, multi_path, provenance_path]
    files: dict[str, Any] = {}
    errors: list[str] = []
    for path in paths:
        actual = sha256_file(path)
        expected = expected_sha256.get(path.name)
        item = {"bytes": path.stat().st_size, "sha256": actual, "expected_sha256": expected}
        if actual != expected:
            item["error"] = "SHA-256 mismatch"
            errors.append(f"frozen prediction changed: {path.name}")
        files[path.name] = item

    coco = read_json(coco_path)
    if not isinstance(coco, list):
        raise ContractError("COCO prediction input must be an array")
    association = []
    with association_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                association.append(json.loads(line))
    single = _read_csv(single_path)
    multi = _read_csv(multi_path)
    provenance = read_json(provenance_path)

    detection_keys = [
        (int(row["scene_id"]), int(row["image_id"]), int(row["category_id"]))
        for row in coco
    ]
    scores = [float(row["score"]) for row in coco]
    times = [float(row["time"]) for row in coco if "time" in row]
    bbox_invalid = sum(
        not isinstance(row.get("bbox"), list)
        or len(row["bbox"]) != 4
        or any(not math.isfinite(float(value)) for value in row["bbox"])
        or float(row["bbox"][2]) <= 0
        or float(row["bbox"][3]) <= 0
        for row in coco
    )
    detection_indices = [int(row["detection_index"]) for row in association]
    association_mismatch = sum(
        index < 0
        or index >= len(coco)
        or (
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["category_id"]),
        )
        != detection_keys[index]
        for row, index in zip(association, detection_indices)
    )
    single_audit = _pose_csv_audit(single)
    multi_audit = _pose_csv_audit(multi)
    total_images = int(frozen_public_summary["image_count"])
    public_scenes = [int(value) for value in frozen_public_summary["scene_ids"]]
    public_objects = [int(value) for value in frozen_public_summary["object_ids"]]
    unique_images = {(key[0], key[1]) for key in detection_keys}
    unique_scenes = {key[0] for key in detection_keys}
    unique_objects = {key[2] for key in detection_keys}
    return {
        "schema_version": STRUCTURAL_AUDIT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "label_free_predictions_and_public_aggregate_domains_only",
        "files": files,
        "errors": errors,
        "ready": not errors,
        "coco": {
            "row_count": len(coco),
            "unique_scene_ids": sorted(unique_scenes),
            "unique_image_count": len(unique_images),
            "unique_object_ids": sorted(unique_objects),
            "unique_key_count": len(set(detection_keys)),
            "duplicate_key_count": len(detection_keys) - len(set(detection_keys)),
            "rows_per_image": dict(sorted(Counter(f"{a:06d}/{b:06d}" for a, b, _ in detection_keys).items())),
            "bbox_invalid_count": bbox_invalid,
            "segmentation_rle_count": sum(isinstance(row.get("segmentation"), dict) for row in coco),
            "score_min": min(scores) if scores else None,
            "score_median": _quantile(scores, 0.5),
            "score_max": max(scores) if scores else None,
            "time_seconds_min": min(times) if times else None,
            "time_seconds_max": max(times) if times else None,
        },
        "association": {
            "row_count": len(association),
            "covered_detection_count": len(set(detection_indices)),
            "missing_detection_indices": sorted(set(range(len(coco))) - set(detection_indices)),
            "join_mismatch_count": association_mismatch,
            "track_count": len({str(row["predicted_track_id"]) for row in association}),
            "target_view_count": sum(bool(row.get("is_target_view")) for row in association),
            "view_rank_min": min((int(row["view_rank"]) for row in association), default=None),
            "view_rank_max": max((int(row["view_rank"]) for row in association), default=None),
        },
        "single_view": single_audit,
        "multi_view": multi_audit,
        "provenance": {
            "producer": provenance.get("producer"),
            "producer_version": provenance.get("producer_version"),
            "label_access_count": provenance.get("label_access_count"),
            "uses_gt_visible_mask": provenance.get("uses_gt_visible_mask"),
            "uses_oracle_association": provenance.get("uses_oracle_association"),
        },
        "coverage": {
            "image": {"covered": len(unique_images), "total": total_images, "fraction": len(unique_images) / total_images},
            "scene_domain": {"covered": len(unique_scenes), "total": len(public_scenes), "fraction": len(unique_scenes) / len(public_scenes)},
            "object_domain": {"covered": len(unique_objects), "total": len(public_objects), "fraction": len(unique_objects) / len(public_objects)},
        },
        "diagnosis": {
            "primary": "global_eight_item_fixed_object_smoke_enumerator",
            "geometry_observation": "all_emitted_poses_are_finite_legal_se3_in_bop_mm_representation",
            "unit_chain": "cad_mm_to_foundationpose_m; depth_and_camera_mm_to_m; model_to_camera_output_m; bop_translation_m_to_mm",
            "missing_prediction_behavior": "non_enumerated_targets_create_no_work_item_and_no_explicit_failure_record",
        },
    }


def build_prefreeze_receipt(
    *,
    protocol_path: Path,
    implementation_commit: str,
    repository_tree: str,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if len(implementation_commit) != 40 or not all(c in "0123456789abcdef" for c in implementation_commit):
        raise ContractError("Implementation commit must be a full Git SHA")
    if len(repository_tree) != 40 or not all(c in "0123456789abcdef" for c in repository_tree):
        raise ContractError("Repository tree must be a full Git SHA")
    receipt = {
        "schema_version": PREFREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "state": "frozen_before_any_development_label_access",
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": implementation_commit,
        "repository_tree": repository_tree,
        "development_label_access_count_at_freeze": 0,
        "xyzibd_val_label_access_count": 0,
        "official_evaluator_invocation_count": 0,
        "selection": protocol["development_split"]["selection"],
        "archives": protocol["development_split"]["archives"],
        "network_contract": protocol["development_split"]["network_contract"],
        "immutable_boundaries": protocol["immutable_boundaries"],
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output_path, receipt)
    return receipt


def verify_prefreeze_receipt(path: Path, protocol_path: Path) -> dict[str, Any]:
    receipt = read_json(path)
    if receipt.get("schema_version") != PREFREEZE_SCHEMA:
        raise ContractError("Pre-freeze receipt schema mismatch")
    if receipt.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Pre-freeze receipt protocol mismatch")
    expected = dict(receipt)
    actual_lock = expected.pop("lock_sha256", None)
    if actual_lock != canonical_sha256(expected):
        raise ContractError("Pre-freeze receipt hash is invalid")
    if receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise ContractError("Protocol changed after pre-freeze")
    if receipt.get("development_label_access_count_at_freeze") != 0:
        raise ContractError("Development labels were opened before freeze")
    if receipt.get("xyzibd_val_label_access_count") != 0:
        raise ContractError("XYZ-IBD validation labels were accessed")
    return receipt
