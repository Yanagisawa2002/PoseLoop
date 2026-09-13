"""Run the frozen A-R9 detector output through FoundationPose and evaluate it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pose_accuracy_recovery_prep.real_instance_detector_v1.runtime import (
    _unpack_masks,
)


SCHEMA = "poseloop.a-r9-foundationpose-e2e.protocol.v1"
PROTOCOL_ID = "poseloop.pose-accuracy-recovery.a-r9-foundationpose-e2e.v1"
INPUT_SCHEMA = "poseloop.a-r9-foundationpose-e2e.input-manifest.v1"
RUN_LOCK_SCHEMA = "poseloop.a-r9-foundationpose-e2e.run-lock.v1"
RESULT_SCHEMA = "poseloop.a-r9-foundationpose-e2e.result.v1"
RECEIPT_SCHEMA = "poseloop.a-r9-foundationpose-e2e.receipt.v1"
EVALUATION_SCHEMA = "poseloop.a-r9-foundationpose-e2e.evaluation.v1"
EXPECTED_RUN_XYZIBD_SHA256 = (
    "979622519b98f6249cb69af932180495716cd047303533943c10ceedb37c5b4a"
)
EXPECTED_MEMORY_PATCH_SHA256 = (
    "1acf3bad5e64950831d1b137d8fa89ca1182c482952cfb42d293581077a7f1d4"
)


class ContractError(RuntimeError):
    """An immutable experiment contract was violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _without_lock(value: Mapping[str, Any], lock_key: str) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key != lock_key}


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ContractError(f"Blank JSONL row {line_number}: {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractError(f"Non-object JSONL row {line_number}: {path}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read JSONL: {path}") from exc
    return rows


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bound_file(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"Bound asset is missing or is a symlink: {resolved}")
    relative = (
        resolved.relative_to(root.resolve()).as_posix()
        if root is not None
        else resolved.as_posix()
    )
    return {
        "relative_path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _resolve_bound(root: Path, record: Mapping[str, Any]) -> Path:
    relative = record.get("relative_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ContractError("Bound asset relative path is invalid")
    path = (root.resolve() / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ContractError("Bound asset escapes its root")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(record.get("size_bytes", -1))
        or _sha256_file(path) != record.get("sha256")
    ):
        raise ContractError(f"Bound asset changed: {relative}")
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path.resolve())
    if (
        protocol.get("schema_version") != SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
    ):
        raise ContractError("A-R9 FoundationPose protocol identity changed")
    dataset = protocol.get("dataset")
    detector = protocol.get("upstream_detector")
    foundationpose = protocol.get("foundationpose")
    boundary = protocol.get("runtime_boundary")
    failure = protocol.get("failure_policy")
    if not all(
        isinstance(value, dict)
        for value in (dataset, detector, foundationpose, boundary, failure)
    ):
        raise ContractError("A-R9 FoundationPose protocol section is missing")
    if (
        dataset.get("scenes") != [10, 25, 30, 40, 65]
        or dataset.get("image_ids") != [0, 10, 20, 30, 40]
        or int(dataset.get("frame_count", -1)) != 25
        or int(dataset.get("ground_truth_instance_count", -1)) != 770
        or dataset.get("scene9_read_permitted") is not False
        or protocol.get("scene9_replay_permitted") is not False
        or protocol.get("sealed_claim_permitted") is not False
    ):
        raise ContractError("Frozen development split or scene-9 boundary changed")
    scene_objects = {
        str(key): int(value)
        for key, value in dataset.get("scene_object_ids", {}).items()
    }
    if scene_objects != {"10": 2, "25": 1, "30": 5, "40": 4, "65": 6}:
        raise ContractError("Frozen task-target object map changed")
    for key in (
        "a9_dataset_manifest_sha256",
        "a9_prediction_manifest_sha256",
        "a9_protocol_sha256",
    ):
        if not _is_sha256(dataset.get(key)):
            raise ContractError(f"Invalid frozen dataset digest: {key}")
    if (
        float(detector.get("operating_score_threshold", -1)) != 0.25
        or int(detector.get("expected_ranked_prediction_count", -1)) != 997
        or int(detector.get("expected_operating_prediction_count", -1)) != 820
        or detector.get("threshold_tuning_permitted") is not False
    ):
        raise ContractError("A-R9 operating policy changed")
    if (
        foundationpose.get("commit") != "a1b694b83e633c2cb6115b9063d940a687759392"
        or int(foundationpose.get("iterations", -1)) != 5
        or int(foundationpose.get("seed", -1)) != 0
        or int(foundationpose.get("candidate_count", -1)) != 252
        or foundationpose.get("resource_batches")
        != {"warp": 32, "refine": 32, "score_data": 8, "score_feature": 32}
        or foundationpose.get("candidate_pruning_permitted") is not False
        or foundationpose.get("parameter_tuning_permitted") is not False
    ):
        raise ContractError("Frozen FoundationPose inference policy changed")
    for checkpoint_key in ("refiner_checkpoint", "scorer_checkpoint"):
        checkpoint = foundationpose.get(checkpoint_key)
        if (
            not isinstance(checkpoint, dict)
            or not _is_sha256(checkpoint.get("sha256"))
            or int(checkpoint.get("size_bytes", 0)) <= 0
        ):
            raise ContractError(
                f"Invalid FoundationPose checkpoint lock: {checkpoint_key}"
            )
    if (
        boundary.get("prediction_before_label_open_required") is not True
        or boundary.get("task_target_object_id_is_public_input") is not True
        or boundary.get("proposal_to_gt_association_permitted") is not False
        or any(
            int(boundary.get(key, -1)) != 0
            for key in (
                "runtime_gt_path_open_count",
                "runtime_evaluator_path_open_count",
                "runtime_official_scorer_run_count",
            )
        )
    ):
        raise ContractError("Runtime label isolation boundary changed")
    if (
        int(failure.get("oracle_diagnostic_run_limit", -1)) != 1
        or int(failure.get("rescue_run_limit", -1)) != 1
        or failure.get("threshold_tuning_for_rescue_permitted") is not False
        or failure.get("second_rescue_permitted") is not False
    ):
        raise ContractError("One-oracle/one-rescue policy changed")
    return protocol


def _camera_record(path: Path, image_id: int) -> dict[str, Any]:
    camera = _read_json(path)
    row = camera.get(str(image_id))
    if not isinstance(row, dict):
        raise ContractError("Public camera row is missing")
    intrinsics = np.asarray(row.get("cam_K"), dtype=np.float64)
    if intrinsics.size != 9:
        raise ContractError("Public camera intrinsics changed")
    intrinsics = intrinsics.reshape(3, 3)
    depth_scale = float(row.get("depth_scale", float("nan")))
    if (
        not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-8, rtol=0.0)
        or not math.isfinite(depth_scale)
        or depth_scale <= 0
    ):
        raise ContractError("Public camera calibration is invalid")
    return {
        "camera_intrinsics_row_major": intrinsics.tolist(),
        "raw_depth_scale_to_mm": depth_scale,
    }


def freeze_inputs(
    *,
    protocol_path: Path,
    dataset_root: Path,
    dataset_manifest_path: Path,
    predictions_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    dataset_root = dataset_root.resolve()
    dataset_manifest_path = dataset_manifest_path.resolve()
    predictions_root = predictions_root.resolve()
    output_root = output_root.resolve()
    protocol = load_protocol(protocol_path)
    if output_root.exists():
        raise ContractError("Input freeze output root is create-only")
    if (
        _sha256_file(dataset_manifest_path)
        != protocol["dataset"]["a9_dataset_manifest_sha256"]
    ):
        raise ContractError("Frozen A-R9 dataset manifest changed")
    prediction_manifest_path = predictions_root / "prediction-manifest.json"
    if (
        _sha256_file(prediction_manifest_path)
        != protocol["dataset"]["a9_prediction_manifest_sha256"]
    ):
        raise ContractError("Frozen A-R9 prediction manifest changed")
    upstream_manifest = _read_json(dataset_manifest_path)
    prediction_manifest = _read_json(prediction_manifest_path)
    if (
        int(prediction_manifest.get("frame_count", -1)) != 25
        or prediction_manifest.get("scene9_read_count") != 0
        or prediction_manifest.get("fixed_evaluation_gt_access_count") != 0
        or prediction_manifest.get("protocol_sha256")
        != protocol["dataset"]["a9_protocol_sha256"]
    ):
        raise ContractError("A-R9 prediction boundary changed")
    source_rows = {
        str(row["frame_id"]): row
        for row in upstream_manifest.get("frames", {}).get("fixed_evaluation", [])
    }
    prediction_rows = {
        str(row["frame_id"]): row for row in prediction_manifest.get("frames", [])
    }
    expected_frames = [
        f"s{scene_id:06d}-i{image_id:06d}"
        for scene_id in protocol["dataset"]["scenes"]
        for image_id in protocol["dataset"]["image_ids"]
    ]
    if sorted(source_rows) != sorted(expected_frames) or sorted(
        prediction_rows
    ) != sorted(expected_frames):
        raise ContractError("A-R9 frame coverage changed")
    items: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    threshold = float(protocol["upstream_detector"]["operating_score_threshold"])
    for frame_id in expected_frames:
        source = source_rows[frame_id]
        predicted = prediction_rows[frame_id]
        scene_id = int(source["scene_id"])
        image_id = int(source["image_id"])
        if (
            scene_id == 9
            or int(predicted["scene_id"]) != scene_id
            or int(predicted["image_id"]) != image_id
        ):
            raise ContractError("A-R9 frame identity changed")
        object_id = int(protocol["dataset"]["scene_object_ids"][str(scene_id)])
        rgb_path = dataset_root / source["rgb"]["relative_path"]
        if _sha256_file(rgb_path) != source["rgb"][
            "sha256"
        ] or rgb_path.stat().st_size != int(source["rgb"]["size_bytes"]):
            raise ContractError(f"A-R9 RGB changed: {frame_id}")
        scene_root = dataset_root / "val" / f"{scene_id:06d}"
        depth_path = scene_root / "depth_realsense" / f"{image_id:06d}.png"
        camera_path = scene_root / "scene_camera_realsense.json"
        cad_path = dataset_root / "models" / f"obj_{object_id:06d}.ply"
        metadata_path = _resolve_bound(predictions_root, predicted["metadata"])
        masks_path = _resolve_bound(predictions_root, predicted["masks"])
        metadata = _read_json(metadata_path)
        masks = _unpack_masks(masks_path)
        scores = [float(score) for score in metadata.get("scores", [])]
        if (
            metadata.get("frame_id") != frame_id
            or int(metadata.get("prediction_count", -1)) != len(masks)
            or len(scores) != len(masks)
        ):
            raise ContractError(f"A-R9 prediction frame bundle changed: {frame_id}")
        try:
            import cv2
        except ImportError as exc:
            raise ContractError("Input freezing requires OpenCV") from exc
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None or depth_raw.ndim != 2:
            raise ContractError(f"Cannot decode public depth: {frame_id}")
        camera = _camera_record(camera_path, image_id)
        depth_m = (
            depth_raw.astype(np.float32)
            * float(camera["raw_depth_scale_to_mm"])
            * 0.001
        )
        frame_item_ids: list[str] = []
        for prediction_index, (score, mask) in enumerate(
            zip(scores, masks, strict=True)
        ):
            if score < threshold:
                continue
            mask = np.ascontiguousarray(mask, dtype=bool)
            if mask.shape != depth_raw.shape:
                raise ContractError(f"A-R9 mask shape changed: {frame_id}")
            nonzero = int(mask.sum())
            valid = mask & np.isfinite(depth_m) & (depth_m > 0)
            valid_count = int(valid.sum())
            if nonzero <= 0:
                raise ContractError(f"A-R9 operating mask is empty: {frame_id}")
            item_id = f"{frame_id}-p{prediction_index:06d}"
            item = {
                "item_id": item_id,
                "frame_id": frame_id,
                "scene_id": scene_id,
                "image_id": image_id,
                "object_id": object_id,
                "prediction_index": prediction_index,
                "detector_score": score,
                "mask_nonzero_pixels": nonzero,
                "valid_depth_pixels": valid_count,
                "mask_sha256": hashlib.sha256(
                    mask.astype(np.uint8).tobytes()
                ).hexdigest(),
            }
            items.append(item)
            frame_item_ids.append(item_id)
        frames.append(
            {
                "frame_id": frame_id,
                "scene_id": scene_id,
                "image_id": image_id,
                "object_id": object_id,
                "height": int(source["height"]),
                "width": int(source["width"]),
                "rgb": _bound_file(rgb_path, root=dataset_root),
                "depth": _bound_file(depth_path, root=dataset_root),
                "camera": _bound_file(camera_path, root=dataset_root),
                "cad": _bound_file(cad_path, root=dataset_root),
                "prediction_metadata": _bound_file(
                    metadata_path, root=predictions_root
                ),
                "prediction_masks": _bound_file(masks_path, root=predictions_root),
                **camera,
                "item_ids": frame_item_ids,
            }
        )
    if len(items) != int(
        protocol["upstream_detector"]["expected_operating_prediction_count"]
    ):
        raise ContractError(f"Expected 820 operating predictions, got {len(items)}")
    manifest: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_role": "ALREADY_CONSUMED_REAL_DEVELOPMENT",
        "dataset_root": dataset_root.as_posix(),
        "predictions_root": predictions_root.as_posix(),
        "dataset_manifest": _bound_file(dataset_manifest_path),
        "prediction_manifest": _bound_file(
            prediction_manifest_path, root=predictions_root
        ),
        "frame_count": len(frames),
        "item_count": len(items),
        "frames": frames,
        "items": items,
        "runtime_boundary": {
            "label_access_count": 0,
            "gt_path_open_count": 0,
            "evaluator_path_open_count": 0,
            "official_scorer_run_count": 0,
            "scene9_read_count": 0,
        },
    }
    manifest["manifest_lock_sha256"] = _canonical_sha256(
        _without_lock(manifest, "manifest_lock_sha256")
    )
    output_root.mkdir(parents=True)
    manifest_path = output_root / "input-manifest.json"
    _write_json_atomic(manifest_path, manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "stage": "freeze-inputs",
        "status": "PASS_FROZEN_A9_INPUTS",
        "created_at_utc": _utc_now(),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_lock_sha256": manifest["manifest_lock_sha256"],
        "frame_count": len(frames),
        "item_count": len(items),
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "official_scorer_run_count": 0,
        "scene9_read_count": 0,
    }
    _write_json_atomic(output_root / "freeze-receipt.json", receipt)
    return receipt


def validate_input_manifest(
    path: Path, protocol_path: Path, *, verify_assets: bool
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    manifest = _read_json(path.resolve())
    if (
        manifest.get("schema_version") != INPUT_SCHEMA
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("protocol_sha256") != _sha256_file(protocol_path.resolve())
        or manifest.get("dataset_role") != "ALREADY_CONSUMED_REAL_DEVELOPMENT"
        or int(manifest.get("frame_count", -1)) != 25
        or int(manifest.get("item_count", -1)) != 820
        or manifest.get("manifest_lock_sha256")
        != _canonical_sha256(_without_lock(manifest, "manifest_lock_sha256"))
    ):
        raise ContractError("Frozen A-R9 input manifest changed")
    boundary = manifest.get("runtime_boundary")
    if not isinstance(boundary, dict) or any(
        int(boundary.get(key, -1)) != 0
        for key in (
            "label_access_count",
            "gt_path_open_count",
            "evaluator_path_open_count",
            "official_scorer_run_count",
            "scene9_read_count",
        )
    ):
        raise ContractError("Frozen runtime access boundary changed")
    frames = manifest.get("frames")
    items = manifest.get("items")
    if not isinstance(frames, list) or not isinstance(items, list):
        raise ContractError("Frozen input inventory is missing")
    item_ids = [str(item.get("item_id", "")) for item in items]
    if len(items) != 820 or len(set(item_ids)) != 820 or any(not item_id for item_id in item_ids):
        raise ContractError("Frozen item IDs are missing or duplicated")
    frame_ids = [str(frame.get("frame_id", "")) for frame in frames]
    if len(frames) != 25 or len(set(frame_ids)) != 25 or any(not frame_id for frame_id in frame_ids):
        raise ContractError("Frozen frame IDs are missing or duplicated")
    if any(str(item.get("frame_id", "")) not in set(frame_ids) for item in items):
        raise ContractError("Frozen item refers to an unknown frame")
    if any(int(item.get("scene_id", -1)) == 9 for item in items):
        raise ContractError("Scene 9 entered the A-R9 FoundationPose input")
    if verify_assets:
        dataset_root = Path(str(manifest["dataset_root"])).resolve()
        predictions_root = Path(str(manifest["predictions_root"])).resolve()
        seen: set[tuple[str, str]] = set()
        for frame in frames:
            for key in ("rgb", "depth", "camera", "cad"):
                record = frame[key]
                cache_key = ("dataset", str(record["relative_path"]))
                if cache_key not in seen:
                    _resolve_bound(dataset_root, record)
                    seen.add(cache_key)
            for key in ("prediction_metadata", "prediction_masks"):
                record = frame[key]
                cache_key = ("predictions", str(record["relative_path"]))
                if cache_key not in seen:
                    _resolve_bound(predictions_root, record)
                    seen.add(cache_key)
        if (
            manifest["prediction_manifest"]["sha256"]
            != protocol["dataset"]["a9_prediction_manifest_sha256"]
        ):
            raise ContractError(
                "Input manifest does not bind the frozen A-R9 predictions"
            )
    return manifest


def _implementation_identity(protocol_path: Path) -> dict[str, Any]:
    root = _repo_root()
    relative_paths = [
        "pose_accuracy_recovery_prep/a9_foundationpose_e2e/__init__.py",
        "pose_accuracy_recovery_prep/a9_foundationpose_e2e/__main__.py",
        "pose_accuracy_recovery_prep/a9_foundationpose_e2e/runtime.py",
        "scripts/provision_a9_foundationpose_env.sh",
        "scripts/run_r1_sealed_inference.py",
        "scripts/run_xyzibd_batch.py",
    ]
    files = {relative: _sha256_file(root / relative) for relative in relative_paths}
    if files["scripts/run_r1_sealed_inference.py"] != EXPECTED_MEMORY_PATCH_SHA256:
        raise ContractError("Frozen FoundationPose memory patch source changed")
    if files["scripts/run_xyzibd_batch.py"] != EXPECTED_RUN_XYZIBD_SHA256:
        raise ContractError("Frozen XYZ-IBD runtime source changed")
    files[protocol_path.resolve().relative_to(root.resolve()).as_posix()] = (
        _sha256_file(protocol_path.resolve())
    )
    return {"files_sha256": files, "identity_sha256": _canonical_sha256(files)}


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_foundationpose(
    protocol: Mapping[str, Any], foundationpose_root: Path
) -> dict[str, Any]:
    foundationpose_root = foundationpose_root.resolve()
    fp = protocol["foundationpose"]
    if _git_head(foundationpose_root) != fp["commit"]:
        raise ContractError("FoundationPose checkout commit changed")
    status = subprocess.run(
        ["git", "-C", str(foundationpose_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    allowed_untracked = {
        "mycppbuild/",
        "weights/",
        "__pycache__/",
        "learning/__pycache__/",
        "learning/training/__pycache__/",
    }
    unexpected = [
        line
        for line in status.splitlines()
        if not (
            line.startswith("?? ")
            and any(line[3:].startswith(prefix) for prefix in allowed_untracked)
        )
    ]
    if unexpected:
        raise ContractError(
            f"FoundationPose checkout has unexpected changes: {unexpected[:3]}"
        )
    records: dict[str, Any] = {}
    for key in ("refiner_checkpoint", "scorer_checkpoint"):
        lock = fp[key]
        path = foundationpose_root / lock["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(lock["size_bytes"])
            or _sha256_file(path) != lock["sha256"]
        ):
            raise ContractError(f"FoundationPose {key} changed")
        records[key] = _bound_file(path, root=foundationpose_root)
    config_locks = {
        "refiner_config": (
            foundationpose_root / "weights/2023-10-28-18-33-37/config.yml",
            fp["refiner_config_sha256"],
        ),
        "scorer_config": (
            foundationpose_root / "weights/2024-01-11-20-02-45/config.yml",
            fp["scorer_config_sha256"],
        ),
    }
    for key, (path, digest) in config_locks.items():
        if _sha256_file(path) != digest:
            raise ContractError(f"FoundationPose {key} changed")
        records[key] = _bound_file(path, root=foundationpose_root)
    return records


def _load_existing_results(
    path: Path, run_lock_sha256: str, item_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    if not rows or rows[0] != {
        "record_type": "metadata",
        "run_lock_sha256": run_lock_sha256,
    }:
        raise ContractError("Existing result metadata differs from the run lock")
    completed: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if (
            row.get("schema_version") != RESULT_SCHEMA
            or row.get("record_type") != "prediction"
        ):
            raise ContractError("Existing result row schema changed")
        item_id = str(row.get("item_id", ""))
        if item_id not in item_ids or item_id in completed:
            raise ContractError("Existing result item is unknown or duplicated")
        if row.get("run_lock_sha256") != run_lock_sha256:
            raise ContractError("Existing result row run lock changed")
        completed[item_id] = row
    return completed


def _pose_list(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7, rtol=0.0)
        or matrix[2, 3] <= 0
    ):
        raise ContractError("FoundationPose returned an invalid SE(3) pose")
    return matrix.tolist()


def preflight_primary_output(
    output_root: Path, *, resume: bool, expected: dict[str, Any]
) -> dict[str, Any] | None:
    """Reject incompatible output before importing or allocating GPU models."""
    if not output_root.exists():
        if resume:
            raise ContractError("Cannot resume an absent primary output root")
        return None
    if not resume:
        raise ContractError("Primary output root exists; use --resume only for the same run")
    lock_path = output_root / "run-lock.json"
    if not lock_path.is_file():
        raise ContractError("Primary resume run lock is missing")
    lock = _read_json(lock_path)
    if (
        lock.get("run_lock_sha256") != _canonical_sha256(_without_lock(lock, "run_lock_sha256"))
        or any(lock.get(key) != value for key, value in expected.items())
    ):
        raise ContractError("Primary resume run lock changed")
    completion = output_root / "completion-receipt.json"
    if completion.exists():
        receipt = _read_json(completion)
        predictions = output_root / "predictions.jsonl"
        if (
            receipt.get("run_lock_sha256") != lock["run_lock_sha256"]
            or not predictions.is_file()
            or receipt.get("predictions_sha256") != _sha256_file(predictions)
        ):
            raise ContractError("Completed primary predictions or run lock changed")
        return receipt
    return None


def run_primary(
    *,
    protocol_path: Path,
    manifest_path: Path,
    foundationpose_root: Path,
    output_root: Path,
    implementation_commit: str,
    resume: bool,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    manifest_path = manifest_path.resolve()
    foundationpose_root = foundationpose_root.resolve()
    output_root = output_root.resolve()
    protocol = load_protocol(protocol_path)
    manifest = validate_input_manifest(manifest_path, protocol_path, verify_assets=True)
    root = _repo_root()
    implementation = _implementation_identity(protocol_path)
    if len(implementation_commit) != 40 or any(
        ch not in "0123456789abcdef" for ch in implementation_commit
    ):
        raise ContractError("Implementation commit must be a full lowercase Git SHA")
    if _git_head(root) != implementation_commit:
        raise ContractError(
            "Implementation commit does not match the deployed checkout"
        )
    implementation_paths = sorted(implementation["files_sha256"])
    implementation_status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", *implementation_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if implementation_status.strip():
        raise ContractError("Implementation files differ from the deployed Git commit")
    foundationpose_assets = _verify_foundationpose(protocol, foundationpose_root)
    completion = preflight_primary_output(
        output_root, resume=resume, expected={
            "schema_version": RUN_LOCK_SCHEMA,
            "protocol_sha256": _sha256_file(protocol_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "manifest_lock_sha256": manifest["manifest_lock_sha256"],
            "implementation_commit": implementation_commit,
            "implementation": implementation,
            "foundationpose_commit": protocol["foundationpose"]["commit"],
            "foundationpose_assets": foundationpose_assets,
        },
    )
    if completion is not None:
        return completion
    try:
        import cv2
        import nvdiffrast.torch as dr
        import torch
        import trimesh
    except ImportError as exc:
        raise ContractError(
            f"FoundationPose runtime import failed: {type(exc).__name__}"
        ) from exc
    if not torch.cuda.is_available():
        raise ContractError("CUDA is unavailable")
    scripts = root / "scripts"
    for source in (foundationpose_root, scripts):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    original_cwd = Path.cwd()
    os.chdir(foundationpose_root)
    try:
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from learning.training.predict_score import ScorePredictor
        from run_r1_sealed_inference import (
            install_memory_bounded_refine_forward,
            install_memory_bounded_score_data,
            install_memory_bounded_score_forward,
            install_memory_bounded_warp,
        )
        from run_xyzibd_batch import clear_per_sample_estimator_state
        from Utils import set_seed
    finally:
        os.chdir(original_cwd)
    gpu_index = 0
    torch.cuda.set_device(gpu_index)
    batches = protocol["foundationpose"]["resource_batches"]
    install_memory_bounded_warp(int(batches["warp"]))
    set_seed(int(protocol["foundationpose"]["seed"]))
    scorer = ScorePredictor()
    install_memory_bounded_score_data(scorer, int(batches["score_data"]))
    install_memory_bounded_score_forward(scorer, int(batches["score_feature"]))
    refiner = PoseRefinePredictor()
    install_memory_bounded_refine_forward(refiner, int(batches["refine"]))
    glctx = dr.RasterizeCudaContext(device=gpu_index)
    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(gpu_index),
        "gpu_uuid": subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    run_lock: dict[str, Any] = {
        "schema_version": RUN_LOCK_SCHEMA,
        "protocol_sha256": _sha256_file(protocol_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_lock_sha256": manifest["manifest_lock_sha256"],
        "implementation_commit": implementation_commit,
        "implementation": implementation,
        "foundationpose_commit": protocol["foundationpose"]["commit"],
        "foundationpose_assets": foundationpose_assets,
        "environment": environment,
        "frozen_inference": {
            "iterations": 5,
            "seed": 0,
            "candidate_count": 252,
            "resource_batches": dict(batches),
            "primary_register_only_no_trace_replay": True,
        },
        "runtime_boundary": {
            "label_access_count": 0,
            "gt_path_open_count": 0,
            "evaluator_path_open_count": 0,
            "official_scorer_run_count": 0,
            "scene9_read_count": 0,
        },
    }
    run_lock["run_lock_sha256"] = _canonical_sha256(
        _without_lock(run_lock, "run_lock_sha256")
    )
    run_lock_path = output_root / "run-lock.json"
    results_path = output_root / "predictions.jsonl"
    if output_root.exists():
        if not resume:
            raise ContractError(
                "Primary output root exists; use --resume only for the same run"
            )
        if not run_lock_path.is_file() or _read_json(run_lock_path) != run_lock:
            raise ContractError("Primary resume run lock changed")
    else:
        if resume:
            raise ContractError("Cannot resume an absent primary output root")
        output_root.mkdir(parents=True)
        _write_json_atomic(run_lock_path, run_lock)
        _append_jsonl(
            results_path,
            {"record_type": "metadata", "run_lock_sha256": run_lock["run_lock_sha256"]},
        )
    completion_path = output_root / "completion-receipt.json"
    if completion_path.exists():
        receipt = _read_json(completion_path)
        if receipt.get("predictions_sha256") != _sha256_file(results_path):
            raise ContractError("Completed primary predictions changed")
        return receipt
    item_by_id = {str(item["item_id"]): item for item in manifest["items"]}
    completed = _load_existing_results(
        results_path, run_lock["run_lock_sha256"], set(item_by_id)
    )
    frame_by_id = {str(frame["frame_id"]): frame for frame in manifest["frames"]}
    dataset_root = Path(str(manifest["dataset_root"])).resolve()
    predictions_root = Path(str(manifest["predictions_root"])).resolve()
    estimator: Any | None = None
    estimator_object_id: int | None = None
    mesh_cache: dict[int, Any] = {}
    frame_cache: dict[str, dict[str, Any]] = {}

    def load_frame(frame_id: str) -> dict[str, Any]:
        if frame_id in frame_cache:
            return frame_cache[frame_id]
        frame = frame_by_id[frame_id]
        rgb_path = _resolve_bound(dataset_root, frame["rgb"])
        depth_path = _resolve_bound(dataset_root, frame["depth"])
        masks_path = _resolve_bound(predictions_root, frame["prediction_masks"])
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw_depth is None or raw_depth.ndim != 2:
            raise ContractError(f"Cannot decode runtime RGB-D: {frame_id}")
        masks = _unpack_masks(masks_path)
        depth_m = np.ascontiguousarray(
            raw_depth.astype(np.float32)
            * float(frame["raw_depth_scale_to_mm"])
            * 0.001,
            dtype=np.float32,
        )
        value = {
            "rgb": np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            "depth_m": depth_m,
            "masks": masks,
            "K": np.asarray(frame["camera_intrinsics_row_major"], dtype=np.float64),
        }
        frame_cache.clear()
        frame_cache[frame_id] = value
        return value

    def object_mesh(object_id: int) -> Any:
        if object_id not in mesh_cache:
            frame = next(
                frame
                for frame in manifest["frames"]
                if int(frame["object_id"]) == object_id
            )
            path = _resolve_bound(dataset_root, frame["cad"])
            mesh = trimesh.load(path, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                raise ContractError(f"Invalid CAD for object {object_id}")
            if not np.all(np.asarray(mesh.extents) > 1.0):
                raise ContractError(f"CAD is not in millimetres for object {object_id}")
            mesh.apply_scale(0.001)
            _ = mesh.vertex_normals
            mesh_cache[object_id] = mesh
        return mesh_cache[object_id]

    started = time.monotonic()
    ordered_items = list(manifest["items"])
    for ordinal, item in enumerate(ordered_items, 1):
        item_id = str(item["item_id"])
        if item_id in completed:
            continue
        object_id = int(item["object_id"])
        frame = load_frame(str(item["frame_id"]))
        mesh = object_mesh(object_id)
        if estimator is None:
            estimator = FoundationPose(
                model_pts=mesh.vertices.copy(),
                model_normals=mesh.vertex_normals.copy(),
                symmetry_tfs=None,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                glctx=glctx,
                debug=0,
                debug_dir=str(output_root / "foundationpose_debug"),
            )
            if int(len(estimator.rot_grid)) != 252:
                raise ContractError("FoundationPose rotation grid changed")
            estimator.rot_grid = estimator.rot_grid[:252].clone()
            estimator_object_id = object_id
        elif estimator_object_id != object_id:
            estimator.reset_object(
                model_pts=mesh.vertices.copy(),
                model_normals=mesh.vertex_normals.copy(),
                symmetry_tfs=None,
                mesh=mesh,
            )
            estimator_object_id = object_id
        mask = np.ascontiguousarray(
            frame["masks"][int(item["prediction_index"])], dtype=bool
        )
        measured_mask_sha = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        if measured_mask_sha != item["mask_sha256"]:
            raise ContractError(f"Runtime mask changed: {item_id}")
        clear_per_sample_estimator_state(estimator)
        set_seed(int(protocol["foundationpose"]["seed"]))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        item_started = time.monotonic()
        fatal_exception: Exception | None = None
        try:
            pose = estimator.register(
                K=frame["K"],
                rgb=frame["rgb"],
                depth=frame["depth_m"],
                ob_mask=mask,
                ob_id=object_id,
                iteration=int(protocol["foundationpose"]["iterations"]),
            )
            torch.cuda.synchronize()
            pose_value = _pose_list(pose)
            if estimator.poses is None or estimator.scores is None:
                raise ContractError("FoundationPose did not expose fresh candidates")
            scores = np.asarray(
                estimator.scores.detach().cpu(), dtype=np.float64
            ).reshape(-1)
            if scores.shape != (252,) or not np.isfinite(scores).all():
                raise ContractError("FoundationPose candidate scores changed")
            row: dict[str, Any] = {
                "record_type": "prediction",
                "schema_version": RESULT_SCHEMA,
                "run_lock_sha256": run_lock["run_lock_sha256"],
                "item_id": item_id,
                "frame_id": item["frame_id"],
                "scene_id": item["scene_id"],
                "image_id": item["image_id"],
                "object_id": object_id,
                "prediction_index": item["prediction_index"],
                "detector_score": item["detector_score"],
                "status": "success",
                "predicted_model_to_camera_pose_m": pose_value,
                "foundationpose_top_score": float(scores[0]),
                "foundationpose_top_score_margin": float(scores[0] - scores[1]),
                "candidate_count": 252,
                "registration_seconds": time.monotonic() - item_started,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "mask_sha256": measured_mask_sha,
                "label_access_count": 0,
                "gt_path_open_count": 0,
                "evaluator_path_open_count": 0,
                "official_scorer_run_count": 0,
                "scene9_read_count": 0,
            }
        except Exception as exc:
            torch.cuda.empty_cache()
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                fatal_exception = exc
            row = {
                "record_type": "prediction",
                "schema_version": RESULT_SCHEMA,
                "run_lock_sha256": run_lock["run_lock_sha256"],
                "item_id": item_id,
                "frame_id": item["frame_id"],
                "scene_id": item["scene_id"],
                "image_id": item["image_id"],
                "object_id": object_id,
                "prediction_index": item["prediction_index"],
                "detector_score": item["detector_score"],
                "status": "failure",
                "failure_type": type(exc).__name__,
                "failure_message": " ".join(str(exc).split())[:500],
                "registration_seconds": time.monotonic() - item_started,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "mask_sha256": measured_mask_sha,
                "label_access_count": 0,
                "gt_path_open_count": 0,
                "evaluator_path_open_count": 0,
                "official_scorer_run_count": 0,
                "scene9_read_count": 0,
            }
        _append_jsonl(results_path, row)
        completed[item_id] = row
        print(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "total": len(ordered_items),
                    "item_id": item_id,
                    "status": row["status"],
                    "seconds": row["registration_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if fatal_exception is not None:
            raise ContractError(
                "Primary stopped after the first CUDA OOM; partial evidence is resumable only after oracle diagnosis"
            ) from fatal_exception
    if set(completed) != set(item_by_id):
        raise ContractError("Primary run ended without exact item coverage")
    status_counts = Counter(str(row["status"]) for row in completed.values())
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "stage": "primary-inference",
        "status": "COMPLETE",
        "created_at_utc": _utc_now(),
        "run_lock_sha256": run_lock["run_lock_sha256"],
        "predictions_sha256": _sha256_file(results_path),
        "item_count": len(completed),
        "status_counts": dict(sorted(status_counts.items())),
        "wall_time_seconds": time.monotonic() - started,
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "official_scorer_run_count": 0,
        "scene9_read_count": 0,
    }
    _write_json_atomic(completion_path, receipt)
    return receipt


def _standard_ap(
    labels: Sequence[bool], scores: Sequence[float], total_gt: int, keys: Sequence[str]
) -> float:
    if len(labels) != len(scores) or len(labels) != len(keys):
        raise ContractError("AP inputs differ in length")
    order = sorted(
        range(len(labels)), key=lambda index: (-float(scores[index]), str(keys[index]))
    )
    if total_gt <= 0 or not order:
        return 0.0
    tp = np.cumsum(np.asarray([1.0 if labels[index] else 0.0 for index in order]))
    fp = np.cumsum(np.asarray([0.0 if labels[index] else 1.0 for index in order]))
    recall = tp / float(total_gt)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def _pose_edges(vertices_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = vertices_m.min(axis=0)
    maximum = vertices_m.max(axis=0)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    edges = []
    for left in range(8):
        for right in range(left + 1, 8):
            if int(np.sum(corners[left] != corners[right])) == 1:
                edges.append((left, right))
    return corners, np.asarray(edges, dtype=np.int32)


def _draw_poses(
    image: np.ndarray,
    poses: Iterable[Sequence[Sequence[float]]],
    K: np.ndarray,
    vertices_m: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    import cv2

    output = image.copy()
    corners, edges = _pose_edges(vertices_m)
    for raw_pose in poses:
        pose = np.asarray(raw_pose, dtype=np.float64).reshape(4, 4)
        camera = (pose[:3, :3] @ corners.T + pose[:3, 3:4]).T
        if np.any(camera[:, 2] <= 1e-6):
            continue
        projected = (K @ camera.T).T
        pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
        for left, right in edges:
            cv2.line(
                output, tuple(pixels[left]), tuple(pixels[right]), color, 2, cv2.LINE_AA
            )
    return output


def _draw_masks(image: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    import cv2

    output = image.copy()
    for mask in masks:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, (0, 210, 255), 2, cv2.LINE_AA)
    return output


def evaluate(
    *,
    protocol_path: Path,
    manifest_path: Path,
    primary_root: Path,
    dataset_root: Path,
    toolkit_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    manifest_path = manifest_path.resolve()
    primary_root = primary_root.resolve()
    dataset_root = dataset_root.resolve()
    toolkit_root = toolkit_root.resolve()
    output_root = output_root.resolve()
    protocol = load_protocol(protocol_path)
    manifest = validate_input_manifest(manifest_path, protocol_path, verify_assets=True)
    if output_root.exists():
        raise ContractError("Evaluation output root is create-only")
    completion = _read_json(primary_root / "completion-receipt.json")
    predictions_path = primary_root / "predictions.jsonl"
    if (
        completion.get("stage") != "primary-inference"
        or completion.get("status") != "COMPLETE"
        or completion.get("predictions_sha256") != _sha256_file(predictions_path)
        or any(
            int(completion.get(key, -1)) != 0
            for key in (
                "label_access_count",
                "gt_path_open_count",
                "evaluator_path_open_count",
                "official_scorer_run_count",
                "scene9_read_count",
            )
        )
    ):
        raise ContractError("Primary completion receipt is absent or unsafe")
    rows = _read_jsonl(predictions_path)
    if len(rows) != 821 or rows[0].get("record_type") != "metadata":
        raise ContractError("Primary prediction coverage changed")
    result_by_id = {str(row["item_id"]): row for row in rows[1:]}
    if len(result_by_id) != 820:
        raise ContractError("Primary prediction IDs are duplicated")
    scripts_root = _repo_root() / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    if str(toolkit_root) not in sys.path:
        sys.path.insert(0, str(toolkit_root))
    from scripts.evaluate_m1 import (
        BOP_TOOLKIT_COMMIT,
        load_object_evaluation_data,
        load_official_models,
        official_errors,
        toolkit_commit,
    )

    if toolkit_commit(toolkit_root) != BOP_TOOLKIT_COMMIT:
        raise ContractError("Pinned BOP Toolkit changed")
    model_params, model_info = load_official_models(dataset_root)
    object_data = {
        object_id: load_object_evaluation_data(object_id, model_params, model_info)
        for object_id in sorted(
            set(int(frame["object_id"]) for frame in manifest["frames"])
        )
    }
    import cv2
    import trimesh

    items_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in manifest["items"]:
        items_by_frame[str(item["frame_id"])].append(item)
    all_joint_labels: list[bool] = []
    all_scores: list[float] = []
    all_keys: list[str] = []
    assigned_errors: list[tuple[float, float] | None] = []
    per_scene_counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {"gt": 0, "joint": 0, "pred": 0}
    )
    per_frame: list[dict[str, Any]] = []
    matched_pose_rows: list[dict[str, Any]] = []
    mask_iou50_match_count = 0
    runtime_success = sum(row.get("status") == "success" for row in rows[1:])
    runtime_completion_fraction = runtime_success / 820.0
    output_root.mkdir(parents=True)
    visual_root = output_root / "visualizations"
    visual_root.mkdir()
    for frame in manifest["frames"]:
        frame_id = str(frame["frame_id"])
        scene_id = int(frame["scene_id"])
        image_id = int(frame["image_id"])
        object_id = int(frame["object_id"])
        scene_root = dataset_root / "val" / f"{scene_id:06d}"
        gt_rows = _read_json(scene_root / "scene_gt_realsense.json").get(str(image_id))
        if not isinstance(gt_rows, list) or not gt_rows:
            raise ContractError(f"Development GT is missing: {frame_id}")
        gt_masks: list[np.ndarray] = []
        gt_poses: list[list[list[float]]] = []
        for gt_index, gt_row in enumerate(gt_rows):
            if int(gt_row.get("obj_id", -1)) != object_id:
                raise ContractError(f"Task-target object mapping changed: {frame_id}")
            mask_path = (
                scene_root
                / "mask_visib_realsense"
                / f"{image_id:06d}_{gt_index:06d}.png"
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or not np.any(mask):
                raise ContractError(
                    f"Development visible mask is missing: {frame_id}/{gt_index}"
                )
            gt_masks.append(mask > 0)
            rotation = np.asarray(gt_row["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            translation_m = (
                np.asarray(gt_row["cam_t_m2c"], dtype=np.float64).reshape(3) * 0.001
            )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation_m
            gt_poses.append(pose.tolist())
        pred_items = sorted(
            items_by_frame[frame_id], key=lambda item: int(item["prediction_index"])
        )
        masks_path = _resolve_bound(
            Path(str(manifest["predictions_root"])), frame["prediction_masks"]
        )
        all_masks = _unpack_masks(masks_path)
        pred_masks = [all_masks[int(item["prediction_index"])] for item in pred_items]
        from pose_accuracy_recovery_prep.real_causal_ablation_v1.runtime import (
            _greedy_matches,
            _iou_matrix,
        )

        matches = _greedy_matches(
            _iou_matrix(gt_masks, pred_masks),
            float(protocol["evaluation"]["mask_iou_threshold"]),
        )
        mask_iou50_match_count += len(matches)
        errors_by_pred: dict[int, tuple[float, float, int, float]] = {}
        K = np.asarray(frame["camera_intrinsics_row_major"], dtype=np.float64)
        for gt_index, pred_index, iou in matches:
            item = pred_items[pred_index]
            result = result_by_id[str(item["item_id"])]
            if result.get("status") != "success":
                continue
            mssd_mm, mspd_px = official_errors(
                np.asarray(
                    result["predicted_model_to_camera_pose_m"], dtype=np.float64
                ),
                np.asarray(gt_poses[gt_index], dtype=np.float64),
                K,
                object_data[object_id],
            )
            normalized_mssd = mssd_mm / float(object_data[object_id]["diameter_mm"])
            errors_by_pred[pred_index] = (
                normalized_mssd,
                mspd_px,
                gt_index,
                float(iou),
            )
            matched_pose_rows.append(
                {
                    "item_id": item["item_id"],
                    "frame_id": frame_id,
                    "gt_index": gt_index,
                    "mask_iou": float(iou),
                    "normalized_mssd": normalized_mssd,
                    "mspd_px": mspd_px,
                }
            )
        joint_threshold = protocol["evaluation"]["joint_pose_threshold"]
        frame_joint = 0
        frame_assigned: list[tuple[float, float] | None] = [None] * len(gt_masks)
        predicted_poses = []
        for pred_index, item in enumerate(pred_items):
            result = result_by_id[str(item["item_id"])]
            if result.get("status") == "success":
                predicted_poses.append(result["predicted_model_to_camera_pose_m"])
            error = errors_by_pred.get(pred_index)
            is_joint = bool(
                error is not None
                and error[0] < float(joint_threshold["mssd_fraction_of_diameter"])
                and error[1] < float(joint_threshold["mspd_pixels"])
            )
            if error is not None:
                frame_assigned[error[2]] = (error[0], error[1])
            frame_joint += int(is_joint)
            all_joint_labels.append(is_joint)
            all_scores.append(float(item["detector_score"]))
            all_keys.append(str(item["item_id"]))
        assigned_errors.extend(frame_assigned)
        per_scene_counts[scene_id]["gt"] += len(gt_masks)
        per_scene_counts[scene_id]["pred"] += len(pred_items)
        per_scene_counts[scene_id]["joint"] += frame_joint
        per_frame.append(
            {
                "frame_id": frame_id,
                "scene_id": scene_id,
                "gt_count": len(gt_masks),
                "prediction_count": len(pred_items),
                "mask_iou50_matches": len(matches),
                "pose_evaluated_matches": len(errors_by_pred),
                "joint_pose_successes": frame_joint,
                "joint_pose_recall": frame_joint / len(gt_masks),
            }
        )
        rgb_path = _resolve_bound(dataset_root, frame["rgb"])
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ContractError(f"Cannot decode visualization RGB: {frame_id}")
        cad_path = _resolve_bound(dataset_root, frame["cad"])
        mesh = trimesh.load(cad_path, force="mesh", process=False)
        vertices_m = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
        panels = [
            _draw_masks(bgr, pred_masks),
            _draw_poses(bgr, predicted_poses, K, vertices_m, (255, 0, 255)),
            _draw_poses(bgr, gt_poses, K, vertices_m, (0, 255, 0)),
        ]
        labels = ["A-R9 masks", "FoundationPose output", "development GT"]
        for panel, label in zip(panels, labels, strict=True):
            cv2.rectangle(panel, (0, 0), (430, 38), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        canvas = np.concatenate(panels, axis=1)
        canvas = cv2.resize(
            canvas,
            (canvas.shape[1] // 2, canvas.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        if not cv2.imwrite(
            str(visual_root / f"{frame_id}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]
        ):
            raise ContractError(f"Cannot write visualization: {frame_id}")
    total_gt = len(assigned_errors)
    if total_gt != int(protocol["dataset"]["ground_truth_instance_count"]):
        raise ContractError(f"Expected 770 evaluation GT instances, got {total_gt}")
    joint_count = sum(all_joint_labels)
    joint_precision = joint_count / len(all_joint_labels)
    joint_recall = joint_count / total_gt
    joint_f1 = (
        0.0
        if joint_precision + joint_recall == 0
        else 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
    )
    joint_ap = _standard_ap(all_joint_labels, all_scores, total_gt, all_keys)
    mssd_recalls = {
        str(threshold): sum(
            error is not None and error[0] < float(threshold)
            for error in assigned_errors
        )
        / total_gt
        for threshold in protocol["evaluation"][
            "mssd_ar_thresholds_fraction_of_diameter"
        ]
    }
    mspd_recalls = {
        str(threshold): sum(
            error is not None and error[1] < float(threshold)
            for error in assigned_errors
        )
        / total_gt
        for threshold in protocol["evaluation"]["mspd_ar_thresholds_pixels"]
    }
    ar_mssd = float(np.mean(list(mssd_recalls.values())))
    ar_mspd = float(np.mean(list(mspd_recalls.values())))
    combined_ar = (ar_mssd + ar_mspd) / 2.0
    scene_rows = []
    positive_scene_count = 0
    positive_threshold = float(
        protocol["promotion_gate"]["per_scene_joint_recall_positive_threshold"]
    )
    for scene_id in protocol["dataset"]["scenes"]:
        counts = per_scene_counts[int(scene_id)]
        recall = counts["joint"] / counts["gt"]
        positive_scene_count += int(recall >= positive_threshold)
        scene_rows.append(
            {"scene_id": int(scene_id), **counts, "joint_pose_recall": recall}
        )
    gate = protocol["promotion_gate"]
    checks = {
        "runtime_completion_fraction_min": runtime_completion_fraction
        >= float(gate["runtime_completion_fraction_min"]),
        "joint_pose_recall_min": joint_recall >= float(gate["joint_pose_recall_min"]),
        "joint_pose_ap_min": joint_ap >= float(gate["joint_pose_ap_min"]),
        "combined_ar_mssd_mspd_min": combined_ar
        >= float(gate["combined_ar_mssd_mspd_min"]),
        "positive_scene_count_min": positive_scene_count
        >= int(gate["positive_scene_count_min"]),
    }
    passed = all(checks.values())
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "PASS_PACKAGE_AND_CLOSE"
        if passed
        else "NO_GO_ORACLE_DIAGNOSIS_PERMITTED_ONCE",
        "protocol_sha256": _sha256_file(protocol_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "primary_completion_receipt_sha256": _sha256_file(
            primary_root / "completion-receipt.json"
        ),
        "predictions_sha256": _sha256_file(predictions_path),
        "dataset_role": "ALREADY_CONSUMED_REAL_DEVELOPMENT",
        "labels_opened_after_primary_completion": True,
        "frame_count": 25,
        "ground_truth_instance_count": total_gt,
        "prediction_count": 820,
        "runtime_success_count": runtime_success,
        "runtime_completion_fraction": runtime_completion_fraction,
        "mask_iou50_match_count": mask_iou50_match_count,
        "pose_evaluated_match_count": len(matched_pose_rows),
        "joint_pose_success_count": joint_count,
        "joint_pose_precision": joint_precision,
        "joint_pose_recall": joint_recall,
        "joint_pose_f1": joint_f1,
        "joint_pose_ap": joint_ap,
        "ar_mssd": ar_mssd,
        "ar_mspd": ar_mspd,
        "combined_ar_mssd_mspd": combined_ar,
        "mssd_threshold_recalls": mssd_recalls,
        "mspd_threshold_recalls": mspd_recalls,
        "positive_scene_count": positive_scene_count,
        "per_scene": scene_rows,
        "per_frame": per_frame,
        "promotion_checks": checks,
        "oracle_diagnostic_runs_used": 0,
        "rescue_runs_used": 0,
        "scene9_read_count": 0,
        "scene9_replay_permitted": False,
        "sealed_claim_permitted": False,
        "claim": protocol["evaluation"]["claim"],
    }
    result_path = output_root / "evaluation-result.json"
    _write_json_atomic(result_path, result)
    with (output_root / "matched-pose-errors.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "item_id",
                "frame_id",
                "gt_index",
                "mask_iou",
                "normalized_mssd",
                "mspd_px",
            ],
        )
        writer.writeheader()
        writer.writerows(matched_pose_rows)
    report = f"""# A-R9 to FoundationPose end-to-end development result

Status: **{result["status"]}**

This is a custom evaluation on already-consumed XYZ-IBD RealSense development
frames. It is not official BOP leaderboard AR/AP, not sealed, and scene 9 was
not read.

| Metric | Result |
| --- | ---: |
| Runtime completion | {runtime_success}/820 ({runtime_completion_fraction:.4f}) |
| Mask IoU50 matched poses | {len(matched_pose_rows)}/770 |
| Joint pose precision | {joint_precision:.4f} |
| Joint pose recall | {joint_recall:.4f} |
| Joint pose F1 | {joint_f1:.4f} |
| Joint pose AP | {joint_ap:.4f} |
| AR MSSD | {ar_mssd:.4f} |
| AR MSPD | {ar_mspd:.4f} |
| Combined AR MSSD/MSPD | {combined_ar:.4f} |
| Positive scenes | {positive_scene_count}/5 |

Joint pose success requires detector-mask IoU >= 0.50, normalized MSSD < 0.10,
and MSPD < 10 px. All promotion conditions are AND gates frozen before the
run. The 25 triptychs show A-R9 masks, FoundationPose CAD projections, and
development GT projections.
"""
    (output_root / "RESULT.md").write_text(report, encoding="utf-8", newline="\n")
    result["evaluation_result_sha256"] = _sha256_file(result_path)
    return result


def package_evidence(
    *,
    protocol_path: Path,
    freeze_root: Path,
    primary_root: Path,
    evaluation_root: Path,
    archive_path: Path,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    freeze_root = freeze_root.resolve()
    primary_root = primary_root.resolve()
    evaluation_root = evaluation_root.resolve()
    archive_path = archive_path.resolve()
    if archive_path.exists():
        raise ContractError("Evidence archive is create-only")
    result = _read_json(evaluation_root / "evaluation-result.json")
    if result.get("status") != "PASS_PACKAGE_AND_CLOSE":
        raise ContractError("Only a passing end-to-end run may be packaged")
    members: list[tuple[Path, str]] = [
        (protocol_path, f"protocol/{protocol_path.name}")
    ]
    for root, prefix in (
        (freeze_root, "freeze"),
        (primary_root, "primary"),
        (evaluation_root, "evaluation"),
    ):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if "foundationpose_debug" in relative:
                continue
            members.append((path, f"{prefix}/{relative}"))
    sums = {name: _sha256_file(path) for path, name in members}
    temporary_sums = archive_path.with_name(
        f".{archive_path.name}.SHA256SUMS.tmp-{os.getpid()}"
    )
    temporary_sums.parent.mkdir(parents=True, exist_ok=True)
    temporary_sums.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
        newline="\n",
    )
    members.append((temporary_sums, "SHA256SUMS"))
    temporary_archive = archive_path.with_name(
        f".{archive_path.name}.tmp-{os.getpid()}"
    )
    try:
        with tarfile.open(temporary_archive, "w:gz") as archive:
            for path, name in members:
                archive.add(path, arcname=name, recursive=False)
        os.replace(temporary_archive, archive_path)
    finally:
        temporary_sums.unlink(missing_ok=True)
        temporary_archive.unlink(missing_ok=True)
    return {
        "status": "PASS_EVIDENCE_PACKAGED",
        "archive": archive_path.as_posix(),
        "size_bytes": archive_path.stat().st_size,
        "sha256": _sha256_file(archive_path),
        "member_count": len(members),
    }


def _parser() -> argparse.ArgumentParser:
    root = _repo_root()
    default_protocol = (
        root / "protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("contract-check")
    check.add_argument("--protocol", type=Path, default=default_protocol)
    freeze = commands.add_parser("freeze-inputs")
    freeze.add_argument("--protocol", type=Path, default=default_protocol)
    freeze.add_argument("--dataset-root", type=Path, required=True)
    freeze.add_argument("--dataset-manifest", type=Path, required=True)
    freeze.add_argument("--predictions-root", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    validate = commands.add_parser("validate-inputs")
    validate.add_argument("--protocol", type=Path, default=default_protocol)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--verify-assets", action="store_true")
    run = commands.add_parser("run-primary")
    run.add_argument("--protocol", type=Path, default=default_protocol)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--foundationpose-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--implementation-commit", required=True)
    run.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--protocol", type=Path, default=default_protocol)
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--primary-root", type=Path, required=True)
    evaluation.add_argument("--dataset-root", type=Path, required=True)
    evaluation.add_argument("--toolkit-root", type=Path, required=True)
    evaluation.add_argument("--output-root", type=Path, required=True)
    package = commands.add_parser("package")
    package.add_argument("--protocol", type=Path, default=default_protocol)
    package.add_argument("--freeze-root", type=Path, required=True)
    package.add_argument("--primary-root", type=Path, required=True)
    package.add_argument("--evaluation-root", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "contract-check":
        protocol = load_protocol(args.protocol)
        result = {
            "status": "PASS_A9_FOUNDATIONPOSE_E2E_CONTRACT",
            "protocol_sha256": _sha256_file(args.protocol.resolve()),
            "protocol_id": protocol["protocol_id"],
            "scene9_replay_permitted": False,
            "oracle_diagnostic_run_limit": 1,
            "rescue_run_limit": 1,
        }
    elif args.command == "freeze-inputs":
        result = freeze_inputs(
            protocol_path=args.protocol,
            dataset_root=args.dataset_root,
            dataset_manifest_path=args.dataset_manifest,
            predictions_root=args.predictions_root,
            output_root=args.output_root,
        )
    elif args.command == "validate-inputs":
        manifest = validate_input_manifest(
            args.manifest, args.protocol, verify_assets=args.verify_assets
        )
        result = {
            "status": "PASS_FROZEN_A9_INPUTS",
            "frame_count": manifest["frame_count"],
            "item_count": manifest["item_count"],
            "manifest_lock_sha256": manifest["manifest_lock_sha256"],
        }
    elif args.command == "run-primary":
        result = run_primary(
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            foundationpose_root=args.foundationpose_root,
            output_root=args.output_root,
            implementation_commit=args.implementation_commit,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        result = evaluate(
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            primary_root=args.primary_root,
            dataset_root=args.dataset_root,
            toolkit_root=args.toolkit_root,
            output_root=args.output_root,
        )
    else:
        result = package_evidence(
            protocol_path=args.protocol,
            freeze_root=args.freeze_root,
            primary_root=args.primary_root,
            evaluation_root=args.evaluation_root,
            archive_path=args.archive,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
