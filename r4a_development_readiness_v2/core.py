"""Readiness v2: validate the official nested pose schema without exporting labels."""

from __future__ import annotations

import json
import math
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from r4a_development_repair.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
)

from . import PROTOCOL_ID


PROTOCOL_SCHEMA = "poseloop.r4a.development-readiness.protocol.v2"
PREFREEZE_SCHEMA = "poseloop.r4a.development-readiness.prefreeze.v2"


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("R4-A readiness v2 protocol schema mismatch")
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("state") != "frozen_before_v2_development_label_parse":
        raise ContractError("R4-A readiness v2 protocol identity/state mismatch")
    failure = protocol.get("immutable_v1_failure", {})
    for key in (
        "protocol_sha256",
        "prefreeze_receipt_sha256",
        "prefreeze_receipt_lock_sha256",
        "job_log_sha256",
        "job_exit_sha256",
        "failure_receipt_sha256",
        "failure_receipt_lock_sha256",
    ):
        if not _valid_hash(failure.get(key)):
            raise ContractError(f"Invalid immutable v1 failure hash: {key}")
    if failure.get("exit_code") != 1 or failure.get("rerun_permitted") is not False:
        raise ContractError("Readiness v1 failure must remain sealed and non-rerunnable")
    diagnosis = protocol.get("schema_only_diagnosis", {})
    if not _valid_hash(diagnosis.get("sha256")) or diagnosis.get("pose_values_recorded") is not False:
        raise ContractError("Readiness schema diagnosis is not label-safe")
    if diagnosis.get("row_count") != 173:
        raise ContractError("Readiness schema diagnosis row count changed")
    source = protocol.get("frozen_slice", {})
    for key in ("plan_sha256", "extraction_result_sha256", "slice_prefreeze_sha256"):
        if not _valid_hash(source.get(key)):
            raise ContractError(f"Invalid frozen slice hash: {key}")
    if source.get("entry_count") != 188 or source.get("target_count") != 173:
        raise ContractError("Frozen slice counts changed")
    if protocol.get("selection", {}).get("scene_ids") != [0, 1, 2]:
        raise ContractError("Readiness v2 scene selection changed")
    if protocol.get("selection", {}).get("image_ids") != [0]:
        raise ContractError("Readiness v2 image selection changed")
    if protocol.get("output_namespace") != "development_readiness_v2":
        raise ContractError("Readiness v2 output namespace changed")
    boundaries = protocol.get("boundaries", {})
    if boundaries.get("xyzibd_val_access_permitted") is not False:
        raise ContractError("Validation access became permitted")
    if boundaries.get("evaluator_invocation_permitted") is not False:
        raise ContractError("Evaluator invocation became permitted")
    if boundaries.get("gt_pose_export_to_prediction_manifest") is not False:
        raise ContractError("GT pose export became permitted")
    return protocol


def _evidence_hashes(
    *,
    plan_path: Path,
    extraction_result_path: Path,
    slice_prefreeze_path: Path,
    v1_protocol_path: Path,
    v1_prefreeze_path: Path,
    v1_job_log_path: Path,
    v1_job_exit_path: Path,
    v1_failure_receipt_path: Path,
    schema_diagnosis_path: Path,
) -> dict[str, dict[str, str]]:
    return {
        "frozen_slice": {
            "plan_sha256": sha256_file(plan_path),
            "extraction_result_sha256": sha256_file(extraction_result_path),
            "slice_prefreeze_sha256": sha256_file(slice_prefreeze_path),
        },
        "immutable_v1_failure": {
            "protocol_sha256": sha256_file(v1_protocol_path),
            "prefreeze_receipt_sha256": sha256_file(v1_prefreeze_path),
            "job_log_sha256": sha256_file(v1_job_log_path),
            "job_exit_sha256": sha256_file(v1_job_exit_path),
            "failure_receipt_sha256": sha256_file(v1_failure_receipt_path),
        },
        "schema_only_diagnosis": {"sha256": sha256_file(schema_diagnosis_path)},
    }


def _verify_evidence(protocol: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    for section, rows in actual.items():
        expected = protocol[section]
        for key, value in rows.items():
            if expected.get(key) != value:
                raise ContractError(f"Readiness v2 evidence hash mismatch: {section}.{key}")


def _model_manifest(models_root: Path, allowed_objects: set[int]) -> list[dict[str, Any]]:
    rows = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(models_root.glob("obj_*.ply"))
    ]
    if len(rows) != 15:
        raise ContractError(f"Public model count changed: {len(rows)}")
    expected_names = {f"obj_{object_id:06d}.ply" for object_id in allowed_objects}
    if {row["name"] for row in rows} != expected_names:
        raise ContractError("Public model object domain changed")
    return rows


def build_prefreeze_receipt(
    *,
    protocol_path: Path,
    plan_path: Path,
    extraction_result_path: Path,
    slice_prefreeze_path: Path,
    v1_protocol_path: Path,
    v1_prefreeze_path: Path,
    v1_job_log_path: Path,
    v1_job_exit_path: Path,
    v1_failure_receipt_path: Path,
    schema_diagnosis_path: Path,
    models_root: Path,
    implementation_commit: str,
    repository_tree: str,
    repository_tracked_clean: bool,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    actual = _evidence_hashes(
        plan_path=plan_path,
        extraction_result_path=extraction_result_path,
        slice_prefreeze_path=slice_prefreeze_path,
        v1_protocol_path=v1_protocol_path,
        v1_prefreeze_path=v1_prefreeze_path,
        v1_job_log_path=v1_job_log_path,
        v1_job_exit_path=v1_job_exit_path,
        v1_failure_receipt_path=v1_failure_receipt_path,
        schema_diagnosis_path=schema_diagnosis_path,
    )
    _verify_evidence(protocol, actual)
    failure = read_json(v1_failure_receipt_path)
    if failure.get("lock_sha256") != protocol["immutable_v1_failure"]["failure_receipt_lock_sha256"]:
        raise ContractError("Readiness v1 failure lock changed")
    if failure.get("rerun_permitted") is not False or failure.get("exit_code") != 1:
        raise ContractError("Readiness v1 failure receipt changed")
    v1_prefreeze = read_json(v1_prefreeze_path)
    if v1_prefreeze.get("lock_sha256") != protocol["immutable_v1_failure"]["prefreeze_receipt_lock_sha256"]:
        raise ContractError("Readiness v1 prefreeze lock changed")
    if v1_prefreeze.get("development_label_files_opened_at_freeze") != 0:
        raise ContractError("Readiness v1 labels were opened before its freeze")
    if v1_job_exit_path.read_text(encoding="utf-8").strip() != "1":
        raise ContractError("Readiness v1 exit code content changed")
    diagnosis = read_json(schema_diagnosis_path)
    if diagnosis.get("pose_values_recorded") is not False or diagnosis.get("row_count") != 173:
        raise ContractError("Schema-only diagnosis content changed")
    for value in (implementation_commit, repository_tree):
        if len(value) != 40 or not all(c in "0123456789abcdef" for c in value):
            raise ContractError("Invalid implementation Git binding")
    if not repository_tracked_clean:
        raise ContractError("Readiness v2 repository has tracked changes")
    try:
        from bop_toolkit_lib import inout
    except ImportError as error:  # pragma: no cover - GPU-A smoke
        raise ContractError(f"Pinned BOP toolkit import failed: {error}") from error
    loader_symbols_present = all(hasattr(inout, name) for name in ("load_scene_gt", "load_scene_camera"))
    if not loader_symbols_present:
        raise ContractError("Pinned BOP toolkit loader symbols are missing")
    allowed_objects = {int(value) for value in protocol["public_model_object_ids"]}
    receipt = {
        "schema_version": PREFREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "state": protocol["state"],
        "role": "DEVELOPMENT_ONLY",
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": implementation_commit,
        "repository_tree": repository_tree,
        "repository_tracked_clean": True,
        "evidence_hashes": actual,
        "public_model_root": str(models_root.resolve()),
        "public_model_manifest": _model_manifest(models_root, allowed_objects),
        "toolkit": {
            "expected_upstream_commit": protocol["pinned_toolkit"]["commit"],
            "expected_source_tree_sha256": protocol["pinned_toolkit"]["source_tree_sha256"],
            "inout_module_path": str(Path(inout.__file__).resolve()),
            "loader_symbols_present": loader_symbols_present,
            "evaluator_entrypoints_permitted": False,
        },
        "output_namespace": protocol["output_namespace"],
        "development_label_files_opened_at_freeze": 0,
        "development_label_bytes_read_at_freeze": 0,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
        "gt_pose_export_to_prediction_manifest": False,
        "score_guided_pose_selection": False,
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output_path, receipt)
    return receipt


def _verify_prefreeze(
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    repo_root: Path,
    models_root: Path,
    evidence_hashes: Mapping[str, Any],
) -> None:
    if receipt.get("schema_version") != PREFREEZE_SCHEMA or receipt.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Readiness v2 prefreeze identity mismatch")
    unlocked = dict(receipt)
    lock = unlocked.pop("lock_sha256", None)
    if lock != canonical_sha256(unlocked):
        raise ContractError("Readiness v2 prefreeze lock mismatch")
    if receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise ContractError("Readiness v2 prefreeze protocol mismatch")
    if receipt.get("implementation_commit") != _git(repo_root, "rev-parse", "HEAD"):
        raise ContractError("Readiness v2 implementation commit changed")
    if receipt.get("repository_tree") != _git(repo_root, "rev-parse", "HEAD^{tree}"):
        raise ContractError("Readiness v2 repository tree changed")
    if _git(repo_root, "status", "--short", "--untracked-files=no"):
        raise ContractError("Readiness v2 repository has tracked changes")
    if receipt.get("evidence_hashes") != evidence_hashes:
        raise ContractError("Readiness v2 prefreeze evidence binding changed")
    allowed_objects = {int(value) for value in protocol["public_model_object_ids"]}
    if receipt.get("public_model_manifest") != _model_manifest(models_root, allowed_objects):
        raise ContractError("Readiness v2 public model manifest changed")
    if receipt.get("development_label_files_opened_at_freeze") != 0:
        raise ContractError("Development labels were opened before v2 freeze")
    if receipt.get("xyzibd_val_access_count") != 0 or receipt.get("evaluator_invocation_count") != 0:
        raise ContractError("Readiness v2 prefreeze boundary changed")


def _png_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ContractError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ContractError(f"Invalid PNG dimensions: {path}")
    return width, height


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(math.isfinite(float(item)) for item in value)


def _finite_matrix(value: Any, rows: int, columns: int) -> bool:
    return isinstance(value, list) and len(value) == rows and all(_finite_vector(row, columns) for row in value)


def _flatten(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [float(value)]


def _loaded_pose_valid(row: Mapping[str, Any]) -> bool:
    rotation = row.get("cam_R_m2c")
    translation = row.get("cam_t_m2c")
    if tuple(getattr(rotation, "shape", ())) != (3, 3):
        return False
    if tuple(getattr(translation, "shape", ())) not in {(3,), (3, 1)}:
        return False
    values = _flatten(rotation) + _flatten(translation)
    return len(values) == 12 and all(math.isfinite(value) for value in values)


def _verify_extraction(result: Mapping[str, Any], plan: Mapping[str, Any]) -> int:
    files = result.get("files")
    if not isinstance(files, list) or len(files) != int(plan["entry_count"]):
        raise ContractError("Extraction result file count mismatch")
    if {str(row["name"]) for row in plan["entries"]} != {str(row["entry"]) for row in files}:
        raise ContractError("Extraction result entry domain differs from plan")
    for row in files:
        path = Path(row["output_path"]).resolve()
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ContractError(f"Extracted file integrity mismatch: {path}")
    return len(files)


def audit_readiness(
    *,
    protocol_path: Path,
    plan_path: Path,
    extraction_result_path: Path,
    slice_prefreeze_path: Path,
    v1_protocol_path: Path,
    v1_prefreeze_path: Path,
    v1_job_log_path: Path,
    v1_job_exit_path: Path,
    v1_failure_receipt_path: Path,
    schema_diagnosis_path: Path,
    repo_root: Path,
    prefreeze_receipt_path: Path,
    data_root: Path,
    models_root: Path,
    output_path: Path,
    target_manifest_path: Path,
    label_access_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    actual = _evidence_hashes(
        plan_path=plan_path,
        extraction_result_path=extraction_result_path,
        slice_prefreeze_path=slice_prefreeze_path,
        v1_protocol_path=v1_protocol_path,
        v1_prefreeze_path=v1_prefreeze_path,
        v1_job_log_path=v1_job_log_path,
        v1_job_exit_path=v1_job_exit_path,
        v1_failure_receipt_path=v1_failure_receipt_path,
        schema_diagnosis_path=schema_diagnosis_path,
    )
    _verify_evidence(protocol, actual)
    receipt = read_json(prefreeze_receipt_path)
    _verify_prefreeze(receipt, protocol, protocol_path, repo_root, models_root, actual)
    plan = read_json(plan_path)
    result = read_json(extraction_result_path)
    extracted_count = _verify_extraction(result, plan)
    try:
        from bop_toolkit_lib import inout
    except ImportError as error:  # pragma: no cover - GPU-A smoke
        raise ContractError(f"Pinned BOP toolkit import failed: {error}") from error

    root = data_root.resolve()
    if "val" in {part.lower() for part in root.parts}:
        raise ContractError("Readiness v2 data root points at validation")
    allowed_objects = {int(value) for value in protocol["public_model_object_ids"]}
    label_access: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    per_scene: Counter[int] = Counter()
    per_object: Counter[int] = Counter()
    model_files: dict[int, dict[str, Any]] = {}
    loader_scenes = 0
    for scene_id in protocol["selection"]["scene_ids"]:
        scene = root / "train_pbr" / f"{scene_id:06d}"
        paths = {
            "scene_gt": scene / "scene_gt.json",
            "scene_gt_info": scene / "scene_gt_info.json",
            "scene_camera": scene / "scene_camera.json",
        }
        for role, path in paths.items():
            if not path.is_file():
                raise ContractError(f"Required development JSON is missing: {path}")
            label_access.append({"role": role, "path": str(path), "sha256": sha256_file(path)})
        scene_gt = read_json(paths["scene_gt"])
        scene_info = read_json(paths["scene_gt_info"])
        scene_camera = read_json(paths["scene_camera"])
        toolkit_gt = inout.load_scene_gt(str(paths["scene_gt"]))
        toolkit_camera = inout.load_scene_camera(str(paths["scene_camera"]))
        loader_scenes += 1
        for image_id in protocol["selection"]["image_ids"]:
            key = str(image_id)
            gt_rows = scene_gt.get(key)
            info_rows = scene_info.get(key)
            camera = scene_camera.get(key)
            if not isinstance(gt_rows, list) or not isinstance(info_rows, list) or len(gt_rows) != len(info_rows):
                raise ContractError(f"GT/info count mismatch: scene={scene_id} image={image_id}")
            if image_id not in toolkit_gt or image_id not in toolkit_camera:
                raise ContractError("Pinned toolkit loader omitted a selected image")
            if len(toolkit_gt[image_id]) != len(gt_rows):
                raise ContractError("Pinned toolkit loader target count mismatch")
            if not isinstance(camera, dict) or not _finite_vector(camera.get("cam_K"), 9):
                raise ContractError("Selected camera entry has invalid cam_K")
            depth_scale = float(camera.get("depth_scale", 0.0))
            if not math.isfinite(depth_scale) or depth_scale <= 0:
                raise ContractError("Selected camera entry has invalid depth_scale")
            gray = scene / "gray" / f"{image_id:06d}.png"
            depth = scene / "depth" / f"{image_id:06d}.png"
            image_shape = _png_shape(gray)
            if _png_shape(depth) != image_shape:
                raise ContractError("Gray/depth dimensions differ")
            for instance_id, (gt, info, loaded_gt) in enumerate(zip(gt_rows, info_rows, toolkit_gt[image_id])):
                object_id = int(gt["obj_id"])
                if object_id not in allowed_objects:
                    raise ContractError(f"GT object is outside the public model domain: {object_id}")
                if not _finite_matrix(gt.get("cam_R_m2c"), 3, 3) or not _finite_vector(gt.get("cam_t_m2c"), 3):
                    raise ContractError("Official raw development GT pose schema is invalid")
                if not _loaded_pose_valid(loaded_gt):
                    raise ContractError("Pinned toolkit development GT pose schema is invalid")
                visibility = float(info.get("visib_fract", math.nan))
                if not math.isfinite(visibility) or not 0.0 <= visibility <= 1.0:
                    raise ContractError("Development visibility fraction is invalid")
                mask = scene / "mask_visib" / f"{image_id:06d}_{instance_id:06d}.png"
                if _png_shape(mask) != image_shape:
                    raise ContractError(f"Mask/image dimensions differ: {mask}")
                label_access.append({"role": "mask_visib_png_header", "path": str(mask), "sha256": sha256_file(mask)})
                model = models_root.resolve() / f"obj_{object_id:06d}.ply"
                if not model.is_file():
                    raise ContractError(f"Public CAD model is missing: {model}")
                model_files.setdefault(
                    object_id,
                    {"path": str(model), "bytes": model.stat().st_size, "sha256": sha256_file(model)},
                )
                targets.append(
                    {
                        "target_id": f"s{scene_id:06d}-i{image_id:06d}-n{instance_id:06d}-o{object_id:06d}",
                        "scene_id": scene_id,
                        "image_id": image_id,
                        "instance_id": instance_id,
                        "object_id": object_id,
                        "gray_path": str(gray),
                        "depth_path": str(depth),
                        "mask_visib_path": str(mask),
                        "model_path": str(model),
                        "depth_scale": depth_scale,
                        "camera_intrinsics": [float(value) for value in camera["cam_K"]],
                    }
                )
                per_scene[scene_id] += 1
                per_object[object_id] += 1
    if len(targets) != protocol["frozen_slice"]["target_count"]:
        raise ContractError(f"Readiness v2 target count mismatch: {len(targets)}")
    if len(per_scene) != 3 or len(per_object) < 2:
        raise ContractError("Readiness v2 scene/object grouping gate failed")

    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_manifest_path.with_name(target_manifest_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in targets:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(target_manifest_path)
    access = {
        "schema_version": "poseloop.r4a.development-label-access.v2",
        "role": "DEVELOPMENT_ONLY",
        "opened_files": label_access,
        "opened_file_count": len(label_access),
        "development_asset_files_hashed": extracted_count,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
        "gt_pose_export_to_prediction_manifest": False,
        "score_guided_pose_selection": False,
    }
    write_json_atomic(label_access_path, access)
    audit = {
        "schema_version": "poseloop.r4a.development-readiness.v2",
        "protocol_id": PROTOCOL_ID,
        "status": "ready",
        "prefreeze_receipt_sha256": sha256_file(prefreeze_receipt_path),
        "extracted_file_count": extracted_count,
        "target_count": len(targets),
        "scene_count": len(per_scene),
        "object_count": len(per_object),
        "targets_per_scene": {str(key): value for key, value in sorted(per_scene.items())},
        "targets_per_object": {str(key): value for key, value in sorted(per_object.items())},
        "model_files": {str(key): value for key, value in sorted(model_files.items())},
        "pinned_toolkit_loader_scene_count": loader_scenes,
        "official_loader_schema_pass": loader_scenes == 3,
        "target_manifest_path": str(target_manifest_path.resolve()),
        "target_manifest_sha256": sha256_file(target_manifest_path),
        "label_access_path": str(label_access_path.resolve()),
        "label_access_sha256": sha256_file(label_access_path),
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
        "gate_pass": len(targets) == 173 and len(per_scene) == 3 and len(per_object) >= 2 and loader_scenes == 3,
    }
    write_json_atomic(output_path, audit)
    return audit
