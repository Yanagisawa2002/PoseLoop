"""Readiness audit that parses only the frozen development slice."""

from __future__ import annotations

import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic

from . import PROTOCOL_ID


PROTOCOL_SCHEMA = "poseloop.r4a.development-readiness.protocol.v1"


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A readiness protocol identity mismatch")
    if protocol.get("state") != "frozen_before_development_label_parse":
        raise ContractError("R4-A readiness protocol is not frozen")
    source = protocol.get("frozen_slice", {})
    for key in ("protocol_sha256", "prefreeze_receipt_sha256", "plan_sha256", "extraction_result_sha256", "extraction_job_log_sha256", "extraction_job_exit_sha256"):
        if not _valid_hash(source.get(key)):
            raise ContractError(f"Invalid frozen slice hash: {key}")
    if source.get("entry_count") != 188 or source.get("target_count") != 173:
        raise ContractError("Frozen slice counts changed")
    selection = protocol.get("selection", {})
    if selection.get("scene_ids") != [0, 1, 2] or selection.get("image_ids") != [0]:
        raise ContractError("Readiness selection changed")
    if selection.get("input_modality") != "gray":
        raise ContractError("Readiness modality changed")
    boundary = protocol.get("boundaries", {})
    if boundary.get("xyzibd_val_access_permitted") is not False or boundary.get("evaluator_invocation_permitted") is not False:
        raise ContractError("Readiness validation/evaluator boundary changed")
    if boundary.get("gt_pose_export_to_prediction_manifest") is not False:
        raise ContractError("GT pose export became permitted")
    return protocol


def _png_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ContractError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ContractError(f"Invalid PNG dimensions: {path}")
    return width, height


def _finite_sequence(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(math.isfinite(float(item)) for item in value)


def _verify_extraction(result: Mapping[str, Any], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = result.get("files")
    if not isinstance(files, list) or len(files) != int(plan["entry_count"]):
        raise ContractError("Extraction result file count mismatch")
    plan_names = {str(row["name"]) for row in plan["entries"]}
    result_names = {str(row["entry"]) for row in files}
    if plan_names != result_names:
        raise ContractError("Extraction result entry domain differs from plan")
    verified = []
    for row in files:
        path = Path(row["output_path"]).resolve()
        if not path.is_file():
            raise ContractError(f"Extracted file is missing: {path}")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ContractError(f"Extracted file SHA mismatch: {path}")
        verified.append({"entry": row["entry"], "path": str(path), "bytes": path.stat().st_size, "sha256": actual})
    return verified


def audit_readiness(
    *,
    protocol_path: Path,
    plan_path: Path,
    extraction_result_path: Path,
    prefreeze_receipt_path: Path,
    data_root: Path,
    models_root: Path,
    output_path: Path,
    target_manifest_path: Path,
    label_access_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    frozen = protocol["frozen_slice"]
    expected_hashes = {
        "plan": frozen["plan_sha256"],
        "extraction_result": frozen["extraction_result_sha256"],
        "prefreeze_receipt": frozen["prefreeze_receipt_sha256"],
    }
    actual_hashes = {
        "plan": sha256_file(plan_path),
        "extraction_result": sha256_file(extraction_result_path),
        "prefreeze_receipt": sha256_file(prefreeze_receipt_path),
    }
    if actual_hashes != expected_hashes:
        raise ContractError(f"Readiness frozen input hash mismatch: {actual_hashes!r}")
    plan = read_json(plan_path)
    result = read_json(extraction_result_path)
    verified_files = _verify_extraction(result, plan)

    try:
        from bop_toolkit_lib import inout
    except ImportError as error:  # pragma: no cover - exercised on GPU-A
        raise ContractError(f"Pinned BOP toolkit import failed: {error}") from error

    root = data_root.resolve()
    model_root = models_root.resolve()
    if "val" in {part.lower() for part in root.parts}:
        raise ContractError("Readiness data root points at a validation path")
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
            if not isinstance(camera, dict) or not _finite_sequence(camera.get("cam_K"), 9):
                raise ContractError("Selected camera entry has invalid cam_K")
            depth_scale = float(camera.get("depth_scale", 0.0))
            if not math.isfinite(depth_scale) or depth_scale <= 0:
                raise ContractError("Selected camera entry has invalid depth_scale")
            gray = scene / "gray" / f"{image_id:06d}.png"
            depth = scene / "depth" / f"{image_id:06d}.png"
            image_shape = _png_shape(gray)
            if _png_shape(depth) != image_shape:
                raise ContractError("Gray/depth dimensions differ")
            for instance_id, (gt, info) in enumerate(zip(gt_rows, info_rows)):
                object_id = int(gt["obj_id"])
                if object_id not in allowed_objects:
                    raise ContractError(f"GT object is outside the public model domain: {object_id}")
                if not _finite_sequence(gt.get("cam_R_m2c"), 9) or not _finite_sequence(gt.get("cam_t_m2c"), 3):
                    raise ContractError("Development GT pose schema is invalid")
                visibility = float(info.get("visib_fract", math.nan))
                if not math.isfinite(visibility) or not 0.0 <= visibility <= 1.0:
                    raise ContractError("Development visibility fraction is invalid")
                mask = scene / "mask_visib" / f"{image_id:06d}_{instance_id:06d}.png"
                if _png_shape(mask) != image_shape:
                    raise ContractError(f"Mask/image dimensions differ: {mask}")
                label_access.append({"role": "mask_visib_png_header", "path": str(mask), "sha256": sha256_file(mask)})
                model = model_root / f"obj_{object_id:06d}.ply"
                if not model.is_file():
                    raise ContractError(f"Public CAD model is missing: {model}")
                model_files.setdefault(object_id, {"path": str(model), "bytes": model.stat().st_size, "sha256": sha256_file(model)})
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
    if len(targets) != frozen["target_count"]:
        raise ContractError(f"Readiness target count mismatch: {len(targets)}")
    if len(per_scene) != 3 or len(per_object) < 2:
        raise ContractError("Readiness scene/object grouping gate failed")

    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_manifest_path.with_name(target_manifest_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in targets:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(target_manifest_path)
    access = {
        "schema_version": "poseloop.r4a.development-label-access.v1",
        "role": "DEVELOPMENT_ONLY",
        "opened_files": label_access,
        "opened_file_count": len(label_access),
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
        "gt_pose_export_to_prediction_manifest": False,
        "score_guided_pose_selection": False,
    }
    write_json_atomic(label_access_path, access)
    audit = {
        "schema_version": "poseloop.r4a.development-readiness.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "ready",
        "frozen_input_hashes": actual_hashes,
        "extracted_file_count": len(verified_files),
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
