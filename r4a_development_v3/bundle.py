"""Build isolated no-GT inference and GPU-A-only evaluator bundles for R4-A v3."""

from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from r4a_development_repair.core import ContractError, canonical_sha256, read_json, sha256_file, write_json_atomic

from . import PROTOCOL_ID
from .assets import bbox_from_mask, coco_rle, depth_component_mask, selected_camera
from .contract import load_protocol
from .id_job import EXACT_SELECTION_SCHEMA


INFERENCE_MANIFEST_SCHEMA = "poseloop.r4a.no-gt-inference-manifest.v3"
HANDOFF_SCHEMA = "poseloop.r4a.gpu-c-handoff.v3"
EVALUATOR_MANIFEST_SCHEMA = "poseloop.r4a.evaluator-only-manifest.v3"


def _require_new_directory(path: Path) -> Path:
    root = path.resolve()
    if root.exists() and any(root.iterdir()):
        raise ContractError(f"R4-A v3 output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if "val" in {part.lower() for part in root.parts}:
        raise ContractError("R4-A v3 output path contains forbidden validation component")
    return root


def _copy(source: Path, target: Path) -> str:
    if not source.is_file():
        raise ContractError(f"R4-A v3 source asset is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return sha256_file(target)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _member_hashes(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    skip = excluded or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in skip:
            continue
        rows.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _write_sums(root: Path, rows: list[dict[str, Any]]) -> Path:
    output = root / "SHA256SUMS"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row['sha256']}  {row['relative_path']}\n")
    return output


def _deterministic_targz(root: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(item for item in root.rglob("*") if item.is_file()):
                    relative = path.relative_to(root).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.size = path.stat().st_size
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    temporary.replace(output)
    return {"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": sha256_file(output)}


def _forbidden_inference_scan(root: Path) -> None:
    forbidden_names = {"scene_gt.json", "scene_gt_info.json", "models_info.json"}
    if any(path.name in forbidden_names or "mask_visib" in path.as_posix() for path in root.rglob("*")):
        raise ContractError("No-GT inference bundle contains a forbidden label filename")
    forbidden_keys = {"cam_R_m2c", "cam_t_m2c", "visib_fract", "pose_error", "gt_pose"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            overlap = forbidden_keys.intersection(value)
            if overlap:
                raise ContractError(f"No-GT inference JSON contains forbidden keys: {sorted(overlap)}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for path in root.rglob("*.json"):
        walk(read_json(path))
    for path in root.rglob("*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    walk(json.loads(line))


def build_inference_bundle(
    *,
    protocol_path: Path,
    exact_selection_path: Path,
    asset_root: Path,
    models_root: Path,
    stage_root: Path,
    archive_path: Path,
    descriptor_path: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    import cv2

    load_protocol(protocol_path)
    exact = read_json(exact_selection_path)
    if exact.get("schema_version") != EXACT_SELECTION_SCHEMA or exact.get("gate_pass") is not True:
        raise ContractError("R4-A v3 inference bundle requires a passing exact selection")
    stage = _require_new_directory(stage_root)
    contracts = stage / "contracts"
    contracts.mkdir(parents=True)
    targets = list(exact["targets"])
    coco_rows: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    shared_hashes: dict[str, str] = {}
    for detection_index, target in enumerate(targets):
        scene_id = int(target["scene_id"])
        image_id = int(target["image_id"])
        object_id = int(target["object_id"])
        item_id = f"s{scene_id:06d}-i{image_id:06d}-o{object_id:06d}"
        source_scene = asset_root.resolve() / "train_pbr" / f"{scene_id:06d}"
        relative = {
            "rgb": f"dataset/rgb/{scene_id:06d}/{image_id:06d}.png",
            "depth": f"dataset/depth/{scene_id:06d}/{image_id:06d}.png",
            "mask": f"dataset/masks/{item_id}.png",
            "camera": f"dataset/cameras/{scene_id:06d}/{image_id:06d}.json",
            "model": f"dataset/models/obj_{object_id:06d}.ply",
        }
        source_rgb = source_scene / "gray" / f"{image_id:06d}.png"
        source_depth = source_scene / "depth" / f"{image_id:06d}.png"
        source_camera = source_scene / "scene_camera.json"
        source_model = models_root.resolve() / f"obj_{object_id:06d}.ply"
        for role, source in (("rgb", source_rgb), ("depth", source_depth), ("model", source_model)):
            destination = stage / relative[role]
            shared_hashes.setdefault(relative[role], _copy(source, destination) if not destination.exists() else sha256_file(destination))
        camera = selected_camera(source_camera, image_id)
        camera_path = stage / relative["camera"]
        if not camera_path.exists():
            write_json_atomic(camera_path, camera)
        shared_hashes.setdefault(relative["camera"], sha256_file(camera_path))
        raw_depth = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
        if raw_depth is None:
            raise ContractError(f"OpenCV could not load exact depth: {source_depth}")
        mask = depth_component_mask(raw_depth)
        mask_path = stage / relative["mask"]
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), mask.astype("uint8") * 255):
            raise ContractError(f"OpenCV could not write predicted mask: {mask_path}")
        segmentation = coco_rle(mask)
        coco_rows.append(
            {
                "scene_id": scene_id,
                "image_id": image_id,
                "category_id": object_id,
                "detection_index": detection_index,
                "score": 1.0,
                "bbox": bbox_from_mask(mask),
                "segmentation": segmentation,
            }
        )
        prepared.append({"item_id": item_id, "target": target, "relative": relative, "camera": camera, "segmentation": segmentation})
    coco_path = contracts / "coco-predictions.json"
    write_json_atomic(coco_path, coco_rows)
    coco_sha = sha256_file(coco_path)
    workload = []
    for detection_index, item in enumerate(prepared):
        target = item["target"]
        relative = item["relative"]
        workload.append(
            {
                "item_id": item["item_id"],
                "object_id": int(target["object_id"]),
                "scene_id": int(target["scene_id"]),
                "image_id": int(target["image_id"]),
                "detection_index": detection_index,
                "category_id": int(target["object_id"]),
                "coco_score": 1.0,
                "coco_prediction_sha256": coco_sha,
                "segmentation_sha256": canonical_sha256(item["segmentation"]),
                "sensor": "synthetic_pbr",
                "rgb_relative_path": relative["rgb"],
                "depth_relative_path": relative["depth"],
                "camera_relative_path": relative["camera"],
                "model_relative_path": relative["model"],
                "rgb_sha256": shared_hashes[relative["rgb"]],
                "depth_sha256": shared_hashes[relative["depth"]],
                "camera_sha256": shared_hashes[relative["camera"]],
                "model_sha256": shared_hashes[relative["model"]],
                "camera_intrinsics": item["camera"]["camera_intrinsics"],
                "depth_scale": item["camera"]["depth_scale"],
                "camera_world_to_camera_pose_m": item["camera"]["camera_world_to_camera_pose_m"],
            }
        )
    workload_path = contracts / "workload.jsonl"
    _write_jsonl(workload_path, workload)
    protocol_copy = contracts / "protocol.json"
    _copy(protocol_path, protocol_copy)
    object_counts = Counter(int(row["object_id"]) for row in workload)
    manifest = {
        "schema_version": INFERENCE_MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": implementation_commit,
        "role": "NO_OBJECT_POSE_GT_INFERENCE",
        "item_count": len(workload),
        "unique_scene_image_object_count": len({(row["scene_id"], row["image_id"], row["object_id"]) for row in workload}),
        "scene_count": len({row["scene_id"] for row in workload}),
        "object_count": len(object_counts),
        "per_object_count": {str(key): value for key, value in sorted(object_counts.items())},
        "coverage": 1.0,
        "manifest_order": [row["item_id"] for row in workload],
        "allowed_roles": ["RGB", "depth", "predicted_input_mask_COCO_RLE", "camera", "CAD"],
        "label_access_count": 0,
        "uses_gt_visible_mask": False,
        "uses_oracle_association": False,
        "fixed_coco_score_semantics": "constant input-mask confidence only; never an evaluator or GT score",
        "forbidden_roles_absent": ["object_pose_GT", "visibility_GT", "evaluator", "evaluator_score", "sealed"],
        "workload_relative_path": "contracts/workload.jsonl",
        "workload_sha256": sha256_file(workload_path),
        "coco_relative_path": "contracts/coco-predictions.json",
        "coco_sha256": coco_sha,
    }
    manifest_path = contracts / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    payload_members = _member_hashes(stage, excluded={"contracts/handoff.json", "SHA256SUMS"})
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": implementation_commit,
        "manifest_relative_path": "contracts/manifest.json",
        "manifest_sha256": sha256_file(manifest_path),
        "workload_relative_path": "contracts/workload.jsonl",
        "workload_sha256": sha256_file(workload_path),
        "coco_relative_path": "contracts/coco-predictions.json",
        "coco_sha256": coco_sha,
        "coverage": manifest["coverage"],
        "label_access_count": 0,
        "payload_member_hashes": payload_members,
        "self_hash_location": "SHA256SUMS",
        "sha256sums_self_exclusion": True,
    }
    handoff_path = contracts / "handoff.json"
    write_json_atomic(handoff_path, handoff)
    sums_rows = _member_hashes(stage, excluded={"SHA256SUMS"})
    sums_path = _write_sums(stage, sums_rows)
    _forbidden_inference_scan(stage)
    archive = _deterministic_targz(stage, archive_path)
    descriptor = {
        "schema_version": "poseloop.r4a.gpu-c-transfer-descriptor.v3",
        **manifest,
        "archive": archive,
        "archive_absolute_local_path": str(archive_path.resolve()),
        "internal_sha256sums_sha256": sha256_file(sums_path),
        "handoff_sha256": sha256_file(handoff_path),
        "extraction_root_layout": {"root": ".", "dataset": "dataset/", "contracts": "contracts/"},
        "forbidden_gt_evaluator_score_sealed_members_absent": True,
    }
    write_json_atomic(descriptor_path, descriptor)
    return descriptor


def build_evaluator_bundle(
    *,
    protocol_path: Path,
    exact_selection_path: Path,
    asset_root: Path,
    models_root: Path,
    models_info_path: Path,
    stage_root: Path,
    archive_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    load_protocol(protocol_path)
    exact = read_json(exact_selection_path)
    if exact.get("schema_version") != EXACT_SELECTION_SCHEMA or exact.get("gate_pass") is not True:
        raise ContractError("R4-A v3 evaluator bundle requires a passing exact selection")
    stage = _require_new_directory(stage_root)
    split = stage / "xyzibd" / "train_pbr_r4a_v3"
    grouped: dict[int, list[dict[str, Any]]] = {}
    for target in exact["targets"]:
        grouped.setdefault(int(target["scene_id"]), []).append(target)
    target_rows = []
    access = []
    for scene_id, targets in sorted(grouped.items()):
        source = asset_root.resolve() / "train_pbr" / f"{scene_id:06d}"
        scene_gt_path = source / "scene_gt.json"
        scene_info_path = source / "scene_gt_info.json"
        scene_camera_path = source / "scene_camera.json"
        scene_gt = read_json(scene_gt_path)
        scene_info = read_json(scene_info_path)
        scene_camera = read_json(scene_camera_path)
        access.extend(
            {"role": role, "path": str(path), "sha256": sha256_file(path)}
            for role, path in (
                ("scene_gt", scene_gt_path),
                ("scene_gt_info", scene_info_path),
                ("scene_camera", scene_camera_path),
            )
        )
        filtered_gt: dict[str, list[Any]] = {}
        filtered_info: dict[str, list[Any]] = {}
        filtered_camera: dict[str, Any] = {}
        for target in targets:
            image_id = int(target["image_id"])
            ordinal = int(target["instance_ordinal"])
            key = str(image_id)
            gt_rows = scene_gt.get(key)
            info_rows = scene_info.get(key)
            if not isinstance(gt_rows, list) or not isinstance(info_rows, list) or ordinal >= len(gt_rows) or ordinal >= len(info_rows):
                raise ContractError("Exact evaluator target ordinal is outside official development labels")
            if int(gt_rows[ordinal].get("obj_id", -1)) != int(target["object_id"]):
                raise ContractError("Exact evaluator target object differs from the ID-only selection")
            filtered_gt.setdefault(key, []).append(gt_rows[ordinal])
            filtered_info.setdefault(key, []).append(info_rows[ordinal])
            filtered_camera[key] = scene_camera[key]
            target_rows.append({"scene_id": scene_id, "im_id": image_id, "obj_id": int(target["object_id"]), "inst_count": 1})
        scene_out = split / f"{scene_id:06d}"
        write_json_atomic(scene_out / "scene_gt.json", filtered_gt)
        write_json_atomic(scene_out / "scene_gt_info.json", filtered_info)
        write_json_atomic(scene_out / "scene_camera.json", filtered_camera)
    models_out = stage / "xyzibd" / "models"
    for object_id in exact["object_ids"]:
        _copy(models_root.resolve() / f"obj_{int(object_id):06d}.ply", models_out / f"obj_{int(object_id):06d}.ply")
    models_info = read_json(models_info_path)
    filtered_models_info = {str(object_id): models_info[str(object_id)] for object_id in exact["object_ids"]}
    write_json_atomic(models_out / "models_info.json", filtered_models_info)
    targets_path = stage / "xyzibd" / "train_pbr_r4a_v3_targets_bop24.json"
    write_json_atomic(targets_path, target_rows)
    bundle_manifest = {
        "schema_version": EVALUATOR_MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "role": "GPU_A_EVALUATOR_ONLY_DEVELOPMENT",
        "must_not_leave_gpu_a": True,
        "target_count": len(target_rows),
        "scene_count": len(grouped),
        "object_count": len(exact["object_ids"]),
        "target_file": str(targets_path.relative_to(stage).as_posix()),
        "label_access": access,
        "label_access_count": len(access),
        "xyzibd_val_access_count": 0,
        "score_guided_selection": False,
        "member_hashes": _member_hashes(stage),
    }
    write_json_atomic(stage / "evaluator-manifest.json", bundle_manifest)
    sums = _member_hashes(stage, excluded={"SHA256SUMS"})
    _write_sums(stage, sums)
    archive = _deterministic_targz(stage, archive_path)
    result = {**bundle_manifest, "archive": archive, "manifest_sha256": sha256_file(stage / "evaluator-manifest.json")}
    write_json_atomic(manifest_path, result)
    return result


__all__ = ["build_evaluator_bundle", "build_inference_bundle"]
