"""Development-only target enumeration and geometry conversion.

The functions in this file may consume the explicitly frozen XYZ-IBD
``train_pbr`` slice.  They reject validation paths and require a verified
pre-freeze receipt before any label-bearing JSON or mask is opened.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import ContractError, read_json, verify_prefreeze_receipt


def _reject_validation_path(path: Path) -> None:
    parts = {part.lower() for part in path.resolve().parts}
    if "val" in parts or any("official_once" in part or "score" in part for part in parts):
        raise ContractError(f"Forbidden validation/evaluation path in R4-A: {path}")


def enumerate_targets(
    *,
    dataset_root: Path,
    scene_ids: Sequence[int],
    image_ids: Sequence[int],
    prefreeze_receipt: Path,
    protocol_path: Path,
) -> list[dict[str, Any]]:
    """Enumerate every GT instance in a predeclared development slice.

    Object identity and visible masks are development-only oracle inputs.  They
    are used identically by before/after geometry diagnostics and never select
    among pose candidates.
    """

    verify_prefreeze_receipt(prefreeze_receipt, protocol_path)
    root = dataset_root.resolve()
    _reject_validation_path(root)
    targets: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene_root = root / "train_pbr" / f"{scene_id:06d}"
        gt_path = scene_root / "scene_gt.json"
        info_path = scene_root / "scene_gt_info.json"
        camera_path = scene_root / "scene_camera.json"
        for path in (gt_path, info_path, camera_path):
            _reject_validation_path(path)
            if not path.is_file():
                raise ContractError(f"Required development file is missing: {path}")
        scene_gt = read_json(gt_path)
        scene_info = read_json(info_path)
        scene_camera = read_json(camera_path)
        for image_id in image_ids:
            key = str(image_id)
            gt_rows = scene_gt.get(key)
            info_rows = scene_info.get(key)
            camera = scene_camera.get(key)
            if not isinstance(gt_rows, list) or not isinstance(info_rows, list):
                raise ContractError(f"Missing development GT rows: scene={scene_id} image={image_id}")
            if len(gt_rows) != len(info_rows):
                raise ContractError("scene_gt and scene_gt_info instance counts differ")
            if not isinstance(camera, dict):
                raise ContractError("Development camera entry is missing")
            for instance_id, (gt, info) in enumerate(zip(gt_rows, info_rows)):
                object_id = int(gt["obj_id"])
                mask_path = scene_root / "mask_visib" / f"{image_id:06d}_{instance_id:06d}.png"
                _reject_validation_path(mask_path)
                targets.append(
                    {
                        "target_id": f"s{scene_id:06d}-i{image_id:06d}-n{instance_id:06d}-o{object_id:06d}",
                        "scene_id": scene_id,
                        "image_id": image_id,
                        "instance_id": instance_id,
                        "object_id": object_id,
                        "model_relative_path": f"models/obj_{object_id:06d}.ply",
                        "rgb_relative_path": f"train_pbr/{scene_id:06d}/rgb/{image_id:06d}.jpg",
                        "depth_relative_path": f"train_pbr/{scene_id:06d}/depth/{image_id:06d}.png",
                        "mask_visib_relative_path": mask_path.relative_to(root).as_posix(),
                        "camera": camera,
                        "visibility_fraction": float(info.get("visib_fract", 0.0)),
                    }
                )
    keys = [row["target_id"] for row in targets]
    if not targets or len(keys) != len(set(keys)):
        raise ContractError("Development target enumeration is empty or duplicated")
    return targets


def camera_world_to_camera_pose_m(camera: Mapping[str, Any]) -> list[list[float]]:
    rotation = camera.get("cam_R_w2c")
    translation = camera.get("cam_t_w2c")
    if rotation is None and translation is None:
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    if not isinstance(rotation, list) or len(rotation) != 9:
        raise ContractError("cam_R_w2c must contain nine values")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ContractError("cam_t_w2c must contain three values")
    pose = [
        [float(rotation[row * 3 + col]) for col in range(3)]
        + [float(translation[row]) * 0.001]
        for row in range(3)
    ]
    pose.append([0.0, 0.0, 0.0, 1.0])
    if not all(math.isfinite(value) for row in pose for value in row):
        raise ContractError("Camera pose contains non-finite values")
    return pose


def depth_to_m(raw_depth: Any, depth_scale: float) -> Any:
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ContractError("depth_scale must be finite and positive")
    return raw_depth * (depth_scale * 0.001)


def foundationpose_to_bop_row(
    *,
    scene_id: int,
    image_id: int,
    object_id: int,
    score: float,
    predicted_model_to_camera_pose_m: Sequence[Sequence[float]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    if len(predicted_model_to_camera_pose_m) != 4 or any(len(row) != 4 for row in predicted_model_to_camera_pose_m):
        raise ContractError("FoundationPose output must be a 4x4 model-to-camera pose")
    rotation = [float(predicted_model_to_camera_pose_m[row][col]) for row in range(3) for col in range(3)]
    translation_mm = [float(predicted_model_to_camera_pose_m[row][3]) * 1000.0 for row in range(3)]
    values = [*rotation, *translation_mm, float(score), float(elapsed_seconds)]
    if not all(math.isfinite(value) for value in values):
        raise ContractError("BOP result row contains non-finite values")
    return {
        "scene_id": int(scene_id),
        "im_id": int(image_id),
        "obj_id": int(object_id),
        "score": float(score),
        "R": " ".join(f"{value:.9g}" for value in rotation),
        "t": " ".join(f"{value:.9g}" for value in translation_mm),
        "time": float(elapsed_seconds),
    }


def coverage_audit(targets: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target_keys = {
        (int(row["scene_id"]), int(row["image_id"]), int(row["object_id"]), int(row["instance_id"]))
        for row in targets
    }
    result_keys = {
        (int(row["scene_id"]), int(row["image_id"]), int(row["object_id"]), int(row["instance_id"]))
        for row in results
        if row.get("status") == "success"
    }
    missing = sorted(target_keys - result_keys)
    extra = sorted(result_keys - target_keys)
    fraction = len(target_keys & result_keys) / len(target_keys) if target_keys else 0.0
    return {
        "target_count": len(target_keys),
        "successful_result_count": len(result_keys),
        "coverage_fraction": fraction,
        "missing": missing,
        "extra": extra,
        "gate_pass": bool(target_keys) and fraction == 1.0 and not extra,
    }
