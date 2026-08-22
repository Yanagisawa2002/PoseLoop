#!/usr/bin/env python3
"""Run one real XYZ-IBD instance through upstream FoundationPose registration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FOUNDATIONPOSE_COMMIT = "a1b694b83e633c2cb6115b9063d940a687759392"
CHECKPOINT_SOURCES = {
    (
        "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",
        "39a143ab8ff830a593f92b7ddf5d66c99e59bb44",
    ),
    (
        "gpue/foundationpose-weights",
        "42d49e0633d245b3cf4dea6b1e7ec2b31d5b7654",
    ),
}
CHECKPOINTS = {
    "refiner": {
        "path": "weights/2023-10-28-18-33-37/model_best.pth",
        "size": 68_220_109,
        "sha256": "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
    },
    "scorer": {
        "path": "weights/2024-01-11-20-02-45/model_best.pth",
        "size": 190_229_389,
        "sha256": "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
    },
}
LAYOUTS = {
    "realsense": {
        "image_dir": "rgb_realsense",
        "depth_dir": "depth_realsense",
        "mask_dir": "mask_visib_realsense",
        "image_kind": "rgb",
    },
    "xyz": {
        "image_dir": "gray_xyz",
        "depth_dir": "depth_xyz",
        "mask_dir": "mask_visib_xyz",
        "image_kind": "gray",
    },
}


@dataclass(frozen=True)
class Candidate:
    scene_id: int
    image_id: int
    gt_index: int
    object_id: int
    modality: str
    visible_fraction: float
    declared_mask_pixels: int
    scene_dir: Path
    gt_entry: dict[str, Any]
    camera_entry: dict[str, Any]


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    mask_path: Path
    depth_path: Path
    mask: Any
    raw_depth: Any
    mask_pixels: int
    valid_depth_pixels: int
    valid_depth_ratio: float
    depth_scale: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    asset_base = Path(os.environ.get("POSELOOP_DATA_ROOT", Path.home() / "datasets"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=asset_base / "xyzibd")
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=repo_root / "third_party" / "FoundationPose",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "artifacts" / "smoke")
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=128)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def configure_file_logging(log_path: Path) -> None:
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_checkpoints(foundationpose_root: Path) -> dict[str, Any]:
    manifest_path = foundationpose_root / "weights" / "poseloop_checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    source_key = (manifest.get("repo_id"), manifest.get("revision"))
    if source_key not in CHECKPOINT_SOURCES:
        raise AssertionError(f"Unapproved checkpoint source/revision: {source_key}")

    measured: dict[str, Any] = {}
    for label, expected in CHECKPOINTS.items():
        checkpoint_path = foundationpose_root / expected["path"]
        size = checkpoint_path.stat().st_size
        digest = sha256_file(checkpoint_path)
        if size != expected["size"]:
            raise AssertionError(
                f"{label} size mismatch: expected {expected['size']}, got {size}"
            )
        if digest != expected["sha256"]:
            raise AssertionError(
                f"{label} SHA-256 mismatch: expected {expected['sha256']}, got {digest}"
            )
        measured[label] = {
            "path": str(checkpoint_path),
            "size_bytes": size,
            "sha256": digest,
        }
        logging.info(
            "Verified %s checkpoint: size=%d sha256=%s", label, size, digest
        )

    return {
        "repo_id": manifest["repo_id"],
        "repo_type": manifest["repo_type"],
        "revision": manifest["revision"],
        "models": measured,
    }


def verify_dataset_structure(data_root: Path) -> dict[str, Any]:
    marker = data_root / ".poseloop_xyzibd_complete"
    models_info = data_root / "models" / "models_info.json"
    val_root = data_root / "val"
    required = [marker, models_info, val_root]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"XYZ-IBD structure is incomplete: {missing}")

    base_camera_files = sorted(data_root.glob("camera*.json"))
    model_files = sorted((data_root / "models").glob("obj_*.ply"))
    scene_dirs = sorted(path for path in val_root.iterdir() if path.is_dir())
    if not base_camera_files or not model_files or not scene_dirs:
        raise AssertionError("XYZ-IBD base, models, or validation scenes are empty")

    return {
        "root": str(data_root),
        "completion_marker": str(marker),
        "base_camera_files": [str(path) for path in base_camera_files],
        "model_count": len(model_files),
        "validation_scene_count": len(scene_dirs),
    }


def enumerate_candidates(data_root: Path, modality: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    val_root = data_root / "val"
    for scene_dir in sorted(
        (path for path in val_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        scene_id = int(scene_dir.name)
        gt_path = scene_dir / f"scene_gt_{modality}.json"
        info_path = scene_dir / f"scene_gt_info_{modality}.json"
        camera_path = scene_dir / f"scene_camera_{modality}.json"
        if not (gt_path.is_file() and info_path.is_file() and camera_path.is_file()):
            continue

        scene_gt = read_json(gt_path)
        scene_info = read_json(info_path)
        scene_camera = read_json(camera_path)
        if set(scene_gt) != set(scene_info):
            raise AssertionError(f"GT/info frame mismatch in {scene_dir} ({modality})")

        for image_key, gt_entries in scene_gt.items():
            info_entries = scene_info[image_key]
            if len(gt_entries) != len(info_entries):
                raise AssertionError(
                    f"GT/info instance mismatch in scene {scene_id}, image {image_key}"
                )
            if image_key not in scene_camera:
                raise AssertionError(
                    f"Missing camera entry in scene {scene_id}, image {image_key}"
                )
            for gt_index, (gt_entry, info_entry) in enumerate(
                zip(gt_entries, info_entries, strict=True)
            ):
                candidates.append(
                    Candidate(
                        scene_id=scene_id,
                        image_id=int(image_key),
                        gt_index=gt_index,
                        object_id=int(gt_entry["obj_id"]),
                        modality=modality,
                        visible_fraction=float(info_entry["visib_fract"]),
                        declared_mask_pixels=int(info_entry["px_count_visib"]),
                        scene_dir=scene_dir,
                        gt_entry=gt_entry,
                        camera_entry=scene_camera[image_key],
                    )
                )

    candidates.sort(
        key=lambda item: (
            -item.visible_fraction,
            -item.declared_mask_pixels,
            item.scene_id,
            item.image_id,
            item.gt_index,
        )
    )
    return candidates


def select_candidate(
    data_root: Path,
    modality: str,
    cv2: Any,
    np: Any,
    max_candidates: int,
) -> tuple[Selection | None, dict[str, Any]]:
    candidates = enumerate_candidates(data_root, modality)
    best_ratio = 0.0
    best_mask_pixels = 0
    inspected = 0

    for candidate in candidates[:max_candidates]:
        layout = LAYOUTS[modality]
        stem = f"{candidate.image_id:06d}"
        mask_path = (
            candidate.scene_dir
            / layout["mask_dir"]
            / f"{stem}_{candidate.gt_index:06d}.png"
        )
        depth_path = candidate.scene_dir / layout["depth_dir"] / f"{stem}.png"
        if not mask_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(
                f"Selected metadata references missing depth/mask: {depth_path}, {mask_path}"
            )

        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if mask_image is None or raw_depth is None:
            raise RuntimeError(f"OpenCV could not read {mask_path} or {depth_path}")
        if mask_image.ndim == 3:
            mask_image = mask_image[..., 0]
        if raw_depth.ndim != 2 or mask_image.shape != raw_depth.shape:
            raise AssertionError(
                f"Depth/mask shape mismatch: {raw_depth.shape}, {mask_image.shape}"
            )

        mask = mask_image > 0
        mask_pixels = int(mask.sum())
        if mask_pixels != candidate.declared_mask_pixels:
            raise AssertionError(
                "Visible-mask pixel count differs from scene_gt_info: "
                f"{mask_pixels} != {candidate.declared_mask_pixels}"
            )
        depth_scale = float(candidate.camera_entry["depth_scale"])
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise AssertionError(f"Invalid BOP depth_scale: {depth_scale}")
        physical_depth_mm = raw_depth.astype(np.float32) * depth_scale
        valid = mask & np.isfinite(physical_depth_mm) & (physical_depth_mm > 0)
        valid_pixels = int(valid.sum())
        ratio = valid_pixels / mask_pixels if mask_pixels else 0.0

        inspected += 1
        best_ratio = max(best_ratio, ratio)
        best_mask_pixels = max(best_mask_pixels, mask_pixels)
        logging.info(
            "Candidate modality=%s scene=%06d image=%06d gt=%d obj=%d "
            "visib=%.6f mask=%d valid_depth=%d ratio=%.6f",
            modality,
            candidate.scene_id,
            candidate.image_id,
            candidate.gt_index,
            candidate.object_id,
            candidate.visible_fraction,
            mask_pixels,
            valid_pixels,
            ratio,
        )
        if mask_pixels >= 1000 and valid_pixels >= 1000 and ratio >= 0.25:
            return (
                Selection(
                    candidate=candidate,
                    mask_path=mask_path,
                    depth_path=depth_path,
                    mask=np.ascontiguousarray(mask),
                    raw_depth=np.ascontiguousarray(raw_depth),
                    mask_pixels=mask_pixels,
                    valid_depth_pixels=valid_pixels,
                    valid_depth_ratio=ratio,
                    depth_scale=depth_scale,
                ),
                {
                    "candidate_count": len(candidates),
                    "inspected_count": inspected,
                    "best_valid_depth_ratio": best_ratio,
                    "largest_mask_pixels": best_mask_pixels,
                },
            )

    return (
        None,
        {
            "candidate_count": len(candidates),
            "inspected_count": inspected,
            "best_valid_depth_ratio": best_ratio,
            "largest_mask_pixels": best_mask_pixels,
        },
    )


def load_selected_image(selection: Selection, cv2: Any, np: Any) -> tuple[Any, Path]:
    candidate = selection.candidate
    layout = LAYOUTS[candidate.modality]
    image_path = (
        candidate.scene_dir
        / layout["image_dir"]
        / f"{candidate.image_id:06d}.png"
    )
    if layout["image_kind"] == "rgb":
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"OpenCV could not read RGB image: {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        gray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if gray is None:
            raise RuntimeError(f"OpenCV could not read grayscale image: {image_path}")
        if gray.ndim == 2:
            gray_plane = gray
        elif gray.ndim == 3 and gray.shape[2] == 3:
            if not (
                np.array_equal(gray[..., 0], gray[..., 1])
                and np.array_equal(gray[..., 1], gray[..., 2])
            ):
                raise AssertionError(f"XYZ gray image channels differ: {image_path}")
            gray_plane = gray[..., 0]
        else:
            raise AssertionError(f"Unexpected XYZ gray image shape: {gray.shape}")
        image = np.repeat(gray_plane[..., None], 3, axis=2)

    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise AssertionError(f"Expected contiguous uint8 HxWx3 input, got {image}")
    return image, image_path


def assert_transform(transform: Any, label: str, np: Any, atol: float) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise AssertionError(f"{label} is not a finite 4x4 transform")
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=atol)
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=atol)
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise AssertionError(f"{label} rotation determinant is not +1")


def load_mesh_and_gt(
    data_root: Path, selection: Selection, trimesh: Any, np: Any
) -> tuple[Any, Any, dict[str, Any], Path]:
    candidate = selection.candidate
    model_path = data_root / "models" / f"obj_{candidate.object_id:06d}.ply"
    mesh = trimesh.load(model_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise AssertionError(f"CAD model is not a non-empty Trimesh: {model_path}")

    models_info = read_json(data_root / "models" / "models_info.json")
    object_info = models_info[str(candidate.object_id)]
    raw_extents_mm = np.asarray(mesh.extents, dtype=np.float64).copy()
    info_extents_mm = np.asarray(
        [object_info["size_x"], object_info["size_y"], object_info["size_z"]],
        dtype=np.float64,
    )
    extent_ratio = raw_extents_mm / info_extents_mm
    if not np.all((extent_ratio > 0.5) & (extent_ratio < 2.0)):
        raise AssertionError(
            "Raw mesh extents are inconsistent with millimetre-scale models_info"
        )
    if not np.all(raw_extents_mm > 1.0):
        raise AssertionError("Raw mesh does not appear to be expressed in millimetres")

    mesh.apply_scale(0.001)
    scaled_extents_m = np.asarray(mesh.extents, dtype=np.float64)
    np.testing.assert_allclose(
        scaled_extents_m, raw_extents_mm * 0.001, rtol=1e-7, atol=1e-9
    )
    if not (0.001 < float(scaled_extents_m.max()) < 2.0):
        raise AssertionError("Scaled mesh extent is implausible for metre units")

    raw_rotation = np.asarray(candidate.gt_entry["cam_R_m2c"], dtype=np.float64)
    raw_translation_mm = np.asarray(
        candidate.gt_entry["cam_t_m2c"], dtype=np.float64
    ).reshape(3)
    gt_pose = np.eye(4, dtype=np.float64)
    gt_pose[:3, :3] = raw_rotation.reshape(3, 3)
    gt_pose[:3, 3] = raw_translation_mm * 0.001
    np.testing.assert_allclose(
        gt_pose[:3, 3], raw_translation_mm * 0.001, rtol=0, atol=1e-12
    )
    assert_transform(gt_pose, "GT pose", np, atol=1e-5)

    conversion = {
        "mesh_input_unit": "millimetres",
        "mesh_millimetres_to_metres_factor": 0.001,
        "mesh_raw_extents_mm": raw_extents_mm.tolist(),
        "mesh_models_info_extents_mm": info_extents_mm.tolist(),
        "mesh_scaled_extents_m": scaled_extents_m.tolist(),
        "gt_translation_input_unit": "millimetres",
        "gt_translation_millimetres_to_metres_factor": 0.001,
        "gt_translation_raw_mm": raw_translation_mm.tolist(),
    }
    return mesh, gt_pose, conversion, model_path


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    foundationpose_root = args.foundationpose_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / "run.log"

    if not (foundationpose_root / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose checkout not found: {foundationpose_root}")
    os.chdir(foundationpose_root)
    sys.path.insert(0, str(foundationpose_root))

    import cv2
    import nvdiffrast.torch as dr
    import numpy as np
    import open3d
    import pytorch3d
    import torch
    import trimesh
    import warp
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    from Utils import draw_posed_3d_box, draw_xyz_axis, set_seed

    configure_file_logging(run_log_path)
    logging.info("PoseLoop M0 XYZ-IBD smoke inference started")
    logging.info("Data root: %s", data_root)
    logging.info("FoundationPose root: %s", foundationpose_root)
    logging.info("Output directory: %s", output_dir)

    if args.iteration < 1:
        raise ValueError("--iteration must be at least 1")
    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    torch.cuda.set_device(0)

    commit = git_commit(foundationpose_root)
    if commit != FOUNDATIONPOSE_COMMIT:
        raise AssertionError(
            f"FoundationPose commit mismatch: expected {FOUNDATIONPOSE_COMMIT}, got {commit}"
        )
    checkpoints = verify_checkpoints(foundationpose_root)
    dataset = verify_dataset_structure(data_root)

    realsense_selection, realsense_stats = select_candidate(
        data_root, "realsense", cv2, np, args.max_candidates
    )
    fallback_reason: str | None = None
    xyz_stats: dict[str, Any] | None = None
    if realsense_selection is not None:
        selection = realsense_selection
    else:
        fallback_reason = (
            "No inspected RealSense candidate met mask/depth thresholds: "
            f"{realsense_stats}"
        )
        logging.warning("%s", fallback_reason)
        selection, xyz_stats = select_candidate(
            data_root, "xyz", cv2, np, args.max_candidates
        )
        if selection is None:
            raise RuntimeError(
                f"No usable RealSense or XYZ candidate; "
                f"realsense={realsense_stats}, xyz={xyz_stats}"
            )

    candidate = selection.candidate
    rgb, image_path = load_selected_image(selection, cv2, np)
    height, width = selection.raw_depth.shape
    if rgb.shape[:2] != (height, width) or selection.mask.shape != (height, width):
        raise AssertionError("RGB, depth, and mask shapes do not match")
    if not selection.mask.any():
        raise AssertionError("Selected visible mask is empty")

    raw_depth_float = selection.raw_depth.astype(np.float32)
    depth_mm = raw_depth_float * selection.depth_scale
    depth_m = np.ascontiguousarray(depth_mm * 0.001, dtype=np.float32)
    np.testing.assert_allclose(
        depth_m[selection.mask],
        (
            selection.raw_depth[selection.mask].astype(np.float32)
            * selection.depth_scale
            * 0.001
        ),
        rtol=1e-6,
        atol=1e-7,
    )
    valid_depth = (
        selection.mask & np.isfinite(depth_m) & (depth_m > 0)
    )
    if int(valid_depth.sum()) != selection.valid_depth_pixels:
        raise AssertionError("Depth validity changed during unit conversion")
    median_masked_depth_m = float(np.median(depth_m[valid_depth]))
    if not 0.05 < median_masked_depth_m < 5.0:
        raise AssertionError(
            f"Masked physical depth is implausible in metres: {median_masked_depth_m}"
        )

    camera_values = np.asarray(
        candidate.camera_entry["cam_K"], dtype=np.float64
    ).reshape(-1)
    if camera_values.size != 9 or not np.isfinite(camera_values).all():
        raise AssertionError("cam_K must contain nine finite row-major values")
    camera_intrinsics = np.ascontiguousarray(camera_values.reshape(3, 3))
    np.testing.assert_allclose(camera_intrinsics[2], [0, 0, 1], atol=1e-8)
    if camera_intrinsics[0, 0] <= 0 or camera_intrinsics[1, 1] <= 0:
        raise AssertionError("Camera focal lengths must be positive")
    if not (
        0 <= camera_intrinsics[0, 2] < width
        and 0 <= camera_intrinsics[1, 2] < height
    ):
        raise AssertionError("Camera principal point is outside the image")

    mesh, gt_pose, mesh_gt_conversion, model_path = load_mesh_and_gt(
        data_root, selection, trimesh, np
    )
    gt_z_m = float(gt_pose[2, 3])
    depth_to_gt_ratio = gt_z_m / median_masked_depth_m
    if not 0.1 < depth_to_gt_ratio < 10.0:
        raise AssertionError(
            "GT translation and physical depth are inconsistent; possible double scaling"
        )

    logging.info(
        "Selected sample scene=%06d image=%06d gt=%d object=%d modality=%s",
        candidate.scene_id,
        candidate.image_id,
        candidate.gt_index,
        candidate.object_id,
        candidate.modality,
    )
    logging.info("RGB/gray path: %s", image_path)
    logging.info("Raw depth path: %s", selection.depth_path)
    logging.info("Visible mask path: %s", selection.mask_path)
    logging.info("CAD model path: %s", model_path)
    logging.info("Camera intrinsics: %s", camera_intrinsics.tolist())
    logging.info("Raw BOP depth_scale: %.12g", selection.depth_scale)
    logging.info("GT model-to-camera pose (metres): %s", gt_pose.tolist())

    set_seed(0)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext(device=0)
    estimator = FoundationPose(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
        debug_dir=str(output_dir),
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    registration_started = time.perf_counter()
    predicted_pose = estimator.register(
        K=camera_intrinsics,
        rgb=rgb,
        depth=depth_m,
        ob_mask=selection.mask,
        ob_id=candidate.object_id,
        iteration=args.iteration,
    )
    torch.cuda.synchronize()
    registration_seconds = time.perf_counter() - registration_started
    peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated())

    predicted_pose = np.asarray(predicted_pose, dtype=np.float64).reshape(4, 4)
    assert_transform(predicted_pose, "Predicted pose", np, atol=5e-3)
    if not hasattr(estimator, "poses") or not hasattr(estimator, "scores"):
        raise AssertionError("Registration returned through the early fallback path")
    estimator_poses = np.asarray(estimator.poses.detach().cpu(), dtype=np.float64)
    estimator_scores = np.asarray(estimator.scores.detach().cpu(), dtype=np.float64)
    if not np.isfinite(estimator_poses).all() or not np.isfinite(estimator_scores).all():
        raise AssertionError("Upstream registration produced non-finite poses or scores")

    translation_error_mm = float(
        1000.0 * np.linalg.norm(predicted_pose[:3, 3] - gt_pose[:3, 3])
    )
    cosine = float(
        np.clip(
            (
                np.trace(predicted_pose[:3, :3] @ gt_pose[:3, :3].T)
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )
    rotation_error_degrees = float(np.degrees(np.arccos(cosine)))

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    corners = np.asarray(
        list(itertools.product(*zip(bounds[0], bounds[1], strict=True))),
        dtype=np.float64,
    )
    corners_h = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
    corners_in_camera = (predicted_pose @ corners_h.T).T[:, :3]
    if not np.all(corners_in_camera[:, 2] > 0):
        raise AssertionError("Predicted 3D bounding box is not in front of the camera")

    overlay = draw_posed_3d_box(
        camera_intrinsics,
        rgb.copy(),
        predicted_pose,
        bounds,
        line_color=(0, 255, 0),
        linewidth=3,
    )
    overlay = draw_xyz_axis(
        overlay,
        ob_in_cam=predicted_pose,
        scale=0.5 * float(mesh.extents.max()),
        K=camera_intrinsics,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    changed_overlay_pixels = int(np.any(overlay != rgb, axis=2).sum())
    if changed_overlay_pixels < 20:
        raise AssertionError(
            f"Overlay changed too few pixels to be visible: {changed_overlay_pixels}"
        )
    overlay_path = output_dir / "overlay.png"
    if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write overlay: {overlay_path}")

    result = {
        "schema_version": 1,
        "foundationpose": {
            "commit_sha": commit,
            "root": str(foundationpose_root),
        },
        "checkpoints": checkpoints,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_device_total_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
            "numpy": np.__version__,
            "pytorch3d": getattr(pytorch3d, "__version__", package_version("pytorch3d")),
            "nvdiffrast": package_version("nvdiffrast"),
            "trimesh": trimesh.__version__,
            "open3d": open3d.__version__,
            "opencv": cv2.__version__,
            "warp": warp.__version__,
        },
        "dataset": dataset,
        "selection": {
            "scene_id": f"{candidate.scene_id:06d}",
            "image_id": f"{candidate.image_id:06d}",
            "gt_instance_index": candidate.gt_index,
            "object_id": candidate.object_id,
            "sensor_modality": candidate.modality,
            "visible_fraction": candidate.visible_fraction,
            "mask_pixel_count": selection.mask_pixels,
            "valid_depth_pixel_count": selection.valid_depth_pixels,
            "valid_depth_ratio_inside_mask": selection.valid_depth_ratio,
            "realsense_search": realsense_stats,
            "xyz_search": xyz_stats,
            "xyz_fallback_reason": fallback_reason,
        },
        "inputs": {
            "image": str(image_path),
            "raw_depth": str(selection.depth_path),
            "visible_mask": str(selection.mask_path),
            "cad_model": str(model_path),
            "camera_intrinsics_row_major": camera_intrinsics.tolist(),
            "raw_bop_depth_scale": selection.depth_scale,
        },
        "unit_conversions": {
            "raw_depth_to_physical_millimetres": {
                "formula": "raw_depth * depth_scale",
                "depth_scale": selection.depth_scale,
            },
            "depth_millimetres_to_metres_factor": 0.001,
            "median_valid_masked_depth_m": median_masked_depth_m,
            **mesh_gt_conversion,
        },
        "predicted_model_to_camera_pose_m": predicted_pose.tolist(),
        "gt_model_to_camera_pose_m": gt_pose.tolist(),
        "metrics": {
            "translation_error_mm": translation_error_mm,
            "rotation_error_degrees_non_symmetry_aware": rotation_error_degrees,
            "accuracy_claim": (
                "None; this is one smoke-test sample and the rotation metric is "
                "not symmetry-aware."
            ),
        },
        "runtime": {
            "registration_iteration_count": args.iteration,
            "registration_seconds": registration_seconds,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "pose_hypothesis_count": int(estimator_poses.shape[0]),
        },
        "outputs": {
            "overlay": str(overlay_path),
            "result": str(output_dir / "result.json"),
            "run_log": str(run_log_path),
            "overlay_changed_pixel_count": changed_overlay_pixels,
        },
    }
    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    logging.info(
        "Registration completed in %.3f s, peak CUDA memory=%d bytes",
        registration_seconds,
        peak_cuda_memory_bytes,
    )
    logging.info(
        "Predicted translation m=%s; GT translation m=%s",
        predicted_pose[:3, 3].tolist(),
        gt_pose[:3, 3].tolist(),
    )
    logging.info(
        "Predicted model-to-camera pose (metres): %s", predicted_pose.tolist()
    )
    logging.info(
        "Raw translation error=%.3f mm; rotation error (non-symmetry-aware)=%.3f deg",
        translation_error_mm,
        rotation_error_degrees,
    )
    logging.info("Wrote %s", overlay_path)
    logging.info("Wrote %s", result_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("PoseLoop M0 XYZ-IBD smoke inference failed")
        raise
