#!/usr/bin/env python3
"""Render occlusion-aware CAD visibility features for M4-R1 on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
RENDER_RESOLUTION = 256
OCCLUSION_FEATURE_NAMES = (
    "occ_target_visible_surface_fraction",
    "occ_candidate_visible_surface_fraction",
    "occ_new_surface_fraction",
    "occ_lost_surface_fraction",
    "occ_union_surface_fraction",
    "occ_visible_surface_jaccard",
    "occ_overlap_over_target",
    "occ_candidate_silhouette_fraction",
    "occ_silhouette_ratio",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "artifacts" / "r1" / "m4_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=output_root / "preacquisition_features.jsonl",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--output", type=Path, default=output_root / "cad_occlusion_features.jsonl"
    )
    parser.add_argument(
        "--summary", type=Path, default=output_root / "cad_occlusion_summary.json"
    )
    parser.add_argument("--resolution", type=int, default=RENDER_RESOLUTION)
    return parser.parse_args()


def _unit(values: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Viewing direction is degenerate")
    return vector / norm


def _direction(features: Mapping[str, Any], prefix: str) -> np.ndarray:
    return _unit([features[f"{prefix}_{axis}"] for axis in ("x", "y", "z")])


def _direction_key(direction: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(direction, decimals=12))


def _camera_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(reference, direction))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = _unit(np.cross(reference, direction))
    up = _unit(np.cross(direction, right))
    return right, up


def _load_mesh(dataset_root: Path, object_id: int) -> dict[str, Any]:
    import trimesh

    path = dataset_root / "models" / f"obj_{object_id:06d}.ply"
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Not a triangle mesh: {path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if len(faces) == 0 or np.any(areas <= 0.0):
        raise ValueError(f"Invalid CAD geometry: {path}")
    centre = 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
    vertices = vertices - centre
    radius = float(np.max(np.linalg.norm(vertices, axis=1)))
    if radius <= 0.0:
        raise ValueError(f"Degenerate CAD radius: {path}")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "areas": areas,
        "surface_area": float(np.sum(areas)),
        "radius": radius,
    }


def _clip_positions(
    vertices: np.ndarray, radius: float, directions: list[np.ndarray]
) -> np.ndarray:
    batches = []
    scale = radius / 0.9
    for direction in directions:
        right, up = _camera_basis(direction)
        x = vertices @ right / scale
        y = vertices @ up / scale
        # Larger dot(direction) is closer to the camera; negate it so the
        # nearest surface has the smaller clip-space depth.
        z = -(vertices @ direction) / scale
        batches.append(
            np.column_stack((x, y, z, np.ones(len(vertices), dtype=np.float64)))
        )
    return np.asarray(batches, dtype=np.float32)


def render_object(
    context: Any,
    mesh: Mapping[str, Any],
    directions: list[np.ndarray],
    resolution: int,
) -> dict[tuple[float, float, float], dict[str, Any]]:
    import nvdiffrast.torch as dr
    import torch

    positions = torch.as_tensor(
        _clip_positions(mesh["vertices"], float(mesh["radius"]), directions),
        dtype=torch.float32,
        device="cuda",
    )
    triangles = torch.as_tensor(
        mesh["faces"], dtype=torch.int32, device="cuda"
    )
    raster, _ = dr.rasterize(
        context, positions, triangles, resolution=[resolution, resolution]
    )
    triangle_ids = raster[..., 3].to(torch.int64).cpu().numpy() - 1
    output = {}
    normals = np.asarray(mesh["normals"], dtype=np.float64)
    areas = np.asarray(mesh["areas"], dtype=np.float64)
    for index, direction in enumerate(directions):
        ids = triangle_ids[index]
        valid_ids = ids[ids >= 0]
        visible = np.zeros(len(areas), dtype=bool)
        if len(valid_ids):
            visible[np.unique(valid_ids)] = True
        # Closed meshes should expose camera-facing triangles.  This filter
        # removes rare back-facing IDs caused by two-sided rasterization.
        visible &= normals @ direction > 1e-9
        output[_direction_key(direction)] = {
            "visible_faces": visible,
            "visible_surface_area": float(np.sum(areas[visible])),
            "silhouette_pixels": int(np.count_nonzero(ids >= 0)),
        }
    return output


def pair_features(
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
    face_areas: np.ndarray,
    surface_area: float,
    resolution: int,
) -> dict[str, float]:
    target_visible = np.asarray(target["visible_faces"], dtype=bool)
    candidate_visible = np.asarray(candidate["visible_faces"], dtype=bool)
    overlap = target_visible & candidate_visible
    union = target_visible | candidate_visible
    new = candidate_visible & ~target_visible
    lost = target_visible & ~candidate_visible
    target_area = float(np.sum(face_areas[target_visible]))
    candidate_area = float(np.sum(face_areas[candidate_visible]))
    overlap_area = float(np.sum(face_areas[overlap]))
    union_area = float(np.sum(face_areas[union]))
    canvas_pixels = resolution * resolution
    features = {
        "occ_target_visible_surface_fraction": target_area / surface_area,
        "occ_candidate_visible_surface_fraction": candidate_area / surface_area,
        "occ_new_surface_fraction": float(np.sum(face_areas[new])) / surface_area,
        "occ_lost_surface_fraction": float(np.sum(face_areas[lost])) / surface_area,
        "occ_union_surface_fraction": union_area / surface_area,
        "occ_visible_surface_jaccard": overlap_area / max(union_area, 1e-12),
        "occ_overlap_over_target": overlap_area / max(target_area, 1e-12),
        "occ_candidate_silhouette_fraction": int(candidate["silhouette_pixels"])
        / canvas_pixels,
        "occ_silhouette_ratio": int(candidate["silhouette_pixels"])
        / max(int(target["silhouette_pixels"]), 1),
    }
    if set(features) != set(OCCLUSION_FEATURE_NAMES):
        raise AssertionError("Occlusion feature contract mismatch")
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Non-finite CAD occlusion feature")
    return {name: float(features[name]) for name in OCCLUSION_FEATURE_NAMES}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    allowed_roots = [
        (repo_root / "artifacts" / "r1" / "m4_r1").resolve(),
        (repo_root / "artifacts" / "r2" / "m4_r2").resolve(),
    ]
    feature_path = args.features.resolve()
    output_path = args.output.resolve()
    summary_path = args.summary.resolve()
    selected_root = next(
        (root for root in allowed_roots if feature_path.is_relative_to(root)), None
    )
    if not feature_path.is_file() or selected_root is None:
        raise ValueError("CAD renderer input must be an existing R1/R2 M4 feature stream")
    if not output_path.is_relative_to(selected_root) or not summary_path.is_relative_to(
        selected_root
    ):
        raise ValueError("CAD renderer outputs must stay beside their R1/R2 M4 inputs")
    if args.resolution < 32 or args.resolution > 1024:
        raise ValueError("Render resolution must be in [32, 1024]")
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    rows = load_jsonl(feature_path)
    by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("record_type") != "m4_r1_preacquisition_feature":
            raise ValueError("Renderer input contains a non-feature row")
        by_object[int(row["object_id"])].append(row)

    import nvdiffrast.torch as dr
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for M4-R1 CAD visibility rendering")
    context = dr.RasterizeCudaContext(device=torch.cuda.current_device())
    rendered_by_object = {}
    meshes = {}
    for object_id, object_rows in sorted(by_object.items()):
        mesh = _load_mesh(dataset_root, object_id)
        meshes[object_id] = mesh
        unique: dict[tuple[float, float, float], np.ndarray] = {}
        for row in object_rows:
            features = row["features"]
            for prefix in ("target_view_direction_object", "candidate_view_direction_object"):
                direction = _direction(features, prefix)
                unique[_direction_key(direction)] = direction
        directions = [unique[key] for key in sorted(unique)]
        rendered_by_object[object_id] = render_object(
            context, mesh, directions, args.resolution
        )
        print(
            f"CAD visibility object={object_id}: directions={len(directions)}, "
            f"faces={len(mesh['faces'])}",
            flush=True,
        )

    output_rows = []
    for row in rows:
        object_id = int(row["object_id"])
        features = row["features"]
        target_direction = _direction(features, "target_view_direction_object")
        candidate_direction = _direction(features, "candidate_view_direction_object")
        rendered = rendered_by_object[object_id]
        mesh = meshes[object_id]
        output_rows.append(
            {
                "record_type": "m4_r1_cad_occlusion_feature",
                "schema_version": SCHEMA_VERSION,
                "group_id": row["group_id"],
                "object_id": object_id,
                "physical_instance_id": row["physical_instance_id"],
                "candidate_slot": row["candidate_slot"],
                "features": pair_features(
                    rendered[_direction_key(target_direction)],
                    rendered[_direction_key(candidate_direction)],
                    np.asarray(mesh["areas"], dtype=np.float64),
                    float(mesh["surface_area"]),
                    args.resolution,
                ),
            }
        )
    write_jsonl_atomic(output_path, output_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M4-R1 CAD visibility",
        "renderer": "nvdiffrast CUDA z-buffer",
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "resolution": args.resolution,
        "input_feature_stream": {
            "path": str(feature_path),
            "sha256": sha256_file(feature_path),
            "contains_candidate_outcome": False,
        },
        "candidate_outcome_stream_read": False,
        "row_count": len(output_rows),
        "feature_names": list(OCCLUSION_FEATURE_NAMES),
        "objects": {
            str(object_id): {
                "model_path": str(mesh["path"]),
                "model_sha256": mesh["sha256"],
                "face_count": len(mesh["faces"]),
            }
            for object_id, mesh in sorted(meshes.items())
        },
    }
    summary["configuration_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json_atomic(summary_path, summary)
    print(f"CAD occlusion rows={len(output_rows)}, sha256={sha256_file(output_path)}")
    print(output_path)


if __name__ == "__main__":
    main()
