"""Label-free baseline/improved visualization planning and optional rendering."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from .common import (
    PREP_PROTOCOL_ID,
    VISUALIZATION_PLAN_SCHEMA,
    PrepError,
    canonical_bytes,
    canonical_sha256,
    read_jsonl,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
)
from .manifest import asset_path, validate_manifest
from .producer import load_protocol
from .results import prediction_projection


VISUALIZATION_RECEIPT_SCHEMA = "poseloop.r4c.prep.visualization-receipt.v1"


def _result_index(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path.resolve()):
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in indexed:
            raise PrepError("Visualization results contain blank or duplicate item ID")
        prediction_projection(row)
        indexed[item_id] = row
    return indexed


def build_visualization_plan(
    *,
    protocol_path: Path,
    manifest_path: Path,
    baseline_path: Path,
    improved_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
    )
    baseline = _result_index(baseline_path)
    improved = _result_index(improved_path)
    expected_ids = [row["item_id"] for row in items]
    expected_set = set(expected_ids)
    if set(baseline) != expected_set or set(improved) != expected_set:
        raise PrepError("Visualization result coverage is not exactly 100 percent")
    frames: list[dict[str, Any]] = []
    for ordinal, item in enumerate(items):
        item_id = item["item_id"]
        left = prediction_projection(baseline[item_id])
        right = prediction_projection(improved[item_id])
        input_fingerprint = canonical_sha256(
            {
                "sample_key": item["sample_key"],
                "mask_variant_id": item["mask_variant_id"],
                "frame_size": item["frame_size"],
                "inputs": item["inputs"],
                "camera_intrinsics": item["camera_intrinsics"],
            }
        )
        frames.append(
            {
                "ordinal": ordinal,
                "item_id": item_id,
                "sample_key": item["sample_key"],
                "mask_variant_id": item["mask_variant_id"],
                "input_fingerprint": input_fingerprint,
                "baseline_prediction_fingerprint": canonical_sha256(left),
                "improved_prediction_fingerprint": canonical_sha256(right),
                "prediction_projection_exact_equal": (
                    canonical_bytes(left) == canonical_bytes(right)
                ),
                "relative_png_path": f"frames/{ordinal:06d}-{item_id}.png",
            }
        )
    plan = {
        "schema_version": VISUALIZATION_PLAN_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
        "baseline_results_sha256": sha256_file(baseline_path.resolve()),
        "improved_results_sha256": sha256_file(improved_path.resolve()),
        "source_contract": manifest["source_contract"],
        "coverage": {
            "expected": len(items),
            "planned": len(frames),
            "percent": 100.0,
            "missing": [],
            "duplicate": [],
        },
        "layout": protocol["visualization"]["layout"],
        "layers": protocol["visualization"]["layers"],
        "frames": frames,
        "video": {
            "relative_path": "baseline-improved-label-free.mp4",
            **protocol["visualization"]["video"],
        },
        "render_dependencies": ["cv2", "numpy", "trimesh"],
        "render_dependencies_required_for_plan": False,
        "pixel_equality_claim": protocol["visualization"][
            "pixel_equality_claim"
        ],
        "accuracy_claim": "unavailable-label-free-content-visualization",
        "label_access_count": 0,
        "official_scorer_run": False,
        "auto_deploy": False,
    }
    write_json_atomic(output_path, plan)
    return plan


def _decode_mask(path: Path, plugin_id: str, cv2: Any, np: Any) -> Any:
    if plugin_id == "manifest-mask-file-v1":
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PrepError(
                    f"Cannot decode manifest JSON mask file: {path.name}"
                ) from exc
            if not isinstance(value, dict) or set(value) != {
                "width",
                "height",
                "mask",
                "score",
            }:
                raise PrepError("Manifest JSON mask fields differ")
            width, height, flat = value["width"], value["height"], value["mask"]
            if (
                type(width) is not int
                or type(height) is not int
                or width <= 0
                or height <= 0
                or not isinstance(flat, list)
                or len(flat) != width * height
                or any(type(cell) is not int or cell not in (0, 1) for cell in flat)
            ):
                raise PrepError("Manifest JSON mask payload is invalid")
            return np.asarray(flat, dtype=np.uint8).reshape(height, width).astype(bool)
        decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise PrepError(f"Cannot decode manifest mask file: {path.name}")
        return np.asarray(decoded) > 0
    if plugin_id != "coco-rle-inline-v1":
        raise PrepError(f"Unsupported visualization mask plugin: {plugin_id}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepError(f"Cannot decode COCO RLE mask artifact: {path.name}") from exc
    segmentation = value.get("segmentation") if isinstance(value, dict) else None
    if not isinstance(segmentation, dict) or set(segmentation) != {"counts", "size"}:
        raise PrepError("COCO RLE mask artifact must contain only a segmentation object")
    size, counts = segmentation["size"], segmentation["counts"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(number) is not int or number <= 0 for number in size)
    ):
        raise PrepError("COCO RLE size is invalid")
    height, width = size
    if isinstance(counts, str):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise PrepError(
                "Compressed COCO RLE requires an already-installed pycocotools runtime"
            ) from exc
        return np.asarray(
            mask_utils.decode(
                {"counts": counts.encode("ascii"), "size": [height, width]}
            )
        ).astype(bool)
    if not isinstance(counts, list) or any(
        type(number) is not int or number < 0 for number in counts
    ):
        raise PrepError("COCO RLE counts are invalid")
    flat = np.zeros(height * width, dtype=bool)
    cursor, active = 0, False
    for count in counts:
        stop = cursor + count
        if stop > len(flat):
            raise PrepError("COCO RLE exceeds its declared size")
        if active:
            flat[cursor:stop] = True
        cursor, active = stop, not active
    if cursor != len(flat):
        raise PrepError("COCO RLE does not fill its declared size")
    return flat.reshape((width, height)).T


def _project(points: Any, pose: Any, intrinsics: Any, np: Any) -> tuple[Any, Any]:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    camera_points = (pose @ homogeneous.T).T[:, :3]
    valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected = (intrinsics @ camera_points[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def _overlay(
    *,
    bgr: Any,
    mask: Any,
    vertices_m: Any,
    edges: Any,
    pose: Any,
    intrinsics: Any,
    cv2: Any,
    np: Any,
) -> Any:
    image = np.asarray(bgr, dtype=np.uint8).copy()
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != image.shape[:2] or not binary.any():
        raise PrepError("Input mask is empty or differs from RGB size")
    source = image[binary].astype(np.uint16)
    mask_color = np.asarray([180, 0, 180], dtype=np.uint16)
    image[binary] = ((3 * source + 2 * mask_color) // 5).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (255, 0, 255), 1, lineType=cv2.LINE_8)
    pixels, valid = _project(vertices_m, pose, intrinsics, np)
    edge_count = 0
    for first, second in edges:
        if not (valid[first] and valid[second]):
            continue
        endpoints = np.clip(
            np.rint(pixels[[first, second]]), -1_000_000, 1_000_000
        ).astype(np.int32)
        cv2.line(
            image,
            tuple(endpoints[0]),
            tuple(endpoints[1]),
            (255, 255, 0),
            1,
            cv2.LINE_8,
        )
        edge_count += 1
    if edge_count == 0:
        raise PrepError("All CAD edges project behind the camera")
    extent = float(np.max(np.ptp(vertices_m, axis=0)))
    if not math.isfinite(extent) or extent <= 0:
        raise PrepError("CAD extent is invalid")
    length = 0.5 * extent
    axes = np.asarray(
        [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]],
        dtype=np.float64,
    )
    axis_pixels, axis_valid = _project(axes, pose, intrinsics, np)
    if not axis_valid[0]:
        raise PrepError("Predicted object origin is behind the camera")
    origin = tuple(np.rint(axis_pixels[0]).astype(np.int32))
    for index, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), 1):
        if axis_valid[index]:
            endpoint = tuple(np.rint(axis_pixels[index]).astype(np.int32))
            cv2.line(image, origin, endpoint, color, 3, cv2.LINE_8)
    return image


def _panel_label(image: Any, label: str, cv2: Any) -> Any:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (260, 42), (0, 0, 0), -1, cv2.LINE_8)
    cv2.putText(
        panel,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_8,
    )
    return panel


def _mesh(path: Path, trimesh: Any, np: Any, scale: float) -> tuple[Any, Any]:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise PrepError("CAD scene contains no triangle mesh")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise PrepError("CAD model is empty")
    edges = np.asarray(loaded.edges_unique, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise PrepError("CAD model has no renderable edges")
    edges = np.sort(edges, axis=1)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    if len(edges) > 4000:
        edges = edges[np.linspace(0, len(edges) - 1, 4000, dtype=np.int64)]
    return np.asarray(loaded.vertices, dtype=np.float64) * scale, edges


def _write_video(path: Path, frames: list[Any], fps: float, cv2: Any) -> dict[str, Any]:
    if not frames:
        raise PrepError("Cannot create an empty visualization video")
    height, width = frames[0].shape[:2]
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.mp4"
    )
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise PrepError("OpenCV mp4v writer is unavailable")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise PrepError("Visualization frame sizes differ")
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(temporary))
    count = 0
    while True:
        ok, decoded = capture.read()
        if not ok:
            break
        if decoded.shape[:2] != (height, width):
            capture.release()
            raise PrepError("Decoded video frame size differs")
        count += 1
    capture.release()
    if count != len(frames):
        if temporary.exists():
            temporary.unlink()
        raise PrepError("Decoded video frame count differs")
    os.replace(temporary, target)
    return {
        "relative_path": target.name,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "frame_count": count,
        "width": width,
        "height": height,
        "fps": fps,
        "codec": "mp4v",
        "canonical_equivalence_artifact": False,
    }


def render_visualization(
    *,
    protocol_path: Path,
    manifest_path: Path,
    asset_root: Path,
    baseline_path: Path,
    improved_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise PrepError(
            "Visualization rendering requires already-installed cv2, numpy, and trimesh"
        ) from exc
    protocol = load_protocol(protocol_path)
    root = output_root.resolve()
    plan_path = root / "visualization-plan.json"
    plan = build_visualization_plan(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        baseline_path=baseline_path,
        improved_path=improved_path,
        output_path=plan_path,
    )
    _, items = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
        asset_root=asset_root,
        verify_assets=True,
    )
    baseline = _result_index(baseline_path)
    improved = _result_index(improved_path)
    item_by_id = {row["item_id"]: row for row in items}
    frame_arrays: list[Any] = []
    frame_receipts: list[dict[str, Any]] = []
    mesh_cache: dict[str, tuple[Any, Any]] = {}
    for frame_plan in plan["frames"]:
        item_id = frame_plan["item_id"]
        item = item_by_id[item_id]
        rgb_path = asset_path(asset_root, item["inputs"]["rgb"])
        mask_path = asset_path(asset_root, item["inputs"]["mask"])
        cad_path = asset_path(asset_root, item["inputs"]["cad"])
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (
            item["frame_size"]["height"],
            item["frame_size"]["width"],
        ):
            raise PrepError(f"RGB decode or frame size differs: {item_id}")
        mask = _decode_mask(
            mask_path, item["mask_provenance"]["plugin_id"], cv2, np
        )
        nonzero = int(np.count_nonzero(mask))
        coverage = item["mask_provenance"]["coverage"]
        if nonzero != coverage["nonzero_pixels"] or not math.isclose(
            nonzero / mask.size,
            float(coverage["fraction"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PrepError(f"Decoded mask coverage differs: {item_id}")
        cad_sha = item["inputs"]["cad"]["sha256"]
        if cad_sha not in mesh_cache:
            mesh_cache[cad_sha] = _mesh(
                cad_path,
                trimesh,
                np,
                float(protocol["visualization"]["cad_scale_metres"]),
            )
        vertices, edges = mesh_cache[cad_sha]
        intrinsics = np.asarray(item["camera_intrinsics"], dtype=np.float64)
        left_content = _overlay(
            bgr=bgr,
            mask=mask,
            vertices_m=vertices,
            edges=edges,
            pose=np.asarray(
                baseline[item_id]["predicted_model_to_camera_pose_m"],
                dtype=np.float64,
            ),
            intrinsics=intrinsics,
            cv2=cv2,
            np=np,
        )
        right_content = _overlay(
            bgr=bgr,
            mask=mask,
            vertices_m=vertices,
            edges=edges,
            pose=np.asarray(
                improved[item_id]["predicted_model_to_camera_pose_m"],
                dtype=np.float64,
            ),
            intrinsics=intrinsics,
            cv2=cv2,
            np=np,
        )
        left = _panel_label(left_content, "BASELINE", cv2)
        right = _panel_label(right_content, "IMPROVED", cv2)
        side_by_side = np.concatenate([left, right], axis=1)
        ok, encoded = cv2.imencode(
            ".png",
            side_by_side,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not ok:
            raise PrepError(f"Cannot encode visualization PNG: {item_id}")
        frame_path = root / frame_plan["relative_png_path"]
        write_bytes_atomic(frame_path, encoded.tobytes())
        frame_receipts.append(
            {
                **frame_plan,
                "baseline_pixel_sha256": hashlib.sha256(
                    left.tobytes(order="C")
                ).hexdigest(),
                "improved_pixel_sha256": hashlib.sha256(
                    right.tobytes(order="C")
                ).hexdigest(),
                "overlay_content_pixel_equal": bool(
                    np.array_equal(left_content, right_content)
                ),
                "panel_pixels_equal_including_labels": bool(
                    np.array_equal(left, right)
                ),
                "side_by_side_pixel_sha256": hashlib.sha256(
                    side_by_side.tobytes(order="C")
                ).hexdigest(),
                "png_sha256": sha256_file(frame_path),
                "png_bytes": frame_path.stat().st_size,
            }
        )
        frame_arrays.append(side_by_side)
    video = _write_video(
        root / plan["video"]["relative_path"],
        frame_arrays,
        float(plan["video"]["fps"]),
        cv2,
    )
    receipt = {
        "schema_version": VISUALIZATION_RECEIPT_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "plan_sha256": sha256_file(plan_path),
        "coverage": {
            "expected": len(items),
            "rendered": len(frame_receipts),
            "percent": 100.0,
            "missing": [],
            "duplicate": [],
        },
        "frames": frame_receipts,
        "video": video,
        "opened_asset_roles": ["public_rgb", "input_mask", "public_cad"],
        "all_manifest_assets_hash_verified": True,
        "dataset_root_scanned": False,
        "pixel_equality_claim": "no-regression-only-not-accuracy",
        "accuracy_claim": "unavailable-label-free-content-visualization",
        "label_access_count": 0,
        "official_scorer_run": False,
    }
    write_json_atomic(root / "visualization-receipt.json", receipt)
    return receipt
