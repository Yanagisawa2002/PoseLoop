"""Exact-entry extraction and deterministic depth-mask generation for R4-A v3."""

from __future__ import annotations

import binascii
import math
from pathlib import Path
from typing import Any, Mapping

from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic
from r4a_development_repair.development import camera_world_to_camera_pose_m
from r4a_development_repair.splitzip import Entry, Part, SplitZip
from r4a_development_repair_v4.splitzip import ShortBodyRetryRangeClient

from . import PROTOCOL_ID
from .contract import ASSET_PLAN_SCHEMA, ASSET_PREFREEZE_SCHEMA, load_protocol, safe_archive_relative, verify_prefreeze


ASSET_RESULT_SCHEMA = "poseloop.r4a.development-asset-extraction.v3"


def _crc32(path: Path) -> int:
    result = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result = binascii.crc32(block, result)
    return result & 0xFFFFFFFF


def _entry(row: Mapping[str, Any]) -> Entry:
    return Entry(
        name=str(row["name"]),
        disk_start=int(row["disk_start"]),
        local_offset=int(row["local_offset"]),
        compressed_size=int(row["compressed_size"]),
        uncompressed_size=int(row["uncompressed_size"]),
        compression=int(row["compression"]),
        crc32=int(row["crc32"]),
        flags=int(row["flags"]),
    )


def extract_assets(
    *,
    protocol_path: Path,
    asset_plan_path: Path,
    prefreeze_path: Path,
    output_root: Path,
    cache_root: Path,
    result_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    receipt = verify_prefreeze(prefreeze_path, schema=ASSET_PREFREEZE_SCHEMA, protocol_path=protocol_path)
    plan = read_json(asset_plan_path)
    if plan.get("schema_version") != ASSET_PLAN_SCHEMA or plan.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A v3 asset extraction plan mismatch")
    if receipt.get("asset_plan_sha256") != sha256_file(asset_plan_path):
        raise ContractError("R4-A v3 asset plan changed after freeze")
    archives = protocol["official_distribution"]["archives"]
    parts = [Part(row["filename"], row["url"], int(row["bytes"]), row["sha256"]) for row in archives]
    limits = protocol["id_discovery_download_contract"]
    client = ShortBodyRetryRangeClient(
        maximum_bytes=int(protocol["exact_asset_selection"]["maximum_network_bytes_including_headers"]),
        chunk_bytes=int(limits["range_chunk_bytes"]),
        cache_root=cache_root.resolve(),
        maximum_attempts=int(limits["maximum_attempts_per_chunk"]),
        retry_statuses=limits["retry_http_statuses"],
        backoff_seconds=limits["retry_backoff_seconds"],
    )
    archive = SplitZip(parts, client)
    root = output_root.resolve()
    files = []
    for index, row in enumerate(plan["entries"]):
        path = root / safe_archive_relative(str(row["name"]))
        size = int(row["uncompressed_size"])
        crc = int(row["crc32"])
        if path.is_file() and path.stat().st_size == size and _crc32(path) == crc:
            status = "resumed_verified"
        else:
            archive.extract_entry(_entry(row), path)
            status = "downloaded_verified"
        files.append(
            {
                "index": index,
                "entry": row["name"],
                "role": row["role"],
                "scene_id": row["scene_id"],
                "image_id": row["image_id"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": status,
            }
        )
        print(f"ASSET_PROGRESS {index + 1}/{len(plan['entries'])} {status} {row['name']}", flush=True)
    result = {
        "schema_version": ASSET_RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "asset_plan_sha256": sha256_file(asset_plan_path),
        "prefreeze_sha256": sha256_file(prefreeze_path),
        "entry_count": len(files),
        "files": files,
        "network_bytes_downloaded": client.bytes_downloaded,
        "cache_bytes_reused": client.bytes_from_cache,
        "network_requests": client.requests,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
    }
    write_json_atomic(result_path, result)
    return result


def depth_component_mask(raw_depth: Any) -> Any:
    """Exact label-blind mask algorithm frozen by GPU-C commit 9d75eb1."""

    import cv2
    import numpy as np

    if raw_depth.ndim != 2:
        raise ContractError("Depth image must be two-dimensional")
    height, width = raw_depth.shape
    valid = np.isfinite(raw_depth) & (raw_depth > 0)
    y0, y1 = height // 8, height - height // 8
    x0, x1 = width // 8, width - width // 8
    roi = np.zeros_like(valid)
    roi[y0:y1, x0:x1] = True
    central_values = raw_depth[valid & roi]
    if len(central_values) < 64:
        raise ContractError("Too few valid depth pixels for audited prediction")
    threshold = float(np.quantile(central_values.astype(np.float64), 0.35))
    candidate = (valid & roi & (raw_depth <= threshold)).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
    choices: list[tuple[float, int, int]] = []
    image_center = np.asarray([width / 2.0, height / 2.0])
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 64:
            continue
        distance = float(np.linalg.norm(centroids[label] - image_center))
        choices.append((distance / max(width, height) - area / (height * width), -area, label))
    if not choices:
        raise ContractError("Depth foreground predictor found no stable component")
    return np.ascontiguousarray(labels == min(choices)[2])


def coco_rle(mask: Any) -> dict[str, Any]:
    """Encode an uncompressed COCO RLE in column-major order."""

    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ContractError("COCO RLE mask must be two-dimensional")
    flat = binary.T.reshape(-1)
    counts = []
    current = 0
    run = 0
    for value in flat:
        bit = int(value != 0)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = bit
    counts.append(run)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def bbox_from_mask(mask: Any) -> list[float]:
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ContractError("Predicted input mask is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def selected_camera(camera_path: Path, image_id: int) -> dict[str, Any]:
    camera = read_json(camera_path).get(str(image_id))
    if not isinstance(camera, dict):
        raise ContractError("Selected development camera row is missing")
    intrinsics = camera.get("cam_K")
    if not isinstance(intrinsics, list) or len(intrinsics) != 9:
        raise ContractError("Selected development camera intrinsics are invalid")
    values = [float(value) for value in intrinsics]
    depth_scale = float(camera.get("depth_scale", 0.0))
    if not all(math.isfinite(value) for value in values) or not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ContractError("Selected development camera contains non-finite values")
    return {
        "camera_intrinsics": values,
        "depth_scale": depth_scale,
        "camera_world_to_camera_pose_m": camera_world_to_camera_pose_m(camera),
    }


__all__ = [
    "ASSET_RESULT_SCHEMA",
    "bbox_from_mask",
    "coco_rle",
    "depth_component_mask",
    "extract_assets",
    "selected_camera",
]
