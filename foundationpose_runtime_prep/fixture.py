"""Create a deterministic, label-free 720p PREP fixture bundle."""

from __future__ import annotations

import binascii
import json
import struct
import zlib
from pathlib import Path
from typing import Any

from .common import (
    PREP_PROTOCOL_ID,
    RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2,
    PrepError,
    canonical_sha256,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
)
from .manifest import (
    A_SOURCE_PROTOCOL_SHA256,
    V2_BOUNDARY_FIXED,
    V2_VARIANT_IDS,
    V2_VARIANT_INPUT_ROLES,
    expected_v2_coverage,
    validate_manifest,
)
from .producer import (
    FIXTURE_CHECKPOINTS,
    FIXTURE_IMPLEMENTATION_COMMIT,
    FIXTURE_IMPLEMENTATION_SHA256,
    FIXTURE_MODEL_SHA256,
)


FIXTURE_SCHEMA = "poseloop.r4c.prep.synthetic-fixture.v2"
WIDTH = 1280
HEIGHT = 720


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png(*, width: int, height: int, channels: int, rows: list[bytes]) -> bytes:
    if channels not in {1, 3} or len(rows) != height:
        raise PrepError("Synthetic PNG dimensions are invalid")
    expected = width * channels
    if any(len(row) != expected for row in rows):
        raise PrepError("Synthetic PNG row width differs")
    header = struct.pack(">IIBBBBB", width, height, 8, 0 if channels == 1 else 2, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _rgb_png(sample_index: int) -> bytes:
    background = bytes((32 + sample_index * 20, 48, 72)) * WIDTH
    foreground = bytes((180, 110 + sample_index * 20, 40))
    rows: list[bytes] = []
    for y in range(HEIGHT):
        if 190 <= y < 530:
            row = bytearray(background)
            left, right = 420 + sample_index * 35, 850 + sample_index * 35
            row[left * 3 : right * 3] = foreground * (right - left)
            rows.append(bytes(row))
        else:
            rows.append(background)
    return _png(width=WIDTH, height=HEIGHT, channels=3, rows=rows)


def _gray_png(value: int) -> bytes:
    row = bytes((value,)) * WIDTH
    return _png(width=WIDTH, height=HEIGHT, channels=1, rows=[row] * HEIGHT)


def _mask_png(sample_index: int, variant_index: int) -> tuple[bytes, int]:
    left = 430 + sample_index * 35 + variant_index * 4
    right = 840 + sample_index * 35 - variant_index * 4
    top = 200 + variant_index * 3
    bottom = 520 - variant_index * 3
    rows: list[bytes] = []
    for y in range(HEIGHT):
        row = bytearray(WIDTH)
        if top <= y < bottom:
            row[left:right] = b"\xff" * (right - left)
        rows.append(bytes(row))
    return (
        _png(width=WIDTH, height=HEIGHT, channels=1, rows=rows),
        (right - left) * (bottom - top),
    )


def _mask_json(sample_index: int) -> tuple[bytes, int]:
    left = 430 + sample_index * 35
    right = 840 + sample_index * 35
    top = 200
    bottom = 520
    flattened = [0] * (WIDTH * HEIGHT)
    foreground = [1] * (right - left)
    for y in range(top, bottom):
        offset = y * WIDTH
        flattened[offset + left : offset + right] = foreground
    nonzero = (right - left) * (bottom - top)
    payload = {
        "width": WIDTH,
        "height": HEIGHT,
        "mask": flattened,
        "score": 1.0,
    }
    return json.dumps(payload, separators=(",", ":")).encode("ascii"), nonzero


def _asset(
    root: Path,
    *,
    relative_path: str,
    role: str,
    payload: bytes,
) -> dict[str, Any]:
    path = root / relative_path
    write_bytes_atomic(path, payload)
    return {
        "role": role,
        "relative_path": relative_path,
        "bytes": len(payload),
        "sha256": sha256_file(path),
    }


def _cad_payload(sample_index: int) -> bytes:
    scale = 50 + 5 * sample_index
    return (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 8\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 12\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        f"{-scale} {-scale} {-scale}\n{scale} {-scale} {-scale}\n"
        f"{scale} {scale} {-scale}\n{-scale} {scale} {-scale}\n"
        f"{-scale} {-scale} {scale}\n{scale} {-scale} {scale}\n"
        f"{scale} {scale} {scale}\n{-scale} {scale} {scale}\n"
        "3 0 1 2\n3 0 2 3\n3 4 6 5\n3 4 7 6\n"
        "3 0 4 5\n3 0 5 1\n3 1 5 6\n3 1 6 2\n"
        "3 2 6 7\n3 2 7 3\n3 3 7 4\n3 3 4 0\n"
    ).encode("ascii")


def build_fixture(*, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    """Write one deterministic bundle and immediately verify every listed asset."""

    root = output_root.resolve()
    manifest_path = root / "manifest.json"
    asset_root = root
    if root.exists() and any(root.iterdir()):
        raise PrepError("Synthetic fixture output root is not empty")
    variants = [
        {
            "mask_variant_id": variant_id,
            "input_role": V2_VARIANT_INPUT_ROLES[variant_id],
            "plugin_id": "manifest-mask-file-v1",
            "source_boundary": "opaque-upstream-input-only",
            "development_control": variant_id in V2_VARIANT_IDS[:2],
        }
        for variant_id in V2_VARIANT_IDS
    ]
    items: list[dict[str, Any]] = []
    for sample_index in range(2):
        camera_payload = (
            '{"K":[900.0,0.0,640.0,0.0,900.0,360.0,0.0,0.0,1.0]}'
        ).encode("ascii")
        shared = {
            "rgb": _asset(
                asset_root,
                relative_path=f"assets/rgb/{sample_index:06d}.png",
                role="public_rgb",
                payload=_rgb_png(sample_index),
            ),
            "depth": _asset(
                asset_root,
                relative_path=f"assets/depth/{sample_index:06d}.png",
                role="public_depth",
                payload=_gray_png(200 - sample_index * 10),
            ),
            "camera": _asset(
                asset_root,
                relative_path=f"assets/camera/{sample_index:06d}.json",
                role="public_camera",
                payload=camera_payload,
            ),
            "cad": _asset(
                asset_root,
                relative_path=f"assets/cad/{sample_index + 1:06d}.ply",
                role="public_cad",
                payload=_cad_payload(sample_index),
            ),
        }
        mask_payload, nonzero = _mask_json(sample_index)
        mask = _asset(
            asset_root,
            relative_path=(
                f"assets/input_masks/runtime-input/{sample_index:06d}.json"
            ),
            role="input_mask",
            payload=mask_payload,
        )
        for variant_index, variant in enumerate(variants):
            item = {
                "item_id": (
                    f"s{sample_index:06d}-i{sample_index:06d}-"
                    f"o{sample_index + 1:06d}-{variant['mask_variant_id']}"
                ),
                "sample_key": {
                    "scene_id": sample_index,
                    "image_id": sample_index,
                    "object_id": sample_index + 1,
                },
                "mask_variant_id": variant["mask_variant_id"],
                "frame_size": {"width": WIDTH, "height": HEIGHT},
                "inputs": {**shared, "mask": mask},
                "camera_intrinsics": [
                    [900.0, 0.0, 640.0],
                    [0.0, 900.0, 360.0],
                    [0.0, 0.0, 1.0],
                ],
                "depth_scale": 0.001,
                "mask_provenance": {
                    "plugin_id": variant["plugin_id"],
                    "input_role": variant["input_role"],
                    "source_artifact_sha256": mask["sha256"],
                    "generator_id": "pose-accuracy-recovery-a-export",
                    "generator_version": "v2",
                    "model_sha256": None,
                    "score": (1.0, 1.0, 0.9, 0.8, 0.7)[variant_index],
                    "coverage": {
                        "nonzero_pixels": nonzero,
                        "fraction": nonzero / (WIDTH * HEIGHT),
                    },
                    "derivation_class": (
                        "gt-derived-development-control"
                        if variant["development_control"]
                        else {
                            "predicted_mask": "predicted-segmentation",
                            "depth_component_mask": "depth-component",
                            "bbox_mask": "bbox-from-predicted-detection",
                        }[variant["mask_variant_id"]]
                    ),
                    "development_control": variant["development_control"],
                    "source_path_disclosed_to_gpu_c": False,
                    "gpu_c_resolves_derivation": False,
                    "label_access_count_on_gpu_c": 0,
                },
            }
            items.append(item)
    upstream_identity = {
        "fixture_schema": FIXTURE_SCHEMA,
        "sample_count": 2,
        "mask_variant_ids": [row["mask_variant_id"] for row in variants],
    }
    runtime_identity = {
        "implementation_commit": FIXTURE_IMPLEMENTATION_COMMIT,
        "implementation_sha256": FIXTURE_IMPLEMENTATION_SHA256,
        "model_sha256": FIXTURE_MODEL_SHA256,
        "checkpoint_sha256": dict(FIXTURE_CHECKPOINTS),
    }
    manifest = {
        "schema_version": RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2,
        "protocol_id": PREP_PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "input_kind": "SYNTHETIC_COMMITTED_FIXTURE",
        "source_contract": {
            "protocol_id": "poseloop.pose-accuracy-recovery.development-prep.v1",
            "protocol_sha256": A_SOURCE_PROTOCOL_SHA256,
            "bundle_sha256": canonical_sha256(
                {"upstream": upstream_identity, "purpose": "fixture-only"}
            ),
            "a_manifest_lock_sha256": canonical_sha256(
                {"upstream": upstream_identity, "role": "gpu-a-fixture"}
            ),
        },
        "producer_runtime_lock": runtime_identity,
        "mask_variants": variants,
        "coverage": expected_v2_coverage(items),
        "items": items,
        "boundary": {
            "contains_gt_derived_control_inputs": True,
            **V2_BOUNDARY_FIXED,
        },
    }
    manifest["manifest_lock_sha256"] = canonical_sha256(manifest)
    write_json_atomic(manifest_path, manifest)
    validated, normalized = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
        asset_root=asset_root,
        verify_assets=True,
    )
    receipt = {
        "schema_version": FIXTURE_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "manifest_relative_path": "manifest.json",
        "manifest_sha256": sha256_file(manifest_path),
        "asset_root_relative_path": ".",
        "item_count": len(normalized),
        "coverage": validated["coverage"],
        "frame_size": {"width": WIDTH, "height": HEIGHT},
        "synthetic_only": True,
        "label_access_count": 0,
        "official_scorer_run": False,
        "auto_deploy": False,
    }
    write_json_atomic(root / "fixture-receipt.json", receipt)
    return receipt
