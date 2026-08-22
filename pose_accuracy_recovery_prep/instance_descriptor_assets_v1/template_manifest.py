"""Create-only import of the frozen A-R4 RGBA render evidence."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image

from pose_accuracy_recovery_prep.core import ContractError

from . import OBJECT_IDS, TEMPLATE_IMPORT_RECEIPT_SCHEMA, VIEW_COUNT
from .contracts import (
    A_R4_CONTENT_AUDIT_INTERNAL_IDENTITY,
    A_R4_RENDER_PROVENANCE,
    DEFAULT_PROBES,
    RuntimeProbes,
    TEMPLATE_MANIFEST_SCHEMA,
    _asset_from_path,
    _exact,
    canonical_sha256,
    create_only_json,
    read_json,
    sha256_file,
    validate_a_r4_safe_evidence,
    validate_template_manifest,
)

_SOURCE_RECEIPT_ARCHIVE_MEMBERS = {
    "authorization": "receipts/one-time-render-authorization-v1r4-attempt3.json",
    "attempt": "attempts/poseloop_ga_cnos_v1r4_attempt_003/attempt-receipt.json",
    "content_audit": "receipts/postrender-content-audit-v1r4-attempt3.json",
}

_CONTENT_AUDIT_BOUNDARY = {
    "depth_path_open_count": 0,
    "dinov2_model_import_count": 0,
    "evaluator_path_open_count": 0,
    "fastsam_import_count": 0,
    "foundationpose_run_count": 0,
    "gpu_c_call_count": 0,
    "gt_path_open_count": 0,
    "label_access_count": 0,
    "mask_path_open_count": 0,
    "mask_visib_path_open_count": 0,
    "official_scorer_run_count": 0,
    "sealed_split_access_count": 0,
}


def _content_audit_inventory(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    audit = _exact(
        read_json(path, "frozen A-R4 content audit"),
        {
            "attempt_receipt_lock_sha256",
            "boundary_counters",
            "created_utc",
            "exit_code",
            "forbidden_runs",
            "gates",
            "images",
            "implementation_commit",
            "implementation_tree",
            "object_count",
            "objects",
            "png_count",
            "request_lock_sha256",
            "route_lock_sha256",
            "schema_version",
            "server_lifecycle_action",
            "status",
        },
        "frozen A-R4 content audit",
    )
    if (
        audit["schema_version"] != "poseloop.a-r4.postrender-content-audit.v1"
        or audit["status"] != "PASS"
        or audit["implementation_commit"]
        != A_R4_RENDER_PROVENANCE["implementation_commit"]
        or audit["implementation_tree"] != A_R4_RENDER_PROVENANCE["implementation_tree"]
        or audit["route_lock_sha256"] != A_R4_RENDER_PROVENANCE["route_lock_sha256"]
        or audit["request_lock_sha256"]
        != A_R4_RENDER_PROVENANCE["runtime_request_lock_sha256"]
        or audit["attempt_receipt_lock_sha256"]
        != A_R4_CONTENT_AUDIT_INTERNAL_IDENTITY["attempt_receipt_lock_sha256"]
        or audit["object_count"] != len(OBJECT_IDS)
        or audit["png_count"] != len(OBJECT_IDS) * VIEW_COUNT
        or audit["exit_code"] != 0
        or audit["server_lifecycle_action"] != "NONE"
        or audit["boundary_counters"] != _CONTENT_AUDIT_BOUNDARY
        or audit["forbidden_runs"]
        != {
            "FastSAM": 0,
            "FoundationPose": 0,
            "GPU-C": 0,
            "descriptor": 0,
            "evaluator_scorer": 0,
            "producer": 0,
        }
        or audit["gates"]
        != {
            "alpha_foreground": "strictly_between_zero_and_full_frame",
            "frame_exact": [640, 480],
            "minimum_unique_png_sha256_per_object": 2,
            "mode_exact": "RGBA",
            "object_coverage": "5_of_5",
            "rgb_foreground": "strictly_between_zero_and_full_frame",
        }
        or not isinstance(audit["created_utc"], str)
        or not audit["created_utc"]
    ):
        raise ContractError("Frozen A-R4 content-audit identity/gates changed")
    images = audit["images"]
    if not isinstance(images, list) or len(images) != len(OBJECT_IDS) * VIEW_COUNT:
        raise ContractError("Frozen A-R4 content-audit image coverage changed")
    inventory: dict[tuple[int, int], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    cursor = 0
    for object_id in OBJECT_IDS:
        alpha_counts: list[int] = []
        rgb_counts: list[int] = []
        hashes: set[str] = set()
        for view_index in range(VIEW_COUNT):
            row = _exact(
                images[cursor],
                {
                    "alpha_foreground_coverage",
                    "alpha_foreground_pixels",
                    "bytes",
                    "height",
                    "mode",
                    "pass",
                    "relative_path",
                    "rgb_foreground_coverage",
                    "rgb_foreground_pixels",
                    "sha256",
                    "width",
                },
                f"frozen A-R4 content-audit image[{cursor}]",
            )
            expected_suffix = (
                "objects",
                f"obj_{object_id:06d}",
                f"{view_index:06d}.png",
            )
            relative = row["relative_path"]
            parts = PurePosixPath(relative).parts if isinstance(relative, str) else ()
            alpha_pixels = row["alpha_foreground_pixels"]
            rgb_pixels = row["rgb_foreground_pixels"]
            if (
                len(parts) < 3
                or parts[-3:] != expected_suffix
                or row["width"] != 640
                or row["height"] != 480
                or row["mode"] != "RGBA"
                or row["pass"] is not True
                or not isinstance(row["bytes"], int)
                or isinstance(row["bytes"], bool)
                or row["bytes"] <= 0
                or not isinstance(row["sha256"], str)
                or len(row["sha256"]) != 64
                or any(
                    character not in "0123456789abcdef" for character in row["sha256"]
                )
                or not isinstance(alpha_pixels, int)
                or isinstance(alpha_pixels, bool)
                or not 0 < alpha_pixels < 640 * 480
                or not isinstance(rgb_pixels, int)
                or isinstance(rgb_pixels, bool)
                or not 0 < rgb_pixels <= alpha_pixels
                or row["alpha_foreground_coverage"] != alpha_pixels / (640 * 480)
                or row["rgb_foreground_coverage"] != rgb_pixels / (640 * 480)
                or row["sha256"] in hashes
            ):
                raise ContractError("Frozen A-R4 content-audit image row changed")
            hashes.add(row["sha256"])
            alpha_counts.append(alpha_pixels)
            rgb_counts.append(rgb_pixels)
            inventory[(object_id, view_index)] = row
            cursor += 1
        summaries.append(
            {
                "all_nonempty_nonfull_alpha": True,
                "all_nonempty_nonfull_rgb": True,
                "all_rgba": True,
                "alpha_pixels_max": max(alpha_counts),
                "alpha_pixels_min": min(alpha_counts),
                "object_id": object_id,
                "png_count": VIEW_COUNT,
                "rgb_pixels_max": max(rgb_counts),
                "rgb_pixels_min": min(rgb_counts),
                "unique_png_sha256_count": len(hashes),
            }
        )
    if audit["objects"] != summaries:
        raise ContractError("Frozen A-R4 content-audit object summaries changed")
    return inventory


def _copy_create_only(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ContractError(f"Missing frozen A-R4 source asset: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    except FileExistsError as exc:
        raise ContractError(
            f"A-R4 import destination is create-only: {destination}"
        ) from exc
    if source.stat().st_size != destination.stat().st_size or sha256_file(
        source
    ) != sha256_file(destination):
        raise ContractError("A-R4 source/destination copy identity changed")


def _inspect_rgba(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            if (
                image.format != "PNG"
                or image.mode != "RGBA"
                or image.size != (640, 480)
            ):
                raise ContractError("A-R4 source render must be 640x480 RGBA PNG")
            pixels = np.asarray(image)
            pil_bbox = image.getbbox()
            alpha_bbox = image.getchannel("A").getbbox()
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot decode A-R4 source render: {path}") from exc
    alpha = pixels[:, :, 3] > 0
    rgb = np.any(pixels[:, :, :3] > 0, axis=2)
    alpha_pixels = int(alpha.sum())
    rgb_nonzero_pixels = int(np.logical_and(alpha, rgb).sum())
    if (
        alpha_pixels <= 0
        or alpha_pixels >= 640 * 480
        or rgb_nonzero_pixels <= 0
        or np.any(pixels[:, :, :3][~alpha] != 0)
        or pil_bbox is None
        or pil_bbox != alpha_bbox
    ):
        raise ContractError("A-R4 source render content/bbox contract changed")
    return alpha_pixels, rgb_nonzero_pixels


def build_template_manifest(
    *,
    data_root: Path,
    source_png_root: Path,
    authorization_receipt: Path,
    attempt_receipt: Path,
    content_audit: Path,
    safe_archive: Path,
    safe_member_inventory: Path,
    deployment_inventory: Path,
    cad_paths: Mapping[int, Path],
    manifest_output: Path,
    import_receipt_output: Path,
    probes: RuntimeProbes = DEFAULT_PROBES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy exact A-R4 evidence into a new root and self-lock its A-R5 manifest."""

    root = data_root.resolve()
    if manifest_output.resolve() != root / "contracts" / "rgba-template-manifest.json":
        raise ContractError("A-R4 template manifest output path changed")
    if (
        import_receipt_output.resolve()
        != root / "receipts" / "a-r4-template-import.json"
    ):
        raise ContractError("A-R4 template import receipt path changed")
    safe_evidence = validate_a_r4_safe_evidence(
        safe_archive=safe_archive,
        member_inventory=safe_member_inventory,
        deployment_inventory=deployment_inventory,
        probes=probes,
    )
    canonical_cads = {
        item["object_id"]: item for item in safe_evidence["canonical_cads"]
    }
    receipt_sources = {
        "authorization": authorization_receipt,
        "attempt": attempt_receipt,
        "content_audit": content_audit,
    }
    expected_receipt_hashes = dict(probes.source_receipt_hashes())
    if set(expected_receipt_hashes) != set(receipt_sources):
        raise ContractError("A-R4 source receipt identity probe changed")
    for role, source in receipt_sources.items():
        if not source.is_file():
            raise ContractError(f"Frozen A-R4 {role} receipt is missing")
        raw_lock = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
        archive_member = _SOURCE_RECEIPT_ARCHIVE_MEMBERS[role]
        if (
            raw_lock["sha256"] != expected_receipt_hashes[role]
            or safe_evidence["payload_members"].get(archive_member) != raw_lock
        ):
            raise ContractError(f"Frozen A-R4 {role} receipt/archive identity changed")
    content_inventory = _content_audit_inventory(content_audit)
    for audit_row in content_inventory.values():
        archive_member = audit_row["relative_path"]
        if safe_evidence["payload_members"].get(archive_member) != {
            "bytes": audit_row["bytes"],
            "sha256": audit_row["sha256"],
        }:
            raise ContractError("A-R4 content audit/safe archive PNG pair changed")
    for object_id in OBJECT_IDS:
        cad = cad_paths.get(object_id)
        canonical = canonical_cads[object_id]
        if (
            cad is None
            or not cad.is_file()
            or {
                "bytes": cad.stat().st_size,
                "sha256": sha256_file(cad),
            }
            != {"bytes": canonical["bytes"], "sha256": canonical["sha256"]}
        ):
            raise ContractError(
                f"Provided CAD does not match frozen A-R4 archive: object {object_id}"
            )

    frozen_sources = {
        "safe_archive": (
            safe_archive,
            root / "evidence" / "a-r4" / "safe-evidence.tar.gz",
            "a_r4_safe_evidence_archive",
        ),
        "safe_member_inventory": (
            safe_member_inventory,
            root / "evidence" / "a-r4" / "safe-evidence-members.json",
            "a_r4_safe_member_inventory",
        ),
        "deployment_inventory": (
            deployment_inventory,
            root / "evidence" / "a-r4" / "deployment-final-inventory.json",
            "a_r4_deployment_inventory",
        ),
    }
    copied_safe_evidence: dict[str, dict[str, Any]] = {}
    for name, (source, destination, role) in frozen_sources.items():
        _copy_create_only(source, destination)
        copied_safe_evidence[name] = _asset_from_path(
            destination, data_root=root, role=role, probes=probes
        )
    copied_receipts: dict[str, dict[str, Any]] = {}
    for role, source in receipt_sources.items():
        destination = root / "evidence" / "a-r4" / f"{role}.json"
        _copy_create_only(source, destination)
        copied_receipts[role] = _asset_from_path(
            destination,
            data_root=root,
            role=f"a_r4_{role}_receipt",
            probes=probes,
        )

    expected_object_directories = {f"obj_{object_id:06d}" for object_id in OBJECT_IDS}
    observed_object_directories = {
        path.name for path in source_png_root.iterdir() if path.is_dir()
    }
    if observed_object_directories != expected_object_directories:
        raise ContractError("Frozen A-R4 source PNG object coverage changed")
    objects: list[dict[str, Any]] = []
    source_png_inventory: list[dict[str, Any]] = []
    for object_id in OBJECT_IDS:
        source_dir = source_png_root / f"obj_{object_id:06d}"
        expected_names = [f"{index:06d}.png" for index in range(VIEW_COUNT)]
        if (
            sorted(path.name for path in source_dir.iterdir() if path.is_file())
            != expected_names
        ):
            raise ContractError("Frozen A-R4 source PNG view coverage changed")
        cad = cad_paths.get(object_id)
        if cad is None:
            raise ContractError("A-R4 template import CAD mapping is incomplete")
        cad_asset = _asset_from_path(
            cad, data_root=root, role="target_cad", probes=probes
        )
        canonical_cad = canonical_cads[object_id]
        if {"bytes": cad_asset["bytes"], "sha256": cad_asset["sha256"]} != {
            "bytes": canonical_cad["bytes"],
            "sha256": canonical_cad["sha256"],
        }:
            raise ContractError("A-R4 CAD asset probe differs from safe archive")
        views: list[dict[str, Any]] = []
        hashes: set[str] = set()
        for view_index, name in enumerate(expected_names):
            source = source_dir / name
            source_lock = {
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
            alpha_pixels, rgb_nonzero_pixels = _inspect_rgba(source)
            audit_row = content_inventory[(object_id, view_index)]
            if (
                source_lock
                != {"bytes": audit_row["bytes"], "sha256": audit_row["sha256"]}
                or alpha_pixels != audit_row["alpha_foreground_pixels"]
                or rgb_nonzero_pixels != audit_row["rgb_foreground_pixels"]
            ):
                raise ContractError(
                    "Source PNG does not match frozen A-R4 content audit"
                )
            destination = root / "assets" / "templates" / f"obj_{object_id:06d}" / name
            _copy_create_only(source, destination)
            destination_asset = _asset_from_path(
                destination,
                data_root=root,
                role="cad_template_rgba",
                probes=probes,
            )
            if source_lock != {
                "bytes": destination_asset["bytes"],
                "sha256": destination_asset["sha256"],
            }:
                raise ContractError("A-R4 PNG import bytes changed")
            if destination_asset["sha256"] in hashes:
                raise ContractError("A-R4 per-object source views are not unique")
            hashes.add(destination_asset["sha256"])
            views.append(
                {
                    "view_index": view_index,
                    "rgba": destination_asset,
                    "alpha_pixels": alpha_pixels,
                    "rgb_nonzero_pixels": rgb_nonzero_pixels,
                }
            )
            source_png_inventory.append(
                {
                    "object_id": object_id,
                    "view_index": view_index,
                    "source": {
                        "archive_member_path": audit_row["relative_path"],
                        **source_lock,
                    },
                    "destination": destination_asset,
                }
            )
        objects.append(
            {
                "object_id": object_id,
                "cad_sha256": cad_asset["sha256"],
                "views": views,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": TEMPLATE_MANIFEST_SCHEMA,
        "a_r4_render_provenance": dict(A_R4_RENDER_PROVENANCE),
        "frame_size": {"height": 480, "width": 640, "mode": "RGBA"},
        "object_count": len(OBJECT_IDS),
        "views_per_object": VIEW_COUNT,
        "objects": objects,
        "template_manifest_lock_sha256": "pending",
    }
    manifest["template_manifest_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "template_manifest_lock_sha256"
        }
    )
    create_only_json(manifest_output, manifest)
    validate_template_manifest(manifest, data_root=root, probes=probes)
    manifest_asset = _asset_from_path(
        manifest_output,
        data_root=root,
        role="rgba_template_manifest",
        probes=probes,
    )
    import_receipt: dict[str, Any] = {
        "schema_version": TEMPLATE_IMPORT_RECEIPT_SCHEMA,
        "status": "PASS_A_R4_EXACT_5_OBJECTS_X_42_RGBA_IMPORTED_CREATE_ONLY",
        "a_r4_render_provenance": dict(A_R4_RENDER_PROVENANCE),
        "safe_evidence": copied_safe_evidence,
        "source_receipts": copied_receipts,
        "canonical_cad_inventory": safe_evidence["canonical_cads"],
        "source_png_count": len(source_png_inventory),
        "source_png_inventory": source_png_inventory,
        "template_manifest": manifest_asset,
        "a_r4_root_modified": False,
        "boundary": {
            "label_access_count": 0,
            "depth_access_count": 0,
            "evaluator_access_count": 0,
            "sealed_access_count": 0,
            "foundationpose_run_count": 0,
        },
        "template_import_receipt_lock_sha256": "pending",
    }
    import_receipt["template_import_receipt_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in import_receipt.items()
            if key != "template_import_receipt_lock_sha256"
        }
    )
    create_only_json(import_receipt_output, import_receipt)
    return manifest, import_receipt


__all__ = ["build_template_manifest"]
