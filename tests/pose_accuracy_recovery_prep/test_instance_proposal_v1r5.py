from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
    write_json,
)
from pose_accuracy_recovery_prep.instance_proposal_v1r5 import (
    FRAME_MANIFEST_SCHEMA,
    PROTOCOL_ID,
    PROTOCOL_SCHEMA,
    RUNTIME_REQUEST_SCHEMA,
)
from pose_accuracy_recovery_prep.instance_proposal_v1r5 import contracts
from pose_accuracy_recovery_prep.instance_proposal_v1r5.adapter import (
    CatalogScore,
    InstanceProposal,
    OfficialCnosCatalogAdapter,
    audit_runtime_module_origins,
    ensure_ultralytics_yolo_compat,
    rank_catalog,
)
from pose_accuracy_recovery_prep.instance_proposal_v1r5.cli import main
from pose_accuracy_recovery_prep.instance_proposal_v1r5.contracts import (
    A_R4_RENDER_PROVENANCE,
    BOUNDARY_ZERO,
    CATALOG_OBJECT_IDS,
    FRAME_KEYS,
    PINNED_CNOS_ARCHIVE_BYTES,
    PINNED_CNOS_ARCHIVE_SHA256,
    PINNED_CNOS_COMMIT,
    PINNED_CNOS_REPOSITORY,
    PINNED_CNOS_TREE,
    PINNED_DINOV2_COMMIT,
    PINNED_DINOV2_IDENTITY_RECEIPT_SHA256,
    PINNED_DINOV2_REPOSITORY,
    PINNED_DINOV2_TREE,
    PINNED_DINOV2_VITL14_BYTES,
    PINNED_DINOV2_VITL14_SHA256,
    PINNED_DINOV2_VITL14_URL,
    SOURCE_CHECKOUT_MANIFEST_SCHEMA,
    VISUALIZATION_ROLES,
    validate_source_checkout_manifest,
    validate_frame_manifest,
    validate_output_bundle,
    validate_protocol,
    validate_runtime_request,
)
from pose_accuracy_recovery_prep.instance_proposal_v1r5.producer import (
    PlannedCrash,
    run_producer,
    validate_run_receipt,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT / "protocols" / "poseloop_pose_accuracy_recovery_instance_proposal_v1r5.json"
)
FASTSAM_SHA256 = "752cadc2828edb1cd4bc4f9eb587100631af06ea2108f4c9ed56df4755701e76"
REAL_VALIDATE_CURRENT_IMPLEMENTATION = contracts._validate_current_implementation


def _asset(
    path: Path, root: Path, role: str, *, sha256: str | None = None
) -> dict[str, Any]:
    return {
        "role": role,
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256 or sha256_file(path),
    }


def _locked(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )
    return value


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _test_git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _protocol() -> dict[str, Any]:
    paths = [
        "pose_accuracy_recovery_prep/core.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/adapter.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/__init__.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/__main__.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/adapter.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/cli.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/contracts.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/producer.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/cnos_catalog_config_v1r5.json",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/SERVER_RUNBOOK_A_R5.md",
    ]
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "role": "DEVELOPMENT_ONLY_LABEL_BLIND_INSTANCE_PROPOSALS",
        "auto_deploy": False,
        "accuracy_claim_permitted": False,
        "predecessor": {
            "a_r3_read_only": True,
            "a_r4_read_only": True,
            "b2_p1_design_reference_only": "5162dfa",
        },
        "a_r4_render_provenance": copy.deepcopy(A_R4_RENDER_PROVENANCE),
        "development_slice": {
            "workload_sha256": contracts.WORKLOAD_SHA256,
            "frame_count": 10,
            "scene_count": 5,
            "catalog_object_ids": list(CATALOG_OBJECT_IDS),
            "frame_keys": [
                {"scene_id": scene_id, "image_id": image_id}
                for scene_id, image_id in FRAME_KEYS
            ],
            "historical_object_stratification_is_provenance_only": True,
            "runtime_target_identity_exposed": False,
            "single_object_fallback_permitted": False,
        },
        "method": {
            "instance_proposals": "official_cnos_fastsam_x_independent_instance_masks",
            "catalog_recognition": "official_cnos_dinov2_vitl14_top5_template_cosine",
            "catalog_scope": "all_five_objects_for_every_proposal",
            "normalization": "raw_cosine_plus_one_divide_two_without_clamp",
            "gt_association_permitted": False,
            "whole_foreground_proposal_permitted": False,
            "foundationpose_participates": False,
        },
        "model_provenance": {
            "dinov2_repository": PINNED_DINOV2_REPOSITORY,
            "dinov2_commit": PINNED_DINOV2_COMMIT,
            "dinov2_tree": PINNED_DINOV2_TREE,
            "dinov2_vitl14_url": PINNED_DINOV2_VITL14_URL,
            "dinov2_vitl14_bytes": PINNED_DINOV2_VITL14_BYTES,
            "dinov2_vitl14_sha256": PINNED_DINOV2_VITL14_SHA256,
            "remote_identity_receipt_sha256": PINNED_DINOV2_IDENTITY_RECEIPT_SHA256,
        },
        "runtime_boundary": copy.deepcopy(BOUNDARY_ZERO),
        "execution_policy": {
            "create_only": True,
            "resume_completed_prefix_only": True,
            "planned_crash_permitted": True,
            "failed_attempt_overwrite_permitted": False,
            "result_export_permitted": False,
            "model_run_authorized_now": False,
            "server_connection_authorized_now": False,
        },
        "visualization_contract": {
            "roles": list(VISUALIZATION_ROLES),
            "must_be_scene_content": True,
            "charts_satisfy_contract": False,
        },
        "wrapper_files": [
            {
                "relative_path": relative,
                "bytes": (ROOT / relative).stat().st_size,
                "sha256": sha256_file(ROOT / relative),
            }
            for relative in paths
        ],
        "protocol_lock_sha256": "pending",
    }
    return _locked(protocol, "protocol_lock_sha256")


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("a-r5-data")
    protocol = _protocol()
    frames: list[dict[str, Any]] = []
    for ordinal, (scene_id, image_id) in enumerate(FRAME_KEYS):
        item_id = f"scene-{scene_id:06d}-image-{image_id:06d}"
        rgb_path = root / "inputs" / "rgb" / f"{item_id}.png"
        rgb = np.zeros((1080, 1440, 3), dtype=np.uint8)
        rgb[:, :, 1] = 20 + ordinal
        rgb[100:300, 120 + ordinal : 320 + ordinal] = (120, 40, 200)
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(rgb_path)
        depth_path = root / "inputs" / "depth" / f"{item_id}.png"
        depth = np.zeros((1080, 1440), dtype=np.uint16)
        depth[80:900, 100:1300] = 900 + ordinal
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(depth, mode="I;16").save(depth_path)
        camera_path = root / "inputs" / "camera" / f"{item_id}.json"
        write_json(
            camera_path,
            {
                "schema_version": "poseloop.public-depth-free-camera.v1r5",
                "frame_size": {"height": 1080, "width": 1440},
                "camera_intrinsics": [
                    1000.0,
                    0.0,
                    720.0,
                    0.0,
                    1000.0,
                    540.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "camera_world_to_camera_pose_m": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "coordinate_convention": "opencv_x_right_y_down_z_forward_row_major",
            },
        )
        frames.append(
            {
                "item_id": item_id,
                "frame_key": {"scene_id": scene_id, "image_id": image_id},
                "inputs": {
                    "rgb": _asset(rgb_path, root, "rgb"),
                    "depth": _asset(depth_path, root, "raw_sensor_depth"),
                    "camera": _asset(camera_path, root, "public_camera"),
                },
            }
        )
    manifest = _locked(
        {
            "schema_version": FRAME_MANIFEST_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "protocol_lock_sha256": protocol["protocol_lock_sha256"],
            "role": "DEVELOPMENT_ONLY_LABEL_BLIND_FRAME_INPUT",
            "frame_size": {"height": 1080, "width": 1440},
            "selection_provenance": {
                "source_workload_sha256": contracts.WORKLOAD_SHA256,
                "id_only_selection": True,
                "historical_object_ids_removed_before_runtime": True,
                "runtime_target_identity_exposed": False,
                "raw_sensor_depth_usage": "disk_bound_for_b2_v2_interface_not_opened_by_fastsam_or_cad_ranking",
            },
            "frame_count": 10,
            "scene_count": 5,
            "frames": frames,
            "frame_manifest_lock_sha256": "pending",
        },
        "frame_manifest_lock_sha256",
    )
    manifest_path = root / "contracts" / "frame-manifest.json"
    write_json(manifest_path, manifest)

    cad_assets: dict[int, dict[str, Any]] = {}
    template_objects: list[dict[str, Any]] = []
    for object_id in CATALOG_OBJECT_IDS:
        cad_path = root / "assets" / "cad" / f"obj_{object_id:06d}.ply"
        _write_bytes(cad_path, f"ply\nobject {object_id}\n".encode())
        cad_assets[object_id] = _asset(cad_path, root, "target_cad")
        views: list[dict[str, Any]] = []
        for view_index in range(42):
            template_path = (
                root
                / "assets"
                / "templates"
                / f"obj_{object_id:06d}"
                / f"{view_index:06d}.png"
            )
            rgba = np.zeros((480, 640, 4), dtype=np.uint8)
            x = 20 + (view_index * 11) % 500
            y = 20 + (view_index * 7) % 360
            rgba[y : y + 50, x : x + 60, :3] = (20 * object_id, 40 + view_index, 120)
            rgba[y : y + 50, x : x + 60, 3] = 255
            template_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, mode="RGBA").save(template_path)
            views.append(
                {
                    "view_index": view_index,
                    "rgba": _asset(template_path, root, "cad_template_rgba"),
                    "alpha_pixels": 3000,
                    "rgb_nonzero_pixels": 3000,
                }
            )
        template_objects.append(
            {
                "object_id": object_id,
                "cad_sha256": cad_assets[object_id]["sha256"],
                "views": views,
            }
        )
    template_manifest = _locked(
        {
            "schema_version": "poseloop.pose-accuracy-recovery.cnos-rgba-template-manifest.v1r5",
            "a_r4_render_provenance": copy.deepcopy(A_R4_RENDER_PROVENANCE),
            "frame_size": {"height": 480, "width": 640, "mode": "RGBA"},
            "object_count": 5,
            "views_per_object": 42,
            "objects": template_objects,
            "template_manifest_lock_sha256": "pending",
        },
        "template_manifest_lock_sha256",
    )
    template_manifest_path = root / "contracts" / "rgba-template-manifest.json"
    write_json(template_manifest_path, template_manifest)

    adapter_config = (
        ROOT
        / "pose_accuracy_recovery_prep"
        / "instance_proposal_v1r5"
        / "cnos_catalog_config_v1r5.json"
    )
    deployed_config = root / "contracts" / "cnos_catalog_config_v1r5.json"
    deployed_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(adapter_config, deployed_config)
    descriptor_receipt_path = root / "receipts" / "descriptor-generation-receipt.json"
    _write_bytes(descriptor_receipt_path, b'{"status":"PASS_FIXTURE"}\n')
    descriptor_receipt = _asset(
        descriptor_receipt_path, root, "descriptor_generation_receipt"
    )
    catalog: list[dict[str, Any]] = []
    for object_id in CATALOG_OBJECT_IDS:
        descriptor_path = root / "assets" / "descriptors" / f"obj_{object_id:06d}.pth"
        _write_bytes(descriptor_path, f"fixture descriptor {object_id}".encode())
        catalog.append(
            {
                "object_id": object_id,
                "cad": cad_assets[object_id],
                "descriptor": _asset(descriptor_path, root, "cad_template_descriptors"),
                "descriptor_metadata": {
                    "shape": [42, 1024],
                    "dtype": "float32",
                    "source_template_manifest_sha256": sha256_file(
                        template_manifest_path
                    ),
                    "generation_receipt_sha256": descriptor_receipt["sha256"],
                },
            }
        )
    source_assets: dict[str, dict[str, Any]] = {}
    cnos_archive_path = root / "sources" / "cnos.tar.gz"
    _write_bytes(cnos_archive_path, b"fixture pinned CNOS archive")
    source_assets["cnos.tar.gz"] = {
        "role": "cnos_source_archive",
        "relative_path": "sources/cnos.tar.gz",
        "bytes": PINNED_CNOS_ARCHIVE_BYTES,
        "sha256": PINNED_CNOS_ARCHIVE_SHA256,
    }
    for name, role in (
        ("dinov2.tar.gz", "dinov2_source_archive"),
        ("implementation.tar.gz", "implementation_source_archive"),
    ):
        path = root / "sources" / name
        _write_bytes(path, f"fixture {name}".encode())
        source_assets[name] = _asset(path, root, role)

    cnos_checkout = root / "sources" / "cnos"
    dinov2_checkout = root / "sources" / "dinov2"
    for checkout in (cnos_checkout, dinov2_checkout):
        (checkout / ".git").mkdir(parents=True)
    cnos_files = [
        "src/model/fast_sam.py",
        "src/model/dinov2.py",
        "src/model/utils.py",
    ]
    dinov2_files = [
        "hubconf.py",
        "dinov2/__init__.py",
        "dinov2/hub/__init__.py",
        "dinov2/hub/backbones.py",
    ]
    for relative in cnos_files:
        _write_bytes(cnos_checkout / relative, f"# fixture {relative}\n".encode())
    for relative in dinov2_files:
        _write_bytes(dinov2_checkout / relative, f"# fixture {relative}\n".encode())

    def checkout_manifest(
        *,
        kind: str,
        repository: str,
        commit: str,
        tree: str,
        archive: dict[str, Any],
        checkout: Path,
        files: list[str],
    ) -> dict[str, Any]:
        execution_files = [
            {
                "relative_path": relative,
                "bytes": (checkout / relative).stat().st_size,
                "sha256": sha256_file(checkout / relative),
            }
            for relative in files
        ]
        return _locked(
            {
                "schema_version": SOURCE_CHECKOUT_MANIFEST_SCHEMA,
                "kind": kind,
                "repository": repository,
                "commit": commit,
                "tree": tree,
                "all_clean": True,
                "git_status_porcelain_v1_untracked_files_all": "",
                "source_archive": {
                    "bytes": archive["bytes"],
                    "sha256": archive["sha256"],
                },
                "execution_files": execution_files,
                "execution_inventory_sha256": canonical_sha256(execution_files),
                "checkout_manifest_lock_sha256": "pending",
            },
            "checkout_manifest_lock_sha256",
        )

    for name, role, value in (
        (
            "cnos-checkout-manifest.json",
            "cnos_checkout_manifest",
            checkout_manifest(
                kind="CNOS",
                repository=PINNED_CNOS_REPOSITORY,
                commit=PINNED_CNOS_COMMIT,
                tree=PINNED_CNOS_TREE,
                archive=source_assets["cnos.tar.gz"],
                checkout=cnos_checkout,
                files=cnos_files,
            ),
        ),
        (
            "dinov2-checkout-manifest.json",
            "dinov2_checkout_manifest",
            checkout_manifest(
                kind="DINOV2",
                repository=PINNED_DINOV2_REPOSITORY,
                commit=PINNED_DINOV2_COMMIT,
                tree=PINNED_DINOV2_TREE,
                archive=source_assets["dinov2.tar.gz"],
                checkout=dinov2_checkout,
                files=dinov2_files,
            ),
        ),
    ):
        path = root / "sources" / name
        write_json(path, value)
        source_assets[name] = _asset(path, root, role)
    fastsam_path = root / "models" / "FastSAM-x.pt"
    _write_bytes(fastsam_path, b"fixture FastSAM-x bytes")
    dinov2_path = root / "models" / "dinov2_vitl14.pth"
    _write_bytes(dinov2_path, b"fixture DINOv2 bytes")
    fastsam_asset = _asset(
        fastsam_path, root, "fastsam_x_checkpoint", sha256=FASTSAM_SHA256
    )
    dinov2_asset = {
        "role": "dinov2_vitl14_checkpoint",
        "relative_path": "models/dinov2_vitl14.pth",
        "bytes": PINNED_DINOV2_VITL14_BYTES,
        "sha256": PINNED_DINOV2_VITL14_SHA256,
    }
    request = {
        "schema_version": RUNTIME_REQUEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "frame_manifest": _asset(manifest_path, root, "frame_manifest"),
        "implementation": {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "source_archive": source_assets["implementation.tar.gz"],
            "archive_route": {
                "format": "tar.gz",
                "prefix": f"poseloop-{'a' * 40}/",
                "command": [
                    "git",
                    "archive",
                    "--format=tar.gz",
                    f"--prefix=poseloop-{'a' * 40}/",
                    "a" * 40,
                ],
            },
        },
        "source": {
            "cnos_repository": PINNED_CNOS_REPOSITORY,
            "cnos_commit": PINNED_CNOS_COMMIT,
            "cnos_tree": PINNED_CNOS_TREE,
            "cnos_archive": source_assets["cnos.tar.gz"],
            "cnos_checkout_manifest": source_assets["cnos-checkout-manifest.json"],
            "cnos_checkout_relative_path": "sources/cnos",
            "dinov2_repository": PINNED_DINOV2_REPOSITORY,
            "dinov2_commit": PINNED_DINOV2_COMMIT,
            "dinov2_tree": PINNED_DINOV2_TREE,
            "dinov2_archive": source_assets["dinov2.tar.gz"],
            "dinov2_checkout_manifest": source_assets["dinov2-checkout-manifest.json"],
            "dinov2_checkout_relative_path": "sources/dinov2",
        },
        "models": {
            "fastsam_x_checkpoint": fastsam_asset,
            "dinov2_vitl14_checkpoint": dinov2_asset,
            "composite_model_sha256": canonical_sha256(
                {
                    "fastsam_x_checkpoint": FASTSAM_SHA256,
                    "dinov2_vitl14_checkpoint": PINNED_DINOV2_VITL14_SHA256,
                }
            ),
        },
        "adapter_config": _asset(deployed_config, root, "cnos_catalog_adapter_config"),
        "a_r4_render_provenance": copy.deepcopy(A_R4_RENDER_PROVENANCE),
        "template_manifest": _asset(
            template_manifest_path, root, "rgba_template_manifest"
        ),
        "descriptor_generation_receipt": descriptor_receipt,
        "catalog": catalog,
        "runtime": {
            "device": "cuda:0",
            "proposal_chunk_size": 16,
            "minimum_proposal_chunk_size": 1,
            "max_oom_chunk_reductions": 4,
        },
        "boundary": copy.deepcopy(BOUNDARY_ZERO),
        "runtime_request_lock_sha256": "pending",
    }
    _locked(request, "runtime_request_lock_sha256")
    return {
        "root": root,
        "protocol": protocol,
        "manifest": manifest,
        "request": request,
        "fastsam_path": fastsam_path,
        "dinov2_path": dinov2_path,
        "cnos_archive_path": cnos_archive_path,
        "cnos_checkout": cnos_checkout,
        "dinov2_checkout": dinov2_checkout,
        "dinov2_files": dinov2_files,
    }


@pytest.fixture(scope="module", autouse=True)
def isolated_runtime_identity_fixture(fixture: dict[str, Any]) -> Any:
    real_verify = contracts._verify_disk_asset
    real_git = contracts._git
    virtual_assets = {
        fixture["fastsam_path"].resolve(): (None, FASTSAM_SHA256),
        fixture["dinov2_path"].resolve(): (
            PINNED_DINOV2_VITL14_BYTES,
            PINNED_DINOV2_VITL14_SHA256,
        ),
        fixture["cnos_archive_path"].resolve(): (
            PINNED_CNOS_ARCHIVE_BYTES,
            PINNED_CNOS_ARCHIVE_SHA256,
        ),
    }

    def verify(
        path: Path,
        *,
        bytes_expected: int,
        sha256_expected: str,
        label: str,
        verify_hashes: bool,
    ) -> None:
        resolved = Path(path).resolve()
        if resolved in virtual_assets:
            expected_bytes, expected_sha = virtual_assets[resolved]
            assert resolved.is_file()
            if expected_bytes is not None and bytes_expected != expected_bytes:
                raise ContractError(f"{label} missing or byte count changed")
            if verify_hashes and sha256_expected != expected_sha:
                raise ContractError(f"{label} SHA-256 mismatch")
            return
        real_verify(
            path,
            bytes_expected=bytes_expected,
            sha256_expected=sha256_expected,
            label=label,
            verify_hashes=verify_hashes,
        )

    cnos_checkout = fixture["cnos_checkout"].resolve()
    dinov2_checkout = fixture["dinov2_checkout"].resolve()

    def git(checkout: Path, *arguments: str, label: str) -> str:
        resolved = Path(checkout).resolve()
        if resolved not in {cnos_checkout, dinov2_checkout}:
            return real_git(checkout, *arguments, label=label)
        identity = (
            (PINNED_CNOS_REPOSITORY, PINNED_CNOS_COMMIT, PINNED_CNOS_TREE)
            if resolved == cnos_checkout
            else (PINNED_DINOV2_REPOSITORY, PINNED_DINOV2_COMMIT, PINNED_DINOV2_TREE)
        )
        if arguments == ("remote", "get-url", "origin"):
            return identity[0] + "\n"
        if arguments == ("rev-parse", "HEAD"):
            return identity[1] + "\n"
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return identity[2] + "\n"
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == ("ls-files", "--", "hubconf.py", "dinov2"):
            return "\n".join(fixture["dinov2_files"]) + "\n"
        raise AssertionError((label, arguments))

    fixture["implementation_checks"] = []

    def implementation_check(
        implementation: dict[str, Any],
        *,
        source_archive_path: Path,
        repository_root: Path,
    ) -> None:
        assert implementation["commit"] == "a" * 40
        assert source_archive_path.is_file()
        assert repository_root == ROOT
        fixture["implementation_checks"].append(implementation["commit"])

    patcher = pytest.MonkeyPatch()
    patcher.setattr(contracts, "_verify_disk_asset", verify)
    patcher.setattr(contracts, "_git", git)
    patcher.setattr(contracts, "_validate_current_implementation", implementation_check)
    yield
    patcher.undo()


def _scores(request: dict[str, Any], base: float) -> tuple[CatalogScore, ...]:
    values: list[CatalogScore] = []
    for ordinal, item in enumerate(request["catalog"]):
        raw = base - ordinal * 0.08
        components = (raw + 0.02, raw + 0.01, raw, raw - 0.01, raw - 0.02)
        values.append(
            CatalogScore(
                object_id=item["object_id"],
                descriptor_sha256=item["descriptor"]["sha256"],
                raw_cosine=raw,
                normalized_similarity=(raw + 1.0) / 2.0,
                top5_template_cosines=components,
                top5_template_indices=(0, 1, 2, 3, 4),
            )
        )
    return rank_catalog(values)


class FakeBackend:
    def __init__(self, request: dict[str, Any]) -> None:
        self.request = request
        self.calls: list[str] = []

    def infer_frame(
        self,
        *,
        rgb_relative_path: str,
        proposal_chunk_size: int,
        minimum_chunk_size: int,
    ) -> tuple[list[InstanceProposal], list[dict[str, Any]]]:
        del proposal_chunk_size, minimum_chunk_size
        ordinal = len(self.calls)
        self.calls.append(rgb_relative_path)
        if ordinal == 0:
            return [], []
        first = np.zeros((1080, 1440), dtype=bool)
        first[120:240, 180:320] = True
        proposals = [InstanceProposal(3, first, 0.85, 0.91, _scores(self.request, 0.4))]
        if ordinal in {2, 3, 5, 7, 9}:
            second = np.zeros((1080, 1440), dtype=bool)
            if ordinal == 3:
                second[239:340, 319:430] = True
            else:
                second[500:620, 700:850] = True
            second_cad_base = 0.7 if ordinal == 2 else -0.1
            proposals.append(
                InstanceProposal(
                    8,
                    second,
                    0.72,
                    0.88,
                    _scores(self.request, second_cad_base),
                )
            )
        return proposals, []


def test_protocol_freezes_a_r4_and_wrapper_bytes() -> None:
    protocol = validate_protocol(_protocol(), repository_root=ROOT)
    assert protocol["a_r4_render_provenance"]["rgba_png_count"] == 210
    assert protocol["method"]["foundationpose_participates"] is False


def test_manifest_is_exact_ten_frame_label_blind_slice(fixture: dict[str, Any]) -> None:
    manifest = validate_frame_manifest(
        fixture["manifest"], fixture["protocol"], data_root=fixture["root"]
    )
    assert len(manifest["frames"]) == 10
    assert {frame["frame_key"]["scene_id"] for frame in manifest["frames"]} == {
        0,
        3,
        9,
        12,
        15,
    }
    assert all("object_id" not in frame for frame in manifest["frames"])
    assert all("object_id" not in frame["frame_key"] for frame in manifest["frames"])
    assert all(
        set(frame["inputs"]) == {"rgb", "depth", "camera"}
        for frame in manifest["frames"]
    )


@pytest.mark.parametrize(
    ("trail", "value"),
    [
        (("frames", 0, "inputs", "target_object_id"), 1),
        (("frames", 0, "inputs", "rgb", "relative_path"), "inputs/scene_gt/rgb.png"),
        (("frames", 0, "inputs", "depth", "relative_path"), "inputs/mask/depth.png"),
    ],
)
def test_manifest_rejects_target_gt_and_mask_paths(
    fixture: dict[str, Any], trail: tuple[Any, ...], value: Any
) -> None:
    manifest = copy.deepcopy(fixture["manifest"])
    cursor: Any = manifest
    for key in trail[:-1]:
        cursor = cursor[key]
    cursor[trail[-1]] = value
    _locked(manifest, "frame_manifest_lock_sha256")
    with pytest.raises(ContractError):
        validate_frame_manifest(
            manifest, fixture["protocol"], data_root=fixture["root"]
        )


def test_camera_extra_depth_scale_fails_closed(fixture: dict[str, Any]) -> None:
    camera = (
        fixture["root"]
        / fixture["manifest"]["frames"][0]["inputs"]["camera"]["relative_path"]
    )
    original = camera.read_bytes()
    value = json.loads(original)
    value["depth_scale"] = 0.001
    write_json(camera, value)
    try:
        with pytest.raises(ContractError, match="fields differ|depth|byte count"):
            validate_frame_manifest(
                fixture["manifest"],
                fixture["protocol"],
                data_root=fixture["root"],
                verify_hashes=False,
            )
    finally:
        camera.write_bytes(original)


def test_runtime_request_binds_sources_models_templates_and_catalog(
    fixture: dict[str, Any],
) -> None:
    request = validate_runtime_request(
        fixture["request"],
        fixture["protocol"],
        fixture["manifest"],
        data_root=fixture["root"],
    )
    assert [item["object_id"] for item in request["catalog"]] == [1, 2, 4, 5, 6]
    assert request["boundary"] == BOUNDARY_ZERO
    assert fixture["implementation_checks"]


def test_65_character_dinov2_transcription_fails_closed(
    fixture: dict[str, Any],
) -> None:
    request = copy.deepcopy(fixture["request"])
    request["models"]["dinov2_vitl14_checkpoint"]["sha256"] = (
        "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb11d7162cbf428"
    )
    request["models"]["composite_model_sha256"] = canonical_sha256(
        {
            "fastsam_x_checkpoint": FASTSAM_SHA256,
            "dinov2_vitl14_checkpoint": request["models"]["dinov2_vitl14_checkpoint"][
                "sha256"
            ],
        }
    )
    _locked(request, "runtime_request_lock_sha256")
    with pytest.raises(ContractError, match="lowercase SHA-256"):
        validate_runtime_request(
            request,
            fixture["protocol"],
            fixture["manifest"],
            data_root=fixture["root"],
        )


def test_current_implementation_commit_tree_clean_status_and_archive_are_rebuilt(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "implementation"
    checkout.mkdir()
    _test_git(checkout, "init")
    _test_git(checkout, "config", "user.email", "fixture@poseloop.invalid")
    _test_git(checkout, "config", "user.name", "PoseLoop Fixture")
    _write_bytes(checkout / "reviewed.py", b"VALUE = 1\n")
    _test_git(checkout, "add", "reviewed.py")
    _test_git(checkout, "commit", "-m", "fixture")
    commit = _test_git(checkout, "rev-parse", "HEAD")
    tree = _test_git(checkout, "rev-parse", "HEAD^{tree}")
    archive = tmp_path / "implementation.tar.gz"
    prefix = f"poseloop-{commit}/"
    _test_git(
        checkout,
        "archive",
        "--format=tar.gz",
        f"--prefix={prefix}",
        f"--output={archive}",
        commit,
    )
    implementation = {
        "commit": commit,
        "tree": tree,
        "source_archive": {
            "role": "implementation_source_archive",
            "relative_path": "sources/implementation.tar.gz",
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
        "archive_route": {
            "format": "tar.gz",
            "prefix": prefix,
            "command": [
                "git",
                "archive",
                "--format=tar.gz",
                f"--prefix={prefix}",
                commit,
            ],
        },
    }
    REAL_VALIDATE_CURRENT_IMPLEMENTATION(
        implementation,
        source_archive_path=archive,
        repository_root=checkout,
    )
    wrong = copy.deepcopy(implementation)
    wrong["commit"] = "f" * 40
    wrong["archive_route"]["prefix"] = f"poseloop-{'f' * 40}/"
    wrong["archive_route"]["command"][-2] = f"--prefix=poseloop-{'f' * 40}/"
    wrong["archive_route"]["command"][-1] = "f" * 40
    with pytest.raises(ContractError, match="exact clean commit/tree"):
        REAL_VALIDATE_CURRENT_IMPLEMENTATION(
            wrong,
            source_archive_path=archive,
            repository_root=checkout,
        )


def test_source_manifest_rejects_untracked_shadow_and_wrong_file_identity(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "cnos"
    checkout.mkdir()
    _test_git(checkout, "init")
    _test_git(checkout, "config", "user.email", "fixture@poseloop.invalid")
    _test_git(checkout, "config", "user.name", "PoseLoop Fixture")
    repository = "https://github.com/example/cnos-fixture"
    _test_git(checkout, "remote", "add", "origin", repository)
    execution_files = []
    for relative in (
        "src/model/fast_sam.py",
        "src/model/dinov2.py",
        "src/model/utils.py",
    ):
        path = checkout / relative
        _write_bytes(path, f"# {relative}\n".encode())
        execution_files.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _test_git(checkout, "add", ".")
    _test_git(checkout, "commit", "-m", "fixture")
    commit = _test_git(checkout, "rev-parse", "HEAD")
    tree = _test_git(checkout, "rev-parse", "HEAD^{tree}")
    archive = {"bytes": 17, "sha256": "a" * 64}
    manifest = _locked(
        {
            "schema_version": SOURCE_CHECKOUT_MANIFEST_SCHEMA,
            "kind": "CNOS",
            "repository": repository,
            "commit": commit,
            "tree": tree,
            "all_clean": True,
            "git_status_porcelain_v1_untracked_files_all": "",
            "source_archive": archive,
            "execution_files": execution_files,
            "execution_inventory_sha256": canonical_sha256(execution_files),
            "checkout_manifest_lock_sha256": "pending",
        },
        "checkout_manifest_lock_sha256",
    )
    validate_source_checkout_manifest(
        manifest,
        checkout=checkout,
        kind="CNOS",
        repository=repository,
        commit=commit,
        tree=tree,
        source_archive=archive,
    )
    shadow = checkout / "src" / "model" / "shadow.py"
    _write_bytes(shadow, b"# untracked shadow\n")
    try:
        with pytest.raises(ContractError, match="exact and clean"):
            validate_source_checkout_manifest(
                manifest,
                checkout=checkout,
                kind="CNOS",
                repository=repository,
                commit=commit,
                tree=tree,
                source_archive=archive,
            )
    finally:
        shadow.unlink()
    wrong = copy.deepcopy(manifest)
    wrong["execution_files"][0]["sha256"] = "0" * 64
    wrong["execution_inventory_sha256"] = canonical_sha256(wrong["execution_files"])
    _locked(wrong, "checkout_manifest_lock_sha256")
    with pytest.raises(ContractError, match="execution file changed"):
        validate_source_checkout_manifest(
            wrong,
            checkout=checkout,
            kind="CNOS",
            repository=repository,
            commit=commit,
            tree=tree,
            source_archive=archive,
        )


def test_runtime_module_origin_audit_rejects_checkout_outside_module(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    def records(checkout: Path, relatives: list[str]) -> dict[str, Any]:
        return {
            "checkout": checkout,
            "files": {
                relative: {
                    "relative_path": relative,
                    "bytes": (checkout / relative).stat().st_size,
                    "sha256": sha256_file(checkout / relative),
                }
                for relative in relatives
            },
        }

    locks = {
        "CNOS": records(
            fixture["cnos_checkout"],
            ["src/model/fast_sam.py", "src/model/dinov2.py", "src/model/utils.py"],
        ),
        "DINOV2": records(fixture["dinov2_checkout"], fixture["dinov2_files"]),
    }
    registry = {
        "src.model.fast_sam": SimpleNamespace(
            __file__=str(fixture["cnos_checkout"] / "src/model/fast_sam.py")
        ),
        "src.model.dinov2": SimpleNamespace(
            __file__=str(fixture["cnos_checkout"] / "src/model/dinov2.py")
        ),
        "src.model.utils": SimpleNamespace(
            __file__=str(fixture["cnos_checkout"] / "src/model/utils.py")
        ),
        "dinov2": SimpleNamespace(
            __file__=str(fixture["dinov2_checkout"] / "dinov2/__init__.py")
        ),
        "dinov2.hub.backbones": SimpleNamespace(
            __file__=str(fixture["dinov2_checkout"] / "dinov2/hub/backbones.py")
        ),
    }
    audited = audit_runtime_module_origins(locks, module_registry=registry)
    assert audited["src.model.fast_sam"] == "src/model/fast_sam.py"
    registry["dinov2.hub"] = SimpleNamespace(
        __file__=None,
        __spec__=SimpleNamespace(
            submodule_search_locations=[str(fixture["dinov2_checkout"] / "dinov2/hub")]
        ),
    )
    audited = audit_runtime_module_origins(locks, module_registry=registry)
    assert audited["dinov2.hub.backbones"] == "dinov2/hub/backbones.py"
    registry["dinov2.hub"].__spec__.submodule_search_locations = [str(tmp_path)]
    with pytest.raises(ContractError, match="namespace was loaded outside"):
        audit_runtime_module_origins(locks, module_registry=registry)
    registry["dinov2.hub"].__spec__.submodule_search_locations = [
        str(fixture["dinov2_checkout"] / "dinov2/hub")
    ]
    outside = tmp_path / "backbones.py"
    shutil.copyfile(fixture["dinov2_checkout"] / "dinov2/hub/backbones.py", outside)
    registry["dinov2.hub.backbones"] = SimpleNamespace(__file__=str(outside))
    with pytest.raises(ContractError, match="outside locked checkout"):
        audit_runtime_module_origins(locks, module_registry=registry)


def test_current_ultralytics_route_gets_exact_legacy_cnos_alias() -> None:
    predictor = object()
    ultralytics = SimpleNamespace()
    modules = {
        "ultralytics": ultralytics,
        "ultralytics.models.yolo.segment.predict": SimpleNamespace(
            SegmentationPredictor=predictor
        ),
    }

    def import_module(name: str) -> Any:
        return modules[name]

    assert (
        ensure_ultralytics_yolo_compat(import_module=import_module)
        == "CURRENT_MODELS_YOLO_ALIAS"
    )
    assert ultralytics.yolo.v8.segment.SegmentationPredictor is predictor
    assert (
        ensure_ultralytics_yolo_compat(import_module=import_module)
        == "NATIVE_LEGACY_YOLO_ROUTE"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["source"].__setitem__("cnos_commit", "f" * 40),
        lambda value: value["models"]["fastsam_x_checkpoint"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda value: value["catalog"][1].__setitem__("object_id", 1),
        lambda value: value["catalog"][0]["descriptor_metadata"].__setitem__(
            "shape", [1, 1024]
        ),
        lambda value: value["catalog"][0]["cad"].__setitem__(
            "relative_path", "assets/scene_gt/obj.ply"
        ),
    ],
)
def test_runtime_identity_or_catalog_swap_fails_closed(
    fixture: dict[str, Any], mutation: Any
) -> None:
    request = copy.deepcopy(fixture["request"])
    mutation(request)
    _locked(request, "runtime_request_lock_sha256")
    with pytest.raises(ContractError):
        validate_runtime_request(
            request, fixture["protocol"], fixture["manifest"], data_root=fixture["root"]
        )


def test_negative_cosines_remain_distinct_and_raw_normalized_order_matches(
    fixture: dict[str, Any],
) -> None:
    ranked = _scores(fixture["request"], -0.1)
    assert ranked[0].raw_cosine == pytest.approx(-0.1)
    assert ranked[1].raw_cosine == pytest.approx(-0.18)
    assert ranked[0].normalized_similarity > ranked[1].normalized_similarity
    assert len({score.normalized_similarity for score in ranked}) == 5


class _CandidateDelegate:
    def __init__(
        self, request: dict[str, Any], *, mismatch: bool = False, zero: bool = False
    ) -> None:
        self.request = request
        self.calls = 0
        self.mismatch = mismatch
        self.zero = zero

    def infer(
        self, *, descriptor_relative_path: str, **_: Any
    ) -> tuple[list[Any], list[dict[str, int]]]:
        from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.adapter import Candidate

        if self.zero:
            raise ContractError(
                "FastSAM produced no valid post-geometric-filter proposals"
            )
        object_index = next(
            index
            for index, item in enumerate(self.request["catalog"])
            if item["descriptor"]["relative_path"] == descriptor_relative_path
        )
        mask = np.zeros((1080, 1440), dtype=bool)
        mask[10:30, 10:30] = True
        if self.mismatch and self.calls == 1:
            mask[50:60, 50:60] = True
        self.calls += 1
        raw = 0.4 - object_index * 0.1
        return [
            Candidate(
                4,
                mask,
                raw,
                (raw + 1) / 2,
                (raw + 0.02, raw + 0.01, raw, raw - 0.01, raw - 0.02),
                (0, 1, 2, 3, 4),
                0.8,
                0.9,
            )
        ], []


def test_catalog_adapter_has_no_target_input_and_scores_all_objects(
    fixture: dict[str, Any],
) -> None:
    delegate = _CandidateDelegate(fixture["request"])
    adapter = OfficialCnosCatalogAdapter(
        data_root=fixture["root"],
        request=fixture["request"],
        device="cpu",
        delegate=delegate,
    )
    proposals, _ = adapter.infer_frame(
        rgb_relative_path="inputs/rgb/x.png",
        proposal_chunk_size=16,
        minimum_chunk_size=1,
    )
    assert delegate.calls == 5
    assert [score.object_id for score in proposals[0].cad_ranking] == [1, 2, 4, 5, 6]


def test_catalog_adapter_rejects_proposal_swap_and_accepts_zero(
    fixture: dict[str, Any],
) -> None:
    mismatched = OfficialCnosCatalogAdapter(
        data_root=fixture["root"],
        request=fixture["request"],
        device="cpu",
        delegate=_CandidateDelegate(fixture["request"], mismatch=True),
    )
    with pytest.raises(ContractError, match="proposal set changed"):
        mismatched.infer_frame(
            rgb_relative_path="inputs/rgb/x.png",
            proposal_chunk_size=16,
            minimum_chunk_size=1,
        )
    zero = OfficialCnosCatalogAdapter(
        data_root=fixture["root"],
        request=fixture["request"],
        device="cpu",
        delegate=_CandidateDelegate(fixture["request"], zero=True),
    )
    assert zero.infer_frame(
        rgb_relative_path="inputs/rgb/x.png",
        proposal_chunk_size=16,
        minimum_chunk_size=1,
    ) == ([], [])


@pytest.fixture(scope="module")
def completed_run(
    fixture: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    output_root = tmp_path_factory.mktemp("a-r5-output")
    backend = FakeBackend(fixture["request"])
    bundle, receipt = run_producer(
        protocol=fixture["protocol"],
        manifest=fixture["manifest"],
        request=fixture["request"],
        data_root=fixture["root"],
        output_root=output_root,
        backend=backend,
    )
    return {
        "output_root": output_root,
        "bundle": bundle,
        "receipt": receipt,
        "backend": backend,
    }


def test_producer_emits_ten_disk_verifiable_rows_and_explicit_states(
    fixture: dict[str, Any], completed_run: dict[str, Any]
) -> None:
    bundle = completed_run["bundle"]
    states = {frame["proposal_state"] for frame in bundle["frames"]}
    assert "NO_PROPOSAL" in states
    assert "ONE_PROPOSAL" in states
    assert "MULTIPLE_PROPOSALS_DISJOINT" in states
    assert "MULTIPLE_PROPOSALS_ADJACENT" in states
    assert len(completed_run["backend"].calls) == 10
    assert all(
        frame["producer_identity"]["fastsam_x_checkpoint_sha256"] == FASTSAM_SHA256
        for frame in bundle["frames"]
    )
    assert all(
        set(frame["visualizations"]) == set(VISUALIZATION_ROLES)
        for frame in bundle["frames"]
    )
    validate_output_bundle(
        bundle,
        fixture["protocol"],
        fixture["manifest"],
        fixture["request"],
        data_root=fixture["root"],
        output_root=completed_run["output_root"],
    )
    validate_run_receipt(
        completed_run["receipt"], bundle, output_root=completed_run["output_root"]
    )


def test_proposal_display_order_is_confidence_first_but_selection_is_cad_first(
    completed_run: dict[str, Any],
) -> None:
    frame = completed_run["bundle"]["frames"][2]
    assert [proposal["proposal_index"] for proposal in frame["proposals"]] == [3, 8]
    assert (
        frame["proposals"][0]["proposal_score"]
        > frame["proposals"][1]["proposal_score"]
    )
    assert (
        frame["proposals"][1]["selected_cad_similarity"]
        > frame["proposals"][0]["selected_cad_similarity"]
    )
    assert frame["selected_proposal_index"] == 8


def test_disk_validator_rejects_mask_tamper(
    fixture: dict[str, Any], completed_run: dict[str, Any]
) -> None:
    frame = next(
        frame for frame in completed_run["bundle"]["frames"] if frame["proposals"]
    )
    path = completed_run["output_root"] / frame["proposals"][0]["mask"]["relative_path"]
    original = path.read_bytes()
    path.write_bytes(original + b"tamper")
    try:
        with pytest.raises(ContractError, match="changed on disk"):
            validate_output_bundle(
                completed_run["bundle"],
                fixture["protocol"],
                fixture["manifest"],
                fixture["request"],
                data_root=fixture["root"],
                output_root=completed_run["output_root"],
            )
    finally:
        path.write_bytes(original)


def test_disk_validator_rejects_raw_depth_tamper(
    fixture: dict[str, Any], completed_run: dict[str, Any]
) -> None:
    path = (
        fixture["root"]
        / fixture["manifest"]["frames"][0]["inputs"]["depth"]["relative_path"]
    )
    original = path.read_bytes()
    path.write_bytes(original + b"tamper")
    try:
        with pytest.raises(ContractError, match="byte count changed"):
            validate_output_bundle(
                completed_run["bundle"],
                fixture["protocol"],
                fixture["manifest"],
                fixture["request"],
                data_root=fixture["root"],
                output_root=completed_run["output_root"],
            )
    finally:
        path.write_bytes(original)


def test_bundle_and_receipt_canonical_locks_fail_closed(
    fixture: dict[str, Any], completed_run: dict[str, Any]
) -> None:
    bundle = copy.deepcopy(completed_run["bundle"])
    bundle["frames"][0]["latency_ms"] += 1
    with pytest.raises(ContractError, match="self-lock"):
        validate_output_bundle(
            bundle,
            fixture["protocol"],
            fixture["manifest"],
            fixture["request"],
            data_root=fixture["root"],
            output_root=completed_run["output_root"],
        )
    receipt = copy.deepcopy(completed_run["receipt"])
    receipt["attempt_count"] += 1
    with pytest.raises(ContractError, match="canonical lock"):
        validate_run_receipt(
            receipt, completed_run["bundle"], output_root=completed_run["output_root"]
        )


def test_planned_crash_resumes_only_unfinished_prefix(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    backend = FakeBackend(fixture["request"])
    with pytest.raises(PlannedCrash):
        run_producer(
            protocol=fixture["protocol"],
            manifest=fixture["manifest"],
            request=fixture["request"],
            data_root=fixture["root"],
            output_root=tmp_path,
            backend=backend,
            planned_crash_after_frames=3,
        )
    assert len(backend.calls) == 3
    bundle, _ = run_producer(
        protocol=fixture["protocol"],
        manifest=fixture["manifest"],
        request=fixture["request"],
        data_root=fixture["root"],
        output_root=tmp_path,
        backend=backend,
        resume=True,
    )
    assert len(backend.calls) == 10
    assert len(bundle["frames"]) == 10


def test_resume_rejects_completed_prefix_tamper(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    backend = FakeBackend(fixture["request"])
    with pytest.raises(PlannedCrash):
        run_producer(
            protocol=fixture["protocol"],
            manifest=fixture["manifest"],
            request=fixture["request"],
            data_root=fixture["root"],
            output_root=tmp_path,
            backend=backend,
            planned_crash_after_frames=1,
        )
    receipt_path = next((tmp_path / "outputs" / "item-receipts").glob("*.json"))
    receipt_path.write_bytes(receipt_path.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="receipt changed"):
        run_producer(
            protocol=fixture["protocol"],
            manifest=fixture["manifest"],
            request=fixture["request"],
            data_root=fixture["root"],
            output_root=tmp_path,
            backend=backend,
            resume=True,
        )


def test_cli_help_and_protocol_validation(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "all-catalog DINOv2" in capsys.readouterr().out
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, _protocol())
    assert (
        main(
            [
                "validate-protocol",
                "--protocol",
                str(protocol_path),
                "--repository-root",
                str(ROOT),
            ]
        )
        == 0
    )


def test_committed_protocol_matches_current_wrapper_bytes_when_present() -> None:
    if PROTOCOL_PATH.is_file():
        validate_protocol(
            json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")), repository_root=ROOT
        )
