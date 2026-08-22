from __future__ import annotations

import copy
import hashlib
import io
import importlib
import importlib.machinery
import json
import shutil
import tarfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
from PIL import Image

from pose_accuracy_recovery_prep.core import ContractError
from pose_accuracy_recovery_prep.instance_descriptor_assets_v1 import (
    FEATURE_DIMENSION,
    OBJECT_IDS,
    VIEW_COUNT,
)
from pose_accuracy_recovery_prep.instance_descriptor_assets_v1.contracts import (
    A_R4_CONTENT_AUDIT_INTERNAL_IDENTITY,
    A_R4_RENDER_PROVENANCE,
    PINNED_CNOS_ARCHIVE_BYTES,
    PINNED_CNOS_ARCHIVE_SHA256,
    PINNED_CNOS_COMMIT,
    PINNED_CNOS_REPOSITORY,
    PINNED_CNOS_TREE,
    PINNED_DINOV2_COMMIT,
    PINNED_DINOV2_REPOSITORY,
    PINNED_DINOV2_TREE,
    PINNED_DINOV2_VITL14_BYTES,
    PINNED_DINOV2_VITL14_SHA256,
    RuntimeProbes,
    audit_descriptor_tensor,
    audit_module_origins,
    build_runtime_request,
    canonical_sha256,
    create_only_json,
    read_json,
    sha256_file,
    validate_protocol,
    validate_a_r4_safe_evidence,
    validate_runtime_request,
)
from pose_accuracy_recovery_prep.instance_descriptor_assets_v1.producer import (
    COMPATIBILITY_BUNDLE_RELATIVE,
    FINAL_RECEIPT_RELATIVE,
    PLANNED_STOP_RELATIVE,
    PlannedStop,
    run_producer,
    validate_success_from_disk,
)
from pose_accuracy_recovery_prep.instance_descriptor_assets_v1.runtime import (
    DescriptorRuntime,
)
from pose_accuracy_recovery_prep.instance_descriptor_assets_v1.template_manifest import (
    build_template_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_instance_descriptor_assets_v1.json"
)


def _torch() -> Any:
    return importlib.import_module("torch")


def _actual_lock(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _execution_files(checkout: Path, paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": relative,
            **_actual_lock(checkout / Path(*relative.split("/"))),
        }
        for relative in sorted(paths)
    ]


@dataclass
class Prepared:
    root: Path
    protocol: dict[str, Any]
    request: dict[str, Any]
    protocol_path: Path
    request_path: Path


def _make_rgba(path: Path, *, object_id: int, view_index: int) -> None:
    pixels = np.zeros((480, 640, 4), dtype=np.uint8)
    x = 20 + (view_index * 13) % 560
    y = 20 + (view_index * 7) % 400
    pixels[y : y + 24, x : x + 28, :3] = (
        20 * object_id,
        40 + view_index,
        90 + object_id,
    )
    pixels[y : y + 24, x : x + 28, 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGBA").save(path)


def _content_audit_fixture(root: Path) -> dict[str, Any]:
    images = []
    objects = []
    for object_id in OBJECT_IDS:
        alpha_counts = []
        rgb_counts = []
        hashes = set()
        for view_index in range(VIEW_COUNT):
            path = (
                root
                / "frozen-r4/objects"
                / f"obj_{object_id:06d}"
                / f"{view_index:06d}.png"
            )
            with Image.open(path) as image:
                pixels = np.asarray(image)
            alpha_pixels = int((pixels[:, :, 3] > 0).sum())
            rgb_pixels = int(
                np.logical_and(
                    pixels[:, :, 3] > 0,
                    np.any(pixels[:, :, :3] > 0, axis=2),
                ).sum()
            )
            sha256 = sha256_file(path)
            alpha_counts.append(alpha_pixels)
            rgb_counts.append(rgb_pixels)
            hashes.add(sha256)
            images.append(
                {
                    "alpha_foreground_coverage": alpha_pixels / (640 * 480),
                    "alpha_foreground_pixels": alpha_pixels,
                    "bytes": path.stat().st_size,
                    "height": 480,
                    "mode": "RGBA",
                    "pass": True,
                    "relative_path": (
                        "attempts/poseloop_ga_cnos_v1r4_attempt_003/objects/"
                        f"obj_{object_id:06d}/{view_index:06d}.png"
                    ),
                    "rgb_foreground_coverage": rgb_pixels / (640 * 480),
                    "rgb_foreground_pixels": rgb_pixels,
                    "sha256": sha256,
                    "width": 640,
                }
            )
        objects.append(
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
    return {
        "attempt_receipt_lock_sha256": A_R4_CONTENT_AUDIT_INTERNAL_IDENTITY[
            "attempt_receipt_lock_sha256"
        ],
        "boundary_counters": {
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
        },
        "created_utc": "2026-08-18T00:00:00Z",
        "exit_code": 0,
        "forbidden_runs": {
            "FastSAM": 0,
            "FoundationPose": 0,
            "GPU-C": 0,
            "descriptor": 0,
            "evaluator_scorer": 0,
            "producer": 0,
        },
        "gates": {
            "alpha_foreground": "strictly_between_zero_and_full_frame",
            "frame_exact": [640, 480],
            "minimum_unique_png_sha256_per_object": 2,
            "mode_exact": "RGBA",
            "object_coverage": "5_of_5",
            "rgb_foreground": "strictly_between_zero_and_full_frame",
        },
        "images": images,
        "implementation_commit": A_R4_RENDER_PROVENANCE["implementation_commit"],
        "implementation_tree": A_R4_RENDER_PROVENANCE["implementation_tree"],
        "object_count": len(OBJECT_IDS),
        "objects": objects,
        "png_count": len(OBJECT_IDS) * VIEW_COUNT,
        "request_lock_sha256": A_R4_RENDER_PROVENANCE["runtime_request_lock_sha256"],
        "route_lock_sha256": A_R4_RENDER_PROVENANCE["route_lock_sha256"],
        "schema_version": "poseloop.a-r4.postrender-content-audit.v1",
        "server_lifecycle_action": "NONE",
        "status": "PASS",
    }


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _safe_evidence_fixture(root: Path, *, variant: str = "base") -> dict[str, Any]:
    frozen = root / "frozen-r4"
    deployment_root = "/fixture/a-r4"
    canonical_cads = []
    cad_assets = []
    for object_id in OBJECT_IDS:
        cad_path = root / "assets/cad" / f"obj_{object_id:06d}.ply"
        lock = _actual_lock(cad_path)
        archive_member_path = f"inputs/cad/obj_{object_id:06d}.ply"
        canonical_cads.append(
            {
                "object_id": object_id,
                "archive_member_path": archive_member_path,
                **lock,
            }
        )
        cad_assets.append(
            {
                "absolute_path": f"{deployment_root}/{archive_member_path}",
                **lock,
            }
        )
    deployment = {
        "cad_assets": cad_assets,
        "deployment_root": deployment_root,
        "fixture_variant": variant,
        "implementation": {
            "commit": A_R4_RENDER_PROVENANCE["implementation_commit"],
            "origin": "/fixture/poseloop",
            "status_porcelain": "",
            "tree": A_R4_RENDER_PROVENANCE["implementation_tree"],
        },
        "protocol": {
            "absolute_path": f"{deployment_root}/protocol.json",
            "bytes": 1,
            "sha256": A_R4_RENDER_PROVENANCE["protocol_sha256"],
        },
        "render": {
            "attempt_receipt": {
                "absolute_path": f"{deployment_root}/attempt-receipt.json",
                "bytes": 1,
                "sha256": A_R4_RENDER_PROVENANCE["attempt_receipt_sha256"],
            },
            "object_count": len(OBJECT_IDS),
            "png_count": len(OBJECT_IDS) * VIEW_COUNT,
            "postrender_audit": {
                "absolute_path": f"{deployment_root}/content-audit.json",
                "bytes": 1,
                "sha256": A_R4_RENDER_PROVENANCE["content_audit_sha256"],
            },
            "status": "PASS",
        },
        "request": {
            "absolute_path": f"{deployment_root}/request.json",
            "bytes": 1,
            "sha256": A_R4_RENDER_PROVENANCE["request_file_sha256"],
        },
        "route_lock_sha256": A_R4_RENDER_PROVENANCE["route_lock_sha256"],
        "schema_version": "poseloop.a-r4.deployment-final-inventory.v1",
        "server_lifecycle_action": "NONE",
        "status": "PASS",
    }
    deployment_path = frozen / "receipts/deployment_inventory.json"
    deployment_path.parent.mkdir(parents=True, exist_ok=True)
    deployment_path.write_bytes(_json_payload(deployment))

    payloads: dict[str, bytes] = {
        "attempts/poseloop_ga_cnos_v1r4_attempt_003/attempt-receipt.json": (
            frozen / "receipts/attempt.json"
        ).read_bytes(),
        "receipts/one-time-render-authorization-v1r4-attempt3.json": (
            frozen / "receipts/authorization.json"
        ).read_bytes(),
        "receipts/postrender-content-audit-v1r4-attempt3.json": (
            frozen / "receipts/content_audit.json"
        ).read_bytes(),
        "receipts/deployment-final-inventory-v1r4.json": deployment_path.read_bytes(),
    }
    for object_id in OBJECT_IDS:
        payloads[f"inputs/cad/obj_{object_id:06d}.ply"] = (
            root / "assets/cad" / f"obj_{object_id:06d}.ply"
        ).read_bytes()
        for view_index in range(VIEW_COUNT):
            relative = (
                "attempts/poseloop_ga_cnos_v1r4_attempt_003/objects/"
                f"obj_{object_id:06d}/{view_index:06d}.png"
            )
            payloads[relative] = (
                frozen / "objects" / f"obj_{object_id:06d}" / f"{view_index:06d}.png"
            ).read_bytes()
    filler_count = 254 - len(payloads)
    assert filler_count > 0
    for index in range(filler_count):
        payloads[f"fixture/filler/{index:06d}.txt"] = f"{variant}:{index}\n".encode()
    members = [
        {
            "relative_path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in sorted(payloads.items())
    ]
    member_inventory = {
        "contains_gt_or_evaluator": False,
        "contains_png_count": len(OBJECT_IDS) * VIEW_COUNT,
        "created_utc": "2026-08-18T00:00:00Z",
        "member_count": len(members),
        "members": members,
        "schema_version": "poseloop.a-r4.safe-evidence-members.v1",
        "server_lifecycle_action": "NONE",
        "status": "FROZEN",
    }
    member_path = frozen / "receipts/safe_members.json"
    member_path.write_bytes(_json_payload(member_inventory))
    sums = "".join(
        f"{row['sha256']}  {row['relative_path']}\n" for row in members
    ).encode()
    archive_path = frozen / "safe-evidence.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive_payloads = {
            **payloads,
            "receipts/safe-evidence-members-v1r4-attempt3.json": (
                member_path.read_bytes()
            ),
            "receipts/SAFE_EVIDENCE_SHA256SUMS": sums,
        }
        for relative, payload in archive_payloads.items():
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    binding = {
        "safe_archive": {**_actual_lock(archive_path), "member_count": 256},
        "member_inventory": {
            **_actual_lock(member_path),
            "schema_version": "poseloop.a-r4.safe-evidence-members.v1",
            "payload_member_count": 254,
            "archive_member_path": (
                "receipts/safe-evidence-members-v1r4-attempt3.json"
            ),
        },
        "deployment_inventory": {
            **_actual_lock(deployment_path),
            "schema_version": "poseloop.a-r4.deployment-final-inventory.v1",
            "archive_member_path": "receipts/deployment-final-inventory-v1r4.json",
        },
        "sha256sums_archive_member_path": "receipts/SAFE_EVIDENCE_SHA256SUMS",
        "canonical_cad_members": canonical_cads,
    }
    binding_path = frozen / "safe-binding.json"
    binding_path.write_bytes(_json_payload(binding))
    return {
        "archive": archive_path,
        "member_inventory": member_path,
        "deployment_inventory": deployment_path,
        "binding": binding,
    }


def _snapshot_files(
    root: Path, kind: str, protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if kind == "IMPLEMENTATION":
        return copy.deepcopy(protocol["wrapper_files"])
    if kind == "CNOS":
        paths = [
            "src/model/dinov2.py",
            "src/model/fast_sam.py",
            "src/model/utils.py",
            "src/utils/bbox_utils.py",
        ]
    else:
        paths = [
            "hubconf.py",
            "dinov2/__init__.py",
            "dinov2/models/__init__.py",
            "dinov2/models/vision_transformer.py",
        ]
    return _execution_files(root, paths)


def _make_probes(root: Path, protocol: Mapping[str, Any]) -> RuntimeProbes:
    root = root.resolve()
    safe_binding = read_json(root / "frozen-r4/safe-binding.json")
    source_receipt_hashes = {
        name: sha256_file(root / "frozen-r4/receipts" / f"{name}.json")
        for name in ("authorization", "attempt", "content_audit")
    }

    virtual_suffixes = {
        "sources/cnos-source.tar.gz": {
            "bytes": PINNED_CNOS_ARCHIVE_BYTES,
            "sha256": PINNED_CNOS_ARCHIVE_SHA256,
        },
        "models/dinov2_vitl14_pretrain.pth": {
            "bytes": PINNED_DINOV2_VITL14_BYTES,
            "sha256": PINNED_DINOV2_VITL14_SHA256,
        },
    }

    def file_lock(path: Path) -> Mapping[str, Any]:
        resolved = path.resolve()
        try:
            suffix = resolved.relative_to(root).as_posix()
        except ValueError:
            suffix = ""
        return virtual_suffixes.get(suffix, _actual_lock(resolved))

    def snapshot(checkout: Path, kind: str) -> Mapping[str, Any]:
        checkout = checkout.resolve()
        request_path = root / "contracts/descriptor-assets-request.json"
        request = read_json(request_path) if request_path.is_file() else None
        files = _snapshot_files(checkout, kind, protocol)
        if kind == "IMPLEMENTATION":
            identity = (
                request["implementation"]
                if request is not None
                else {"commit": "a" * 40, "tree": "b" * 40}
            )
            repository = "https://example.invalid/poseloop"
            status = ""
        elif kind == "CNOS":
            identity = {"commit": PINNED_CNOS_COMMIT, "tree": PINNED_CNOS_TREE}
            repository = PINNED_CNOS_REPOSITORY
            expected_paths = {item["relative_path"] for item in files}
            shadows = [
                path
                for path in checkout.rglob("*.py")
                if path.relative_to(checkout).as_posix() not in expected_paths
            ]
            status = (
                ""
                if not shadows
                else f"?? {shadows[0].relative_to(checkout).as_posix()}\n"
            )
        else:
            identity = {"commit": PINNED_DINOV2_COMMIT, "tree": PINNED_DINOV2_TREE}
            repository = PINNED_DINOV2_REPOSITORY
            expected_paths = {item["relative_path"] for item in files}
            shadows = [
                path
                for path in checkout.rglob("*.py")
                if path.relative_to(checkout).as_posix() not in expected_paths
            ]
            status = (
                ""
                if not shadows
                else f"?? {shadows[0].relative_to(checkout).as_posix()}\n"
            )
        return {
            "absolute_path": str(checkout),
            "repository": repository,
            "commit": identity["commit"],
            "tree": identity["tree"],
            "all_clean": status == "",
            "git_status_porcelain_v1_untracked_files_all": status,
            "execution_files": files,
            "execution_inventory_sha256": canonical_sha256(files),
        }

    def archive_lock(checkout: Path, prefix: str, commit: str) -> Mapping[str, Any]:
        del prefix, commit
        checkout = checkout.resolve()
        if checkout == (root / "sources/cnos").resolve():
            return virtual_suffixes["sources/cnos-source.tar.gz"]
        if checkout == (root / "sources/dinov2").resolve():
            return _actual_lock(root / "sources/dinov2-source.tar.gz")
        return _actual_lock(root / "sources/implementation-source.tar.gz")

    return RuntimeProbes(
        file_lock=file_lock,
        git_snapshot=snapshot,
        archive_lock=archive_lock,
        safe_evidence_binding=lambda: copy.deepcopy(safe_binding),
        source_receipt_hashes=lambda: copy.deepcopy(source_receipt_hashes),
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    root = tmp_path_factory.mktemp("a-r5-p2-base")
    protocol = read_json(PROTOCOL_PATH)

    for object_id in OBJECT_IDS:
        cad = root / "assets/cad" / f"obj_{object_id:06d}.ply"
        cad.parent.mkdir(parents=True, exist_ok=True)
        cad.write_bytes(f"ply\nobject {object_id}\n".encode())
        for view_index in range(VIEW_COUNT):
            _make_rgba(
                root
                / "frozen-r4/objects"
                / f"obj_{object_id:06d}"
                / f"{view_index:06d}.png",
                object_id=object_id,
                view_index=view_index,
            )
    receipt_sources = {}
    for name in ("authorization", "attempt", "content_audit"):
        path = root / "frozen-r4/receipts" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = (
            _content_audit_fixture(root)
            if name == "content_audit"
            else {"fixture": name}
        )
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        receipt_sources[name] = path
    safe_evidence = _safe_evidence_fixture(root)

    cnos = root / "sources/cnos"
    dinov2 = root / "sources/dinov2"
    for relative in (
        "src/model/dinov2.py",
        "src/model/fast_sam.py",
        "src/model/utils.py",
        "src/utils/bbox_utils.py",
    ):
        path = cnos / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture {relative}\n", encoding="utf-8")
    for relative in (
        "hubconf.py",
        "dinov2/__init__.py",
        "dinov2/models/__init__.py",
        "dinov2/models/vision_transformer.py",
    ):
        path = dinov2 / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture {relative}\n", encoding="utf-8")
    for relative, payload in (
        ("sources/implementation-source.tar.gz", b"implementation archive"),
        ("sources/cnos-source.tar.gz", b"virtual pinned CNOS archive"),
        ("sources/dinov2-source.tar.gz", b"DINO archive"),
        ("models/dinov2_vitl14_pretrain.pth", b"virtual official weight"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    probes = _make_probes(root, protocol)
    build_template_manifest(
        data_root=root,
        source_png_root=root / "frozen-r4/objects",
        authorization_receipt=receipt_sources["authorization"],
        attempt_receipt=receipt_sources["attempt"],
        content_audit=receipt_sources["content_audit"],
        safe_archive=safe_evidence["archive"],
        safe_member_inventory=safe_evidence["member_inventory"],
        deployment_inventory=safe_evidence["deployment_inventory"],
        cad_paths={
            object_id: root / "assets/cad" / f"obj_{object_id:06d}.ply"
            for object_id in OBJECT_IDS
        },
        manifest_output=root / "contracts/rgba-template-manifest.json",
        import_receipt_output=root / "receipts/a-r4-template-import.json",
        probes=probes,
    )

    request = build_runtime_request(
        protocol=protocol,
        data_root=root,
        implementation_archive=root / "sources/implementation-source.tar.gz",
        template_manifest=root / "contracts/rgba-template-manifest.json",
        template_import_receipt=root / "receipts/a-r4-template-import.json",
        cnos_checkout=cnos,
        cnos_archive=root / "sources/cnos-source.tar.gz",
        dinov2_checkout=dinov2,
        dinov2_archive=root / "sources/dinov2-source.tar.gz",
        dinov2_weights=root / "models/dinov2_vitl14_pretrain.pth",
        cad_paths={
            object_id: root / "assets/cad" / f"obj_{object_id:06d}.ply"
            for object_id in OBJECT_IDS
        },
        probes=probes,
    )
    request_path = root / "contracts/descriptor-assets-request.json"
    create_only_json(request_path, request)
    protocol_path = root / "contracts/protocol.json"
    shutil.copyfile(PROTOCOL_PATH, protocol_path)
    return Prepared(root, protocol, request, protocol_path, request_path)


@pytest.fixture
def case(tmp_path: Path, prepared: Prepared) -> Prepared:
    root = tmp_path / "case"
    shutil.copytree(prepared.root, root)
    return Prepared(
        root=root,
        protocol=copy.deepcopy(prepared.protocol),
        request=read_json(root / "contracts/descriptor-assets-request.json"),
        protocol_path=root / "contracts/protocol.json",
        request_path=root / "contracts/descriptor-assets-request.json",
    )


def _module_audit(validation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    mapping = {
        "src.model.dinov2": ("cnos_files", "src/model/dinov2.py"),
        "src.model.utils": ("cnos_files", "src/model/utils.py"),
        "src.utils.bbox_utils": ("cnos_files", "src/utils/bbox_utils.py"),
        "dinov2.models.vision_transformer": (
            "dinov2_files",
            "dinov2/models/vision_transformer.py",
        ),
    }
    return {
        name: {"relative_path": relative, **validation[group][relative]}
        for name, (group, relative) in mapping.items()
    }


def _runtime(validation: Mapping[str, Any]) -> DescriptorRuntime:
    torch = _torch()
    return DescriptorRuntime(
        torch=torch,
        descriptor_model=None,
        cropper=None,
        normalize=None,
        device=torch.device("cpu"),
        module_origin_audit=_module_audit(validation),
        weight_strict_load=True,
    )


def _compute(**kwargs: Any) -> Any:
    torch = _torch()
    object_id = kwargs["object_manifest"]["object_id"]
    return torch.full(
        (VIEW_COUNT, FEATURE_DIMENSION), float(object_id), dtype=torch.float32
    )


def _run(case: Prepared, **kwargs: Any) -> dict[str, Any]:
    return run_producer(
        protocol=case.protocol,
        request=case.request,
        protocol_path=case.protocol_path,
        request_path=case.request_path,
        data_root=case.root,
        probes=_make_probes(case.root, case.protocol),
        runtime_factory=_runtime,
        compute_fn=_compute,
        **kwargs,
    )


def test_protocol_and_wrapper_self_lock(prepared: Prepared) -> None:
    validated = validate_protocol(prepared.protocol, repository_root=ROOT)
    assert validated["objects"] == list(OBJECT_IDS)
    assert validated["auto_deploy"] is False


def test_template_import_is_exact_and_create_only(case: Prepared) -> None:
    manifest = read_json(case.root / "contracts/rgba-template-manifest.json")
    receipt = read_json(case.root / "receipts/a-r4-template-import.json")
    assert sum(len(item["views"]) for item in manifest["objects"]) == 210
    assert receipt["source_png_count"] == 210
    with pytest.raises(ContractError, match="create-only"):
        build_template_manifest(
            data_root=case.root,
            source_png_root=case.root / "frozen-r4/objects",
            authorization_receipt=case.root / "frozen-r4/receipts/authorization.json",
            attempt_receipt=case.root / "frozen-r4/receipts/attempt.json",
            content_audit=case.root / "frozen-r4/receipts/content_audit.json",
            safe_archive=case.root / "frozen-r4/safe-evidence.tar.gz",
            safe_member_inventory=case.root / "frozen-r4/receipts/safe_members.json",
            deployment_inventory=case.root
            / "frozen-r4/receipts/deployment_inventory.json",
            cad_paths={
                value: case.root / "assets/cad" / f"obj_{value:06d}.ply"
                for value in OBJECT_IDS
            },
            manifest_output=case.root / "contracts/rgba-template-manifest.json",
            import_receipt_output=case.root / "receipts/a-r4-template-import.json",
            probes=_make_probes(case.root, case.protocol),
        )


def test_template_import_rejects_png_not_bound_to_content_audit(
    tmp_path: Path, prepared: Prepared
) -> None:
    root = tmp_path / "audit-mismatch"
    shutil.copytree(prepared.root / "frozen-r4", root / "frozen-r4")
    shutil.copytree(prepared.root / "assets/cad", root / "assets/cad")
    _make_rgba(
        root / "frozen-r4/objects/obj_000001/000000.png",
        object_id=1,
        view_index=41,
    )
    receipt_paths = {
        name: root / "frozen-r4/receipts" / f"{name}.json"
        for name in ("authorization", "attempt", "content_audit")
    }
    with pytest.raises(ContractError, match="does not match frozen A-R4"):
        build_template_manifest(
            data_root=root,
            source_png_root=root / "frozen-r4/objects",
            authorization_receipt=receipt_paths["authorization"],
            attempt_receipt=receipt_paths["attempt"],
            content_audit=receipt_paths["content_audit"],
            safe_archive=root / "frozen-r4/safe-evidence.tar.gz",
            safe_member_inventory=root / "frozen-r4/receipts/safe_members.json",
            deployment_inventory=root / "frozen-r4/receipts/deployment_inventory.json",
            cad_paths={
                value: root / "assets/cad" / f"obj_{value:06d}.ply"
                for value in OBJECT_IDS
            },
            manifest_output=root / "contracts/rgba-template-manifest.json",
            import_receipt_output=root / "receipts/a-r4-template-import.json",
            probes=_make_probes(root, prepared.protocol),
        )


@pytest.mark.parametrize("swap_kind", ["fake_bytes", "object_swap"])
def test_template_import_rejects_cad_not_bound_to_safe_archive(
    tmp_path: Path, prepared: Prepared, swap_kind: str
) -> None:
    root = tmp_path / f"cad-{swap_kind}"
    shutil.copytree(prepared.root / "frozen-r4", root / "frozen-r4")
    shutil.copytree(prepared.root / "assets/cad", root / "assets/cad")
    cad_paths = {
        value: root / "assets/cad" / f"obj_{value:06d}.ply" for value in OBJECT_IDS
    }
    if swap_kind == "fake_bytes":
        cad_paths[1].write_bytes(b"ply\nreviewer fake CAD\n")
    else:
        cad_paths[1] = cad_paths[2]
    receipt_paths = {
        name: root / "frozen-r4/receipts" / f"{name}.json"
        for name in ("authorization", "attempt", "content_audit")
    }
    with pytest.raises(ContractError, match="does not match frozen A-R4 archive"):
        build_template_manifest(
            data_root=root,
            source_png_root=root / "frozen-r4/objects",
            authorization_receipt=receipt_paths["authorization"],
            attempt_receipt=receipt_paths["attempt"],
            content_audit=receipt_paths["content_audit"],
            safe_archive=root / "frozen-r4/safe-evidence.tar.gz",
            safe_member_inventory=root / "frozen-r4/receipts/safe_members.json",
            deployment_inventory=root / "frozen-r4/receipts/deployment_inventory.json",
            cad_paths=cad_paths,
            manifest_output=root / "contracts/rgba-template-manifest.json",
            import_receipt_output=root / "receipts/a-r4-template-import.json",
            probes=_make_probes(root, prepared.protocol),
        )


@pytest.mark.parametrize(
    "evidence_name", ["safe_archive", "member_inventory", "deployment_inventory"]
)
def test_safe_evidence_tamper_fails(
    tmp_path: Path, prepared: Prepared, evidence_name: str
) -> None:
    root = tmp_path / f"tamper-{evidence_name}"
    shutil.copytree(prepared.root / "frozen-r4", root / "frozen-r4")
    paths = {
        "safe_archive": root / "frozen-r4/safe-evidence.tar.gz",
        "member_inventory": root / "frozen-r4/receipts/safe_members.json",
        "deployment_inventory": root / "frozen-r4/receipts/deployment_inventory.json",
    }
    paths[evidence_name].write_bytes(paths[evidence_name].read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="bytes/SHA changed"):
        validate_a_r4_safe_evidence(
            safe_archive=paths["safe_archive"],
            member_inventory=paths["member_inventory"],
            deployment_inventory=paths["deployment_inventory"],
            probes=_make_probes(root, prepared.protocol),
        )


@pytest.mark.parametrize(
    "crossed_name", ["safe_archive", "member_inventory", "deployment_inventory"]
)
def test_safe_evidence_cross_pair_fails(
    tmp_path: Path, prepared: Prepared, crossed_name: str
) -> None:
    alternate_root = tmp_path / f"alternate-{crossed_name}"
    shutil.copytree(prepared.root, alternate_root)
    alternate = _safe_evidence_fixture(alternate_root, variant="alternate")
    base_paths = {
        "safe_archive": prepared.root / "frozen-r4/safe-evidence.tar.gz",
        "member_inventory": prepared.root / "frozen-r4/receipts/safe_members.json",
        "deployment_inventory": prepared.root
        / "frozen-r4/receipts/deployment_inventory.json",
    }
    alternate_paths = {
        "safe_archive": alternate["archive"],
        "member_inventory": alternate["member_inventory"],
        "deployment_inventory": alternate["deployment_inventory"],
    }
    selected = dict(base_paths)
    selected[crossed_name] = alternate_paths[crossed_name]
    hybrid_binding = read_json(prepared.root / "frozen-r4/safe-binding.json")
    hybrid_binding[
        {
            "safe_archive": "safe_archive",
            "member_inventory": "member_inventory",
            "deployment_inventory": "deployment_inventory",
        }[crossed_name]
    ] = copy.deepcopy(
        alternate["binding"][
            {
                "safe_archive": "safe_archive",
                "member_inventory": "member_inventory",
                "deployment_inventory": "deployment_inventory",
            }[crossed_name]
        ]
    )
    base_probes = _make_probes(prepared.root, prepared.protocol)
    probes = RuntimeProbes(
        file_lock=base_probes.file_lock,
        git_snapshot=base_probes.git_snapshot,
        archive_lock=base_probes.archive_lock,
        safe_evidence_binding=lambda: copy.deepcopy(hybrid_binding),
    )
    with pytest.raises(ContractError, match="diverge|changed"):
        validate_a_r4_safe_evidence(
            safe_archive=selected["safe_archive"],
            member_inventory=selected["member_inventory"],
            deployment_inventory=selected["deployment_inventory"],
            probes=probes,
        )


def test_runtime_request_validates_full_provenance(case: Prepared) -> None:
    value = validate_runtime_request(
        case.request,
        case.protocol,
        data_root=case.root,
        probes=_make_probes(case.root, case.protocol),
    )
    assert len(value["cnos_files"]) == 4
    assert len(value["dinov2_files"]) == 4
    assert value["request"]["boundary"]["label_access_count"] == 0


def test_pinned_cnos_transport_does_not_require_cross_platform_reencoding(
    case: Prepared,
) -> None:
    base = _make_probes(case.root, case.protocol)

    def archive_lock(checkout: Path, prefix: str, commit: str) -> Mapping[str, Any]:
        if checkout.resolve() == (case.root / "sources/cnos").resolve():
            return {"bytes": 1, "sha256": "0" * 64}
        return base.archive_lock(checkout, prefix, commit)

    probes = RuntimeProbes(
        file_lock=base.file_lock,
        git_snapshot=base.git_snapshot,
        archive_lock=archive_lock,
        safe_evidence_binding=base.safe_evidence_binding,
        source_receipt_hashes=base.source_receipt_hashes,
    )
    value = validate_runtime_request(
        case.request,
        case.protocol,
        data_root=case.root,
        probes=probes,
    )
    assert (
        value["request"]["source"]["cnos"]["source_archive"]
        == (case.request["source"]["cnos"]["source_archive"])
    )


def test_unpinned_dinov2_archive_still_requires_host_reconstruction(
    case: Prepared,
) -> None:
    base = _make_probes(case.root, case.protocol)

    def archive_lock(checkout: Path, prefix: str, commit: str) -> Mapping[str, Any]:
        if checkout.resolve() == (case.root / "sources/dinov2").resolve():
            return {"bytes": 1, "sha256": "0" * 64}
        return base.archive_lock(checkout, prefix, commit)

    probes = RuntimeProbes(
        file_lock=base.file_lock,
        git_snapshot=base.git_snapshot,
        archive_lock=archive_lock,
        safe_evidence_binding=base.safe_evidence_binding,
        source_receipt_hashes=base.source_receipt_hashes,
    )
    with pytest.raises(
        ContractError, match="DINOV2 source archive cannot be reconstructed"
    ):
        validate_runtime_request(
            case.request,
            case.protocol,
            data_root=case.root,
            probes=probes,
        )


def test_wrong_or_65_digit_weight_identity_fails(case: Prepared) -> None:
    for wrong in ("0" * 64, PINNED_DINOV2_VITL14_SHA256 + "0"):
        request = copy.deepcopy(case.request)
        request["model"]["dinov2_vitl14_checkpoint"]["sha256"] = wrong
        request["runtime_request_lock_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in request.items()
                if key != "runtime_request_lock_sha256"
            }
        )
        with pytest.raises(ContractError):
            validate_runtime_request(
                request,
                case.protocol,
                data_root=case.root,
                probes=_make_probes(case.root, case.protocol),
            )


def test_template_asset_swap_fails(case: Prepared) -> None:
    path = case.root / "assets/templates/obj_000001/000000.png"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="bytes/SHA"):
        validate_runtime_request(
            case.request,
            case.protocol,
            data_root=case.root,
            probes=_make_probes(case.root, case.protocol),
        )


def test_source_shadow_and_source_byte_tamper_fail(case: Prepared) -> None:
    shadow = case.root / "sources/cnos/src/shadow.py"
    shadow.write_text("# shadow\n", encoding="utf-8")
    with pytest.raises(ContractError, match="source shadow"):
        validate_runtime_request(
            case.request,
            case.protocol,
            data_root=case.root,
            probes=_make_probes(case.root, case.protocol),
        )
    shadow.unlink()
    source = case.root / "sources/dinov2/dinov2/models/vision_transformer.py"
    source.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ContractError, match="source bytes"):
        validate_runtime_request(
            case.request,
            case.protocol,
            data_root=case.root,
            probes=_make_probes(case.root, case.protocol),
        )


def test_module_origin_audit_rejects_outside_checkout(
    case: Prepared, tmp_path: Path
) -> None:
    validation = validate_runtime_request(
        case.request,
        case.protocol,
        data_root=case.root,
        probes=_make_probes(case.root, case.protocol),
    )
    registry: dict[str, Any] = {}
    for name, record in _module_audit(validation).items():
        module = types.ModuleType(name)
        root = (
            validation["cnos_checkout"]
            if name.startswith("src.")
            else validation["dinov2_checkout"]
        )
        module.__file__ = str(root / Path(*record["relative_path"].split("/")))
        registry[name] = module
    assert (
        len(
            audit_module_origins(
                registry,
                cnos_checkout=validation["cnos_checkout"],
                cnos_files=validation["cnos_files"],
                dinov2_checkout=validation["dinov2_checkout"],
                dinov2_files=validation["dinov2_files"],
            )
        )
        == 4
    )
    namespace = types.ModuleType("dinov2.models")
    namespace.__spec__ = importlib.machinery.ModuleSpec(
        "dinov2.models", loader=None, is_package=True
    )
    namespace.__spec__.submodule_search_locations = [
        str(validation["dinov2_checkout"] / "dinov2/models")
    ]
    registry["dinov2.models"] = namespace
    assert (
        len(
            audit_module_origins(
                registry,
                cnos_checkout=validation["cnos_checkout"],
                cnos_files=validation["cnos_files"],
                dinov2_checkout=validation["dinov2_checkout"],
                dinov2_files=validation["dinov2_files"],
            )
        )
        == 4
    )
    outside_namespace = tmp_path / "dinov2/models"
    outside_namespace.mkdir(parents=True)
    namespace.__spec__.submodule_search_locations = [str(outside_namespace)]
    with pytest.raises(ContractError, match="namespace loaded outside"):
        audit_module_origins(
            registry,
            cnos_checkout=validation["cnos_checkout"],
            cnos_files=validation["cnos_files"],
            dinov2_checkout=validation["dinov2_checkout"],
            dinov2_files=validation["dinov2_files"],
        )
    namespace.__spec__.submodule_search_locations = [
        str(validation["dinov2_checkout"] / "dinov2/models")
    ]
    registry["dinov2.models.vision_transformer"].__file__ = str(
        case.root / "outside.py"
    )
    with pytest.raises(ContractError, match="outside bound checkout"):
        audit_module_origins(
            registry,
            cnos_checkout=validation["cnos_checkout"],
            cnos_files=validation["cnos_files"],
            dinov2_checkout=validation["dinov2_checkout"],
            dinov2_files=validation["dinov2_files"],
        )


@pytest.mark.parametrize(
    "tensor_kind",
    ["wrong_shape", "wrong_dtype", "nonfinite"],
)
def test_valid_file_with_wrong_tensor_contract_fails(
    tmp_path: Path, tensor_kind: str
) -> None:
    torch = _torch()
    tensor = {
        "wrong_shape": torch.zeros((41, FEATURE_DIMENSION), dtype=torch.float32),
        "wrong_dtype": torch.zeros(
            (VIEW_COUNT, FEATURE_DIMENSION), dtype=torch.float64
        ),
        "nonfinite": torch.full(
            (VIEW_COUNT, FEATURE_DIMENSION), float("nan"), dtype=torch.float32
        ),
    }[tensor_kind]
    path = tmp_path / "descriptor.pth"
    torch.save(tensor, path)
    with pytest.raises(ContractError, match="finite float32"):
        audit_descriptor_tensor(path, torch_module=torch)


def test_orphan_tensor_without_sidecar_fails(case: Prepared) -> None:
    torch = _torch()
    path = case.root / "assets/descriptors/obj_000001.pth"
    path.parent.mkdir(parents=True)
    torch.save(torch.zeros((VIEW_COUNT, FEATURE_DIMENSION), dtype=torch.float32), path)
    with pytest.raises(ContractError, match="without its request-specific sidecar"):
        _run(case)


def test_planned_stop_resume_and_independent_validation(case: Prepared) -> None:
    torch = _torch()
    with pytest.raises(PlannedStop):
        _run(case, planned_stop_after_object_count=2)
    assert (case.root / PLANNED_STOP_RELATIVE).is_file()
    result = _run(case, resume=True)
    assert result["descriptor_count"] == 5
    validation = validate_success_from_disk(
        protocol=case.protocol,
        request=case.request,
        protocol_path=case.protocol_path,
        request_path=case.request_path,
        data_root=case.root,
        probes=_make_probes(case.root, case.protocol),
        torch_module=torch,
    )
    assert validation["rgba_count"] == 210
    bundle = read_json(case.root / COMPATIBILITY_BUNDLE_RELATIVE)
    assert len(bundle["catalog_snippet"]) == 5
    receipt_sha = validation["descriptor_generation_receipt"]["sha256"]
    assert {
        item["descriptor_metadata"]["generation_receipt_sha256"]
        for item in bundle["catalog_snippet"]
    } == {receipt_sha}


def test_resume_rejects_missing_planned_stop(case: Prepared) -> None:
    with pytest.raises(PlannedStop):
        _run(case, planned_stop_after_object_count=1)
    (case.root / PLANNED_STOP_RELATIVE).unlink()
    with pytest.raises(ContractError, match="planned-stop prefix"):
        _run(case, resume=True)


def test_cross_request_resume_rejected(case: Prepared) -> None:
    with pytest.raises(PlannedStop):
        _run(case, planned_stop_after_object_count=1)
    alternate = copy.deepcopy(case.request)
    alternate["implementation"]["commit"] = "c" * 40
    alternate["implementation"]["tree"] = "d" * 40
    prefix = f"poseloop-{'c' * 40}/"
    alternate["implementation"]["archive_route"] = {
        "format": "tar.gz",
        "prefix": prefix,
        "command": [
            "git",
            "archive",
            "--format=tar.gz",
            f"--prefix={prefix}",
            "c" * 40,
        ],
    }
    alternate["runtime_request_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in alternate.items()
            if key != "runtime_request_lock_sha256"
        }
    )
    case.request = alternate
    case.request_path.write_text(
        json.dumps(alternate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ContractError, match="cross-request"):
        _run(case, resume=True)


def test_disk_validator_rejects_descriptor_tamper(case: Prepared) -> None:
    torch = _torch()
    _run(case)
    path = case.root / "assets/descriptors/obj_000004.pth"
    torch.save(torch.ones((VIEW_COUNT, FEATURE_DIMENSION), dtype=torch.float32), path)
    with pytest.raises(ContractError):
        validate_success_from_disk(
            protocol=case.protocol,
            request=case.request,
            protocol_path=case.protocol_path,
            request_path=case.request_path,
            data_root=case.root,
            probes=_make_probes(case.root, case.protocol),
            torch_module=torch,
        )


def test_final_receipt_and_bundle_are_create_only(case: Prepared) -> None:
    _run(case)
    assert (case.root / FINAL_RECEIPT_RELATIVE).is_file()
    with pytest.raises(ContractError, match="already exist"):
        _run(case)
