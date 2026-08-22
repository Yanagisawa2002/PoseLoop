"""Fail-closed contracts for A-R5-P2 descriptor asset production."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

from pose_accuracy_recovery_prep.core import ContractError
from pose_accuracy_recovery_prep.instance_proposal_v1r5.contracts import (
    A_R4_RENDER_PROVENANCE,
    BOUNDARY_ZERO,
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
    PINNED_DINOV2_VITL14_URL,
)

from . import (
    FEATURE_DIMENSION,
    OBJECT_IDS,
    PROTOCOL_ID,
    PROTOCOL_SCHEMA,
    REQUEST_SCHEMA,
    TEMPLATE_IMPORT_RECEIPT_SCHEMA,
    VIEW_COUNT,
)

TEMPLATE_HEIGHT = 480
TEMPLATE_WIDTH = 640
TEMPLATE_MANIFEST_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-rgba-template-manifest.v1r5"
)
A_R5_PROTOCOL_ID = "poseloop.pose-accuracy-recovery.development.instance-proposal.v1r5"
A_R4_CONTENT_AUDIT_INTERNAL_IDENTITY = {
    "content_audit_sha256": A_R4_RENDER_PROVENANCE["content_audit_sha256"],
    "attempt_receipt_lock_sha256": (
        "b36a9aacfe57f58292c0aaaaca2333fb763b4c38b50a3bdf723a60d9621fc570"
    ),
}
A_R4_SOURCE_RECEIPT_HASHES = {
    "authorization": A_R4_RENDER_PROVENANCE["authorization_receipt_sha256"],
    "attempt": A_R4_RENDER_PROVENANCE["attempt_receipt_sha256"],
    "content_audit": A_R4_RENDER_PROVENANCE["content_audit_sha256"],
}
A_R4_SAFE_EVIDENCE_BINDING = {
    "safe_archive": {
        "bytes": 1653423,
        "sha256": "c997f4f6d65f2398527b55bcdffbb40cf4881783cbe50580e712697646f118bb",
        "member_count": 256,
    },
    "member_inventory": {
        "bytes": 54361,
        "sha256": "31100eb821a6230cc9700984dad81b786d63d58f26456e55e4c0622b9fda42e3",
        "schema_version": "poseloop.a-r4.safe-evidence-members.v1",
        "payload_member_count": 254,
        "archive_member_path": ("receipts/safe-evidence-members-v1r4-attempt3.json"),
    },
    "deployment_inventory": {
        "bytes": 9065,
        "sha256": A_R4_RENDER_PROVENANCE["deployment_inventory_sha256"],
        "schema_version": "poseloop.a-r4.deployment-final-inventory.v1",
        "archive_member_path": "receipts/deployment-final-inventory-v1r4.json",
    },
    "sha256sums_archive_member_path": "receipts/SAFE_EVIDENCE_SHA256SUMS",
    "canonical_cad_members": [
        {
            "object_id": 1,
            "archive_member_path": "inputs/cad/obj_000001.ply",
            "bytes": 127492,
            "sha256": (
                "177bf8313fc8feb749e86ab8953d9f97102842e54a08c63f32b8adfb663321e1"
            ),
        },
        {
            "object_id": 2,
            "archive_member_path": "inputs/cad/obj_000002.ply",
            "bytes": 319240,
            "sha256": (
                "c6cc5f76b73782415a44362ee8853542f1da580fc49aac8b165767cfc6e2a24e"
            ),
        },
        {
            "object_id": 4,
            "archive_member_path": "inputs/cad/obj_000004.ply",
            "bytes": 658929,
            "sha256": (
                "2885f809073970da906e5192d848dde4f15f844e1ef1e80d13a6ad7a603c9ef9"
            ),
        },
        {
            "object_id": 5,
            "archive_member_path": "inputs/cad/obj_000005.ply",
            "bytes": 679427,
            "sha256": (
                "33dbf9fec3fa2a68fe6ec847507221feac8f00c002f61619bce1344395383a01"
            ),
        },
        {
            "object_id": 6,
            "archive_member_path": "inputs/cad/obj_000006.ply",
            "bytes": 708278,
            "sha256": (
                "0e9ce0f575f104f97c4baf857b5449101eecc9658382b04f893905858a75eece"
            ),
        },
    ],
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_DATA_TOKEN = re.compile(
    r"(^|[\\/_.-])(scene_gt(?:_info)?|mask_visib|depth|evaluator|sealed|"
    r"foundationpose|ground_truth|oracle)([\\/_.-]|$)",
    re.IGNORECASE,
)

_WRAPPER_PATHS = {
    "pose_accuracy_recovery_prep/core.py",
    "pose_accuracy_recovery_prep/instance_proposal_v1r5/contracts.py",
    "pose_accuracy_recovery_prep/instance_descriptor_assets_v1/__init__.py",
    "pose_accuracy_recovery_prep/instance_descriptor_assets_v1/__main__.py",
    "pose_accuracy_recovery_prep/instance_descriptor_assets_v1/contracts.py",
    "pose_accuracy_recovery_prep/instance_descriptor_assets_v1/template_manifest.py",
    "pose_accuracy_recovery_prep/instance_descriptor_assets_v1/runtime.py",
    "pose_accuracy_recovery_prep/instance_descriptor_assets_v1/producer.py",
    "pose_accuracy_recovery_prep/instance_descriptor_assets_v1/cli.py",
    "pose_accuracy_recovery_prep/instance_descriptor_assets_v1/"
    "SERVER_RUNBOOK_A_R5_P2.md",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ContractError(f"Output is create-only: {path}") from exc


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ContractError(f"{label} fields changed")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ContractError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def _git_oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_OID.fullmatch(value) is None:
        raise ContractError(f"{label} must be lowercase 40-hex Git identity")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _self_lock(value: Mapping[str, Any], field: str, label: str) -> None:
    observed = _sha(value.get(field), f"{label}.{field}")
    expected = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )
    if observed != expected:
        raise ContractError(f"{label} self-lock mismatch")


def _disk_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Missing file: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _run_git(checkout: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise ContractError(f"Git probe failed in {checkout}: {' '.join(args)}")
    return process.stdout


def _default_git_snapshot(checkout: Path, kind: str) -> dict[str, Any]:
    root = checkout.resolve()
    if not (root / ".git").exists():
        raise ContractError(f"{kind} checkout has no .git: {root}")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    try:
        repository = _run_git(root, "remote", "get-url", "origin").strip()
    except ContractError:
        repository = ""
    tracked = [value for value in _run_git(root, "ls-files", "-z").split("\0") if value]
    if kind == "IMPLEMENTATION":
        paths = sorted(_WRAPPER_PATHS)
    else:
        paths = sorted(value for value in tracked if value.endswith(".py"))
        if not paths:
            raise ContractError(f"{kind} checkout has no tracked Python sources")
    execution_files = []
    for relative in paths:
        path = root / Path(*PurePosixPath(relative).parts)
        lock = _disk_lock(path)
        execution_files.append({"relative_path": relative, **lock})
    return {
        "absolute_path": str(root),
        "repository": repository,
        "commit": _run_git(root, "rev-parse", "HEAD").strip(),
        "tree": _run_git(root, "rev-parse", "HEAD^{tree}").strip(),
        "all_clean": status == "",
        "git_status_porcelain_v1_untracked_files_all": status,
        "execution_files": execution_files,
        "execution_inventory_sha256": canonical_sha256(execution_files),
    }


def _default_archive_lock(checkout: Path, prefix: str, commit: str) -> dict[str, Any]:
    process = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "archive",
            "--format=tar.gz",
            f"--prefix={prefix}",
            commit,
        ],
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise ContractError(f"Cannot reconstruct source archive in {checkout}")
    return {
        "bytes": len(process.stdout),
        "sha256": hashlib.sha256(process.stdout).hexdigest(),
    }


def _default_safe_evidence_binding() -> Mapping[str, Any]:
    return A_R4_SAFE_EVIDENCE_BINDING


def _default_source_receipt_hashes() -> Mapping[str, str]:
    return A_R4_SOURCE_RECEIPT_HASHES


@dataclass(frozen=True)
class RuntimeProbes:
    """Injectable disk/Git probes used only by contract fixtures."""

    file_lock: Callable[[Path], Mapping[str, Any]] = _disk_lock
    git_snapshot: Callable[[Path, str], Mapping[str, Any]] = _default_git_snapshot
    archive_lock: Callable[[Path, str, str], Mapping[str, Any]] = _default_archive_lock
    safe_evidence_binding: Callable[[], Mapping[str, Any]] = (
        _default_safe_evidence_binding
    )
    source_receipt_hashes: Callable[[], Mapping[str, str]] = (
        _default_source_receipt_hashes
    )


DEFAULT_PROBES = RuntimeProbes()


def repository_root_from_package() -> Path:
    return Path(__file__).resolve().parents[2]


def _asset_from_path(
    path: Path,
    *,
    data_root: Path,
    role: str,
    probes: RuntimeProbes,
) -> dict[str, Any]:
    root = data_root.resolve()
    candidate = path.resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError(f"{role} asset escapes data root") from exc
    if _FORBIDDEN_DATA_TOKEN.search(relative):
        raise ContractError(f"Forbidden descriptor input token: {relative}")
    lock = dict(probes.file_lock(candidate))
    if set(lock) != {"bytes", "sha256"}:
        raise ContractError(f"{role} file probe schema changed")
    _positive_int(lock["bytes"], f"{role}.bytes")
    _sha(lock["sha256"], f"{role}.sha256")
    return {"role": role, "relative_path": relative, **lock}


def _resolve_asset(
    asset: Any,
    *,
    data_root: Path,
    role: str,
    root_name: str,
    label: str,
    probes: RuntimeProbes,
) -> tuple[dict[str, Any], Path]:
    value = _exact(
        asset,
        {"role", "relative_path", "bytes", "sha256"},
        label,
    )
    if value["role"] != role:
        raise ContractError(f"{label}.role must be {role}")
    relative_raw = value["relative_path"]
    if not isinstance(relative_raw, str) or "\\" in relative_raw:
        raise ContractError(f"{label}.relative_path must be POSIX-relative")
    relative = PurePosixPath(relative_raw)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != root_name
        or _FORBIDDEN_DATA_TOKEN.search(relative_raw)
    ):
        raise ContractError(f"{label}.relative_path escapes its frozen role root")
    _positive_int(value["bytes"], f"{label}.bytes")
    _sha(value["sha256"], f"{label}.sha256")
    root = data_root.resolve()
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{label} resolves outside data root") from exc
    observed = dict(probes.file_lock(path))
    if observed != {"bytes": value["bytes"], "sha256": value["sha256"]}:
        raise ContractError(f"{label} disk bytes/SHA changed")
    return value, path


def _validate_safe_evidence_binding(raw: Any) -> dict[str, Any]:
    binding = _exact(
        raw,
        {
            "safe_archive",
            "member_inventory",
            "deployment_inventory",
            "sha256sums_archive_member_path",
            "canonical_cad_members",
        },
        "A-R4 safe evidence binding",
    )
    archive = _exact(
        binding["safe_archive"],
        {"bytes", "sha256", "member_count"},
        "A-R4 safe archive binding",
    )
    _positive_int(archive["bytes"], "A-R4 safe archive bytes")
    _positive_int(archive["member_count"], "A-R4 safe archive member count")
    _sha(archive["sha256"], "A-R4 safe archive SHA")
    member_inventory = _exact(
        binding["member_inventory"],
        {
            "bytes",
            "sha256",
            "schema_version",
            "payload_member_count",
            "archive_member_path",
        },
        "A-R4 safe member-inventory binding",
    )
    _positive_int(member_inventory["bytes"], "A-R4 member inventory bytes")
    _positive_int(
        member_inventory["payload_member_count"],
        "A-R4 payload member count",
    )
    _sha(member_inventory["sha256"], "A-R4 member inventory SHA")
    if (
        member_inventory["schema_version"] != "poseloop.a-r4.safe-evidence-members.v1"
        or member_inventory["archive_member_path"]
        != "receipts/safe-evidence-members-v1r4-attempt3.json"
    ):
        raise ContractError("A-R4 safe member-inventory route changed")
    deployment = _exact(
        binding["deployment_inventory"],
        {"bytes", "sha256", "schema_version", "archive_member_path"},
        "A-R4 deployment-inventory binding",
    )
    _positive_int(deployment["bytes"], "A-R4 deployment inventory bytes")
    _sha(deployment["sha256"], "A-R4 deployment inventory SHA")
    if (
        deployment["schema_version"] != "poseloop.a-r4.deployment-final-inventory.v1"
        or deployment["archive_member_path"]
        != "receipts/deployment-final-inventory-v1r4.json"
        or binding["sha256sums_archive_member_path"]
        != "receipts/SAFE_EVIDENCE_SHA256SUMS"
    ):
        raise ContractError("A-R4 deployment/sums evidence route changed")
    cads = binding["canonical_cad_members"]
    if not isinstance(cads, list) or len(cads) != len(OBJECT_IDS):
        raise ContractError("A-R4 canonical CAD evidence coverage changed")
    validated_cads = []
    for index, raw_cad in enumerate(cads):
        cad = _exact(
            raw_cad,
            {"object_id", "archive_member_path", "bytes", "sha256"},
            f"A-R4 canonical CAD[{index}]",
        )
        object_id = OBJECT_IDS[index]
        if (
            cad["object_id"] != object_id
            or cad["archive_member_path"] != f"inputs/cad/obj_{object_id:06d}.ply"
        ):
            raise ContractError("A-R4 canonical CAD object/path mapping changed")
        _positive_int(cad["bytes"], f"A-R4 CAD {object_id} bytes")
        _sha(cad["sha256"], f"A-R4 CAD {object_id} SHA")
        validated_cads.append(cad)
    binding["canonical_cad_members"] = validated_cads
    return binding


def _safe_archive_member_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ContractError(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or _FORBIDDEN_DATA_TOKEN.search(value)
    ):
        raise ContractError(f"{label} is unsafe or crosses the runtime boundary")
    return value


def validate_a_r4_safe_evidence(
    *,
    safe_archive: Path,
    member_inventory: Path,
    deployment_inventory: Path,
    probes: RuntimeProbes = DEFAULT_PROBES,
) -> dict[str, Any]:
    """Recompute the frozen safe archive and cross-bind its two inventories."""

    binding = _validate_safe_evidence_binding(probes.safe_evidence_binding())
    archive_lock = _disk_lock(safe_archive)
    member_lock = _disk_lock(member_inventory)
    deployment_lock = _disk_lock(deployment_inventory)
    if archive_lock != {
        "bytes": binding["safe_archive"]["bytes"],
        "sha256": binding["safe_archive"]["sha256"],
    }:
        raise ContractError("Frozen A-R4 safe archive bytes/SHA changed")
    if member_lock != {
        "bytes": binding["member_inventory"]["bytes"],
        "sha256": binding["member_inventory"]["sha256"],
    }:
        raise ContractError("Frozen A-R4 member inventory bytes/SHA changed")
    if deployment_lock != {
        "bytes": binding["deployment_inventory"]["bytes"],
        "sha256": binding["deployment_inventory"]["sha256"],
    }:
        raise ContractError("Frozen A-R4 deployment inventory bytes/SHA changed")

    inventory = _exact(
        read_json(member_inventory, "frozen A-R4 safe member inventory"),
        {
            "contains_gt_or_evaluator",
            "contains_png_count",
            "created_utc",
            "member_count",
            "members",
            "schema_version",
            "server_lifecycle_action",
            "status",
        },
        "frozen A-R4 safe member inventory",
    )
    if (
        inventory["schema_version"] != binding["member_inventory"]["schema_version"]
        or inventory["status"] != "FROZEN"
        or inventory["contains_gt_or_evaluator"] is not False
        or inventory["contains_png_count"] != len(OBJECT_IDS) * VIEW_COUNT
        or inventory["member_count"]
        != binding["member_inventory"]["payload_member_count"]
        or inventory["server_lifecycle_action"] != "NONE"
        or not isinstance(inventory["created_utc"], str)
        or not inventory["created_utc"]
    ):
        raise ContractError("Frozen A-R4 safe member inventory gates changed")
    rows = inventory["members"]
    if not isinstance(rows, list) or len(rows) != inventory["member_count"]:
        raise ContractError("Frozen A-R4 payload member coverage changed")
    member_map: dict[str, dict[str, Any]] = {}
    validated_rows = []
    for index, raw_row in enumerate(rows):
        row = _exact(
            raw_row,
            {"relative_path", "bytes", "sha256"},
            f"A-R4 safe member[{index}]",
        )
        relative = _safe_archive_member_path(
            row["relative_path"], f"A-R4 safe member[{index}].relative_path"
        )
        if relative in member_map:
            raise ContractError("Frozen A-R4 safe archive has duplicate payload paths")
        if (
            not isinstance(row["bytes"], int)
            or isinstance(row["bytes"], bool)
            or row["bytes"] < 0
        ):
            raise ContractError("Frozen A-R4 payload member bytes are invalid")
        _sha(row["sha256"], f"A-R4 safe member[{index}].sha256")
        record = {"bytes": row["bytes"], "sha256": row["sha256"]}
        member_map[relative] = record
        validated_rows.append(row)

    try:
        with tarfile.open(safe_archive, mode="r:gz") as archive:
            tar_members = archive.getmembers()
            if len(tar_members) != binding["safe_archive"]["member_count"] or any(
                not member.isfile() for member in tar_members
            ):
                raise ContractError(
                    "Frozen A-R4 safe archive member count/type changed"
                )
            tar_map: dict[str, tarfile.TarInfo] = {}
            for index, member in enumerate(tar_members):
                name = _safe_archive_member_path(
                    member.name, f"A-R4 archive member[{index}]"
                )
                if name in tar_map:
                    raise ContractError(
                        "Frozen A-R4 safe archive has duplicate members"
                    )
                tar_map[name] = member
            special_paths = {
                binding["member_inventory"]["archive_member_path"],
                binding["sha256sums_archive_member_path"],
            }
            if set(tar_map) != set(member_map) | special_paths:
                raise ContractError("Frozen A-R4 safe archive/inventory paths diverge")
            for relative, expected in member_map.items():
                stream = archive.extractfile(tar_map[relative])
                if stream is None:
                    raise ContractError("Cannot read frozen A-R4 archive payload")
                payload = stream.read()
                observed = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                if observed != expected or tar_map[relative].size != len(payload):
                    raise ContractError(
                        f"Frozen A-R4 archive payload changed: {relative}"
                    )
            manifest_stream = archive.extractfile(
                tar_map[binding["member_inventory"]["archive_member_path"]]
            )
            sums_stream = archive.extractfile(
                tar_map[binding["sha256sums_archive_member_path"]]
            )
            if manifest_stream is None or sums_stream is None:
                raise ContractError("Frozen A-R4 archive closure members are missing")
            if manifest_stream.read() != member_inventory.read_bytes():
                raise ContractError("A-R4 external/internal member inventories diverge")
            expected_sums = "".join(
                f"{row['sha256']}  {row['relative_path']}\n" for row in validated_rows
            ).encode("utf-8")
            if sums_stream.read() != expected_sums:
                raise ContractError("Frozen A-R4 SHA256SUMS/member inventory diverge")
    except (OSError, tarfile.TarError) as exc:
        raise ContractError("Cannot read frozen A-R4 safe archive") from exc

    deployment_member_path = binding["deployment_inventory"]["archive_member_path"]
    if member_map.get(deployment_member_path) != deployment_lock:
        raise ContractError("A-R4 deployment inventory/archive member diverge")
    deployment = read_json(deployment_inventory, "frozen A-R4 deployment inventory")
    required_deployment = {
        "cad_assets",
        "deployment_root",
        "implementation",
        "protocol",
        "render",
        "request",
        "route_lock_sha256",
        "schema_version",
        "server_lifecycle_action",
        "status",
    }
    if not required_deployment.issubset(deployment):
        raise ContractError("Frozen A-R4 deployment inventory fields changed")
    implementation = deployment["implementation"]
    protocol_asset = deployment["protocol"]
    request_asset = deployment["request"]
    render = deployment["render"]
    if (
        deployment["schema_version"]
        != binding["deployment_inventory"]["schema_version"]
        or deployment["status"] != "PASS"
        or deployment["server_lifecycle_action"] != "NONE"
        or deployment["route_lock_sha256"]
        != A_R4_RENDER_PROVENANCE["route_lock_sha256"]
        or not isinstance(deployment["deployment_root"], str)
        or not deployment["deployment_root"].startswith("/")
        or not isinstance(implementation, Mapping)
        or implementation.get("commit")
        != A_R4_RENDER_PROVENANCE["implementation_commit"]
        or implementation.get("tree") != A_R4_RENDER_PROVENANCE["implementation_tree"]
        or implementation.get("status_porcelain") != ""
        or not isinstance(protocol_asset, Mapping)
        or protocol_asset.get("sha256") != A_R4_RENDER_PROVENANCE["protocol_sha256"]
        or not isinstance(request_asset, Mapping)
        or request_asset.get("sha256") != A_R4_RENDER_PROVENANCE["request_file_sha256"]
        or not isinstance(render, Mapping)
        or render.get("status") != "PASS"
        or render.get("object_count") != len(OBJECT_IDS)
        or render.get("png_count") != len(OBJECT_IDS) * VIEW_COUNT
        or not isinstance(render.get("attempt_receipt"), Mapping)
        or render["attempt_receipt"].get("sha256")
        != A_R4_RENDER_PROVENANCE["attempt_receipt_sha256"]
        or not isinstance(render.get("postrender_audit"), Mapping)
        or render["postrender_audit"].get("sha256")
        != A_R4_RENDER_PROVENANCE["content_audit_sha256"]
    ):
        raise ContractError("Frozen A-R4 deployment identity changed")

    canonical_cads = []
    for raw_cad in binding["canonical_cad_members"]:
        relative = raw_cad["archive_member_path"]
        observed = member_map.get(relative)
        expected = {"bytes": raw_cad["bytes"], "sha256": raw_cad["sha256"]}
        if observed != expected:
            raise ContractError("A-R4 canonical CAD/archive member identity changed")
        canonical_cads.append(dict(raw_cad))
    deployment_cads = deployment["cad_assets"]
    if not isinstance(deployment_cads, list) or len(deployment_cads) != len(
        canonical_cads
    ):
        raise ContractError("A-R4 deployment CAD coverage changed")
    deployment_root = deployment["deployment_root"].rstrip("/")
    for index, canonical in enumerate(canonical_cads):
        deployed = _exact(
            deployment_cads[index],
            {"absolute_path", "bytes", "sha256"},
            f"A-R4 deployment CAD[{index}]",
        )
        if deployed != {
            "absolute_path": f"{deployment_root}/{canonical['archive_member_path']}",
            "bytes": canonical["bytes"],
            "sha256": canonical["sha256"],
        }:
            raise ContractError("A-R4 deployment/archive CAD cross-pair changed")
    return {
        "binding": binding,
        "safe_archive": archive_lock,
        "member_inventory": member_lock,
        "deployment_inventory": deployment_lock,
        "payload_members": member_map,
        "canonical_cads": canonical_cads,
    }


def validate_protocol(
    protocol: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    value = _exact(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "role",
            "auto_deploy",
            "accuracy_claim_permitted",
            "predecessor",
            "a_r4_render_provenance",
            "a_r4_content_audit_internal_identity",
            "a_r4_safe_evidence_binding",
            "objects",
            "descriptor_contract",
            "model_provenance",
            "execution_policy",
            "runtime_boundary",
            "wrapper_files",
            "protocol_lock_sha256",
        },
        "A-R5-P2 protocol",
    )
    if (
        value["schema_version"] != PROTOCOL_SCHEMA
        or value["protocol_id"] != PROTOCOL_ID
        or value["role"] != "DEVELOPMENT_ONLY_DESCRIPTOR_ASSETS"
        or value["auto_deploy"] is not False
        or value["accuracy_claim_permitted"] is not False
        or value["a_r4_render_provenance"] != A_R4_RENDER_PROVENANCE
        or value["a_r4_content_audit_internal_identity"]
        != A_R4_CONTENT_AUDIT_INTERNAL_IDENTITY
        or value["a_r4_safe_evidence_binding"] != A_R4_SAFE_EVIDENCE_BINDING
        or value["objects"] != list(OBJECT_IDS)
        or value["runtime_boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("A-R5-P2 protocol identity/boundary changed")
    if value["predecessor"] != {
        "a_r4_read_only": True,
        "instance_proposal_v1r5_read_only": True,
        "instance_proposal_v1r5_protocol_id": A_R5_PROTOCOL_ID,
        "base_commit": "9e0ac58c1de04e18fa4f56b934ba1878964e99d7",
    }:
        raise ContractError("A-R5-P2 predecessor boundary changed")
    if value["descriptor_contract"] != {
        "object_count": 5,
        "views_per_object": VIEW_COUNT,
        "shape_per_object": [VIEW_COUNT, FEATURE_DIMENSION],
        "dtype": "float32",
        "finite_required": True,
        "model_name": "dinov2_vitl14",
        "token_name": "x_norm_clstoken",
        "proposal_image_size": 224,
        "descriptor_width_size": 640,
        "feature_chunk_size": 16,
        "rgba_preprocess": "official_cnos_pil_bbox_cropresizepad224_rgb_normalize",
    }:
        raise ContractError("A-R5-P2 descriptor semantics changed")
    if value["model_provenance"] != {
        "cnos_repository": PINNED_CNOS_REPOSITORY,
        "cnos_commit": PINNED_CNOS_COMMIT,
        "cnos_tree": PINNED_CNOS_TREE,
        "dinov2_repository": PINNED_DINOV2_REPOSITORY,
        "dinov2_commit": PINNED_DINOV2_COMMIT,
        "dinov2_tree": PINNED_DINOV2_TREE,
        "dinov2_vitl14_url": PINNED_DINOV2_VITL14_URL,
        "dinov2_vitl14_bytes": PINNED_DINOV2_VITL14_BYTES,
        "dinov2_vitl14_sha256": PINNED_DINOV2_VITL14_SHA256,
        "weight_load": "torch_load_weights_only_then_strict_true",
    }:
        raise ContractError("A-R5-P2 official model provenance changed")
    if value["execution_policy"] != {
        "create_only": True,
        "per_object_request_specific_sidecar": True,
        "planned_stop_permitted": True,
        "resume_requires_matching_planned_stop_and_sidecars": True,
        "final_receipt_create_only": True,
        "independent_disk_validation_required": True,
        "server_connection_authorized_now": False,
        "model_run_authorized_now": False,
    }:
        raise ContractError("A-R5-P2 execution policy changed")
    wrappers = value["wrapper_files"]
    if not isinstance(wrappers, list) or len(wrappers) != len(_WRAPPER_PATHS):
        raise ContractError("A-R5-P2 wrapper inventory changed")
    observed_paths: set[str] = set()
    for index, raw in enumerate(wrappers):
        item = _exact(raw, {"relative_path", "bytes", "sha256"}, f"wrapper[{index}]")
        relative = item["relative_path"]
        if not isinstance(relative, str) or relative in observed_paths:
            raise ContractError("A-R5-P2 wrapper path is duplicate/invalid")
        observed_paths.add(relative)
        _positive_int(item["bytes"], f"wrapper[{index}].bytes")
        _sha(item["sha256"], f"wrapper[{index}].sha256")
        if repository_root is not None:
            path = repository_root.resolve() / Path(*PurePosixPath(relative).parts)
            if _disk_lock(path) != {
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }:
                raise ContractError(f"A-R5-P2 wrapper bytes changed: {relative}")
    if observed_paths != _WRAPPER_PATHS:
        raise ContractError("A-R5-P2 wrapper inventory is not exact")
    _self_lock(value, "protocol_lock_sha256", "A-R5-P2 protocol")
    return value


def _source_snapshot(
    raw: Any,
    *,
    data_root: Path,
    kind: str,
    expected_repository: str,
    expected_commit: str,
    expected_tree: str,
    expected_archive: Mapping[str, Any] | None,
    probes: RuntimeProbes,
) -> tuple[dict[str, Any], Path, dict[str, dict[str, Any]]]:
    value = _exact(
        raw,
        {
            "repository",
            "commit",
            "tree",
            "checkout_relative_path",
            "all_clean",
            "git_status_porcelain_v1_untracked_files_all",
            "source_archive",
            "archive_route",
            "execution_files",
            "execution_inventory_sha256",
        },
        f"{kind} source lock",
    )
    if (
        value["repository"] != expected_repository
        or value["commit"] != expected_commit
        or value["tree"] != expected_tree
        or value["all_clean"] is not True
        or value["git_status_porcelain_v1_untracked_files_all"] != ""
    ):
        raise ContractError(f"{kind} source identity/clean state changed")
    _git_oid(value["commit"], f"{kind}.commit")
    _git_oid(value["tree"], f"{kind}.tree")
    relative = value["checkout_relative_path"]
    if (
        not isinstance(relative, str)
        or "\\" in relative
        or PurePosixPath(relative).parts[:1] != ("sources",)
    ):
        raise ContractError(f"{kind} checkout path changed")
    checkout = (data_root.resolve() / Path(*PurePosixPath(relative).parts)).resolve()
    archive, archive_path = _resolve_asset(
        value["source_archive"],
        data_root=data_root,
        role=f"{kind.lower()}_source_archive",
        root_name="sources",
        label=f"{kind}.source_archive",
        probes=probes,
    )
    if expected_archive is not None and {
        "bytes": archive["bytes"],
        "sha256": archive["sha256"],
    } != dict(expected_archive):
        raise ContractError(f"{kind} pinned archive identity changed")
    route = _exact(
        value["archive_route"], {"format", "prefix", "command"}, f"{kind}.archive_route"
    )
    prefix = f"{kind.lower()}-{expected_commit}/"
    command = [
        "git",
        "archive",
        "--format=tar.gz",
        f"--prefix={prefix}",
        expected_commit,
    ]
    if route != {"format": "tar.gz", "prefix": prefix, "command": command}:
        raise ContractError(f"{kind} deterministic archive route changed")
    # The frozen CNOS transport archive was produced on Git for Windows. Its
    # exported text payload uses CRLF and its gzip stream is zlib-version
    # dependent, so Linux `git archive --format=tar.gz` cannot reproduce the
    # byte stream even from the exact commit/tree. For a pinned archive the
    # immutable bytes above, clean commit/tree below, and complete execution
    # source inventory are independent provenance checks. Unpinned archives
    # (currently DINOv2) remain execution-host reproducible.
    if expected_archive is None:
        reconstructed = dict(probes.archive_lock(checkout, prefix, expected_commit))
        if reconstructed != {
            "bytes": archive["bytes"],
            "sha256": archive["sha256"],
        }:
            raise ContractError(f"{kind} source archive cannot be reconstructed")
    files = value["execution_files"]
    if not isinstance(files, list) or not files:
        raise ContractError(f"{kind} full execution source manifest is empty")
    file_map: dict[str, dict[str, Any]] = {}
    for index, raw_file in enumerate(files):
        item = _exact(
            raw_file,
            {"relative_path", "bytes", "sha256"},
            f"{kind}.execution_files[{index}]",
        )
        path_text = item["relative_path"]
        if (
            not isinstance(path_text, str)
            or "\\" in path_text
            or not path_text.endswith(".py")
            or path_text in file_map
        ):
            raise ContractError(f"{kind} execution source path changed")
        path = checkout / Path(*PurePosixPath(path_text).parts)
        observed = dict(probes.file_lock(path))
        expected = {"bytes": item["bytes"], "sha256": item["sha256"]}
        if observed != expected:
            raise ContractError(f"{kind} execution source bytes changed: {path_text}")
        _positive_int(item["bytes"], f"{kind}.execution_files[{index}].bytes")
        _sha(item["sha256"], f"{kind}.execution_files[{index}].sha256")
        file_map[path_text] = expected
    if value["execution_inventory_sha256"] != canonical_sha256(files):
        raise ContractError(f"{kind} execution source manifest lock changed")
    snapshot = dict(probes.git_snapshot(checkout, kind))
    expected_snapshot = {
        "absolute_path": str(checkout),
        "repository": expected_repository,
        "commit": expected_commit,
        "tree": expected_tree,
        "all_clean": True,
        "git_status_porcelain_v1_untracked_files_all": "",
        "execution_files": files,
        "execution_inventory_sha256": value["execution_inventory_sha256"],
    }
    if snapshot != expected_snapshot:
        raise ContractError(f"{kind} checkout changed or contains a source shadow")
    if not archive_path.is_file():
        raise ContractError(f"{kind} source archive disappeared")
    return value, checkout, file_map


def validate_template_manifest(
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    probes: RuntimeProbes = DEFAULT_PROBES,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    value = _exact(
        manifest,
        {
            "schema_version",
            "a_r4_render_provenance",
            "frame_size",
            "object_count",
            "views_per_object",
            "objects",
            "template_manifest_lock_sha256",
        },
        "A-R4 RGBA template manifest",
    )
    if (
        value["schema_version"] != TEMPLATE_MANIFEST_SCHEMA
        or value["a_r4_render_provenance"] != A_R4_RENDER_PROVENANCE
        or value["frame_size"] != {"height": 480, "width": 640, "mode": "RGBA"}
        or value["object_count"] != 5
        or value["views_per_object"] != VIEW_COUNT
    ):
        raise ContractError("A-R4 RGBA template identity changed")
    objects = value["objects"]
    if not isinstance(objects, list) or len(objects) != len(OBJECT_IDS):
        raise ContractError("A-R4 template object coverage changed")
    result: dict[int, dict[str, Any]] = {}
    for object_index, raw in enumerate(objects):
        item = _exact(
            raw,
            {"object_id", "cad_sha256", "views"},
            f"template.objects[{object_index}]",
        )
        object_id = item["object_id"]
        if object_id != OBJECT_IDS[object_index]:
            raise ContractError("A-R4 template object order changed")
        _sha(item["cad_sha256"], f"template object {object_id} CAD")
        views = item["views"]
        if not isinstance(views, list) or len(views) != VIEW_COUNT:
            raise ContractError("Every A-R4 object requires exactly 42 views")
        hashes: set[str] = set()
        for view_index, raw_view in enumerate(views):
            view = _exact(
                raw_view,
                {"view_index", "rgba", "alpha_pixels", "rgb_nonzero_pixels"},
                f"template object {object_id} view {view_index}",
            )
            if view["view_index"] != view_index:
                raise ContractError("A-R4 template view order changed")
            asset, path = _resolve_asset(
                view["rgba"],
                data_root=data_root,
                role="cad_template_rgba",
                root_name="assets",
                label=f"template object {object_id} view {view_index}",
                probes=probes,
            )
            alpha_expected = _positive_int(view["alpha_pixels"], "alpha_pixels")
            rgb_expected = _positive_int(
                view["rgb_nonzero_pixels"], "rgb_nonzero_pixels"
            )
            try:
                with Image.open(path) as image:
                    image.load()
                    if (
                        image.format != "PNG"
                        or image.mode != "RGBA"
                        or image.size != (640, 480)
                    ):
                        raise ContractError("Template must decode as 640x480 RGBA PNG")
                    pixels = np.asarray(image)
                    pil_bbox = image.getbbox()
                    alpha_bbox = image.getchannel("A").getbbox()
            except (OSError, ValueError) as exc:
                raise ContractError("A-R4 RGBA template cannot be decoded") from exc
            alpha = pixels[:, :, 3] > 0
            rgb = np.any(pixels[:, :, :3] > 0, axis=2)
            if (
                int(alpha.sum()) != alpha_expected
                or int(np.logical_and(alpha, rgb).sum()) != rgb_expected
                or alpha_expected >= TEMPLATE_HEIGHT * TEMPLATE_WIDTH
                or np.any(pixels[:, :, :3][~alpha] != 0)
                or pil_bbox is None
                or pil_bbox != alpha_bbox
            ):
                raise ContractError("A-R4 RGBA content/bbox contract changed")
            if asset["sha256"] in hashes:
                raise ContractError("A-R4 per-object template content is not unique")
            hashes.add(asset["sha256"])
        result[object_id] = item
    _self_lock(value, "template_manifest_lock_sha256", "A-R4 template manifest")
    return value, result


def validate_template_import_receipt(
    receipt: Mapping[str, Any],
    *,
    data_root: Path,
    template_manifest: Mapping[str, Any],
    template_manifest_asset: Mapping[str, Any],
    probes: RuntimeProbes = DEFAULT_PROBES,
) -> dict[str, Any]:
    value = _exact(
        receipt,
        {
            "schema_version",
            "status",
            "a_r4_render_provenance",
            "safe_evidence",
            "source_receipts",
            "canonical_cad_inventory",
            "source_png_count",
            "source_png_inventory",
            "template_manifest",
            "a_r4_root_modified",
            "boundary",
            "template_import_receipt_lock_sha256",
        },
        "A-R4 template import receipt",
    )
    if (
        value["schema_version"] != TEMPLATE_IMPORT_RECEIPT_SCHEMA
        or value["status"] != "PASS_A_R4_EXACT_5_OBJECTS_X_42_RGBA_IMPORTED_CREATE_ONLY"
        or value["a_r4_render_provenance"] != A_R4_RENDER_PROVENANCE
        or value["a_r4_root_modified"] is not False
        or value["boundary"]
        != {
            "label_access_count": 0,
            "depth_access_count": 0,
            "evaluator_access_count": 0,
            "sealed_access_count": 0,
            "foundationpose_run_count": 0,
        }
    ):
        raise ContractError("A-R4 template import provenance/boundary changed")
    safe_assets = _exact(
        value["safe_evidence"],
        {"safe_archive", "safe_member_inventory", "deployment_inventory"},
        "A-R4 safe evidence assets",
    )
    safe_archive_asset, safe_archive_path = _resolve_asset(
        safe_assets["safe_archive"],
        data_root=data_root,
        role="a_r4_safe_evidence_archive",
        root_name="evidence",
        label="A-R4 safe archive",
        probes=probes,
    )
    safe_member_asset, safe_member_path = _resolve_asset(
        safe_assets["safe_member_inventory"],
        data_root=data_root,
        role="a_r4_safe_member_inventory",
        root_name="evidence",
        label="A-R4 safe member inventory",
        probes=probes,
    )
    deployment_asset, deployment_path = _resolve_asset(
        safe_assets["deployment_inventory"],
        data_root=data_root,
        role="a_r4_deployment_inventory",
        root_name="evidence",
        label="A-R4 deployment inventory",
        probes=probes,
    )
    safe_validation = validate_a_r4_safe_evidence(
        safe_archive=safe_archive_path,
        member_inventory=safe_member_path,
        deployment_inventory=deployment_path,
        probes=probes,
    )
    if (
        {"bytes": safe_archive_asset["bytes"], "sha256": safe_archive_asset["sha256"]}
        != safe_validation["safe_archive"]
        or {
            "bytes": safe_member_asset["bytes"],
            "sha256": safe_member_asset["sha256"],
        }
        != safe_validation["member_inventory"]
        or {
            "bytes": deployment_asset["bytes"],
            "sha256": deployment_asset["sha256"],
        }
        != safe_validation["deployment_inventory"]
        or value["canonical_cad_inventory"] != safe_validation["canonical_cads"]
    ):
        raise ContractError("A-R4 safe evidence asset closure changed")
    for object_item, canonical in zip(
        template_manifest["objects"], safe_validation["canonical_cads"], strict=True
    ):
        if (
            object_item["object_id"] != canonical["object_id"]
            or object_item["cad_sha256"] != canonical["sha256"]
        ):
            raise ContractError("A-R4 template CAD is not safe-archive-bound")
    source_receipts = _exact(
        value["source_receipts"],
        {"authorization", "attempt", "content_audit"},
        "A-R4 source receipts",
    )
    expected_hashes = dict(probes.source_receipt_hashes())
    if set(expected_hashes) != {"authorization", "attempt", "content_audit"}:
        raise ContractError("A-R4 source receipt identity probe changed")
    for name, expected_hash in expected_hashes.items():
        _sha(expected_hash, f"A-R4 {name} expected receipt SHA")
    receipt_archive_members = {
        "authorization": "receipts/one-time-render-authorization-v1r4-attempt3.json",
        "attempt": ("attempts/poseloop_ga_cnos_v1r4_attempt_003/attempt-receipt.json"),
        "content_audit": "receipts/postrender-content-audit-v1r4-attempt3.json",
    }
    for name, expected_hash in expected_hashes.items():
        asset, _ = _resolve_asset(
            source_receipts[name],
            data_root=data_root,
            role=f"a_r4_{name}_receipt",
            root_name="evidence",
            label=f"A-R4 {name} receipt",
            probes=probes,
        )
        if asset["sha256"] != expected_hash or safe_validation["payload_members"].get(
            receipt_archive_members[name]
        ) != {"bytes": asset["bytes"], "sha256": asset["sha256"]}:
            raise ContractError(f"A-R4 {name} receipt identity changed")
    manifest_asset, manifest_path = _resolve_asset(
        value["template_manifest"],
        data_root=data_root,
        role="rgba_template_manifest",
        root_name="contracts",
        label="template import manifest asset",
        probes=probes,
    )
    if manifest_asset != dict(template_manifest_asset) or read_json(
        manifest_path, "imported template manifest"
    ) != dict(template_manifest):
        raise ContractError("A-R4 import receipt points to a different manifest")
    inventory = value["source_png_inventory"]
    if (
        value["source_png_count"] != len(OBJECT_IDS) * VIEW_COUNT
        or not isinstance(inventory, list)
        or len(inventory) != len(OBJECT_IDS) * VIEW_COUNT
    ):
        raise ContractError("A-R4 source PNG import inventory is incomplete")
    cursor = 0
    for object_item in template_manifest["objects"]:
        for view in object_item["views"]:
            row = _exact(
                inventory[cursor],
                {"object_id", "view_index", "source", "destination"},
                f"source_png_inventory[{cursor}]",
            )
            source = _exact(
                row["source"],
                {"archive_member_path", "bytes", "sha256"},
                f"source_png_inventory[{cursor}].source",
            )
            _positive_int(source["bytes"], "source PNG bytes")
            _sha(source["sha256"], "source PNG SHA")
            expected_archive_member = (
                "attempts/poseloop_ga_cnos_v1r4_attempt_003/objects/"
                f"obj_{object_item['object_id']:06d}/{view['view_index']:06d}.png"
            )
            if (
                row["object_id"] != object_item["object_id"]
                or row["view_index"] != view["view_index"]
                or source["archive_member_path"] != expected_archive_member
                or row["destination"] != view["rgba"]
                or {
                    "bytes": source["bytes"],
                    "sha256": source["sha256"],
                }
                != {
                    "bytes": view["rgba"]["bytes"],
                    "sha256": view["rgba"]["sha256"],
                }
                or safe_validation["payload_members"].get(source["archive_member_path"])
                != {"bytes": source["bytes"], "sha256": source["sha256"]}
            ):
                raise ContractError("A-R4 source/destination PNG inventory changed")
            cursor += 1
    _self_lock(
        value,
        "template_import_receipt_lock_sha256",
        "A-R4 template import receipt",
    )
    return value


def _implementation_snapshot(
    raw: Any,
    *,
    data_root: Path,
    protocol: Mapping[str, Any],
    probes: RuntimeProbes,
) -> dict[str, Any]:
    value = _exact(
        raw,
        {
            "checkout_absolute_path",
            "commit",
            "tree",
            "all_clean",
            "git_status_porcelain_v1_untracked_files_all",
            "source_archive",
            "archive_route",
            "execution_files",
            "execution_inventory_sha256",
        },
        "implementation source lock",
    )
    checkout = repository_root_from_package().resolve()
    if value["checkout_absolute_path"] != str(checkout):
        raise ContractError("A-R5-P2 is not executing from the bound checkout")
    _git_oid(value["commit"], "implementation.commit")
    _git_oid(value["tree"], "implementation.tree")
    if (
        value["all_clean"] is not True
        or value["git_status_porcelain_v1_untracked_files_all"] != ""
    ):
        raise ContractError("A-R5-P2 implementation checkout is not all-clean")
    archive, _ = _resolve_asset(
        value["source_archive"],
        data_root=data_root,
        role="implementation_source_archive",
        root_name="sources",
        label="implementation.source_archive",
        probes=probes,
    )
    prefix = f"poseloop-{value['commit']}/"
    route = {
        "format": "tar.gz",
        "prefix": prefix,
        "command": [
            "git",
            "archive",
            "--format=tar.gz",
            f"--prefix={prefix}",
            value["commit"],
        ],
    }
    if value["archive_route"] != route:
        raise ContractError("A-R5-P2 implementation archive route changed")
    if dict(probes.archive_lock(checkout, prefix, value["commit"])) != {
        "bytes": archive["bytes"],
        "sha256": archive["sha256"],
    }:
        raise ContractError("A-R5-P2 implementation archive is not reproducible")
    expected_files = protocol["wrapper_files"]
    if value["execution_files"] != expected_files or value[
        "execution_inventory_sha256"
    ] != canonical_sha256(expected_files):
        raise ContractError("A-R5-P2 implementation execution inventory changed")
    snapshot = dict(probes.git_snapshot(checkout, "IMPLEMENTATION"))
    expected_snapshot = {
        "absolute_path": str(checkout),
        "repository": snapshot.get("repository", ""),
        "commit": value["commit"],
        "tree": value["tree"],
        "all_clean": True,
        "git_status_porcelain_v1_untracked_files_all": "",
        "execution_files": expected_files,
        "execution_inventory_sha256": value["execution_inventory_sha256"],
    }
    if snapshot != expected_snapshot:
        raise ContractError("A-R5-P2 implementation checkout changed")
    return value


def validate_runtime_request(
    request: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    data_root: Path,
    probes: RuntimeProbes = DEFAULT_PROBES,
) -> dict[str, Any]:
    protocol_value = validate_protocol(
        protocol, repository_root=repository_root_from_package()
    )
    value = _exact(
        request,
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "implementation",
            "template_manifest",
            "template_import_receipt",
            "source",
            "model",
            "catalog",
            "runtime",
            "boundary",
            "runtime_request_lock_sha256",
        },
        "A-R5-P2 runtime request",
    )
    if (
        value["schema_version"] != REQUEST_SCHEMA
        or value["protocol_id"] != PROTOCOL_ID
        or value["protocol_lock_sha256"] != protocol_value["protocol_lock_sha256"]
        or value["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("A-R5-P2 runtime request identity/boundary changed")
    _implementation_snapshot(
        value["implementation"],
        data_root=data_root,
        protocol=protocol_value,
        probes=probes,
    )
    template_asset, template_path = _resolve_asset(
        value["template_manifest"],
        data_root=data_root,
        role="rgba_template_manifest",
        root_name="contracts",
        label="template_manifest",
        probes=probes,
    )
    template, template_objects = validate_template_manifest(
        read_json(template_path, "A-R4 template manifest"),
        data_root=data_root,
        probes=probes,
    )
    import_asset, import_path = _resolve_asset(
        value["template_import_receipt"],
        data_root=data_root,
        role="a_r4_template_import_receipt",
        root_name="receipts",
        label="template_import_receipt",
        probes=probes,
    )
    template_import_receipt = validate_template_import_receipt(
        read_json(import_path, "A-R4 template import receipt"),
        data_root=data_root,
        template_manifest=template,
        template_manifest_asset=template_asset,
        probes=probes,
    )
    source = _exact(value["source"], {"cnos", "dinov2"}, "request.source")
    cnos, cnos_checkout, cnos_files = _source_snapshot(
        source["cnos"],
        data_root=data_root,
        kind="CNOS",
        expected_repository=PINNED_CNOS_REPOSITORY,
        expected_commit=PINNED_CNOS_COMMIT,
        expected_tree=PINNED_CNOS_TREE,
        expected_archive={
            "bytes": PINNED_CNOS_ARCHIVE_BYTES,
            "sha256": PINNED_CNOS_ARCHIVE_SHA256,
        },
        probes=probes,
    )
    dinov2, dinov2_checkout, dinov2_files = _source_snapshot(
        source["dinov2"],
        data_root=data_root,
        kind="DINOV2",
        expected_repository=PINNED_DINOV2_REPOSITORY,
        expected_commit=PINNED_DINOV2_COMMIT,
        expected_tree=PINNED_DINOV2_TREE,
        expected_archive=None,
        probes=probes,
    )
    model = _exact(
        value["model"],
        {"dinov2_vitl14_checkpoint", "official_url", "identity_receipt_sha256"},
        "request.model",
    )
    weight, weight_path = _resolve_asset(
        model["dinov2_vitl14_checkpoint"],
        data_root=data_root,
        role="dinov2_vitl14_checkpoint",
        root_name="models",
        label="DINOv2 ViT-L/14 checkpoint",
        probes=probes,
    )
    if (
        weight["bytes"] != PINNED_DINOV2_VITL14_BYTES
        or weight["sha256"] != PINNED_DINOV2_VITL14_SHA256
        or model["official_url"] != PINNED_DINOV2_VITL14_URL
        or model["identity_receipt_sha256"]
        != "7f417bba08ea1dd8eb5d1481a06d76a45d32410bc7fcf495591dc196c8ef338e"
    ):
        raise ContractError("A-R5-P2 official DINOv2 weight identity changed")
    runtime = _exact(
        value["runtime"],
        {
            "device",
            "model_name",
            "token_name",
            "proposal_image_size",
            "descriptor_width_size",
            "feature_chunk_size",
            "strict_weight_load",
        },
        "request.runtime",
    )
    if runtime != {
        "device": "cuda:0",
        "model_name": "dinov2_vitl14",
        "token_name": "x_norm_clstoken",
        "proposal_image_size": 224,
        "descriptor_width_size": 640,
        "feature_chunk_size": 16,
        "strict_weight_load": True,
    }:
        raise ContractError("A-R5-P2 runtime semantics changed")
    catalog = value["catalog"]
    if not isinstance(catalog, list) or len(catalog) != len(OBJECT_IDS):
        raise ContractError("A-R5-P2 requires exactly five catalog objects")
    catalog_result: list[dict[str, Any]] = []
    for index, raw in enumerate(catalog):
        item = _exact(
            raw,
            {"object_id", "cad", "descriptor_relative_path", "sidecar_relative_path"},
            f"catalog[{index}]",
        )
        object_id = item["object_id"]
        if object_id != OBJECT_IDS[index]:
            raise ContractError("A-R5-P2 catalog object order changed")
        cad, _ = _resolve_asset(
            item["cad"],
            data_root=data_root,
            role="target_cad",
            root_name="assets",
            label=f"catalog[{index}].cad",
            probes=probes,
        )
        if cad["sha256"] != template_objects[object_id]["cad_sha256"]:
            raise ContractError("A-R5-P2 CAD/template identity mismatch")
        expected_descriptor = f"assets/descriptors/obj_{object_id:06d}.pth"
        expected_sidecar = f"receipts/descriptor-assets-v1/obj_{object_id:06d}.json"
        if (
            item["descriptor_relative_path"] != expected_descriptor
            or item["sidecar_relative_path"] != expected_sidecar
        ):
            raise ContractError("A-R5-P2 output layout changed")
        catalog_result.append(item)
    _self_lock(value, "runtime_request_lock_sha256", "A-R5-P2 runtime request")
    return {
        "request": value,
        "protocol": protocol_value,
        "template_manifest": template,
        "template_manifest_asset": template_asset,
        "template_import_receipt": template_import_receipt,
        "template_import_receipt_asset": import_asset,
        "template_objects": template_objects,
        "cnos": cnos,
        "cnos_checkout": cnos_checkout,
        "cnos_files": cnos_files,
        "dinov2": dinov2,
        "dinov2_checkout": dinov2_checkout,
        "dinov2_files": dinov2_files,
        "weight_path": weight_path,
        "catalog": catalog_result,
    }


def _source_entry_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    checkout_relative_path: str,
    source_archive: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    commit = str(snapshot["commit"])
    prefix = f"{kind.lower()}-{commit}/"
    return {
        "repository": snapshot["repository"],
        "commit": commit,
        "tree": snapshot["tree"],
        "checkout_relative_path": checkout_relative_path,
        "all_clean": snapshot["all_clean"],
        "git_status_porcelain_v1_untracked_files_all": snapshot[
            "git_status_porcelain_v1_untracked_files_all"
        ],
        "source_archive": dict(source_archive),
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
        "execution_files": snapshot["execution_files"],
        "execution_inventory_sha256": snapshot["execution_inventory_sha256"],
    }


def build_runtime_request(
    *,
    protocol: Mapping[str, Any],
    data_root: Path,
    implementation_archive: Path,
    template_manifest: Path,
    template_import_receipt: Path,
    cnos_checkout: Path,
    cnos_archive: Path,
    dinov2_checkout: Path,
    dinov2_archive: Path,
    dinov2_weights: Path,
    cad_paths: Mapping[int, Path],
    probes: RuntimeProbes = DEFAULT_PROBES,
) -> dict[str, Any]:
    """Build and immediately revalidate a hash-locked A-R5-P2 request."""

    protocol_value = validate_protocol(
        protocol, repository_root=repository_root_from_package()
    )
    root = data_root.resolve()
    implementation_root = repository_root_from_package()
    implementation_snapshot = dict(
        probes.git_snapshot(implementation_root, "IMPLEMENTATION")
    )
    implementation_asset = _asset_from_path(
        implementation_archive,
        data_root=root,
        role="implementation_source_archive",
        probes=probes,
    )
    implementation_commit = str(implementation_snapshot["commit"])
    implementation_prefix = f"poseloop-{implementation_commit}/"
    template_asset = _asset_from_path(
        template_manifest,
        data_root=root,
        role="rgba_template_manifest",
        probes=probes,
    )
    import_asset = _asset_from_path(
        template_import_receipt,
        data_root=root,
        role="a_r4_template_import_receipt",
        probes=probes,
    )
    cnos_snapshot = dict(probes.git_snapshot(cnos_checkout, "CNOS"))
    dinov2_snapshot = dict(probes.git_snapshot(dinov2_checkout, "DINOV2"))
    cnos_archive_asset = _asset_from_path(
        cnos_archive, data_root=root, role="cnos_source_archive", probes=probes
    )
    dinov2_archive_asset = _asset_from_path(
        dinov2_archive,
        data_root=root,
        role="dinov2_source_archive",
        probes=probes,
    )
    weight_asset = _asset_from_path(
        dinov2_weights,
        data_root=root,
        role="dinov2_vitl14_checkpoint",
        probes=probes,
    )
    catalog = []
    for object_id in OBJECT_IDS:
        path = cad_paths.get(object_id)
        if path is None:
            raise ContractError("A-R5-P2 CAD mapping is incomplete")
        catalog.append(
            {
                "object_id": object_id,
                "cad": _asset_from_path(
                    path, data_root=root, role="target_cad", probes=probes
                ),
                "descriptor_relative_path": f"assets/descriptors/obj_{object_id:06d}.pth",
                "sidecar_relative_path": (
                    f"receipts/descriptor-assets-v1/obj_{object_id:06d}.json"
                ),
            }
        )
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol_value["protocol_lock_sha256"],
        "implementation": {
            "checkout_absolute_path": str(implementation_root.resolve()),
            "commit": implementation_snapshot["commit"],
            "tree": implementation_snapshot["tree"],
            "all_clean": implementation_snapshot["all_clean"],
            "git_status_porcelain_v1_untracked_files_all": implementation_snapshot[
                "git_status_porcelain_v1_untracked_files_all"
            ],
            "source_archive": implementation_asset,
            "archive_route": {
                "format": "tar.gz",
                "prefix": implementation_prefix,
                "command": [
                    "git",
                    "archive",
                    "--format=tar.gz",
                    f"--prefix={implementation_prefix}",
                    implementation_commit,
                ],
            },
            "execution_files": protocol_value["wrapper_files"],
            "execution_inventory_sha256": canonical_sha256(
                protocol_value["wrapper_files"]
            ),
        },
        "template_manifest": template_asset,
        "template_import_receipt": import_asset,
        "source": {
            "cnos": _source_entry_from_snapshot(
                cnos_snapshot,
                checkout_relative_path=cnos_checkout.resolve()
                .relative_to(root)
                .as_posix(),
                source_archive=cnos_archive_asset,
                kind="CNOS",
            ),
            "dinov2": _source_entry_from_snapshot(
                dinov2_snapshot,
                checkout_relative_path=dinov2_checkout.resolve()
                .relative_to(root)
                .as_posix(),
                source_archive=dinov2_archive_asset,
                kind="DINOV2",
            ),
        },
        "model": {
            "dinov2_vitl14_checkpoint": weight_asset,
            "official_url": PINNED_DINOV2_VITL14_URL,
            "identity_receipt_sha256": (
                "7f417bba08ea1dd8eb5d1481a06d76a45d32410bc7fcf495591dc196c8ef338e"
            ),
        },
        "catalog": catalog,
        "runtime": {
            "device": "cuda:0",
            "model_name": "dinov2_vitl14",
            "token_name": "x_norm_clstoken",
            "proposal_image_size": 224,
            "descriptor_width_size": 640,
            "feature_chunk_size": 16,
            "strict_weight_load": True,
        },
        "boundary": dict(BOUNDARY_ZERO),
        "runtime_request_lock_sha256": "pending",
    }
    request["runtime_request_lock_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in request.items()
            if key != "runtime_request_lock_sha256"
        }
    )
    validate_runtime_request(request, protocol_value, data_root=root, probes=probes)
    return request


def audit_module_origins(
    module_registry: Mapping[str, ModuleType],
    *,
    cnos_checkout: Path,
    cnos_files: Mapping[str, Mapping[str, Any]],
    dinov2_checkout: Path,
    dinov2_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind every loaded CNOS/DINO module to the full request source manifests."""

    result: dict[str, dict[str, Any]] = {}

    def inspect(
        name: str, checkout: Path, files: Mapping[str, Mapping[str, Any]]
    ) -> None:
        module = module_registry[name]
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw:
            spec = getattr(module, "__spec__", None)
            locations = getattr(spec, "submodule_search_locations", None)
            if locations is None:
                raise ContractError(f"Runtime module lacks a source origin: {name}")
            observed_locations = list(locations)
            expected_relative = PurePosixPath(*name.split(".")).as_posix()
            if not observed_locations:
                raise ContractError(f"Runtime namespace has no source root: {name}")
            for location in observed_locations:
                if not isinstance(location, str) or not location:
                    raise ContractError(
                        f"Runtime namespace has an invalid source root: {name}"
                    )
                namespace_root = Path(location).resolve()
                try:
                    relative = namespace_root.relative_to(checkout.resolve()).as_posix()
                except ValueError as exc:
                    raise ContractError(
                        f"Runtime namespace loaded outside bound checkout: {name}"
                    ) from exc
                if relative != expected_relative or not namespace_root.is_dir():
                    raise ContractError(
                        f"Runtime namespace source root changed: {name}"
                    )
            # Namespace packages contain no executable bytes of their own.
            # Any imported child with a source file is still audited below.
            return
        path = Path(raw).resolve()
        try:
            relative = path.relative_to(checkout.resolve()).as_posix()
        except ValueError as exc:
            raise ContractError(
                f"Runtime module loaded outside bound checkout: {name}"
            ) from exc
        expected = files.get(relative)
        if expected is None or _disk_lock(path) != dict(expected):
            raise ContractError(f"Runtime module bytes/origin changed: {name}")
        result[name] = {"relative_path": relative, **dict(expected)}

    cnos_names = sorted(
        name for name in module_registry if name == "src" or name.startswith("src.")
    )
    dino_names = sorted(
        name
        for name in module_registry
        if name == "dinov2" or name.startswith("dinov2.")
    )
    for name in cnos_names:
        inspect(name, cnos_checkout, cnos_files)
    for name in dino_names:
        inspect(name, dinov2_checkout, dinov2_files)
    required = {
        "src.model.dinov2",
        "src.model.utils",
        "src.utils.bbox_utils",
        "dinov2.models.vision_transformer",
    }
    if not required.issubset(result):
        raise ContractError("Official CNOS/DINO runtime module audit is incomplete")
    return dict(sorted(result.items()))


def validate_recorded_module_audit(
    audit: Any,
    *,
    cnos_files: Mapping[str, Mapping[str, Any]],
    dinov2_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(audit, Mapping) or not audit:
        raise ContractError("Recorded module-origin audit is missing")
    result: dict[str, dict[str, Any]] = {}
    for name, raw in audit.items():
        if not isinstance(name, str):
            raise ContractError("Recorded module name is invalid")
        item = _exact(raw, {"relative_path", "bytes", "sha256"}, f"module audit {name}")
        source = (
            cnos_files if name == "src" or name.startswith("src.") else dinov2_files
        )
        expected = source.get(item["relative_path"])
        if expected is None or dict(expected) != {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }:
            raise ContractError(f"Recorded module audit is not source-bound: {name}")
        result[name] = item
    required = {
        "src.model.dinov2",
        "src.model.utils",
        "src.utils.bbox_utils",
        "dinov2.models.vision_transformer",
    }
    if not required.issubset(result):
        raise ContractError("Recorded module-origin audit is incomplete")
    return result


def audit_descriptor_tensor(
    path: Path, *, torch_module: Any | None = None
) -> dict[str, Any]:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise ContractError(
                "Torch is required to audit descriptor tensors"
            ) from exc
    try:
        tensor = torch_module.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError(f"Cannot load descriptor tensor: {path}") from exc
    if (
        not torch_module.is_tensor(tensor)
        or list(tensor.shape) != [VIEW_COUNT, FEATURE_DIMENSION]
        or tensor.dtype != torch_module.float32
        or not bool(torch_module.isfinite(tensor).all().item())
    ):
        raise ContractError("Descriptor must be finite float32 [42,1024]")
    return {
        **_disk_lock(path),
        "shape": [VIEW_COUNT, FEATURE_DIMENSION],
        "dtype": "float32",
        "finite": True,
    }


__all__ = [
    "A_R4_CONTENT_AUDIT_INTERNAL_IDENTITY",
    "A_R4_RENDER_PROVENANCE",
    "A_R4_SAFE_EVIDENCE_BINDING",
    "A_R5_PROTOCOL_ID",
    "BOUNDARY_ZERO",
    "DEFAULT_PROBES",
    "RuntimeProbes",
    "audit_descriptor_tensor",
    "audit_module_origins",
    "build_runtime_request",
    "canonical_json_bytes",
    "canonical_sha256",
    "create_only_json",
    "read_json",
    "repository_root_from_package",
    "sha256_file",
    "validate_a_r4_safe_evidence",
    "validate_protocol",
    "validate_recorded_module_audit",
    "validate_runtime_request",
    "validate_template_manifest",
    "validate_template_import_receipt",
]
