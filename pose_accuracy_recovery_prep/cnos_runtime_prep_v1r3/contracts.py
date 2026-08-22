"""Fail-closed contracts for the A-R3 pinned-CNOS renderer wrapper.

The wrapper never edits the CNOS checkout.  It binds both the reproducible
``git archive`` member bytes (CRLF in the pinned repository) and the clean
checkout bytes (LF after the repository's checkout attributes), then imports
only the latter.  The one semantic repair is separately frozen: the existing
path-only diameter helper receives the already hash-bound CAD path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.contracts import (
    git_identity,
    git_is_ancestor,
    git_remote_repository,
)
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
)

from . import GPU_OVERRIDE, OBJECT_IDS, REQUEST_SCHEMA, ROUTE_ID, ROUTE_SCHEMA

BASE_IMPLEMENTATION_COMMIT = "ff9bd13248d4d707c7dc9e41efa4773168db7ba3"
BASE_IMPLEMENTATION_TREE = "0351534d5317505e699f1a954d11523b5a9a72d9"
R2_ROUTE_ID = "poseloop.pose-accuracy-recovery.development.cnos-runtime-prep.v1r2"
R2_ROUTE_LOCK = "ed89bfbdaa5589e22576a94ad020c9ff9b82d4aa5c84273ea4488c45225cdad8"
BLOCKER_RECEIPT = {
    "relative_path": "evidence/predecessor/producer-blocker-closeout-v1r2.json",
    "bytes": 4034,
    "sha256": "b45fa38c1928c9d72717eff570c1bba3a03b4b6a5a065a0f54a478c7ad5b848e",
}
SAFE_ARCHIVE = {
    "relative_path": "evidence/predecessor/poseloop-a-r2-ff9bd132-safe-closeout.tar.gz",
    "bytes": 68089,
    "sha256": "464ea2e2f091167d812f8d5e91bfbd4baf473aeb4e15f3a02363a22c9e02f946",
}

CNOS_REPOSITORY = "https://github.com/nv-nguyen/cnos"
CNOS_COMMIT = "298d1f3366171464ca271659f0e2f7a6eb8e39b4"
CNOS_TREE = "595ba390ad1fdcd2141c8004e505b5da1eb403c9"
CNOS_ARCHIVE = {
    "relative_path": (
        "sources/cnos-298d1f3366171464ca271659f0e2f7a6eb8e39b4-source.tar.gz"
    ),
    "bytes": 17114735,
    "sha256": "07c52c95f31ddae7fce14f5741c2fe5d7eab88369905237b2a52a5f4a8ebd6e5",
}
CNOS_CHECKOUT = "sources/cnos"
POSES_RELATIVE_PATH = "src/poses/predefined_poses/obj_poses_level0.npy"
POSES_LOCK = {
    "relative_path": POSES_RELATIVE_PATH,
    "bytes": 5504,
    "sha256": "9316d96402b82d5057a19405cd6e56f92c9cc1cff0b03aa2a170f8cb83000c10",
}
SOURCE_INTERFACES = (
    {
        "relative_path": "src/poses/pyrender.py",
        "archive_member": {
            "bytes": 5044,
            "sha256": "646a7c32ca31a855520fe97387c0ee544ce305a98b8e4740b60d4f79613d46a1",
            "line_endings": "CRLF",
        },
        "checkout_file": {
            "bytes": 4912,
            "sha256": "8dbe52bbb7c6d2fc419786ce1d8602fca4fe82ce6d80af708ff85feff31ed215",
            "line_endings": "LF",
        },
    },
    {
        "relative_path": "src/utils/trimesh_utils.py",
        "archive_member": {
            "bytes": 2070,
            "sha256": "adf533469f9382ff984994d43309c794bb2f2eff3c01885a45c99a0175f48b86",
            "line_endings": "CRLF",
        },
        "checkout_file": {
            "bytes": 1983,
            "sha256": "d2111f22b1266466e67b5917996b217d7a0923790a95a9ac83283841f1d07a6d",
            "line_endings": "LF",
        },
    },
)

CAD_LOCKS = (
    {
        "object_id": 1,
        "relative_path": "inputs/cad/obj_000001.ply",
        "bytes": 127492,
        "sha256": "177bf8313fc8feb749e86ab8953d9f97102842e54a08c63f32b8adfb663321e1",
    },
    {
        "object_id": 2,
        "relative_path": "inputs/cad/obj_000002.ply",
        "bytes": 319240,
        "sha256": "c6cc5f76b73782415a44362ee8853542f1da580fc49aac8b165767cfc6e2a24e",
    },
    {
        "object_id": 4,
        "relative_path": "inputs/cad/obj_000004.ply",
        "bytes": 658929,
        "sha256": "2885f809073970da906e5192d848dde4f15f844e1ef1e80d13a6ad7a603c9ef9",
    },
    {
        "object_id": 5,
        "relative_path": "inputs/cad/obj_000005.ply",
        "bytes": 679427,
        "sha256": "33dbf9fec3fa2a68fe6ec847507221feac8f00c002f61619bce1344395383a01",
    },
    {
        "object_id": 6,
        "relative_path": "inputs/cad/obj_000006.ply",
        "bytes": 708278,
        "sha256": "0e9ce0f575f104f97c4baf857b5449101eecc9658382b04f893905858a75eece",
    },
)

BOUNDARY_ZERO = {
    "label_access_count": 0,
    "depth_path_open_count": 0,
    "mask_path_open_count": 0,
    "mask_visib_path_open_count": 0,
    "gt_path_open_count": 0,
    "evaluator_path_open_count": 0,
    "sealed_split_access_count": 0,
    "official_scorer_run_count": 0,
    "fastsam_import_count": 0,
    "dinov2_model_import_count": 0,
    "foundationpose_run_count": 0,
    "gpu_c_call_count": 0,
}

FORBIDDEN_PATH_TOKENS = frozenset(
    {
        "depth",
        "mask",
        "mask_visib",
        "gt",
        "ground_truth",
        "evaluator",
        "sealed",
        "scorer",
        "foundationpose",
        "fastsam",
        "dinov2",
    }
)

R2_DEPENDENCY_LOCKS = (
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r2/__init__.py",
        "bytes": 1485,
        "sha256": "81cfe87aa0ecd974ea982a050c8c64842b47e85880c562154869898ea16a9ca4",
    },
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r2/__main__.py",
        "bytes": 48,
        "sha256": "935a1c1166b0c1ea35a82256345000bf2c73ded718d77773bc27a71ecce28f7d",
    },
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r2/cli.py",
        "bytes": 9453,
        "sha256": "6cf967c0688f9fa1e4bef3d298667b43ef698ab3d372786f8aba8fdac43db577",
    },
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r2/contracts.py",
        "bytes": 54665,
        "sha256": "32498a79c79b40d1b0633831ad13cf9002a262d4d80805e094073df8ee463155",
    },
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r2/producer.py",
        "bytes": 5389,
        "sha256": "63b91ac3d456628d1c30e5e0c2db37844b186accb4e4be9cf8a8e1934134ef44",
    },
    {
        "relative_path": (
            "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r2/"
            "SERVER_RUNBOOK_CNOS_RUNTIME_PREP_V1R2.md"
        ),
        "bytes": 5684,
        "sha256": "fe98a5d66c017498652e224707dfff2e85b990407b49872fe65bf382fa569d49",
    },
    {
        "relative_path": (
            "protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r2.json"
        ),
        "bytes": 6729,
        "sha256": "da987f1ca93a9a45faa3335b34698745cbc9f214d66cbc760303d6dd0a4ffd5c",
    },
)

WRAPPER_PATHS = frozenset(
    {
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/__init__.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/__main__.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/cli.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/contracts.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/renderer.py",
        (
            "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/"
            "SERVER_RUNBOOK_CNOS_RUNTIME_PREP_V1R3.md"
        ),
    }
)


@dataclass(frozen=True)
class RuntimeValidation:
    """Disk-verified paths admitted to the renderer process."""

    route: dict[str, Any]
    request: dict[str, Any]
    deployment_root: Path
    implementation_checkout: Path
    cnos_checkout: Path
    poses_path: Path
    venv_bin: Path
    python_executable: Path
    objects: tuple[tuple[int, Path], ...]


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ContractError(
            f"{label} fields differ: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{label} must remain relative without traversal")
    return path


def _resolve(root: Path, value: Any, label: str) -> Path:
    relative = _relative(value, label)
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"{label} escapes deployment root") from exc
    return candidate


def _path_tokens(value: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return {part for part in normalized.split("_") if part}


def _reject_forbidden_path(value: str, label: str) -> None:
    matches = sorted(_path_tokens(value) & FORBIDDEN_PATH_TOKENS)
    if matches:
        raise ContractError(f"{label} contains forbidden runtime tokens: {matches}")


def _disk_asset(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Missing required file: {relative_path}")
    return {
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_relative_asset(
    root: Path, expected: Mapping[str, Any], label: str
) -> Path:
    _exact(expected, {"relative_path", "bytes", "sha256"}, label)
    if not _is_sha256(expected["sha256"]):
        raise ContractError(f"{label}.sha256 must be lowercase SHA-256")
    _reject_forbidden_path(str(expected["relative_path"]), label)
    path = _resolve(root, expected["relative_path"], label)
    if _disk_asset(path, str(expected["relative_path"])) != dict(expected):
        raise ContractError(f"{label} bytes/SHA differ from the frozen lock")
    return path


def _validate_inventory(
    values: Any,
    expected: Sequence[Mapping[str, Any]],
    *,
    label: str,
    repository_root: Path | None,
) -> None:
    if not isinstance(values, list) or values != [dict(item) for item in expected]:
        raise ContractError(f"{label} inventory changed")
    if repository_root is None:
        return
    for item in expected:
        path = _resolve(repository_root, item["relative_path"], label)
        if _disk_asset(path, str(item["relative_path"])) != dict(item):
            raise ContractError(f"Frozen dependency changed: {item['relative_path']}")


def _require_self_lock(value: Mapping[str, Any], field: str, label: str) -> str:
    lock = value.get(field)
    if not _is_sha256(lock):
        raise ContractError(f"{label}.{field} must be lowercase SHA-256")
    expected = canonical_sha256({key: item for key, item in value.items() if key != field})
    if lock != expected:
        raise ContractError(f"{label} self-lock mismatch")
    return str(lock)


def _expected_predecessor() -> dict[str, Any]:
    return {
        "implementation_commit": BASE_IMPLEMENTATION_COMMIT,
        "implementation_tree": BASE_IMPLEMENTATION_TREE,
        "r2_route_id": R2_ROUTE_ID,
        "r2_route_lock_sha256": R2_ROUTE_LOCK,
        "blocker": {
            "status": "BLOCKED_BEFORE_DESCRIPTOR_FREEZE",
            "receipt": dict(BLOCKER_RECEIPT),
            "safe_archive": dict(SAFE_ARCHIVE),
            "immutable": True,
            "reinterpretation_permitted": False,
            "remote_hotfix_permitted": False,
        },
    }


def _expected_source() -> dict[str, Any]:
    return {
        "repository": CNOS_REPOSITORY,
        "commit": CNOS_COMMIT,
        "tree": CNOS_TREE,
        "archive": dict(CNOS_ARCHIVE),
        "checkout_relative_path": CNOS_CHECKOUT,
        "interface_files": [dict(value) for value in SOURCE_INTERFACES],
        "object_poses_level0": dict(POSES_LOCK),
    }


def _expected_wrapper_contract() -> dict[str, Any]:
    return {
        "mode": "THIN_IMPORT_OFFICIAL_RENDER_WITH_PATH_DIAMETER",
        "official_render_function": "src.poses.pyrender.render",
        "official_as_mesh_function": "src.utils.trimesh_utils.as_mesh",
        "official_diameter_function": "src.utils.trimesh_utils.get_obj_diameter",
        "diameter_argument": "hash_bound_cad_file_path",
        "mesh_argument_for_diameter_permitted": False,
        "source_checkout_modification_permitted": False,
        "camera_sampling": {
            "id": "cnos_obj_poses_level0_all",
            "template_level": 0,
            "view_count_per_object": 42,
            "translation_scale": "divide_by_1000_once",
        },
        "renderer": {
            "backend": "pyrender",
            "intrinsic": [
                [572.4114, 0.0, 325.2611],
                [0.0, 573.57043, 242.04899],
                [0.0, 0.0, 1.0],
            ],
            "image_size": {"height": 480, "width": 640},
            "light_intensity": 0.6,
            "standard_non_tless_non_hot3d_branch_only": True,
        },
        "process": {
            "venv_python": "explicit_absolute_venv_bin_python",
            "path_policy": "prepend_exact_bound_venv_bin",
            "pythonpath_policy": "prepend_exact_bound_implementation_checkout",
            "gpu_override": GPU_OVERRIDE,
            "gpu_override_json_type": "string",
            "child_count": 5,
            "child_failure_policy": "nonzero_or_invalid_output_fails_parent",
            "output_policy": "create_only_attempt_root",
        },
    }


def _expected_output_gate() -> dict[str, Any]:
    return {
        "png_count_per_object": 42,
        "filename_sequence": "000000.png-through-000041.png",
        "image_size": {"height": 480, "width": 640},
        "allowed_modes": ["RGB", "RGBA"],
        "require_nonempty_nonfull_rgb_foreground": True,
        "minimum_unique_png_sha256_per_object": 2,
        "object_coverage": "5_of_5",
    }


def validate_route(
    route: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Validate the inert A-R3 route and its frozen predecessor bytes."""
    _exact(
        route,
        {
            "schema_version",
            "route_id",
            "route_lock_sha256",
            "auto_deploy",
            "role",
            "predecessor",
            "source",
            "wrapper_contract",
            "objects",
            "output_gate",
            "boundary",
            "r2_dependency_files",
            "wrapper_files",
        },
        "CNOS R3 route",
    )
    if (
        route["schema_version"] != ROUTE_SCHEMA
        or route["route_id"] != ROUTE_ID
        or route["auto_deploy"] is not False
        or route["role"] != "DEVELOPMENT_ONLY_DESCRIPTOR_RENDERER"
    ):
        raise ContractError("CNOS R3 route identity/state mismatch")
    if route["predecessor"] != _expected_predecessor():
        raise ContractError("CNOS R3 predecessor/blocker binding changed")
    if route["source"] != _expected_source():
        raise ContractError("CNOS R3 pinned source binding changed")
    if route["wrapper_contract"] != _expected_wrapper_contract():
        raise ContractError("CNOS R3 thin-wrapper semantics changed")
    if route["objects"] != [dict(value) for value in CAD_LOCKS]:
        raise ContractError("CNOS R3 five-object CAD lock changed")
    if route["output_gate"] != _expected_output_gate():
        raise ContractError("CNOS R3 output content gate changed")
    if route["boundary"] != BOUNDARY_ZERO:
        raise ContractError("CNOS R3 zero-access boundary changed")
    _validate_inventory(
        route["r2_dependency_files"],
        R2_DEPENDENCY_LOCKS,
        label="CNOS R3 frozen R2 dependencies",
        repository_root=repository_root,
    )
    wrapper_files = route["wrapper_files"]
    if not isinstance(wrapper_files, list):
        raise ContractError("CNOS R3 wrapper file inventory must be a list")
    actual_paths = {
        str(_exact(item, {"relative_path", "bytes", "sha256"}, "wrapper file")["relative_path"])
        for item in wrapper_files
    }
    if actual_paths != WRAPPER_PATHS or len(wrapper_files) != len(WRAPPER_PATHS):
        raise ContractError("CNOS R3 wrapper file inventory changed")
    if repository_root is not None:
        for item in wrapper_files:
            path = _resolve(repository_root, item["relative_path"], "wrapper file")
            if _disk_asset(path, str(item["relative_path"])) != dict(item):
                raise ContractError(f"CNOS R3 wrapper source changed: {item['relative_path']}")
    route_lock = _require_self_lock(route, "route_lock_sha256", "CNOS R3 route")
    return {
        "status": "valid",
        "route_id": ROUTE_ID,
        "route_lock_sha256": route_lock,
        "object_count": len(OBJECT_IDS),
        "expected_png_count": len(OBJECT_IDS) * 42,
        "execution_ready": False,
        "model_imported": False,
        "boundary": dict(BOUNDARY_ZERO),
    }


def _validate_source_archive_members(archive: Path) -> None:
    prefix = f"cnos-{CNOS_COMMIT}/"
    with tarfile.open(archive, "r:gz") as stream:
        for interface in SOURCE_INTERFACES:
            member_name = prefix + str(interface["relative_path"])
            try:
                member = stream.getmember(member_name)
                handle = stream.extractfile(member)
            except (KeyError, tarfile.TarError) as exc:
                raise ContractError(f"Missing pinned CNOS archive member: {member_name}") from exc
            if handle is None:
                raise ContractError(f"Cannot read pinned CNOS archive member: {member_name}")
            payload = handle.read()
            expected = interface["archive_member"]
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ContractError(f"Pinned CNOS archive member bytes changed: {member_name}")


def _validate_checkout_interfaces(checkout: Path) -> None:
    for interface in SOURCE_INTERFACES:
        relative = str(interface["relative_path"])
        path = _resolve(checkout, relative, "CNOS checkout interface")
        expected = {
            "relative_path": relative,
            "bytes": interface["checkout_file"]["bytes"],
            "sha256": interface["checkout_file"]["sha256"],
        }
        if _disk_asset(path, relative) != expected:
            raise ContractError(f"Pinned CNOS checkout interface bytes changed: {relative}")


def _validate_absolute_asset(value: Any, expected_path: Path, label: str) -> None:
    asset = _exact(value, {"absolute_path", "bytes", "sha256"}, label)
    if Path(str(asset["absolute_path"])).resolve() != expected_path.resolve():
        raise ContractError(f"{label}.absolute_path differs from the bound path")
    if not expected_path.is_file():
        raise ContractError(f"Missing {label}: {expected_path}")
    if (
        not _is_sha256(asset["sha256"])
        or asset["bytes"] != expected_path.stat().st_size
        or asset["sha256"] != sha256_file(expected_path)
    ):
        raise ContractError(f"{label} bytes/SHA differ from disk")


def validate_runtime_request(
    route: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    deployment_root: Path,
    repository_root: Path | None = None,
) -> RuntimeValidation:
    """Validate every file admitted to a future renderer child process."""
    validate_route(route, repository_root=repository_root)
    _exact(
        request,
        {
            "schema_version",
            "role",
            "route_lock_sha256",
            "request_lock_sha256",
            "attempt_id",
            "implementation",
            "paths",
            "venv",
            "gpu_override",
            "objects",
            "boundary",
        },
        "CNOS R3 render request",
    )
    if (
        request["schema_version"] != REQUEST_SCHEMA
        or request["role"] != "DEVELOPMENT_ONLY_DESCRIPTOR_RENDERER"
        or request["route_lock_sha256"] != route["route_lock_sha256"]
        or request["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("CNOS R3 render request identity/boundary changed")
    attempt_id = request["attempt_id"]
    if not isinstance(attempt_id, str) or re.fullmatch(
        r"poseloop_ga_cnos_v1r3_[a-z0-9][a-z0-9_-]{0,47}", attempt_id
    ) is None:
        raise ContractError("CNOS R3 attempt_id is outside the frozen namespace")
    _require_self_lock(request, "request_lock_sha256", "CNOS R3 render request")

    root = deployment_root.resolve()
    implementation = _exact(
        request["implementation"],
        {"checkout_relative_path", "approved_commit", "approved_tree"},
        "CNOS R3 implementation",
    )
    if implementation["checkout_relative_path"] != "sources/poseloop":
        raise ContractError("CNOS R3 implementation checkout path changed")
    implementation_checkout = _resolve(
        root, implementation["checkout_relative_path"], "implementation checkout"
    )
    identity = git_identity(implementation_checkout)
    if (
        identity
        != {
            "commit": implementation["approved_commit"],
            "tree": implementation["approved_tree"],
            "clean": True,
        }
        or implementation["approved_commit"] == BASE_IMPLEMENTATION_COMMIT
        or not git_is_ancestor(
            implementation_checkout,
            BASE_IMPLEMENTATION_COMMIT,
            str(implementation["approved_commit"]),
        )
    ):
        raise ContractError("CNOS R3 reviewed implementation identity changed")

    paths = _exact(
        request["paths"],
        {
            "cnos_source_archive",
            "cnos_source_checkout",
            "object_poses_level0",
            "predecessor_blocker_receipt",
            "predecessor_safe_archive",
        },
        "CNOS R3 paths",
    )
    if paths != {
        "cnos_source_archive": CNOS_ARCHIVE["relative_path"],
        "cnos_source_checkout": CNOS_CHECKOUT,
        "object_poses_level0": f"{CNOS_CHECKOUT}/{POSES_RELATIVE_PATH}",
        "predecessor_blocker_receipt": BLOCKER_RECEIPT["relative_path"],
        "predecessor_safe_archive": SAFE_ARCHIVE["relative_path"],
    }:
        raise ContractError("CNOS R3 admitted path set changed")
    for label, value in paths.items():
        _reject_forbidden_path(str(value), f"CNOS R3 paths.{label}")

    source_archive = _validate_relative_asset(root, CNOS_ARCHIVE, "CNOS source archive")
    _validate_source_archive_members(source_archive)
    _validate_relative_asset(root, BLOCKER_RECEIPT, "R2 blocker receipt")
    _validate_relative_asset(root, SAFE_ARCHIVE, "R2 safe archive")
    cnos_checkout = _resolve(root, CNOS_CHECKOUT, "CNOS checkout")
    if git_identity(cnos_checkout) != {
        "commit": CNOS_COMMIT,
        "tree": CNOS_TREE,
        "clean": True,
    } or git_remote_repository(cnos_checkout) != CNOS_REPOSITORY:
        raise ContractError("Pinned CNOS checkout identity/origin changed")
    _validate_checkout_interfaces(cnos_checkout)
    poses_path = _resolve(cnos_checkout, POSES_RELATIVE_PATH, "CNOS level-0 poses")
    if _disk_asset(poses_path, POSES_RELATIVE_PATH) != POSES_LOCK:
        raise ContractError("Pinned CNOS level-0 object poses changed")

    venv = _exact(request["venv"], {"bin_path", "python"}, "CNOS R3 venv")
    if not isinstance(venv["bin_path"], str):
        raise ContractError("CNOS R3 venv.bin_path must be an absolute string path")
    _reject_forbidden_path(venv["bin_path"], "CNOS R3 venv.bin_path")
    venv_bin = Path(venv["bin_path"])
    if not venv_bin.is_absolute() or venv_bin.name != "bin" or not venv_bin.is_dir():
        raise ContractError("CNOS R3 requires an existing absolute venv/bin directory")
    python_executable = venv_bin / "python"
    _validate_absolute_asset(venv["python"], python_executable, "CNOS R3 venv python")
    if request["gpu_override"] != GPU_OVERRIDE or not isinstance(
        request["gpu_override"], str
    ):
        raise ContractError("CNOS R3 GPU override must remain string '0'")

    request_objects = request["objects"]
    expected_objects = [
        {"object_id": item["object_id"], "cad_relative_path": item["relative_path"]}
        for item in CAD_LOCKS
    ]
    if request_objects != expected_objects:
        raise ContractError("CNOS R3 request must cover the exact five CAD objects in order")
    objects: list[tuple[int, Path]] = []
    for item in CAD_LOCKS:
        _reject_forbidden_path(str(item["relative_path"]), "CNOS R3 CAD path")
        asset = {
            key: item[key] for key in ("relative_path", "bytes", "sha256")
        }
        path = _validate_relative_asset(
            root, asset, f"CAD object {item['object_id']}"
        )
        objects.append((int(item["object_id"]), path))
    return RuntimeValidation(
        route=dict(route),
        request=dict(request),
        deployment_root=root,
        implementation_checkout=implementation_checkout,
        cnos_checkout=cnos_checkout,
        poses_path=poses_path,
        venv_bin=venv_bin.resolve(),
        python_executable=python_executable.resolve(),
        objects=tuple(objects),
    )


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_create_only(path: Path, payload: bytes) -> None:
    """Write an immutable attempt artifact; existing paths are never reused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ContractError(f"CNOS R3 create-only artifact already exists: {path}") from exc
