"""Fail-closed contracts for the A-R4 CNOS renderer repair.

A-R4 keeps the official CNOS checkout and all A-R3 bytes immutable.  It binds
the two failed A-R3 attempts, admits the same five CADs, and changes only two
execution semantics: millimetre meshes are recentered after scaling, and a
bound ``venv/bin/python`` entry is executed without resolving away its venv
identity.  The entry and its resolved target are both verified from disk.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import contracts as v1
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3 import contracts as r3
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
)

from . import GPU_OVERRIDE, OBJECT_IDS, REQUEST_SCHEMA, ROLE, ROUTE_ID, ROUTE_SCHEMA

BASE_IMPLEMENTATION_COMMIT = "1aef708bcdc678b3734f25fd35b1381090c7f55d"
BASE_IMPLEMENTATION_TREE = "1311a5fede91ecd26cbc7c272ec9c5867ef72eca"
R3_ROUTE_LOCK = "cb83957708fe25eca809c36ff8c11e85811ce18e214fc4e9b2e40da48f348c77"

SECOND_BLOCKER_RECEIPT = {
    "relative_path": "evidence/predecessor/render-attempt2-geometry-blocker.json",
    "bytes": 4831,
    "sha256": "729f0f2a84a3e9b10c455b249ccb3d09fafae9de594e9797544625cfc9e7056e",
}
FINAL_SAFE_ARCHIVE = {
    "relative_path": (
        "evidence/predecessor/poseloop-a-r3-1aef708b-final-safe-evidence.tar.gz"
    ),
    "bytes": 274172,
    "sha256": "5dffd275caa703531c1a3db420c1f48c0ccd3efcccc9263a11dcf1691550ee03",
}
FINAL_ARCHIVE_MEMBERS = (
    {
        "relative_path": "receipts/render-attempt2-geometry-blocker.json",
        "bytes": 4831,
        "sha256": ("729f0f2a84a3e9b10c455b249ccb3d09fafae9de594e9797544625cfc9e7056e"),
    },
    {
        "relative_path": "receipts/render-attempt2-content-audit-addendum.json",
        "sha256": ("bb4e5f8566f1d747118da6351c54af5f27ddcc3e1fc83a123e69b1a407b7bdb3"),
    },
    {
        "relative_path": (
            "archives/poseloop-a-r3-1aef708b-render-failure-safe-evidence.tar.gz"
        ),
        "bytes": 125100,
        "sha256": ("2d082f486320e214daeea6895075ea5afd4458556db91bcae72677a78be84ac0"),
    },
)

BOUNDARY_ZERO = dict(r3.BOUNDARY_ZERO)
CAD_LOCKS = tuple(copy.deepcopy(r3.CAD_LOCKS))
CNOS_ARCHIVE = copy.deepcopy(r3.CNOS_ARCHIVE)
CNOS_CHECKOUT = r3.CNOS_CHECKOUT
CNOS_COMMIT = r3.CNOS_COMMIT
CNOS_REPOSITORY = r3.CNOS_REPOSITORY
CNOS_TREE = r3.CNOS_TREE
POSES_LOCK = copy.deepcopy(r3.POSES_LOCK)
POSES_RELATIVE_PATH = r3.POSES_RELATIVE_PATH
SOURCE_INTERFACES = tuple(copy.deepcopy(r3.SOURCE_INTERFACES))

R3_DEPENDENCY_LOCKS = (
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/__init__.py",
        "bytes": 914,
        "sha256": "fb56e81ca4b4c8da85e1fe39bf9e46732003e96c25d98008be4354c4c00b6872",
    },
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/__main__.py",
        "bytes": 48,
        "sha256": "935a1c1166b0c1ea35a82256345000bf2c73ded718d77773bc27a71ecce28f7d",
    },
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/cli.py",
        "bytes": 5220,
        "sha256": "354cc865c1f929178574a752915873a08d89c27a78dc0d8d7bb94fd73ce7148c",
    },
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/contracts.py",
        "bytes": 26198,
        "sha256": "3d12e39c269d0d2cae2e172efcfe54b33b3603131fdbf856602ef56176f67239",
    },
    {
        "relative_path": "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/renderer.py",
        "bytes": 15859,
        "sha256": "1e4d06598b9e194336f07bfdf83582c247815e0ebe8768ff66349df36dae3b51",
    },
    {
        "relative_path": (
            "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r3/"
            "SERVER_RUNBOOK_CNOS_RUNTIME_PREP_V1R3.md"
        ),
        "bytes": 5957,
        "sha256": "d4af2a4b8f1151e9823a74d5901538552b4075fbf7231d6bad660086d93a3780",
    },
    {
        "relative_path": (
            "protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r3.json"
        ),
        "bytes": 8796,
        "sha256": "e8ccf79a7ffcdf0ebd21276d890e56d115c68073e80bc9e27428e51596aff45e",
    },
)

WRAPPER_PATHS = frozenset(
    {
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r4/__init__.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r4/__main__.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r4/cli.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r4/contracts.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r4/renderer.py",
        (
            "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r4/"
            "SERVER_RUNBOOK_CNOS_RUNTIME_PREP_V1R4.md"
        ),
    }
)


@dataclass(frozen=True)
class RuntimeValidation:
    """Disk-verified paths admitted to an A-R4 renderer process."""

    route: dict[str, Any]
    request: dict[str, Any]
    deployment_root: Path
    implementation_checkout: Path
    cnos_checkout: Path
    poses_path: Path
    venv_bin: Path
    python_entry: Path
    python_target: Path
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


def _require_self_lock(value: Mapping[str, Any], field: str, label: str) -> str:
    lock = value.get(field)
    if not _is_sha256(lock):
        raise ContractError(f"{label}.{field} must be lowercase SHA-256")
    expected = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )
    if lock != expected:
        raise ContractError(f"{label} self-lock mismatch")
    return str(lock)


def _disk_asset(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Missing required file: {relative_path}")
    return {
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_inventory(
    values: Any,
    expected: Sequence[Mapping[str, Any]],
    *,
    label: str,
    repository_root: Path | None,
) -> None:
    frozen = [dict(item) for item in expected]
    if not isinstance(values, list) or values != frozen:
        raise ContractError(f"{label} inventory changed")
    if repository_root is None:
        return
    for item in expected:
        path = r3._resolve(repository_root, item["relative_path"], label)
        if _disk_asset(path, str(item["relative_path"])) != dict(item):
            raise ContractError(f"Frozen dependency changed: {item['relative_path']}")


def _expected_predecessor() -> dict[str, Any]:
    return {
        "implementation_commit": BASE_IMPLEMENTATION_COMMIT,
        "implementation_tree": BASE_IMPLEMENTATION_TREE,
        "r3_route_id": r3.ROUTE_ID,
        "r3_route_lock_sha256": R3_ROUTE_LOCK,
        "attempt_001": {
            "status": "FAILED_CREATE_ONLY_ATTEMPT",
            "failure": "MISSING_PYRENDER_AFTER_VENV_ENTRY_RESOLUTION",
            "safe_archive": dict(FINAL_ARCHIVE_MEMBERS[2]),
        },
        "attempt_002": {
            "status": "FAILED_CREATE_ONLY_ATTEMPT",
            "failure": "PRE_SCALE_RECENTER_TRANSFORM_UNIT_MISMATCH",
            "blocker_receipt": dict(SECOND_BLOCKER_RECEIPT),
            "final_safe_archive": dict(FINAL_SAFE_ARCHIVE),
            "content_audit_member": dict(FINAL_ARCHIVE_MEMBERS[1]),
            "observed": {
                "png_count": 42,
                "empty_or_transparent_count": 40,
                "visible_count": 2,
                "visible_pixels_each": [4, 4],
                "unique_png_sha256_count": 3,
            },
        },
        "immutable": True,
        "reinterpretation_permitted": False,
        "third_render_attempt_permitted": False,
    }


def _expected_source() -> dict[str, Any]:
    return {
        "repository": CNOS_REPOSITORY,
        "commit": CNOS_COMMIT,
        "tree": CNOS_TREE,
        "archive": dict(CNOS_ARCHIVE),
        "checkout_relative_path": CNOS_CHECKOUT,
        "interface_files": [copy.deepcopy(item) for item in SOURCE_INTERFACES],
        "object_poses_level0": dict(POSES_LOCK),
    }


def _expected_repair_contract() -> dict[str, Any]:
    return {
        "mode": "THIN_IMPORT_OFFICIAL_RENDER_SCALE_THEN_RECENTER",
        "official_render_function": "src.poses.pyrender.render",
        "official_as_mesh_function": "src.utils.trimesh_utils.as_mesh",
        "official_diameter_function": "src.utils.trimesh_utils.get_obj_diameter",
        "diameter_argument": "hash_bound_cad_file_path",
        "mesh_argument_for_diameter_permitted": False,
        "source_checkout_modification_permitted": False,
        "mesh_units": {
            "scale_threshold_strictly_greater_than": 100.0,
            "millimetre_scale_factor": 0.001,
            "scaled_branch_order": [
                "load_hash_bound_cad",
                "diameter_from_hash_bound_cad_path",
                "apply_scale_0.001",
                "read_scaled_mesh_bounding_box_centroid",
                "construct_negative_centroid_recenter",
                "official_render",
            ],
            "unscaled_branch": "recenter_from_original_mesh_centroid",
        },
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
            "pyopengl_platform": "egl",
            "standard_non_tless_non_hot3d_branch_only": True,
        },
        "process": {
            "venv_python_entry": "execute_bound_unresolved_venv_bin_python",
            "venv_python_target": "verify_resolved_target_bytes_and_sha256",
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
        "allowed_modes": ["RGBA"],
        "require_nonempty_nonfull_rgb_foreground": True,
        "require_nonempty_nonfull_alpha_for_rgba": True,
        "minimum_unique_png_sha256_per_object": 2,
        "object_coverage": "5_of_5",
    }


def validate_route(
    route: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Validate the inert A-R4 route and every frozen A-R3 dependency."""
    _exact(
        route,
        {
            "schema_version",
            "route_id",
            "route_lock_sha256",
            "auto_deploy",
            "role",
            "predecessor",
            "official_source",
            "repair_contract",
            "objects",
            "output_gate",
            "boundary",
            "r3_dependency_files",
            "wrapper_files",
        },
        "CNOS R4 route",
    )
    if (
        route["schema_version"] != ROUTE_SCHEMA
        or route["route_id"] != ROUTE_ID
        or route["auto_deploy"] is not False
        or route["role"] != ROLE
    ):
        raise ContractError("CNOS R4 route identity/state mismatch")
    if route["predecessor"] != _expected_predecessor():
        raise ContractError("CNOS R4 predecessor/failure binding changed")
    if route["official_source"] != _expected_source():
        raise ContractError("CNOS R4 official source binding changed")
    if route["repair_contract"] != _expected_repair_contract():
        raise ContractError("CNOS R4 minimal repair semantics changed")
    if route["objects"] != [dict(item) for item in CAD_LOCKS]:
        raise ContractError("CNOS R4 five-object CAD lock changed")
    if route["output_gate"] != _expected_output_gate():
        raise ContractError("CNOS R4 output content/alpha gate changed")
    if route["boundary"] != BOUNDARY_ZERO:
        raise ContractError("CNOS R4 zero-access boundary changed")
    _validate_inventory(
        route["r3_dependency_files"],
        R3_DEPENDENCY_LOCKS,
        label="CNOS R4 frozen R3 dependencies",
        repository_root=repository_root,
    )
    wrapper_files = route["wrapper_files"]
    if not isinstance(wrapper_files, list):
        raise ContractError("CNOS R4 wrapper file inventory must be a list")
    paths = {
        str(
            _exact(item, {"relative_path", "bytes", "sha256"}, "wrapper file")[
                "relative_path"
            ]
        )
        for item in wrapper_files
    }
    if paths != WRAPPER_PATHS or len(wrapper_files) != len(WRAPPER_PATHS):
        raise ContractError("CNOS R4 wrapper file inventory changed")
    if repository_root is not None:
        for item in wrapper_files:
            path = r3._resolve(repository_root, item["relative_path"], "wrapper file")
            if _disk_asset(path, str(item["relative_path"])) != dict(item):
                raise ContractError(
                    f"CNOS R4 wrapper source changed: {item['relative_path']}"
                )
        r3_route_path = r3._resolve(
            repository_root,
            R3_DEPENDENCY_LOCKS[-1]["relative_path"],
            "CNOS R3 route",
        )
        r3.validate_route(
            load_json_object(r3_route_path, "CNOS R3 route"),
            repository_root=repository_root,
        )
    route_lock = _require_self_lock(route, "route_lock_sha256", "CNOS R4 route")
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


def _validate_final_archive_members(archive: Path) -> None:
    try:
        with tarfile.open(archive, "r:gz") as stream:
            for expected in FINAL_ARCHIVE_MEMBERS:
                name = str(expected["relative_path"])
                try:
                    member = stream.getmember(name)
                    handle = stream.extractfile(member)
                except (KeyError, tarfile.TarError) as exc:
                    raise ContractError(
                        f"Missing A-R3 safe archive member: {name}"
                    ) from exc
                if handle is None or not member.isfile():
                    raise ContractError(f"Invalid A-R3 safe archive member: {name}")
                payload = handle.read()
                if "bytes" in expected and len(payload) != expected["bytes"]:
                    raise ContractError(
                        f"A-R3 safe archive member size changed: {name}"
                    )
                if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ContractError(f"A-R3 safe archive member SHA changed: {name}")
    except tarfile.TarError as exc:
        raise ContractError("Cannot read frozen A-R3 final safe archive") from exc


def python_entry_lock(entry: Path) -> dict[str, Any]:
    """Describe a lexical interpreter entry without resolving it for execution."""
    absolute = Path(os.path.abspath(os.fspath(entry)))
    if absolute.is_symlink():
        target_text = os.readlink(absolute)
        encoded = os.fsencode(target_text)
        return {
            "absolute_path": str(absolute),
            "kind": "symlink",
            "link_target": target_text,
            "link_bytes": len(encoded),
            "link_target_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if absolute.is_file():
        return {
            "absolute_path": str(absolute),
            "kind": "regular_file",
            "link_target": None,
            "link_bytes": None,
            "link_target_sha256": None,
        }
    raise ContractError(f"Missing bound venv Python entry: {absolute}")


def python_target_lock(entry: Path) -> dict[str, Any]:
    """Describe the resolved target while preserving the separate entry lock."""
    try:
        target = entry.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"Cannot resolve bound venv Python entry: {entry}") from exc
    if not target.is_file():
        raise ContractError(f"Bound venv Python target is not a file: {target}")
    return {
        "absolute_path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def validate_runtime_request(
    route: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    deployment_root: Path,
    repository_root: Path | None = None,
) -> RuntimeValidation:
    """Validate every file admitted to a future A-R4 renderer child."""
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
        "CNOS R4 render request",
    )
    if (
        request["schema_version"] != REQUEST_SCHEMA
        or request["role"] != ROLE
        or request["route_lock_sha256"] != route["route_lock_sha256"]
        or request["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("CNOS R4 render request identity/boundary changed")
    attempt_id = request["attempt_id"]
    if (
        not isinstance(attempt_id, str)
        or re.fullmatch(r"poseloop_ga_cnos_v1r4_[a-z0-9][a-z0-9_-]{0,47}", attempt_id)
        is None
    ):
        raise ContractError("CNOS R4 attempt_id is outside the frozen namespace")
    _require_self_lock(request, "request_lock_sha256", "CNOS R4 render request")

    root = deployment_root.resolve()
    implementation = _exact(
        request["implementation"],
        {"checkout_relative_path", "approved_commit", "approved_tree"},
        "CNOS R4 implementation",
    )
    if implementation["checkout_relative_path"] != "sources/poseloop":
        raise ContractError("CNOS R4 implementation checkout path changed")
    implementation_checkout = r3._resolve(
        root, implementation["checkout_relative_path"], "implementation checkout"
    )
    identity = v1.git_identity(implementation_checkout)
    if (
        identity
        != {
            "commit": implementation["approved_commit"],
            "tree": implementation["approved_tree"],
            "clean": True,
        }
        or implementation["approved_commit"] == BASE_IMPLEMENTATION_COMMIT
        or not v1.git_is_ancestor(
            implementation_checkout,
            BASE_IMPLEMENTATION_COMMIT,
            str(implementation["approved_commit"]),
        )
    ):
        raise ContractError("CNOS R4 reviewed implementation identity changed")

    paths = _exact(
        request["paths"],
        {
            "cnos_source_archive",
            "cnos_source_checkout",
            "object_poses_level0",
            "predecessor_second_blocker_receipt",
            "predecessor_final_safe_archive",
        },
        "CNOS R4 paths",
    )
    if paths != {
        "cnos_source_archive": CNOS_ARCHIVE["relative_path"],
        "cnos_source_checkout": CNOS_CHECKOUT,
        "object_poses_level0": f"{CNOS_CHECKOUT}/{POSES_RELATIVE_PATH}",
        "predecessor_second_blocker_receipt": SECOND_BLOCKER_RECEIPT["relative_path"],
        "predecessor_final_safe_archive": FINAL_SAFE_ARCHIVE["relative_path"],
    }:
        raise ContractError("CNOS R4 admitted path set changed")
    for label, value in paths.items():
        r3._reject_forbidden_path(str(value), f"CNOS R4 paths.{label}")

    source_archive = r3._validate_relative_asset(
        root, CNOS_ARCHIVE, "CNOS source archive"
    )
    r3._validate_source_archive_members(source_archive)
    r3._validate_relative_asset(root, SECOND_BLOCKER_RECEIPT, "A-R3 blocker receipt")
    final_archive = r3._validate_relative_asset(
        root, FINAL_SAFE_ARCHIVE, "A-R3 final safe archive"
    )
    _validate_final_archive_members(final_archive)
    cnos_checkout = r3._resolve(root, CNOS_CHECKOUT, "CNOS checkout")
    if (
        v1.git_identity(cnos_checkout)
        != {
            "commit": CNOS_COMMIT,
            "tree": CNOS_TREE,
            "clean": True,
        }
        or v1.git_remote_repository(cnos_checkout) != CNOS_REPOSITORY
    ):
        raise ContractError("Pinned CNOS checkout identity/origin changed")
    r3._validate_checkout_interfaces(cnos_checkout)
    poses_path = r3._resolve(cnos_checkout, POSES_RELATIVE_PATH, "CNOS level-0 poses")
    if r3._disk_asset(poses_path, POSES_RELATIVE_PATH) != POSES_LOCK:
        raise ContractError("Pinned CNOS level-0 object poses changed")

    venv = _exact(
        request["venv"],
        {"bin_path", "python_entry", "python_target"},
        "CNOS R4 venv",
    )
    if not isinstance(venv["bin_path"], str):
        raise ContractError("CNOS R4 venv.bin_path must be an absolute string path")
    r3._reject_forbidden_path(venv["bin_path"], "CNOS R4 venv.bin_path")
    venv_bin = Path(venv["bin_path"])
    if not venv_bin.is_absolute() or venv_bin.name != "bin" or not venv_bin.is_dir():
        raise ContractError("CNOS R4 requires an existing absolute venv/bin directory")
    python_entry = venv_bin / "python"
    declared_entry = _exact(
        venv["python_entry"],
        {
            "absolute_path",
            "kind",
            "link_target",
            "link_bytes",
            "link_target_sha256",
        },
        "CNOS R4 Python entry",
    )
    if dict(declared_entry) != python_entry_lock(python_entry):
        raise ContractError("CNOS R4 venv Python entry/link identity changed")
    declared_target = _exact(
        venv["python_target"],
        {"absolute_path", "bytes", "sha256"},
        "CNOS R4 Python target",
    )
    if dict(declared_target) != python_target_lock(python_entry):
        raise ContractError("CNOS R4 venv Python resolved target bytes/SHA changed")
    if request["gpu_override"] != GPU_OVERRIDE or not isinstance(
        request["gpu_override"], str
    ):
        raise ContractError("CNOS R4 GPU override must remain string '0'")

    expected_objects = [
        {"object_id": item["object_id"], "cad_relative_path": item["relative_path"]}
        for item in CAD_LOCKS
    ]
    if request["objects"] != expected_objects:
        raise ContractError(
            "CNOS R4 request must cover the exact five CAD objects in order"
        )
    objects: list[tuple[int, Path]] = []
    for item in CAD_LOCKS:
        path = r3._validate_relative_asset(
            root,
            {key: item[key] for key in ("relative_path", "bytes", "sha256")},
            f"CAD object {item['object_id']}",
        )
        objects.append((int(item["object_id"]), path))
    return RuntimeValidation(
        route=dict(route),
        request=dict(request),
        deployment_root=root,
        implementation_checkout=implementation_checkout,
        cnos_checkout=cnos_checkout,
        poses_path=poses_path,
        venv_bin=Path(os.path.abspath(os.fspath(venv_bin))),
        python_entry=Path(os.path.abspath(os.fspath(python_entry))),
        python_target=python_entry.resolve(strict=True),
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
    """Write a create-only A-R4 artifact and fsync it before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ContractError(
            f"CNOS R4 create-only artifact already exists: {path}"
        ) from exc
