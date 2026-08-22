"""Fail-closed contracts for A-R5 label-blind CNOS instance proposals.

The runtime sees frames and a complete five-object CAD catalogue.  It never
receives the historical per-frame target identity that was used to choose the
development slice.  FastSAM masks are therefore independent proposals and
DINOv2/CAD matching is performed for every proposal against every catalogue
object.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
)

from . import (
    FRAME_MANIFEST_SCHEMA,
    OUTPUT_BUNDLE_SCHEMA,
    PROTOCOL_ID,
    PROTOCOL_SCHEMA,
    RUNTIME_REQUEST_SCHEMA,
)

FRAME_WIDTH = 1440
FRAME_HEIGHT = 1080
TEMPLATE_WIDTH = 640
TEMPLATE_HEIGHT = 480
CATALOG_OBJECT_IDS = (1, 2, 4, 5, 6)
FRAME_KEYS = (
    (0, 0),
    (0, 1),
    (3, 0),
    (3, 1),
    (9, 0),
    (9, 1),
    (12, 0),
    (12, 1),
    (15, 0),
    (15, 1),
)
WORKLOAD_SHA256 = "dd9f9c4ce9b9ca380614064f58e332661aee9ce63c9602918d52ae391267dfb3"

PINNED_CNOS_REPOSITORY = "https://github.com/nv-nguyen/cnos"
PINNED_CNOS_COMMIT = "298d1f3366171464ca271659f0e2f7a6eb8e39b4"
PINNED_CNOS_TREE = "595ba390ad1fdcd2141c8004e505b5da1eb403c9"
PINNED_CNOS_ARCHIVE_BYTES = 17114735
PINNED_CNOS_ARCHIVE_SHA256 = (
    "07c52c95f31ddae7fce14f5741c2fe5d7eab88369905237b2a52a5f4a8ebd6e5"
)
PINNED_DINOV2_REPOSITORY = "https://github.com/facebookresearch/dinov2"
PINNED_DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
PINNED_DINOV2_TREE = "2a27257b79b0633b027a21014bc9360e3c1b3f43"
PINNED_DINOV2_VITL14_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
)
PINNED_DINOV2_VITL14_BYTES = 1217586395
PINNED_DINOV2_VITL14_SHA256 = (
    "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
)
PINNED_DINOV2_IDENTITY_RECEIPT_SHA256 = (
    "7f417bba08ea1dd8eb5d1481a06d76a45d32410bc7fcf495591dc196c8ef338e"
)
SOURCE_CHECKOUT_MANIFEST_SCHEMA = (
    "poseloop.pose-accuracy-recovery.source-checkout-manifest.v1r5"
)
CNOS_EXECUTION_FILES = (
    "src/model/fast_sam.py",
    "src/model/dinov2.py",
    "src/model/utils.py",
)

A_R4_RENDER_PROVENANCE = {
    "implementation_commit": "844a7c48a767e579b3c799851d8ab2a3961283d5",
    "implementation_tree": "da2cbb40f606242087c553260e3c795455226f38",
    "route_lock_sha256": "5c189080581d386f83702131c134aad5fcd07054454f8374811a6f5dac6c53f4",
    "protocol_sha256": "b3d9b1fb810f6928e369a233d3a1a9e7276d0eefa71b6966afa956583b3f4e8d",
    "runtime_request_lock_sha256": "ba15896143c4cef39fd536c738b0e43960cbfca017f9765ccefa233adec4c5650",
    "request_file_sha256": "3aee98342efc47e279811b074083743d0cbe07ec7bf56b26430b5a929bc342c2",
    "preflight_sha256": "3c6b6760cdb93a602d0ab0ece54adda9ecf6735bb705ba4e4cd5c2742bf59c0a",
    "job_sha256": "52ac3d5e935a76a8e932ad03fec4083c23d5bee8186ea37f61fc213f8494d89e",
    "authorization_receipt_sha256": "d32366d74278e49b8e055d9f2a432dcc9320034297b485e42e8e41889d72fd8c",
    "attempt_receipt_sha256": "7efb8e76a1e4d0c602f6c6ed4896a35774fe2575fcd50b69ce0e721f8d8d80cd",
    "attempt_receipt_lock_sha256": "b36a9aacfe57f58292c0aaaaca2333fb763b4c38b50a3bdf723a600d9621fc570",
    "exit_receipt_sha256": "9bbb4ef9f4b9aa94a308c3120a5e4b144eebac13e5413d0dc777235b0b8d2b82",
    "content_audit_sha256": "353dae87b5350511c7d131322eead76c6383ab01916cf3682107657cdfa12816",
    "deployment_inventory_sha256": "1630ce2c5ba906e61fb3f3867248a8b9e345e19b2f59112cb71fc1795585b1fe",
    "safe_archive_bytes": 1653423,
    "safe_archive_sha256": "c997f4f6d65f2398527b55bcdffbb40cf4881783cbe50580e712697646f118bb",
    "safe_archive_member_count": 256,
    "rgba_png_count": 210,
    "objects": [
        {"object_id": 1, "view_count": 42, "unique_content_count": 42},
        {"object_id": 2, "view_count": 42, "unique_content_count": 42},
        {"object_id": 4, "view_count": 42, "unique_content_count": 42},
        {"object_id": 5, "view_count": 42, "unique_content_count": 42},
        {"object_id": 6, "view_count": 42, "unique_content_count": 42},
    ],
    "status": "PASS_RGBA_5_OBJECTS_X_42_VIEWS",
}

BOUNDARY_ZERO = {
    "label_access_count": 0,
    "scene_gt_access_count": 0,
    "scene_gt_info_access_count": 0,
    "source_mask_access_count": 0,
    "mask_visib_access_count": 0,
    "depth_visibility_access_count": 0,
    "evaluator_access_count": 0,
    "sealed_access_count": 0,
    "oracle_association_count": 0,
    "foundationpose_run_count": 0,
    "official_scorer_run_count": 0,
    "downstream_export_count": 0,
}

PROPOSAL_STATES = {
    "NO_PROPOSAL",
    "ONE_PROPOSAL",
    "MULTIPLE_PROPOSALS_DISJOINT",
    "MULTIPLE_PROPOSALS_ADJACENT",
}
VISUALIZATION_ROLES = (
    "rgb_all_instance_contours",
    "selected_mask",
    "cad_topk_overlay",
    "adjacency_overlay",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_INPUT_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:scene[_-]?gt(?:[_-]?info)?|mask[_-]?visib|"
    r"mask|ground[_-]?truth|gt|evaluator|evaluation|sealed|oracle|"
    r"depth(?:[_-]?derived)?[_-]?visibility|target[_-]?object(?:[_-]?id)?|"
    r"association)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


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


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _git_oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_OID_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase 40-hex Git object id")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _unit(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ContractError(f"{label} must be finite in [0, 1]")
    return float(value)


def _raw_cosine(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not -1.0 <= float(value) <= 1.0
    ):
        raise ContractError(f"{label} must be finite in [-1, 1]")
    return float(value)


def normalized_cad_similarity(raw_cosine: Any) -> float:
    """Frozen affine storage transform; deliberately no clamp."""
    return (_raw_cosine(raw_cosine, "raw cosine") + 1.0) / 2.0


def _self_lock(value: Mapping[str, Any], field: str, label: str) -> str:
    lock = _sha(value.get(field), f"{label}.{field}")
    expected = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )
    if lock != expected:
        raise ContractError(f"{label} canonical self-lock mismatch")
    return lock


def _assert_label_blind(value: Any, trail: str = "runtime_input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _FORBIDDEN_INPUT_RE.search(str(key)):
                raise ContractError(f"Forbidden label/visibility key at {trail}.{key}")
            _assert_label_blind(child, f"{trail}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_label_blind(child, f"{trail}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_INPUT_RE.search(
        value.replace("\\", "/")
    ):
        raise ContractError(f"Forbidden label/visibility value at {trail}")


def _relative_path(
    value: Any, *, root: str, label: str, label_blind: bool = True
) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != root
    ):
        raise ContractError(f"{label} must remain under {root}/")
    if label_blind:
        _assert_label_blind(value, label)
    return path


def _resolve(root: Path, relative: PurePosixPath, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"{label} escapes its root") from exc
    return candidate


def repository_root_from_package() -> Path:
    """Resolve the checkout that contains the code executing this validation."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pose_accuracy_recovery_prep" / "instance_proposal_v1r5").is_dir():
        raise ContractError("A-R5 executing package is outside a PoseLoop checkout")
    return root


def _git(checkout: Path, *arguments: str, label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise ContractError(f"{label}: git is unavailable") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit={completed.returncode}"
        raise ContractError(f"{label}: git command failed: {detail}")
    return completed.stdout


def _normalized_repository(value: str) -> str:
    normalized = value.strip().replace("\\", "/").rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def _checkout_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a POSIX checkout-relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ContractError(f"{label} escapes the source checkout")
    return relative


def _source_execution_paths(checkout: Path, kind: str) -> list[str]:
    if kind == "CNOS":
        return list(CNOS_EXECUTION_FILES)
    if kind != "DINOV2":
        raise ContractError("A-R5 source checkout kind changed")
    tracked = _git(
        checkout,
        "ls-files",
        "--",
        "hubconf.py",
        "dinov2",
        label="DINOv2 tracked execution inventory",
    ).splitlines()
    dinov2_python = sorted(
        path for path in tracked if path.startswith("dinov2/") and path.endswith(".py")
    )
    if "hubconf.py" not in tracked or not dinov2_python:
        raise ContractError("DINOv2 checkout lacks hubconf.py or dinov2/**/*.py")
    return ["hubconf.py", *dinov2_python]


def validate_source_checkout_manifest(
    manifest: Mapping[str, Any],
    *,
    checkout: Path,
    kind: str,
    repository: str,
    commit: str,
    tree: str,
    source_archive: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate a clean Git checkout and every Python file used at runtime."""
    label = f"A-R5 {kind} checkout manifest"
    manifest = _exact(
        manifest,
        {
            "schema_version",
            "kind",
            "repository",
            "commit",
            "tree",
            "all_clean",
            "git_status_porcelain_v1_untracked_files_all",
            "source_archive",
            "execution_files",
            "execution_inventory_sha256",
            "checkout_manifest_lock_sha256",
        },
        label,
    )
    archive_binding = _exact(
        manifest["source_archive"], {"bytes", "sha256"}, f"{label}.source_archive"
    )
    if (
        manifest["schema_version"] != SOURCE_CHECKOUT_MANIFEST_SCHEMA
        or manifest["kind"] != kind
        or manifest["repository"] != repository
        or manifest["commit"] != commit
        or manifest["tree"] != tree
        or manifest["all_clean"] is not True
        or manifest["git_status_porcelain_v1_untracked_files_all"] != ""
        or archive_binding
        != {
            "bytes": source_archive["bytes"],
            "sha256": source_archive["sha256"],
        }
    ):
        raise ContractError(f"{label} identity/archive binding changed")
    _git_oid(manifest["commit"], f"{label}.commit")
    _git_oid(manifest["tree"], f"{label}.tree")
    _positive_int(archive_binding["bytes"], f"{label}.source_archive.bytes")
    _sha(archive_binding["sha256"], f"{label}.source_archive.sha256")
    _sha(
        manifest["execution_inventory_sha256"],
        f"{label}.execution_inventory_sha256",
    )
    _self_lock(manifest, "checkout_manifest_lock_sha256", label)

    checkout = checkout.resolve()
    if not checkout.is_dir() or not (checkout / ".git").exists():
        raise ContractError(f"{label} checkout is missing Git metadata")
    actual_repository = _git(
        checkout, "remote", "get-url", "origin", label=f"{label} origin"
    ).strip()
    actual_commit = _git(checkout, "rev-parse", "HEAD", label=f"{label} HEAD").strip()
    actual_tree = _git(
        checkout, "rev-parse", "HEAD^{tree}", label=f"{label} tree"
    ).strip()
    status = _git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        label=f"{label} clean status",
    )
    if (
        _normalized_repository(actual_repository) != _normalized_repository(repository)
        or actual_commit != commit
        or actual_tree != tree
        or status != ""
    ):
        raise ContractError(f"{label} disk Git identity is not exact and clean")

    expected_paths = _source_execution_paths(checkout, kind)
    raw_files = manifest["execution_files"]
    if not isinstance(raw_files, list):
        raise ContractError(f"{label}.execution_files must be a list")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_files):
        item = _exact(
            raw,
            {"relative_path", "bytes", "sha256"},
            f"{label}.execution_files[{index}]",
        )
        relative = _checkout_relative(
            item["relative_path"], f"{label}.execution_files[{index}].relative_path"
        ).as_posix()
        if relative in seen:
            raise ContractError(f"{label} execution inventory contains duplicates")
        seen.add(relative)
        size = _positive_int(item["bytes"], f"{label}.{relative}.bytes")
        digest = _sha(item["sha256"], f"{label}.{relative}.sha256")
        path = (checkout / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            path.relative_to(checkout)
        except ValueError as exc:
            raise ContractError(f"{label} execution file escapes checkout") from exc
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ContractError(f"{label} execution file changed: {relative}")
        parsed.append({"relative_path": relative, "bytes": size, "sha256": digest})
    if [item["relative_path"] for item in parsed] != expected_paths:
        raise ContractError(f"{label} execution file coverage/order changed")
    if manifest["execution_inventory_sha256"] != canonical_sha256(parsed):
        raise ContractError(f"{label} execution inventory lock mismatch")
    return {item["relative_path"]: item for item in parsed}


def load_source_execution_locks(
    request: Mapping[str, Any], *, data_root: Path
) -> dict[str, dict[str, Any]]:
    """Revalidate and return the exact source files allowed to execute."""
    source = request["source"]
    result: dict[str, dict[str, Any]] = {}
    specifications = (
        (
            "CNOS",
            "cnos_checkout_relative_path",
            "cnos_checkout_manifest",
            "cnos_archive",
            PINNED_CNOS_REPOSITORY,
            PINNED_CNOS_COMMIT,
            PINNED_CNOS_TREE,
            "cnos_checkout_manifest",
            "cnos_source_archive",
        ),
        (
            "DINOV2",
            "dinov2_checkout_relative_path",
            "dinov2_checkout_manifest",
            "dinov2_archive",
            PINNED_DINOV2_REPOSITORY,
            PINNED_DINOV2_COMMIT,
            PINNED_DINOV2_TREE,
            "dinov2_checkout_manifest",
            "dinov2_source_archive",
        ),
    )
    for (
        kind,
        checkout_key,
        manifest_key,
        archive_key,
        repository,
        commit,
        tree,
        manifest_role,
        archive_role,
    ) in specifications:
        checkout_relative = _relative_path(
            source[checkout_key], root="sources", label=f"source.{checkout_key}"
        )
        checkout = _resolve(data_root, checkout_relative, f"source.{checkout_key}")
        manifest_asset, manifest_path = _asset(
            source[manifest_key],
            role=manifest_role,
            root_name="sources",
            label=f"source.{manifest_key}",
            disk_root=data_root,
            verify_hashes=True,
        )
        archive_asset, _ = _asset(
            source[archive_key],
            role=archive_role,
            root_name="sources",
            label=f"source.{archive_key}",
            disk_root=data_root,
            verify_hashes=True,
        )
        del manifest_asset
        assert manifest_path is not None
        files = validate_source_checkout_manifest(
            read_json(manifest_path),
            checkout=checkout,
            kind=kind,
            repository=repository,
            commit=commit,
            tree=tree,
            source_archive=archive_asset,
        )
        result[kind] = {"checkout": checkout, "files": files}
    return result


def _validate_current_implementation(
    implementation: Mapping[str, Any],
    *,
    source_archive_path: Path,
    repository_root: Path,
) -> None:
    """Bind the declared implementation to this clean checkout and archive bytes."""
    expected_route = {
        "format": "tar.gz",
        "prefix": f"poseloop-{implementation['commit']}/",
        "command": [
            "git",
            "archive",
            "--format=tar.gz",
            f"--prefix=poseloop-{implementation['commit']}/",
            implementation["commit"],
        ],
    }
    if implementation["archive_route"] != expected_route:
        raise ContractError("A-R5 implementation archive route changed")
    actual_commit = _git(
        repository_root,
        "rev-parse",
        "HEAD",
        label="A-R5 implementation HEAD",
    ).strip()
    actual_tree = _git(
        repository_root,
        "rev-parse",
        "HEAD^{tree}",
        label="A-R5 implementation tree",
    ).strip()
    status = _git(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        label="A-R5 implementation clean status",
    )
    if (
        actual_commit != implementation["commit"]
        or actual_tree != implementation["tree"]
        or status != ""
    ):
        raise ContractError(
            "A-R5 executing implementation is not the exact clean commit/tree"
        )
    with tempfile.TemporaryDirectory(prefix="poseloop-a-r5-archive-") as temporary:
        rebuilt = Path(temporary) / "implementation.tar.gz"
        _git(
            repository_root,
            "archive",
            "--format=tar.gz",
            f"--prefix=poseloop-{implementation['commit']}/",
            f"--output={rebuilt}",
            implementation["commit"],
            label="A-R5 implementation archive reconstruction",
        )
        if rebuilt.stat().st_size != source_archive_path.stat().st_size or sha256_file(
            rebuilt
        ) != sha256_file(source_archive_path):
            raise ContractError(
                "A-R5 implementation source archive is not reproducible"
            )


def _asset(
    value: Any,
    *,
    role: str,
    root_name: str,
    label: str,
    disk_root: Path | None,
    verify_hashes: bool,
) -> tuple[dict[str, Any], Path | None]:
    asset = _exact(value, {"role", "relative_path", "bytes", "sha256"}, label)
    if asset["role"] != role:
        raise ContractError(f"{label}.role must be {role}")
    relative = _relative_path(
        asset["relative_path"], root=root_name, label=f"{label}.relative_path"
    )
    size = _positive_int(asset["bytes"], f"{label}.bytes")
    _sha(asset["sha256"], f"{label}.sha256")
    path: Path | None = None
    if disk_root is not None:
        path = _resolve(disk_root, relative, label)
        _verify_disk_asset(
            path,
            bytes_expected=size,
            sha256_expected=asset["sha256"],
            label=label,
            verify_hashes=verify_hashes,
        )
    return dict(asset), path


def _verify_disk_asset(
    path: Path,
    *,
    bytes_expected: int,
    sha256_expected: str,
    label: str,
    verify_hashes: bool,
) -> None:
    if not path.is_file() or path.stat().st_size != bytes_expected:
        raise ContractError(f"{label} missing or byte count changed")
    if verify_hashes and sha256_file(path) != sha256_expected:
        raise ContractError(f"{label} SHA-256 mismatch")


def _frame_key(value: Any, label: str) -> tuple[int, int]:
    key = _exact(value, {"scene_id", "image_id"}, label)
    result: list[int] = []
    for field in ("scene_id", "image_id"):
        item = key[field]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ContractError(f"{label}.{field} must be a non-negative integer")
        result.append(item)
    return result[0], result[1]


def _verify_rgb(path: Path, label: str) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                raise ContractError(f"{label} must decode as 1440x1080 RGB")
    except (OSError, ValueError) as exc:
        raise ContractError(f"{label} is not a decodable RGB image") from exc


def decoded_image_binding(
    path: Path, *, role: str, relative_path: str
) -> dict[str, Any]:
    """Recompute a disk image binding used by the B2-v2 independent validator."""
    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise ContractError(f"{role} input cannot be decoded") from exc
    if role == "rgb":
        if mode != "RGB" or array.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise ContractError("A-R5 RGB input must be 1440x1080 RGB")
        foreground = np.any(array != 0, axis=2)
        decoded_shape = [FRAME_HEIGHT, FRAME_WIDTH, 3]
    elif role == "raw_sensor_depth":
        if mode not in {"I;16", "I;16L", "I"} or array.shape != (
            FRAME_HEIGHT,
            FRAME_WIDTH,
        ):
            raise ContractError(
                "A-R5 raw sensor depth must be a 1440x1080 16/32-bit integer image"
            )
        if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise ContractError(
                "A-R5 raw sensor depth values must be non-negative integers"
            )
        foreground = array > 0
        decoded_shape = [FRAME_HEIGHT, FRAME_WIDTH]
    else:
        raise ContractError(f"Unsupported decoded image role: {role}")
    locations = np.argwhere(foreground)
    support = None
    if len(locations):
        y_min, x_min = locations.min(axis=0)
        y_max, x_max = locations.max(axis=0)
        support = [int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1]
    return {
        "role": role,
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "decoded_shape": decoded_shape,
        "decoded_mode": mode,
        "foreground_pixels": int(foreground.sum()),
        "support_bbox_xyxy_half_open": support,
    }


def _verify_depth(path: Path, label: str) -> None:
    decoded_image_binding(
        path, role="raw_sensor_depth", relative_path="inputs/depth/verified"
    )


def _verify_public_camera(path: Path, label: str) -> None:
    camera = read_json(path)
    camera = _exact(
        camera,
        {
            "schema_version",
            "frame_size",
            "camera_intrinsics",
            "camera_world_to_camera_pose_m",
            "coordinate_convention",
        },
        label,
    )
    if camera["schema_version"] != "poseloop.public-depth-free-camera.v1r5":
        raise ContractError(f"{label}.schema_version changed")
    if camera["frame_size"] != {"height": FRAME_HEIGHT, "width": FRAME_WIDTH}:
        raise ContractError(f"{label}.frame_size changed")
    intrinsics = camera["camera_intrinsics"]
    if (
        not isinstance(intrinsics, list)
        or len(intrinsics) != 9
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            for item in intrinsics
        )
    ):
        raise ContractError(f"{label}.camera_intrinsics must be nine finite values")
    pose = camera["camera_world_to_camera_pose_m"]
    if (
        not isinstance(pose, list)
        or len(pose) != 16
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            for item in pose
        )
        or pose[12:] != [0.0, 0.0, 0.0, 1.0]
    ):
        raise ContractError(f"{label}.camera_world_to_camera_pose_m changed")
    if camera["coordinate_convention"] != "opencv_x_right_y_down_z_forward_row_major":
        raise ContractError(f"{label}.coordinate_convention changed")
    _assert_label_blind(camera, label)


def validate_protocol(
    protocol: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol_id",
        "role",
        "auto_deploy",
        "accuracy_claim_permitted",
        "predecessor",
        "a_r4_render_provenance",
        "development_slice",
        "method",
        "model_provenance",
        "runtime_boundary",
        "execution_policy",
        "visualization_contract",
        "wrapper_files",
        "protocol_lock_sha256",
    }
    protocol = _exact(protocol, required, "A-R5 protocol")
    if (
        protocol["schema_version"] != PROTOCOL_SCHEMA
        or protocol["protocol_id"] != PROTOCOL_ID
        or protocol["role"] != "DEVELOPMENT_ONLY_LABEL_BLIND_INSTANCE_PROPOSALS"
        or protocol["auto_deploy"] is not False
        or protocol["accuracy_claim_permitted"] is not False
    ):
        raise ContractError("A-R5 protocol identity/role changed")
    predecessor = _exact(
        protocol["predecessor"],
        {"a_r3_read_only", "a_r4_read_only", "b2_p1_design_reference_only"},
        "A-R5 predecessor",
    )
    if predecessor != {
        "a_r3_read_only": True,
        "a_r4_read_only": True,
        "b2_p1_design_reference_only": "5162dfa",
    }:
        raise ContractError("A-R5 predecessor boundary changed")
    if protocol["a_r4_render_provenance"] != A_R4_RENDER_PROVENANCE:
        raise ContractError("A-R4 PASS provenance changed")
    development = _exact(
        protocol["development_slice"],
        {
            "workload_sha256",
            "frame_count",
            "scene_count",
            "catalog_object_ids",
            "frame_keys",
            "historical_object_stratification_is_provenance_only",
            "runtime_target_identity_exposed",
            "single_object_fallback_permitted",
        },
        "A-R5 development slice",
    )
    if (
        development["workload_sha256"] != WORKLOAD_SHA256
        or development["frame_count"] != len(FRAME_KEYS)
        or development["scene_count"] != 5
        or development["catalog_object_ids"] != list(CATALOG_OBJECT_IDS)
        or development["frame_keys"]
        != [
            {"scene_id": scene_id, "image_id": image_id}
            for scene_id, image_id in FRAME_KEYS
        ]
        or development["historical_object_stratification_is_provenance_only"]
        is not True
        or development["runtime_target_identity_exposed"] is not False
        or development["single_object_fallback_permitted"] is not False
    ):
        raise ContractError("A-R5 multi-scene/multi-object slice changed")
    method = _exact(
        protocol["method"],
        {
            "instance_proposals",
            "catalog_recognition",
            "catalog_scope",
            "normalization",
            "gt_association_permitted",
            "whole_foreground_proposal_permitted",
            "foundationpose_participates",
        },
        "A-R5 method",
    )
    if method != {
        "instance_proposals": "official_cnos_fastsam_x_independent_instance_masks",
        "catalog_recognition": "official_cnos_dinov2_vitl14_top5_template_cosine",
        "catalog_scope": "all_five_objects_for_every_proposal",
        "normalization": "raw_cosine_plus_one_divide_two_without_clamp",
        "gt_association_permitted": False,
        "whole_foreground_proposal_permitted": False,
        "foundationpose_participates": False,
    }:
        raise ContractError("A-R5 official CNOS semantics changed")
    model_provenance = _exact(
        protocol["model_provenance"],
        {
            "dinov2_repository",
            "dinov2_commit",
            "dinov2_tree",
            "dinov2_vitl14_url",
            "dinov2_vitl14_bytes",
            "dinov2_vitl14_sha256",
            "remote_identity_receipt_sha256",
        },
        "A-R5 model provenance",
    )
    if model_provenance != {
        "dinov2_repository": PINNED_DINOV2_REPOSITORY,
        "dinov2_commit": PINNED_DINOV2_COMMIT,
        "dinov2_tree": PINNED_DINOV2_TREE,
        "dinov2_vitl14_url": PINNED_DINOV2_VITL14_URL,
        "dinov2_vitl14_bytes": PINNED_DINOV2_VITL14_BYTES,
        "dinov2_vitl14_sha256": PINNED_DINOV2_VITL14_SHA256,
        "remote_identity_receipt_sha256": PINNED_DINOV2_IDENTITY_RECEIPT_SHA256,
    }:
        raise ContractError("A-R5 official DINOv2 source/checkpoint identity changed")
    if protocol["runtime_boundary"] != BOUNDARY_ZERO:
        raise ContractError("A-R5 zero-access boundary changed")
    execution = _exact(
        protocol["execution_policy"],
        {
            "create_only",
            "resume_completed_prefix_only",
            "planned_crash_permitted",
            "failed_attempt_overwrite_permitted",
            "result_export_permitted",
            "model_run_authorized_now",
            "server_connection_authorized_now",
        },
        "A-R5 execution policy",
    )
    if execution != {
        "create_only": True,
        "resume_completed_prefix_only": True,
        "planned_crash_permitted": True,
        "failed_attempt_overwrite_permitted": False,
        "result_export_permitted": False,
        "model_run_authorized_now": False,
        "server_connection_authorized_now": False,
    }:
        raise ContractError("A-R5 local-only/create-only policy changed")
    visual = _exact(
        protocol["visualization_contract"],
        {"roles", "must_be_scene_content", "charts_satisfy_contract"},
        "A-R5 visualization contract",
    )
    if (
        visual["roles"] != list(VISUALIZATION_ROLES)
        or visual["must_be_scene_content"] is not True
        or visual["charts_satisfy_contract"] is not False
    ):
        raise ContractError("A-R5 content visualization contract changed")
    files = protocol["wrapper_files"]
    if not isinstance(files, list) or len(files) < 6:
        raise ContractError("A-R5 wrapper file inventory is incomplete")
    seen: set[str] = set()
    for index, raw in enumerate(files):
        item = _exact(
            raw, {"relative_path", "bytes", "sha256"}, f"wrapper_files[{index}]"
        )
        relative = _relative_path(
            item["relative_path"],
            root="pose_accuracy_recovery_prep",
            label=f"wrapper_files[{index}].relative_path",
        )
        if relative.as_posix() in seen:
            raise ContractError("A-R5 wrapper file inventory contains duplicates")
        seen.add(relative.as_posix())
        _positive_int(item["bytes"], f"wrapper_files[{index}].bytes")
        _sha(item["sha256"], f"wrapper_files[{index}].sha256")
        if repository_root is not None:
            path = _resolve(repository_root, relative, f"wrapper_files[{index}]")
            if (
                not path.is_file()
                or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                raise ContractError(f"A-R5 wrapper file changed: {relative.as_posix()}")
    required_wrappers = {
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
    }
    if seen != required_wrappers:
        raise ContractError("A-R5 wrapper/dependency file inventory is not exact")
    _self_lock(protocol, "protocol_lock_sha256", "A-R5 protocol")
    return dict(protocol)


def validate_frame_manifest(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    data_root: Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    validate_protocol(protocol)
    manifest = _exact(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "role",
            "frame_size",
            "selection_provenance",
            "frame_count",
            "scene_count",
            "frames",
            "frame_manifest_lock_sha256",
        },
        "A-R5 frame manifest",
    )
    if (
        manifest["schema_version"] != FRAME_MANIFEST_SCHEMA
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["protocol_lock_sha256"] != protocol["protocol_lock_sha256"]
        or manifest["role"] != "DEVELOPMENT_ONLY_LABEL_BLIND_FRAME_INPUT"
        or manifest["frame_size"] != {"height": FRAME_HEIGHT, "width": FRAME_WIDTH}
    ):
        raise ContractError("A-R5 frame manifest identity changed")
    provenance = _exact(
        manifest["selection_provenance"],
        {
            "source_workload_sha256",
            "id_only_selection",
            "historical_object_ids_removed_before_runtime",
            "runtime_target_identity_exposed",
            "raw_sensor_depth_usage",
        },
        "frame manifest selection provenance",
    )
    if provenance != {
        "source_workload_sha256": WORKLOAD_SHA256,
        "id_only_selection": True,
        "historical_object_ids_removed_before_runtime": True,
        "runtime_target_identity_exposed": False,
        "raw_sensor_depth_usage": (
            "disk_bound_for_b2_v2_interface_not_opened_by_fastsam_or_cad_ranking"
        ),
    }:
        raise ContractError("A-R5 frame selection provenance changed")
    frames = manifest["frames"]
    if (
        not isinstance(frames, list)
        or len(frames) != len(FRAME_KEYS)
        or manifest["frame_count"] != len(FRAME_KEYS)
        or manifest["scene_count"] != 5
    ):
        raise ContractError("A-R5 frame/scene coverage changed")
    observed: list[tuple[int, int]] = []
    item_ids: set[str] = set()
    for index, raw in enumerate(frames):
        frame = _exact(raw, {"item_id", "frame_key", "inputs"}, f"frames[{index}]")
        key = _frame_key(frame["frame_key"], f"frames[{index}].frame_key")
        expected_item = f"scene-{key[0]:06d}-image-{key[1]:06d}"
        if frame["item_id"] != expected_item or frame["item_id"] in item_ids:
            raise ContractError(
                "A-R5 frame item identity is duplicate or non-canonical"
            )
        inputs = _exact(
            frame["inputs"], {"rgb", "depth", "camera"}, f"frames[{index}].inputs"
        )
        _assert_label_blind(inputs, f"frames[{index}].inputs")
        _, rgb_path = _asset(
            inputs["rgb"],
            role="rgb",
            root_name="inputs",
            label=f"frames[{index}].rgb",
            disk_root=data_root,
            verify_hashes=verify_hashes,
        )
        _, depth_path = _asset(
            inputs["depth"],
            role="raw_sensor_depth",
            root_name="inputs",
            label=f"frames[{index}].depth",
            disk_root=data_root,
            verify_hashes=verify_hashes,
        )
        _, camera_path = _asset(
            inputs["camera"],
            role="public_camera",
            root_name="inputs",
            label=f"frames[{index}].camera",
            disk_root=data_root,
            verify_hashes=verify_hashes,
        )
        if rgb_path is not None:
            _verify_rgb(rgb_path, f"frames[{index}].rgb")
        if depth_path is not None:
            _verify_depth(depth_path, f"frames[{index}].depth")
        if camera_path is not None:
            _verify_public_camera(camera_path, f"frames[{index}].camera")
        observed.append(key)
        item_ids.add(frame["item_id"])
    if observed != list(FRAME_KEYS) or len({scene for scene, _ in observed}) != 5:
        raise ContractError("A-R5 exact frozen frame order/coverage changed")
    _self_lock(manifest, "frame_manifest_lock_sha256", "A-R5 frame manifest")
    return dict(manifest)


def _validate_template_manifest(
    value: Mapping[str, Any], *, data_root: Path, verify_hashes: bool
) -> dict[str, Any]:
    manifest = _exact(
        value,
        {
            "schema_version",
            "a_r4_render_provenance",
            "frame_size",
            "object_count",
            "views_per_object",
            "objects",
            "template_manifest_lock_sha256",
        },
        "A-R5 template manifest",
    )
    if (
        manifest["schema_version"]
        != "poseloop.pose-accuracy-recovery.cnos-rgba-template-manifest.v1r5"
        or manifest["a_r4_render_provenance"] != A_R4_RENDER_PROVENANCE
        or manifest["frame_size"]
        != {"height": TEMPLATE_HEIGHT, "width": TEMPLATE_WIDTH, "mode": "RGBA"}
        or manifest["object_count"] != 5
        or manifest["views_per_object"] != 42
    ):
        raise ContractError("A-R5 template manifest provenance/shape changed")
    objects = manifest["objects"]
    if not isinstance(objects, list) or len(objects) != 5:
        raise ContractError("A-R5 template object coverage changed")
    observed: list[int] = []
    for object_index, raw in enumerate(objects):
        item = _exact(
            raw,
            {"object_id", "cad_sha256", "views"},
            f"template.objects[{object_index}]",
        )
        object_id = item["object_id"]
        _sha(item["cad_sha256"], f"template.objects[{object_index}].cad_sha256")
        views = item["views"]
        if not isinstance(views, list) or len(views) != 42:
            raise ContractError(
                "Every A-R5 catalogue object requires 42 template views"
            )
        hashes: set[str] = set()
        for view_index, raw_view in enumerate(views):
            view = _exact(
                raw_view,
                {"view_index", "rgba", "alpha_pixels", "rgb_nonzero_pixels"},
                f"template.objects[{object_index}].views[{view_index}]",
            )
            if view["view_index"] != view_index:
                raise ContractError("A-R5 template view order changed")
            asset, path = _asset(
                view["rgba"],
                role="cad_template_rgba",
                root_name="assets",
                label=f"template view {object_id}/{view_index}",
                disk_root=data_root,
                verify_hashes=verify_hashes,
            )
            alpha_pixels = _positive_int(view["alpha_pixels"], "template alpha_pixels")
            rgb_pixels = _positive_int(
                view["rgb_nonzero_pixels"], "template rgb_nonzero_pixels"
            )
            if (
                alpha_pixels >= TEMPLATE_WIDTH * TEMPLATE_HEIGHT
                or rgb_pixels > alpha_pixels
            ):
                raise ContractError("A-R5 template content statistics changed")
            if asset["sha256"] in hashes:
                raise ContractError(
                    "A-R5 object template views must have unique content"
                )
            hashes.add(asset["sha256"])
            if path is not None:
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGBA" or image.size != (
                            TEMPLATE_WIDTH,
                            TEMPLATE_HEIGHT,
                        ):
                            raise ContractError(
                                "A-R5 template must decode as 640x480 RGBA"
                            )
                        array = np.asarray(image)
                except (OSError, ValueError) as exc:
                    raise ContractError("A-R5 template PNG cannot be decoded") from exc
                alpha = array[:, :, 3] > 0
                rgb = np.any(array[:, :, :3] > 0, axis=2)
                if (
                    int(alpha.sum()) != alpha_pixels
                    or int(np.logical_and(alpha, rgb).sum()) != rgb_pixels
                ):
                    raise ContractError("A-R5 template content statistics mismatch")
        observed.append(object_id)
    if observed != list(CATALOG_OBJECT_IDS):
        raise ContractError("A-R5 template catalogue order changed")
    _self_lock(manifest, "template_manifest_lock_sha256", "A-R5 template manifest")
    return dict(manifest)


def validate_runtime_request(
    request: Mapping[str, Any],
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    data_root: Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if data_root is None:
        raise ContractError("A-R5 runtime validation requires a disk data root")
    repository_root = repository_root_from_package()
    validate_protocol(protocol, repository_root=repository_root)
    validate_frame_manifest(
        manifest, protocol, data_root=data_root, verify_hashes=verify_hashes
    )
    request = _exact(
        request,
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "frame_manifest",
            "implementation",
            "source",
            "models",
            "adapter_config",
            "a_r4_render_provenance",
            "template_manifest",
            "descriptor_generation_receipt",
            "catalog",
            "runtime",
            "boundary",
            "runtime_request_lock_sha256",
        },
        "A-R5 runtime request",
    )
    if (
        request["schema_version"] != RUNTIME_REQUEST_SCHEMA
        or request["protocol_id"] != PROTOCOL_ID
        or request["protocol_lock_sha256"] != protocol["protocol_lock_sha256"]
        or request["a_r4_render_provenance"] != A_R4_RENDER_PROVENANCE
        or request["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("A-R5 runtime request identity/boundary changed")
    _assert_label_blind(
        {
            key: request[key]
            for key in (
                "frame_manifest",
                "implementation",
                "source",
                "models",
                "adapter_config",
                "template_manifest",
                "descriptor_generation_receipt",
                "catalog",
                "runtime",
            )
        }
    )
    _, frame_path = _asset(
        request["frame_manifest"],
        role="frame_manifest",
        root_name="contracts",
        label="runtime frame manifest",
        disk_root=data_root,
        verify_hashes=verify_hashes,
    )
    if frame_path is not None:
        disk_manifest = read_json(frame_path)
        if disk_manifest != manifest:
            raise ContractError(
                "runtime frame manifest bytes decode to different content"
            )
    implementation = _exact(
        request["implementation"],
        {"commit", "tree", "source_archive", "archive_route"},
        "runtime implementation",
    )
    _git_oid(implementation["commit"], "implementation.commit")
    _git_oid(implementation["tree"], "implementation.tree")
    _, implementation_archive_path = _asset(
        implementation["source_archive"],
        role="implementation_source_archive",
        root_name="sources",
        label="implementation source archive",
        disk_root=data_root,
        verify_hashes=True,
    )
    assert implementation_archive_path is not None
    _validate_current_implementation(
        implementation,
        source_archive_path=implementation_archive_path,
        repository_root=repository_root,
    )
    source = _exact(
        request["source"],
        {
            "cnos_repository",
            "cnos_commit",
            "cnos_tree",
            "cnos_archive",
            "cnos_checkout_manifest",
            "cnos_checkout_relative_path",
            "dinov2_repository",
            "dinov2_commit",
            "dinov2_tree",
            "dinov2_archive",
            "dinov2_checkout_manifest",
            "dinov2_checkout_relative_path",
        },
        "runtime source",
    )
    if (
        source["cnos_repository"] != PINNED_CNOS_REPOSITORY
        or source["cnos_commit"] != PINNED_CNOS_COMMIT
        or source["cnos_tree"] != PINNED_CNOS_TREE
        or source["dinov2_repository"] != PINNED_DINOV2_REPOSITORY
        or source["dinov2_commit"] != PINNED_DINOV2_COMMIT
        or source["dinov2_tree"] != PINNED_DINOV2_TREE
    ):
        raise ContractError("A-R5 official CNOS/DINOv2 source route changed")
    for key in ("cnos_commit", "cnos_tree", "dinov2_commit", "dinov2_tree"):
        _git_oid(source[key], f"source.{key}")
    source_assets: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, Path] = {}
    for key, role in (
        ("cnos_archive", "cnos_source_archive"),
        ("cnos_checkout_manifest", "cnos_checkout_manifest"),
        ("dinov2_archive", "dinov2_source_archive"),
        ("dinov2_checkout_manifest", "dinov2_checkout_manifest"),
    ):
        asset, path = _asset(
            source[key],
            role=role,
            root_name="sources",
            label=f"source.{key}",
            disk_root=data_root,
            verify_hashes=True,
        )
        assert path is not None
        source_assets[key] = asset
        source_paths[key] = path
    if source_assets["cnos_archive"] != {
        "role": "cnos_source_archive",
        "relative_path": source_assets["cnos_archive"]["relative_path"],
        "bytes": PINNED_CNOS_ARCHIVE_BYTES,
        "sha256": PINNED_CNOS_ARCHIVE_SHA256,
    }:
        raise ContractError("A-R5 pinned CNOS source archive identity changed")
    checkout_paths: dict[str, Path] = {}
    for key in ("cnos_checkout_relative_path", "dinov2_checkout_relative_path"):
        relative = _relative_path(source[key], root="sources", label=f"source.{key}")
        checkout_paths[key] = _resolve(data_root, relative, f"source.{key}")
    validate_source_checkout_manifest(
        read_json(source_paths["cnos_checkout_manifest"]),
        checkout=checkout_paths["cnos_checkout_relative_path"],
        kind="CNOS",
        repository=PINNED_CNOS_REPOSITORY,
        commit=PINNED_CNOS_COMMIT,
        tree=PINNED_CNOS_TREE,
        source_archive=source_assets["cnos_archive"],
    )
    validate_source_checkout_manifest(
        read_json(source_paths["dinov2_checkout_manifest"]),
        checkout=checkout_paths["dinov2_checkout_relative_path"],
        kind="DINOV2",
        repository=PINNED_DINOV2_REPOSITORY,
        commit=PINNED_DINOV2_COMMIT,
        tree=PINNED_DINOV2_TREE,
        source_archive=source_assets["dinov2_archive"],
    )
    models = _exact(
        request["models"],
        {"fastsam_x_checkpoint", "dinov2_vitl14_checkpoint", "composite_model_sha256"},
        "runtime models",
    )
    for key, role in (
        ("fastsam_x_checkpoint", "fastsam_x_checkpoint"),
        ("dinov2_vitl14_checkpoint", "dinov2_vitl14_checkpoint"),
    ):
        _asset(
            models[key],
            role=role,
            root_name="models",
            label=f"models.{key}",
            disk_root=data_root,
            verify_hashes=True,
        )
    if (
        models["fastsam_x_checkpoint"]["sha256"]
        != "752cadc2828edb1cd4bc4f9eb587100631af06ea2108f4c9ed56df4755701e76"
    ):
        raise ContractError("A-R5 FastSAM-x checkpoint identity changed")
    if (
        models["dinov2_vitl14_checkpoint"]["bytes"] != PINNED_DINOV2_VITL14_BYTES
        or models["dinov2_vitl14_checkpoint"]["sha256"] != PINNED_DINOV2_VITL14_SHA256
    ):
        raise ContractError("A-R5 official DINOv2 ViT-L/14 checkpoint identity changed")
    expected_composite = canonical_sha256(
        {
            key: models[key]["sha256"]
            for key in ("fastsam_x_checkpoint", "dinov2_vitl14_checkpoint")
        }
    )
    if models["composite_model_sha256"] != expected_composite:
        raise ContractError("A-R5 composite model identity mismatch")
    adapter_asset, adapter_path = _asset(
        request["adapter_config"],
        role="cnos_catalog_adapter_config",
        root_name="contracts",
        label="adapter config",
        disk_root=data_root,
        verify_hashes=verify_hashes,
    )
    if adapter_path is not None:
        committed_config_path = Path(__file__).with_name(
            "cnos_catalog_config_v1r5.json"
        )
        if read_json(adapter_path) != read_json(committed_config_path) or adapter_asset[
            "sha256"
        ] != sha256_file(committed_config_path):
            raise ContractError(
                "A-R5 adapter config differs from reviewed wrapper bytes"
            )
    template_asset, template_path = _asset(
        request["template_manifest"],
        role="rgba_template_manifest",
        root_name="contracts",
        label="template manifest",
        disk_root=data_root,
        verify_hashes=verify_hashes,
    )
    descriptor_receipt, _ = _asset(
        request["descriptor_generation_receipt"],
        role="descriptor_generation_receipt",
        root_name="receipts",
        label="descriptor generation receipt",
        disk_root=data_root,
        verify_hashes=verify_hashes,
    )
    catalog = request["catalog"]
    if not isinstance(catalog, list) or len(catalog) != 5:
        raise ContractError("A-R5 runtime requires the complete five-object catalogue")
    observed: list[int] = []
    cad_hashes: dict[int, str] = {}
    for index, raw in enumerate(catalog):
        item = _exact(
            raw,
            {"object_id", "cad", "descriptor", "descriptor_metadata"},
            f"catalog[{index}]",
        )
        object_id = item["object_id"]
        cad, _ = _asset(
            item["cad"],
            role="target_cad",
            root_name="assets",
            label=f"catalog[{index}].cad",
            disk_root=data_root,
            verify_hashes=verify_hashes,
        )
        descriptor, _ = _asset(
            item["descriptor"],
            role="cad_template_descriptors",
            root_name="assets",
            label=f"catalog[{index}].descriptor",
            disk_root=data_root,
            verify_hashes=verify_hashes,
        )
        metadata = _exact(
            item["descriptor_metadata"],
            {
                "shape",
                "dtype",
                "source_template_manifest_sha256",
                "generation_receipt_sha256",
            },
            f"catalog[{index}].descriptor_metadata",
        )
        if (
            metadata["shape"] != [42, 1024]
            or metadata["dtype"] != "float32"
            or metadata["source_template_manifest_sha256"] != template_asset["sha256"]
            or metadata["generation_receipt_sha256"] != descriptor_receipt["sha256"]
        ):
            raise ContractError("A-R5 descriptor shape/provenance changed")
        observed.append(object_id)
        cad_hashes[object_id] = cad["sha256"]
        _sha(descriptor["sha256"], f"catalog[{index}].descriptor.sha256")
    if observed != list(CATALOG_OBJECT_IDS):
        raise ContractError("A-R5 catalogue order/coverage changed")
    if template_path is not None:
        template = read_json(template_path)
        validated_template = _validate_template_manifest(
            template, data_root=data_root, verify_hashes=verify_hashes
        )
        for item in validated_template["objects"]:
            if item["cad_sha256"] != cad_hashes[item["object_id"]]:
                raise ContractError("A-R5 template/CAD identity mismatch")
    runtime = _exact(
        request["runtime"],
        {
            "device",
            "proposal_chunk_size",
            "minimum_proposal_chunk_size",
            "max_oom_chunk_reductions",
        },
        "runtime settings",
    )
    if (
        not isinstance(runtime["device"], str)
        or not runtime["device"].startswith("cuda:")
        or not isinstance(runtime["proposal_chunk_size"], int)
        or not isinstance(runtime["minimum_proposal_chunk_size"], int)
        or runtime["minimum_proposal_chunk_size"] < 1
        or runtime["proposal_chunk_size"] < runtime["minimum_proposal_chunk_size"]
        or runtime["max_oom_chunk_reductions"] != 4
    ):
        raise ContractError("A-R5 runtime/OOM policy changed")
    _self_lock(request, "runtime_request_lock_sha256", "A-R5 runtime request")
    return dict(request)


def mask_statistics(mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(mask)
    if (
        values.shape != (FRAME_HEIGHT, FRAME_WIDTH)
        or values.dtype != np.bool_
        or not values.any()
    ):
        raise ContractError("A-R5 proposal mask must be non-empty bool 1080x1440")
    ys, xs = np.nonzero(values)
    return {
        "bbox_xyxy_half_open": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        "mask_pixels": int(values.sum()),
        "coverage": int(values.sum()) / (FRAME_WIDTH * FRAME_HEIGHT),
        "connected_components": _count_components_8(values),
    }


def _count_components_8(mask: np.ndarray) -> int:
    data = np.asarray(mask, dtype=bool)
    height, width = data.shape
    visited = np.zeros_like(data)
    components = 0
    for y, x in np.argwhere(data):
        if visited[y, x]:
            continue
        components += 1
        stack = [(int(y), int(x))]
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if (
                        (dy or dx)
                        and 0 <= ny < height
                        and 0 <= nx < width
                        and data[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return components


def proposal_adjacency_pairs(masks: Sequence[np.ndarray]) -> list[list[int]]:
    pairs: list[list[int]] = []
    for left in range(len(masks)):
        expanded = np.pad(np.asarray(masks[left], dtype=bool), 1)
        expanded = np.logical_or.reduce(
            [
                expanded[1 + dy : 1 + dy + FRAME_HEIGHT, 1 + dx : 1 + dx + FRAME_WIDTH]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
        )
        for right in range(left + 1, len(masks)):
            if np.logical_and(expanded, masks[right]).any():
                pairs.append([left, right])
    return pairs


def proposal_state(count: int, adjacency: Sequence[Sequence[int]]) -> str:
    if count == 0:
        return "NO_PROPOSAL"
    if count == 1:
        return "ONE_PROPOSAL"
    return "MULTIPLE_PROPOSALS_ADJACENT" if adjacency else "MULTIPLE_PROPOSALS_DISJOINT"


def _output_asset(value: Any, *, role: str, output_root: Path, label: str) -> Path:
    asset = _exact(value, {"role", "relative_path", "bytes", "sha256"}, label)
    if asset["role"] != role:
        raise ContractError(f"{label}.role changed")
    relative = _relative_path(
        asset["relative_path"],
        root="outputs",
        label=f"{label}.relative_path",
        label_blind=False,
    )
    path = _resolve(output_root, relative, label)
    if (
        not path.is_file()
        or path.stat().st_size != _positive_int(asset["bytes"], f"{label}.bytes")
        or sha256_file(path) != _sha(asset["sha256"], f"{label}.sha256")
    ):
        raise ContractError(f"{label} changed on disk")
    return path


def _read_binary_mask(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "L" or image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                raise ContractError(f"{label} must be 1440x1080 single-channel L")
            array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise ContractError(f"{label} cannot be decoded") from exc
    if set(np.unique(array).tolist()) - {0, 255}:
        raise ContractError(f"{label} must be strictly binary 0/255")
    mask = array == 255
    if not mask.any():
        raise ContractError(f"{label} must be non-empty")
    return mask


def _validate_cad_ranking(
    value: Any, catalog: Mapping[int, Mapping[str, Any]], label: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(CATALOG_OBJECT_IDS):
        raise ContractError(f"{label} must rank all five catalogue objects")
    parsed: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        item = _exact(
            raw,
            {
                "rank",
                "object_id",
                "raw_cosine",
                "normalized_similarity",
                "top5_template_cosines",
                "top5_template_indices",
                "descriptor_sha256",
            },
            f"{label}[{index}]",
        )
        raw_score = _raw_cosine(item["raw_cosine"], f"{label}[{index}].raw_cosine")
        normalized = _unit(
            item["normalized_similarity"], f"{label}[{index}].normalized_similarity"
        )
        cosines = item["top5_template_cosines"]
        indices = item["top5_template_indices"]
        if (
            item["rank"] != index + 1
            or item["object_id"] not in catalog
            or item["descriptor_sha256"]
            != catalog[item["object_id"]]["descriptor"]["sha256"]
            or not isinstance(cosines, list)
            or len(cosines) != 5
            or not isinstance(indices, list)
            or len(indices) != 5
            or len(set(indices)) != 5
            or any(
                not isinstance(item_index, int)
                or isinstance(item_index, bool)
                or not 0 <= item_index < 42
                for item_index in indices
            )
        ):
            raise ContractError(f"{label}[{index}] identity/top-5 evidence changed")
        components = [_raw_cosine(score, f"{label}[{index}].top5") for score in cosines]
        if (
            components != sorted(components, reverse=True)
            or not math.isclose(raw_score, sum(components) / 5.0, abs_tol=1e-7)
            or not math.isclose(
                normalized, normalized_cad_similarity(raw_score), abs_tol=1e-7
            )
        ):
            raise ContractError(f"{label}[{index}] raw/normalized trace changed")
        parsed.append(dict(item))
    expected = sorted(parsed, key=lambda item: (-item["raw_cosine"], item["object_id"]))
    if parsed != expected or [item["object_id"] for item in parsed] != sorted(
        catalog,
        key=lambda object_id: (
            -next(
                entry["raw_cosine"]
                for entry in parsed
                if entry["object_id"] == object_id
            ),
            object_id,
        ),
    ):
        raise ContractError(f"{label} ordering changed")
    return parsed


def validate_output_bundle(
    bundle: Mapping[str, Any],
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    validate_runtime_request(request, protocol, manifest, data_root=data_root)
    bundle = _exact(
        bundle,
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "frame_manifest_lock_sha256",
            "runtime_request_lock_sha256",
            "run_identity_sha256",
            "role",
            "status",
            "frame_count",
            "scene_count",
            "catalog_object_ids",
            "boundary",
            "frames",
            "output_bundle_lock_sha256",
        },
        "A-R5 output bundle",
    )
    if (
        bundle["schema_version"] != OUTPUT_BUNDLE_SCHEMA
        or bundle["protocol_id"] != PROTOCOL_ID
        or bundle["protocol_lock_sha256"] != protocol["protocol_lock_sha256"]
        or bundle["frame_manifest_lock_sha256"]
        != manifest["frame_manifest_lock_sha256"]
        or bundle["runtime_request_lock_sha256"]
        != request["runtime_request_lock_sha256"]
        or bundle["role"] != "DEVELOPMENT_ONLY_LABEL_BLIND_INSTANCE_PROPOSALS"
        or bundle["status"] != "COMPLETE"
        or bundle["frame_count"] != 10
        or bundle["scene_count"] != 5
        or bundle["catalog_object_ids"] != list(CATALOG_OBJECT_IDS)
        or bundle["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("A-R5 output identity/coverage/boundary changed")
    expected_run_identity = canonical_sha256(
        {
            "protocol_lock_sha256": protocol["protocol_lock_sha256"],
            "frame_manifest_lock_sha256": manifest["frame_manifest_lock_sha256"],
            "runtime_request_lock_sha256": request["runtime_request_lock_sha256"],
        }
    )
    if bundle["run_identity_sha256"] != expected_run_identity:
        raise ContractError("A-R5 run identity changed")
    frames = bundle["frames"]
    if not isinstance(frames, list) or len(frames) != 10:
        raise ContractError("A-R5 output frame count changed")
    manifest_by_item = {frame["item_id"]: frame for frame in manifest["frames"]}
    catalog = {item["object_id"]: item for item in request["catalog"]}
    for frame_index, raw in enumerate(frames):
        item = _exact(
            raw,
            {
                "item_id",
                "frame_key",
                "input_hashes",
                "input_bindings",
                "producer_identity",
                "proposal_state",
                "proposal_count",
                "adjacency_pairs",
                "proposals",
                "selected_proposal_index",
                "score_trace",
                "visualizations",
                "oom_events",
                "latency_ms",
            },
            f"output.frames[{frame_index}]",
        )
        expected_manifest = manifest["frames"][frame_index]
        if (
            item["item_id"] != expected_manifest["item_id"]
            or item["frame_key"] != expected_manifest["frame_key"]
        ):
            raise ContractError("A-R5 output frame order/identity changed")
        if item["input_hashes"] != {
            role: expected_manifest["inputs"][role]["sha256"]
            for role in ("rgb", "depth", "camera")
        }:
            raise ContractError("A-R5 output input hash binding changed")
        input_bindings = _exact(
            item["input_bindings"], {"rgb", "depth", "camera"}, "A-R5 input bindings"
        )
        for role, manifest_role, decoded_role in (
            ("rgb", "rgb", "rgb"),
            ("depth", "depth", "raw_sensor_depth"),
        ):
            manifest_asset = expected_manifest["inputs"][manifest_role]
            relative = _relative_path(
                manifest_asset["relative_path"],
                root="inputs",
                label=f"{role} input binding",
            )
            path = _resolve(data_root, relative, f"{role} input binding")
            expected_binding = decoded_image_binding(
                path,
                role=decoded_role,
                relative_path=manifest_asset["relative_path"],
            )
            if input_bindings[role] != expected_binding:
                raise ContractError(f"A-R5 {role} decoded disk binding changed")
        if input_bindings["camera"] != expected_manifest["inputs"]["camera"]:
            raise ContractError("A-R5 public camera disk binding changed")
        expected_producer_identity = {
            "implementation_commit": request["implementation"]["commit"],
            "implementation_tree": request["implementation"]["tree"],
            "implementation_source_archive_sha256": request["implementation"][
                "source_archive"
            ]["sha256"],
            "cnos_commit": request["source"]["cnos_commit"],
            "cnos_tree": request["source"]["cnos_tree"],
            "dinov2_commit": request["source"]["dinov2_commit"],
            "dinov2_tree": request["source"]["dinov2_tree"],
            "adapter_config_sha256": request["adapter_config"]["sha256"],
            "fastsam_x_checkpoint_sha256": request["models"]["fastsam_x_checkpoint"][
                "sha256"
            ],
            "dinov2_vitl14_checkpoint_sha256": request["models"][
                "dinov2_vitl14_checkpoint"
            ]["sha256"],
            "composite_model_sha256": request["models"]["composite_model_sha256"],
            "template_manifest_sha256": request["template_manifest"]["sha256"],
            "descriptor_generation_receipt_sha256": request[
                "descriptor_generation_receipt"
            ]["sha256"],
            "catalog_cad_sha256": {
                str(entry["object_id"]): entry["cad"]["sha256"]
                for entry in request["catalog"]
            },
            "catalog_descriptor_sha256": {
                str(entry["object_id"]): entry["descriptor"]["sha256"]
                for entry in request["catalog"]
            },
        }
        if item["producer_identity"] != expected_producer_identity:
            raise ContractError("A-R5 per-frame producer identity changed")
        proposals = item["proposals"]
        if not isinstance(proposals, list) or item["proposal_count"] != len(proposals):
            raise ContractError("A-R5 proposal count changed")
        masks: list[np.ndarray] = []
        indices: set[int] = set()
        for proposal_rank, raw_proposal in enumerate(proposals, start=1):
            proposal = _exact(
                raw_proposal,
                {
                    "proposal_index",
                    "proposal_rank",
                    "proposal_score",
                    "mask_stability",
                    "bbox_xyxy_half_open",
                    "mask",
                    "cad_ranking",
                    "selected_object_id",
                    "selected_raw_cosine",
                    "selected_cad_similarity",
                },
                f"output.frames[{frame_index}].proposals[{proposal_rank - 1}]",
            )
            index = proposal["proposal_index"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index in indices
                or proposal["proposal_rank"] != proposal_rank
            ):
                raise ContractError("A-R5 proposal identity/rank changed")
            indices.add(index)
            _unit(proposal["proposal_score"], "proposal score")
            _unit(proposal["mask_stability"], "mask stability")
            mask_record = _exact(
                proposal["mask"],
                {
                    "role",
                    "relative_path",
                    "bytes",
                    "sha256",
                    "decoded_shape",
                    "decoded_mode",
                    "foreground_pixels",
                    "support_bbox_xyxy_half_open",
                    "mask_pixels",
                    "coverage",
                    "connected_components",
                },
                "proposal mask",
            )
            path = _output_asset(
                {
                    key: mask_record[key]
                    for key in ("role", "relative_path", "bytes", "sha256")
                },
                role="independent_instance_mask",
                output_root=output_root,
                label="proposal mask",
            )
            mask = _read_binary_mask(path, "proposal mask")
            statistics = mask_statistics(mask)
            if (
                proposal["bbox_xyxy_half_open"] != statistics["bbox_xyxy_half_open"]
                or mask_record["decoded_shape"] != [FRAME_HEIGHT, FRAME_WIDTH]
                or mask_record["decoded_mode"] != "L"
                or mask_record["foreground_pixels"] != statistics["mask_pixels"]
                or mask_record["support_bbox_xyxy_half_open"]
                != statistics["bbox_xyxy_half_open"]
                or mask_record["mask_pixels"] != statistics["mask_pixels"]
                or not math.isclose(
                    mask_record["coverage"], statistics["coverage"], abs_tol=1e-12
                )
                or mask_record["connected_components"]
                != statistics["connected_components"]
            ):
                raise ContractError("A-R5 proposal mask statistics/bbox mismatch")
            ranking = _validate_cad_ranking(
                proposal["cad_ranking"], catalog, "proposal CAD ranking"
            )
            if (
                proposal["selected_object_id"] != ranking[0]["object_id"]
                or not math.isclose(
                    proposal["selected_raw_cosine"],
                    ranking[0]["raw_cosine"],
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    proposal["selected_cad_similarity"],
                    ranking[0]["normalized_similarity"],
                    abs_tol=1e-12,
                )
            ):
                raise ContractError("A-R5 selected object is not CAD rank 1")
            masks.append(mask)
        expected_proposal_order = sorted(
            proposals,
            key=lambda proposal: (
                -proposal["proposal_score"],
                -proposal["mask_stability"],
                proposal["proposal_index"],
            ),
        )
        if proposals != expected_proposal_order:
            raise ContractError("A-R5 FastSAM proposal order changed")
        adjacency = proposal_adjacency_pairs(masks)
        if (
            item["adjacency_pairs"] != adjacency
            or item["proposal_state"] != proposal_state(len(proposals), adjacency)
            or item["proposal_state"] not in PROPOSAL_STATES
        ):
            raise ContractError("A-R5 proposal state/adjacency changed")
        if proposals:
            selected = sorted(
                proposals,
                key=lambda proposal: (
                    -proposal["selected_cad_similarity"],
                    -proposal["proposal_score"],
                    -proposal["mask_stability"],
                    proposal["proposal_index"],
                ),
            )[0]["proposal_index"]
            if item["selected_proposal_index"] != selected:
                raise ContractError("A-R5 selected proposal ranking changed")
        elif item["selected_proposal_index"] is not None:
            raise ContractError("A-R5 zero-proposal frame must not select a proposal")
        trace_path = _output_asset(
            item["score_trace"],
            role="candidate_score_trace",
            output_root=output_root,
            label="score trace",
        )
        trace = read_json(trace_path)
        trace = _exact(
            trace,
            {
                "schema_version",
                "run_identity_sha256",
                "item_id",
                "normalization",
                "catalog_order",
                "proposals",
                "score_trace_lock_sha256",
            },
            "A-R5 score trace",
        )
        if (
            trace["schema_version"]
            != "poseloop.pose-accuracy-recovery.instance-score-trace.v1r5"
            or trace["run_identity_sha256"] != bundle["run_identity_sha256"]
            or trace["normalization"] != "raw_cosine_plus_one_divide_two_without_clamp"
            or trace["catalog_order"] != list(CATALOG_OBJECT_IDS)
            or trace["item_id"] != item["item_id"]
            or trace["proposals"]
            != [
                {
                    "proposal_index": proposal["proposal_index"],
                    "cad_ranking": proposal["cad_ranking"],
                }
                for proposal in proposals
            ]
        ):
            raise ContractError("A-R5 candidate score trace changed")
        _self_lock(trace, "score_trace_lock_sha256", "A-R5 score trace")
        visuals = _exact(
            item["visualizations"], set(VISUALIZATION_ROLES), "A-R5 visualizations"
        )
        for role in VISUALIZATION_ROLES:
            visual_path = _output_asset(
                visuals[role],
                role=role,
                output_root=output_root,
                label=f"visualization {role}",
            )
            try:
                with Image.open(visual_path) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                        raise ContractError(
                            "A-R5 visualization must be actual 1440x1080 RGB scene content"
                        )
            except (OSError, ValueError) as exc:
                raise ContractError("A-R5 visualization cannot be decoded") from exc
        if not isinstance(item["oom_events"], list) or any(
            not isinstance(event, Mapping) for event in item["oom_events"]
        ):
            raise ContractError("A-R5 OOM evidence changed")
        if (
            not isinstance(item["latency_ms"], (int, float))
            or isinstance(item["latency_ms"], bool)
            or not math.isfinite(item["latency_ms"])
            or item["latency_ms"] < 0
        ):
            raise ContractError("A-R5 latency must be finite and non-negative")
        del manifest_by_item[item["item_id"]]
    if manifest_by_item:
        raise ContractError("A-R5 output coverage is incomplete")
    _self_lock(bundle, "output_bundle_lock_sha256", "A-R5 output bundle")
    return dict(bundle)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


__all__ = [
    "A_R4_RENDER_PROVENANCE",
    "BOUNDARY_ZERO",
    "CATALOG_OBJECT_IDS",
    "FRAME_HEIGHT",
    "FRAME_KEYS",
    "FRAME_WIDTH",
    "PROPOSAL_STATES",
    "TEMPLATE_HEIGHT",
    "TEMPLATE_WIDTH",
    "VISUALIZATION_ROLES",
    "WORKLOAD_SHA256",
    "decoded_image_binding",
    "load_json_object",
    "mask_statistics",
    "normalized_cad_similarity",
    "proposal_adjacency_pairs",
    "proposal_state",
    "validate_frame_manifest",
    "validate_output_bundle",
    "validate_protocol",
    "validate_runtime_request",
]
