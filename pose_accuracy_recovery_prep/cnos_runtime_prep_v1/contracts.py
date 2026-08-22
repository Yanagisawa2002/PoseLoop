"""Fail-closed disk preflight for the pinned official CNOS runtime route."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
)
from pose_accuracy_recovery_prep.instance_proposal_v1 import (
    INPUT_MANIFEST_SCHEMA,
    PROTOCOL_ID,
    RUNTIME_LOCK_SCHEMA,
)
from pose_accuracy_recovery_prep.instance_proposal_v1.contracts import (
    FRAME_SIZE,
    validate_protocol,
    validate_runtime_lock,
)

from . import (
    DEPLOYMENT_REQUEST_SCHEMA,
    DEPLOYMENT_SCHEMA,
    RENDER_MANIFEST_SCHEMA,
    ROUTE_ID,
    ROUTE_SCHEMA,
)

PINNED_CNOS_REPOSITORY = "https://github.com/nv-nguyen/cnos"
PINNED_DINOV2_REPOSITORY = "https://github.com/facebookresearch/dinov2"
PINNED_CNOS_COMMIT = "298d1f3366171464ca271659f0e2f7a6eb8e39b4"
PINNED_CNOS_TREE = "595ba390ad1fdcd2141c8004e505b5da1eb403c9"
PINNED_BASE_COMMIT = "b5eea0522321721e08d3bef067f0f53e16b52b24"
ADAPTER_CONFIG_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-adapter-config.v1"
BOUNDARY_ZERO = {
    "label_access_count": 0,
    "gt_path_open_count": 0,
    "evaluator_path_open_count": 0,
    "sealed_split_access_count": 0,
    "official_scorer_run": False,
    "legacy_depth_prompt_read_count": 0,
    "legacy_mask_read_count": 0,
}
DEPLOYMENT_BOUNDARY_ZERO = {
    "label_access_count": 0,
    "gt_path_open_count": 0,
    "evaluator_path_open_count": 0,
    "sealed_split_access_count": 0,
    "official_scorer_run": False,
    "depth_path_open_count": 0,
    "legacy_mask_open_count": 0,
    "old_sam_checkpoint_open_count": 0,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_object(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    if set(value) != keys:
        raise ContractError(
            f"{label} keys mismatch: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git(value: Any, label: str) -> str:
    if not _is_git_object(value):
        raise ContractError(f"{label} must be a full lowercase Git object")
    return value


def _require_lock(value: Mapping[str, Any], field: str, label: str) -> str:
    lock = _require_sha256(value.get(field), f"{label}.{field}")
    unlocked = copy.deepcopy(dict(value))
    unlocked.pop(field, None)
    if canonical_sha256(unlocked) != lock:
        raise ContractError(f"{label} canonical lock mismatch")
    return lock


def _relative(value: Any, label: str, *, required_root: str | None = None) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{label} must remain relative without traversal")
    if required_root is not None and path.parts[0] != required_root:
        raise ContractError(f"{label} must remain under {required_root}/")
    return path


def _resolve(root: Path, relative: PurePosixPath, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"{label} escapes deployment root") from exc
    return candidate


def _normalized_tokens(path: PurePosixPath) -> str:
    return path.as_posix().lower().replace("-", "_").replace(".", "_")


def _reject_forbidden_path(
    path: PurePosixPath, forbidden_inputs: list[str], label: str
) -> None:
    normalized = _normalized_tokens(path)
    matches = [token for token in forbidden_inputs if token in normalized]
    if matches:
        raise ContractError(f"{label} contains forbidden input token(s): {matches}")


def _reject_forbidden_json(
    value: Any, forbidden_inputs: list[str], label: str
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower().replace("-", "_").replace(".", "_")
            matches = [token for token in forbidden_inputs if token in key_text]
            if matches:
                raise ContractError(
                    f"{label} contains forbidden JSON key token(s): {matches}"
                )
            _reject_forbidden_json(
                child, forbidden_inputs, f"{label}.{key}"
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_json(
                child, forbidden_inputs, f"{label}[{index}]"
            )
    elif isinstance(value, str):
        normalized = value.lower().replace("-", "_").replace(".", "_")
        matches = [token for token in forbidden_inputs if token in normalized]
        if matches:
            raise ContractError(
                f"{label} contains forbidden JSON value token(s): {matches}"
            )


def _bind_file(
    root: Path,
    relative_value: Any,
    *,
    label: str,
    required_root: str | None = None,
    forbidden_inputs: list[str] | None = None,
) -> dict[str, Any]:
    relative = _relative(relative_value, label, required_root=required_root)
    if forbidden_inputs is not None:
        _reject_forbidden_path(relative, forbidden_inputs, label)
    candidate = _resolve(root, relative, label)
    if not candidate.is_file():
        raise ContractError(f"Missing required file: {relative.as_posix()}")
    size = candidate.stat().st_size
    if size <= 0:
        raise ContractError(f"Required file is empty: {relative.as_posix()}")
    return {
        "relative_path": relative.as_posix(),
        "bytes": size,
        "sha256": sha256_file(candidate),
    }


def _validate_bound_file(
    root: Path,
    value: Any,
    *,
    label: str,
    required_root: str | None = None,
    forbidden_inputs: list[str] | None = None,
) -> dict[str, Any]:
    asset = _exact(value, {"relative_path", "bytes", "sha256"}, label)
    _require_sha256(asset["sha256"], f"{label}.sha256")
    if (
        not isinstance(asset["bytes"], int)
        or isinstance(asset["bytes"], bool)
        or asset["bytes"] <= 0
    ):
        raise ContractError(f"{label}.bytes must be a positive integer")
    actual = _bind_file(
        root,
        asset["relative_path"],
        label=label,
        required_root=required_root,
        forbidden_inputs=forbidden_inputs,
    )
    if dict(asset) != actual:
        raise ContractError(f"{label} disk bytes/SHA differ from frozen binding")
    return actual


def _git_output(checkout: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"Cannot inspect Git checkout {checkout}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ContractError(f"Git identity inspection failed: {detail}")
    return result.stdout.strip()


def git_identity(checkout: Path) -> dict[str, Any]:
    """Read commit/tree/clean state without modifying the checkout."""
    if not checkout.is_dir():
        raise ContractError(f"Missing source checkout: {checkout}")
    commit = _git_output(checkout, "rev-parse", "HEAD")
    tree = _git_output(checkout, "rev-parse", "HEAD^{tree}")
    status = _git_output(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    _require_git(commit, "source checkout commit")
    _require_git(tree, "source checkout tree")
    if status:
        raise ContractError(f"Source checkout is not clean: {checkout}")
    return {"commit": commit, "tree": tree, "clean": True}


def git_is_ancestor(checkout: Path, ancestor: str, descendant: str) -> bool:
    """Check the approved implementation descends from the frozen parent commit."""
    _require_git(ancestor, "implementation ancestor")
    _require_git(descendant, "implementation descendant")
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError("Cannot inspect PoseLoop implementation ancestry") from exc
    if result.returncode not in {0, 1}:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ContractError(f"PoseLoop ancestry inspection failed: {detail}")
    return result.returncode == 0


def git_remote_repository(checkout: Path) -> str:
    """Resolve origin to one canonical HTTPS repository identity."""
    value = _git_output(checkout, "remote", "get-url", "origin").strip()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if not value.startswith("https://github.com/"):
        raise ContractError("Source origin is not a canonical GitHub repository")
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _validate_adapter_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact(
        value,
        {
            "schema_version",
            "route",
            "frame_size",
            "fastsam",
            "dinov2",
            "postprocessing",
            "renderer",
            "ranking",
            "oom_policy",
        },
        "adapter_config",
    )
    if value["schema_version"] != ADAPTER_CONFIG_SCHEMA:
        raise ContractError("CNOS adapter config schema mismatch")
    if value["route"] != "OFFICIAL_CNOS_FASTSAM_DINOV2_PYRENDER":
        raise ContractError("CNOS adapter route changed")
    if value["frame_size"] != FRAME_SIZE:
        raise ContractError("CNOS adapter frame must remain 1440x1080")
    fastsam = value["fastsam"]
    if fastsam != {
        "configured_conf_threshold": 0.05,
        "effective_wrapper_conf_threshold": 0.25,
        "iou_threshold": 0.9,
        "mask_threshold": 0.5,
        "max_det": 200,
        "segmentor_width_size": 640,
        "stability_high_threshold": 0.55,
        "stability_low_threshold": 0.5,
    }:
        raise ContractError("Frozen FastSAM adapter settings changed")
    dinov2 = value["dinov2"]
    if dinov2 != {
        "descriptor_width_size": 640,
        "feature_chunk_size": 16,
        "model_name": "dinov2_vitl14",
        "normalize_features": True,
        "proposal_image_size": 224,
        "score_storage_transform": (
            "monotonic_affine_raw_cosine_plus_one_divide_two"
        ),
        "template_aggregation": "avg_top_5_cosine",
        "token_name": "x_norm_clstoken",
    }:
        raise ContractError("Frozen DINOv2 adapter settings changed")
    if value["postprocessing"] != {
        "confidence_filter": None,
        "min_box_side_relative": 0.05,
        "min_mask_area_relative": 0.0003,
        "nms": None,
        "target_candidate_policy": (
            "all_post_geometric_filter_proposals_scored_for_target_cad"
        ),
    }:
        raise ContractError("Frozen CNOS postprocessing changed")
    if value["ranking"] != {
        "order": [
            "cad_similarity_desc",
            "proposal_score_desc",
            "mask_stability_desc",
            "proposal_index_asc",
        ],
        "selection": "frozen_rank_1_for_target_object",
    }:
        raise ContractError("Frozen proposal ranking changed")
    oom = value["oom_policy"]
    if oom != {
        "formula_changes_permitted": False,
        "initial_proposal_chunk_size": 16,
        "max_chunk_reductions": 4,
        "minimum_proposal_chunk_size": 1,
        "reduction": "integer_halve_retry_same_slice",
    }:
        raise ContractError("Frozen bounded OOM policy changed")
    renderer = value["renderer"]
    if (
        renderer.get("backend") != "pyrender"
        or renderer.get("template_level") != 0
        or renderer.get("view_count_per_object") != 42
        or renderer.get("view_sampling_id") != "cnos_obj_poses_level0_all"
    ):
        raise ContractError("Frozen PyRender route changed")
    return dict(value)


def validate_route(
    route: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Validate the committed, inert route selection and source evidence."""
    _exact(
        route,
        {
            "schema_version",
            "route_id",
            "route_lock_sha256",
            "base_commit",
            "role",
            "auto_deploy",
            "execution_ready_now",
            "weights_downloaded_now",
            "parent_protocol",
            "source",
            "adapter_config",
            "runtime_route",
            "required_runtime_assets",
            "formula",
            "resume_policy",
            "forbidden_inputs",
            "boundary",
        },
        "route",
    )
    if route["schema_version"] != ROUTE_SCHEMA or route["route_id"] != ROUTE_ID:
        raise ContractError("CNOS runtime route identity mismatch")
    if route["base_commit"] != PINNED_BASE_COMMIT:
        raise ContractError("CNOS route base commit changed")
    if (
        route["role"] != "DEVELOPMENT_ONLY"
        or route["auto_deploy"] is not False
        or route["execution_ready_now"] is not False
        or route["weights_downloaded_now"] is not False
    ):
        raise ContractError("CNOS route must remain inert before asset freeze")
    parent = _exact(
        route["parent_protocol"],
        {"protocol_id", "protocol_lock_sha256", "protocol_file_sha256"},
        "route.parent_protocol",
    )
    if parent["protocol_id"] != PROTOCOL_ID:
        raise ContractError("CNOS route parent protocol changed")
    _require_sha256(parent["protocol_lock_sha256"], "parent protocol lock")
    _require_sha256(parent["protocol_file_sha256"], "parent protocol file SHA")

    source = _exact(
        route["source"],
        {"repository", "commit", "tree", "checkout_relative_path", "archive", "interface_files"},
        "route.source",
    )
    if (
        source["repository"] != PINNED_CNOS_REPOSITORY
        or source["commit"] != PINNED_CNOS_COMMIT
        or source["tree"] != PINNED_CNOS_TREE
        or source["checkout_relative_path"] != "sources/cnos"
    ):
        raise ContractError("Official CNOS source identity changed")
    _validate_bound_metadata(source["archive"], "route.source.archive")
    interfaces = source["interface_files"]
    if not isinstance(interfaces, list) or len(interfaces) != 11:
        raise ContractError("CNOS source interface lock must contain 11 files")
    interface_paths: set[str] = set()
    for index, asset in enumerate(interfaces):
        parsed = _validate_bound_metadata(asset, f"route interface[{index}]")
        relative = _relative(parsed["relative_path"], f"route interface[{index}]")
        if relative.as_posix() in interface_paths:
            raise ContractError("CNOS interface source paths must be unique")
        interface_paths.add(relative.as_posix())

    adapter = _validate_bound_metadata(route["adapter_config"], "route.adapter_config")
    required_forbidden = {
        "depth",
        "depth_component_bbox",
        "bbox_mask",
        "legacy_mask",
        "oracle",
        "ground_truth",
        "evaluator",
        "sealed",
        "sam_vit_b",
        "sam6d_pose",
    }
    forbidden = route["forbidden_inputs"]
    if (
        not isinstance(forbidden, list)
        or not all(isinstance(item, str) and item for item in forbidden)
        or not required_forbidden.issubset(forbidden)
    ):
        raise ContractError("CNOS route forbidden-input boundary is incomplete")
    if route["runtime_route"] != {
        "descriptor": "official_dinov2_vitl14_local_hub_source_and_checkpoint",
        "implementation": "official_cnos_fastsam_dinov2_pyrender",
        "pose_module": None,
        "proposal_generator": "official_fastsam_x",
        "renderer": "official_cnos_pyrender",
    }:
        raise ContractError("CNOS runtime route changed")
    if route["formula"] != {
        "cad_similarity_raw": "l2_normalized_dinov2_cls_cosine_then_avg_top_5_templates",
        "cad_similarity_stored": "monotonic_affine_raw_cosine_plus_one_divide_two",
        "mask_binarization": "resized_fastsam_probability_greater_equal_0.50",
        "mask_stability": "iou_probability_greater_equal_0.50_vs_greater_equal_0.55",
        "proposal_score": "raw_fastsam_box_confidence",
        "ranking_order": [
            "cad_similarity_desc",
            "proposal_score_desc",
            "mask_stability_desc",
            "proposal_index_asc",
        ],
        "target_candidates": (
            "all_post_geometric_filter_fastsam_proposals_scored_for_target_cad"
        ),
    }:
        raise ContractError("CNOS formula changed")
    if route["resume_policy"] != {
        "completed_item_overwrite_permitted": False,
        "formula_changes_on_resume_permitted": False,
        "item_receipt_required": True,
        "output_mode": "atomic_content_addressed_per_item",
        "planned_crash_after_items": None,
        "resume_requires_identical_deployment_and_runtime_locks": True,
    }:
        raise ContractError("CNOS resume policy changed")
    if route["boundary"] != {
        "auto_trigger_gpu_c": False,
        "depth_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "gt_path_open_count": 0,
        "label_access_count": 0,
        "legacy_mask_open_count": 0,
        "official_scorer_run": False,
        "old_sam_checkpoint_open_count": 0,
        "sealed_split_access_count": 0,
        "server_shutdown_permitted": False,
    }:
        raise ContractError("CNOS route access/lifecycle boundary changed")
    required_assets = route["required_runtime_assets"]
    expected_paths = {
        "implementation_source_archive": "sources/poseloop-cnos-runtime-prep-source.tar.gz",
        "implementation_source_checkout": "sources/poseloop",
        "fastsam_checkpoint": "models/FastSAM-x.pt",
        "dinov2_source_archive": "sources/dinov2-source.tar.gz",
        "dinov2_source_checkout": "sources/dinov2",
        "dinov2_checkpoint": "models/dinov2_vitl14_pretrain.pth",
        "render_manifest": "inputs/render/render-manifest.json",
    }
    if not isinstance(required_assets, Mapping) or set(required_assets) != set(
        expected_paths
    ):
        raise ContractError("CNOS required runtime asset slots changed")
    for name, expected_path in expected_paths.items():
        slot = _exact(
            required_assets[name], {"relative_path", "available_now"}, f"slot {name}"
        )
        if slot != {"relative_path": expected_path, "available_now": False}:
            raise ContractError(f"CNOS runtime asset slot changed: {name}")
    route_lock = _require_lock(route, "route_lock_sha256", "route")

    if repository_root is not None:
        adapter_actual = _validate_bound_file(
            repository_root, adapter, label="route.adapter_config"
        )
        adapter_value = _load_object(
            _resolve(
                repository_root,
                _relative(adapter_actual["relative_path"], "route.adapter_config"),
                "route.adapter_config",
            ),
            "adapter config",
        )
        _validate_adapter_config(adapter_value)
        protocol_path = repository_root / "protocols" / "poseloop_pose_accuracy_recovery_instance_proposal_v1.json"
        if not protocol_path.is_file() or sha256_file(protocol_path) != parent[
            "protocol_file_sha256"
        ]:
            raise ContractError("Parent protocol file SHA differs from CNOS route")
    return {
        "status": "valid",
        "route_id": ROUTE_ID,
        "route_lock_sha256": route_lock,
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "execution_ready": False,
        "weights_downloaded": False,
    }


def _validate_bound_metadata(value: Any, label: str) -> dict[str, Any]:
    asset = _exact(value, {"relative_path", "bytes", "sha256"}, label)
    _relative(asset["relative_path"], label)
    _require_sha256(asset["sha256"], f"{label}.sha256")
    if (
        not isinstance(asset["bytes"], int)
        or isinstance(asset["bytes"], bool)
        or asset["bytes"] <= 0
    ):
        raise ContractError(f"{label}.bytes must be positive")
    return dict(asset)


def _expected_samples(protocol: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    expected: dict[str, dict[str, int]] = {}
    for sample in protocol["input_lock"]["samples"]:
        expected[sample["item_id"]] = {
            name: sample[name] for name in ("scene_id", "image_id", "object_id")
        }
    return expected


def _sample_key(value: Any, label: str) -> dict[str, int]:
    key = _exact(value, {"scene_id", "image_id", "object_id"}, label)
    parsed: dict[str, int] = {}
    for name in ("scene_id", "image_id", "object_id"):
        item = key[name]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ContractError(f"{label}.{name} must be a non-negative integer")
        parsed[name] = item
    return parsed


def _validate_request(
    request: Mapping[str, Any], route: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    _exact(
        request,
        {
            "schema_version",
            "role",
            "route_lock_sha256",
            "paths",
            "implementation",
            "input_items",
            "runtime",
            "boundary",
        },
        "deployment_request",
    )
    if (
        request["schema_version"] != DEPLOYMENT_REQUEST_SCHEMA
        or request["role"] != "DEVELOPMENT_ONLY"
        or request["route_lock_sha256"] != route["route_lock_sha256"]
    ):
        raise ContractError("CNOS deployment request identity mismatch")
    if request["boundary"] != DEPLOYMENT_BOUNDARY_ZERO:
        raise ContractError("CNOS deployment request boundary must remain zero")
    paths = _exact(
        request["paths"],
        {
            "implementation_source_archive",
            "implementation_source_checkout",
            "cnos_source_archive",
            "cnos_source_checkout",
            "fastsam_checkpoint",
            "dinov2_source_archive",
            "dinov2_source_checkout",
            "dinov2_checkpoint",
            "adapter_config",
            "source_manifest",
            "workload_manifest",
            "render_manifest",
        },
        "deployment_request.paths",
    )
    expected_paths = {
        "implementation_source_archive": route["required_runtime_assets"][
            "implementation_source_archive"
        ]["relative_path"],
        "implementation_source_checkout": route["required_runtime_assets"][
            "implementation_source_checkout"
        ]["relative_path"],
        "cnos_source_archive": route["source"]["archive"]["relative_path"],
        "cnos_source_checkout": route["source"]["checkout_relative_path"],
        "fastsam_checkpoint": route["required_runtime_assets"]["fastsam_checkpoint"]["relative_path"],
        "dinov2_source_archive": route["required_runtime_assets"]["dinov2_source_archive"]["relative_path"],
        "dinov2_source_checkout": route["required_runtime_assets"]["dinov2_source_checkout"]["relative_path"],
        "dinov2_checkpoint": route["required_runtime_assets"]["dinov2_checkpoint"]["relative_path"],
        "adapter_config": "config/cnos_adapter_config_v1.json",
        "source_manifest": "inputs/manifests/source.json",
        "workload_manifest": "inputs/manifests/workload.json",
        "render_manifest": route["required_runtime_assets"]["render_manifest"]["relative_path"],
    }
    if dict(paths) != expected_paths:
        raise ContractError("CNOS deployment paths differ from frozen route")
    implementation = _exact(
        request["implementation"],
        {"approved_commit", "approved_tree"},
        "deployment_request.implementation",
    )
    _require_git(implementation["approved_commit"], "approved implementation commit")
    _require_git(implementation["approved_tree"], "approved implementation tree")
    if implementation["approved_commit"] == PINNED_BASE_COMMIT:
        raise ContractError("CNOS implementation must be a new commit after the base")
    runtime = _exact(
        request["runtime"],
        {"device", "proposal_chunk_size", "minimum_proposal_chunk_size"},
        "deployment_request.runtime",
    )
    if runtime != {
        "device": "cuda:0",
        "proposal_chunk_size": 16,
        "minimum_proposal_chunk_size": 1,
    }:
        raise ContractError("CNOS runtime execution settings changed")
    expected = _expected_samples(protocol)
    items = request["input_items"]
    if not isinstance(items, list) or len(items) != 10:
        raise ContractError("CNOS deployment request requires exact 10 input items")
    seen: set[str] = set()
    for index, item_value in enumerate(items):
        item = _exact(
            item_value,
            {
                "item_id",
                "sample_key",
                "rgb_path",
                "camera_path",
                "cad_path",
                "descriptor_path",
            },
            f"deployment_request.input_items[{index}]",
        )
        item_id = item["item_id"]
        if item_id not in expected or item_id in seen:
            raise ContractError(f"Unexpected or duplicate CNOS input item: {item_id}")
        seen.add(item_id)
        if _sample_key(item["sample_key"], f"CNOS input {item_id}") != expected[item_id]:
            raise ContractError(f"CNOS input sample key mismatch: {item_id}")
        for name in ("rgb_path", "camera_path", "cad_path", "descriptor_path"):
            relative = _relative(item[name], f"CNOS input {item_id}.{name}", required_root="inputs")
            _reject_forbidden_path(relative, route["forbidden_inputs"], f"CNOS input {item_id}.{name}")
    if seen != set(expected):
        raise ContractError("CNOS request input coverage differs from protocol")
    return {
        "paths": dict(paths),
        "implementation": dict(implementation),
        "runtime": dict(runtime),
        "items": items,
    }


def _validate_render_manifest(
    manifest: Mapping[str, Any],
    *,
    deployment_root: Path,
    protocol: Mapping[str, Any],
    adapter_config_sha256: str,
    forbidden_inputs: list[str],
) -> dict[str, Any]:
    _exact(
        manifest,
        {
            "schema_version",
            "role",
            "render_manifest_lock_sha256",
            "adapter_config_sha256",
            "renderer",
            "objects",
        },
        "render_manifest",
    )
    if (
        manifest["schema_version"] != RENDER_MANIFEST_SCHEMA
        or manifest["role"] != "DEVELOPMENT_ONLY_CAD_RENDER_TEMPLATES"
        or manifest["adapter_config_sha256"] != adapter_config_sha256
    ):
        raise ContractError("CNOS render manifest identity/config mismatch")
    if manifest["renderer"] != {
        "backend": "pyrender",
        "view_sampling_id": "cnos_obj_poses_level0_all",
        "view_count_per_object": 42,
        "template_level": 0,
    }:
        raise ContractError("CNOS render manifest settings changed")
    expected_cad = {
        item["object_id"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for item in protocol["input_lock"]["cad_assets"]
    }
    objects = manifest["objects"]
    if not isinstance(objects, list) or len(objects) != 5:
        raise ContractError("CNOS render manifest must bind exactly five objects")
    parsed: dict[int, dict[str, Any]] = {}
    cad_paths: set[str] = set()
    descriptor_paths: set[str] = set()
    for index, object_value in enumerate(objects):
        item = _exact(
            object_value,
            {"object_id", "cad", "descriptor", "template_inventory_sha256"},
            f"render_manifest.objects[{index}]",
        )
        object_id = item["object_id"]
        if (
            not isinstance(object_id, int)
            or isinstance(object_id, bool)
            or object_id not in expected_cad
            or object_id in parsed
        ):
            raise ContractError("CNOS render object identity is invalid or duplicate")
        _require_sha256(
            item["template_inventory_sha256"],
            f"render object {object_id}.template_inventory_sha256",
        )
        cad = _validate_bound_file(
            deployment_root,
            item["cad"],
            label=f"render object {object_id}.cad",
            required_root="inputs",
            forbidden_inputs=forbidden_inputs,
        )
        descriptor = _validate_bound_file(
            deployment_root,
            item["descriptor"],
            label=f"render object {object_id}.descriptor",
            required_root="inputs",
            forbidden_inputs=forbidden_inputs,
        )
        if {"bytes": cad["bytes"], "sha256": cad["sha256"]} != expected_cad[
            object_id
        ]:
            raise ContractError(f"CNOS render CAD differs from protocol: {object_id}")
        if cad["relative_path"] in cad_paths or descriptor["relative_path"] in descriptor_paths:
            raise ContractError("CNOS render CAD/descriptor paths must be unique per object")
        expected_inventory = canonical_sha256(
            {
                "object_id": object_id,
                "cad": cad,
                "descriptor": descriptor,
                "adapter_config_sha256": adapter_config_sha256,
                "renderer": manifest["renderer"],
            }
        )
        if item["template_inventory_sha256"] != expected_inventory:
            raise ContractError(
                f"CNOS render derivation inventory is not disk-bound: {object_id}"
            )
        cad_paths.add(cad["relative_path"])
        descriptor_paths.add(descriptor["relative_path"])
        parsed[object_id] = {
            "cad": cad,
            "descriptor": descriptor,
            "template_inventory_sha256": item["template_inventory_sha256"],
        }
    _require_lock(manifest, "render_manifest_lock_sha256", "render_manifest")
    return {"objects": parsed, "object_count": 5}


def freeze_deployment(
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    deployment_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute every disk identity and create additive deployment/runtime locks."""
    route_validation = validate_route(route)
    protocol_validation = validate_protocol(protocol)
    if (
        route["parent_protocol"]["protocol_lock_sha256"]
        != protocol_validation["protocol_lock_sha256"]
    ):
        raise ContractError("CNOS route does not bind this parent protocol lock")
    parsed_request = _validate_request(request, route, protocol)
    paths = parsed_request["paths"]
    forbidden = route["forbidden_inputs"]

    implementation_archive = _bind_file(
        deployment_root,
        paths["implementation_source_archive"],
        label="PoseLoop implementation source archive",
    )
    implementation_checkout_rel = _relative(
        paths["implementation_source_checkout"], "PoseLoop implementation checkout"
    )
    implementation_checkout = _resolve(
        deployment_root,
        implementation_checkout_rel,
        "PoseLoop implementation checkout",
    )
    implementation_identity = git_identity(implementation_checkout)
    if implementation_identity != {
        "commit": parsed_request["implementation"]["approved_commit"],
        "tree": parsed_request["implementation"]["approved_tree"],
        "clean": True,
    }:
        raise ContractError("PoseLoop checkout differs from the approved commit/tree")
    if not git_is_ancestor(
        implementation_checkout,
        PINNED_BASE_COMMIT,
        implementation_identity["commit"],
    ):
        raise ContractError("PoseLoop implementation does not descend from frozen base")

    cnos_archive = _bind_file(
        deployment_root, paths["cnos_source_archive"], label="CNOS source archive"
    )
    if cnos_archive != route["source"]["archive"]:
        raise ContractError("CNOS source archive differs from pinned route")
    cnos_checkout_rel = _relative(paths["cnos_source_checkout"], "CNOS checkout")
    cnos_checkout = _resolve(deployment_root, cnos_checkout_rel, "CNOS checkout")
    cnos_identity = git_identity(cnos_checkout)
    if cnos_identity != {
        "commit": route["source"]["commit"],
        "tree": route["source"]["tree"],
        "clean": True,
    }:
        raise ContractError("CNOS checkout commit/tree differs from pinned route")
    for interface in route["source"]["interface_files"]:
        relative = _relative(interface["relative_path"], "CNOS interface")
        candidate = _resolve(cnos_checkout, relative, "CNOS interface")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != interface["bytes"]
            or sha256_file(candidate) != interface["sha256"]
        ):
            raise ContractError(
                f"CNOS source interface differs: {relative.as_posix()}"
            )

    fastsam = _bind_file(
        deployment_root,
        paths["fastsam_checkpoint"],
        label="FastSAM checkpoint",
        required_root="models",
        forbidden_inputs=forbidden,
    )
    dinov2_archive = _bind_file(
        deployment_root, paths["dinov2_source_archive"], label="DINOv2 source archive"
    )
    dinov2_checkout_rel = _relative(paths["dinov2_source_checkout"], "DINOv2 checkout")
    dinov2_checkout = _resolve(deployment_root, dinov2_checkout_rel, "DINOv2 checkout")
    dinov2_identity = git_identity(dinov2_checkout)
    dinov2_repository = git_remote_repository(dinov2_checkout)
    if dinov2_repository != PINNED_DINOV2_REPOSITORY:
        raise ContractError("DINOv2 checkout is not the official repository")
    dinov2_checkpoint = _bind_file(
        deployment_root,
        paths["dinov2_checkpoint"],
        label="DINOv2 checkpoint",
        required_root="models",
        forbidden_inputs=forbidden,
    )
    adapter_config = _bind_file(
        deployment_root, paths["adapter_config"], label="CNOS adapter config"
    )
    if {
        "bytes": adapter_config["bytes"],
        "sha256": adapter_config["sha256"],
    } != {
        "bytes": route["adapter_config"]["bytes"],
        "sha256": route["adapter_config"]["sha256"],
    }:
        raise ContractError("Deployment adapter config differs from committed route")
    adapter_path = _resolve(
        deployment_root,
        _relative(adapter_config["relative_path"], "CNOS adapter config"),
        "CNOS adapter config",
    )
    adapter_value = _validate_adapter_config(_load_object(adapter_path, "adapter config"))

    source_manifest = _bind_file(
        deployment_root,
        paths["source_manifest"],
        label="source manifest",
        required_root="inputs",
        forbidden_inputs=forbidden,
    )
    workload_manifest = _bind_file(
        deployment_root,
        paths["workload_manifest"],
        label="workload manifest",
        required_root="inputs",
        forbidden_inputs=forbidden,
    )
    if workload_manifest["sha256"] != protocol["input_lock"]["workload_sha256"]:
        raise ContractError("Workload manifest SHA differs from parent protocol")
    for label, asset in (
        ("source manifest", source_manifest),
        ("workload manifest", workload_manifest),
    ):
        manifest_path = _resolve(
            deployment_root,
            _relative(asset["relative_path"], label),
            label,
        )
        manifest_value = _load_object(manifest_path, label)
        _reject_forbidden_json(manifest_value, forbidden, label)
    render_manifest_asset = _bind_file(
        deployment_root,
        paths["render_manifest"],
        label="render manifest",
        required_root="inputs",
        forbidden_inputs=forbidden,
    )
    render_manifest_path = _resolve(
        deployment_root,
        _relative(render_manifest_asset["relative_path"], "render manifest"),
        "render manifest",
    )
    render_validation = _validate_render_manifest(
        _load_object(render_manifest_path, "render manifest"),
        deployment_root=deployment_root,
        protocol=protocol,
        adapter_config_sha256=adapter_config["sha256"],
        forbidden_inputs=forbidden,
    )

    runtime_items: list[dict[str, Any]] = []
    expected_samples = _expected_samples(protocol)
    for request_item in parsed_request["items"]:
        item_id = request_item["item_id"]
        key = expected_samples[item_id]
        object_assets = render_validation["objects"][key["object_id"]]
        rgb = _bind_file(
            deployment_root,
            request_item["rgb_path"],
            label=f"{item_id}.rgb",
            required_root="inputs",
            forbidden_inputs=forbidden,
        )
        camera = _bind_file(
            deployment_root,
            request_item["camera_path"],
            label=f"{item_id}.camera",
            required_root="inputs",
            forbidden_inputs=forbidden,
        )
        camera_path = _resolve(
            deployment_root,
            _relative(camera["relative_path"], f"{item_id}.camera"),
            f"{item_id}.camera",
        )
        _reject_forbidden_json(
            _load_object(camera_path, f"{item_id}.camera"),
            forbidden,
            f"{item_id}.camera",
        )
        if request_item["cad_path"] != object_assets["cad"]["relative_path"]:
            raise ContractError(f"CNOS request CAD path swap: {item_id}")
        if request_item["descriptor_path"] != object_assets["descriptor"]["relative_path"]:
            raise ContractError(f"CNOS request descriptor path swap: {item_id}")
        runtime_items.append(
            {
                "item_id": item_id,
                "sample_key": copy.deepcopy(key),
                "inputs": {
                    "rgb": rgb,
                    "camera": camera,
                    "cad": {"object_id": key["object_id"], **object_assets["cad"]},
                    "cad_render_descriptors": {
                        "object_id": key["object_id"],
                        **object_assets["descriptor"],
                    },
                },
            }
        )
    data = {
        "schema_version": INPUT_MANIFEST_SCHEMA,
        "input_manifest_lock_sha256": "pending",
        "source_manifest": source_manifest,
        "workload_manifest": workload_manifest,
        "frame_size": copy.deepcopy(FRAME_SIZE),
        "sample_count": 10,
        "scene_ids": copy.deepcopy(protocol["input_lock"]["scene_ids"]),
        "object_ids": copy.deepcopy(protocol["input_lock"]["object_ids"]),
        "items": runtime_items,
    }
    data["input_manifest_lock_sha256"] = canonical_sha256(
        {key: value for key, value in data.items() if key != "input_manifest_lock_sha256"}
    )
    composite_checkpoint_sha = canonical_sha256(
        {"fastsam": fastsam, "dinov2": dinov2_checkpoint}
    )
    runtime_lock = {
        "schema_version": RUNTIME_LOCK_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol_validation["protocol_lock_sha256"],
        "runtime_lock_sha256": "pending",
        "execution_ready": True,
        "backend": {
            "family": "CAD_CONDITIONED_INSTANCE_PROPOSAL",
            "implementation_name": "official-cnos-fastsam-x-dinov2-vitl14",
            "source_repository": route["source"]["repository"],
            "source_commit": cnos_identity["commit"],
            "source_tree": cnos_identity["tree"],
            "source_archive_sha256": cnos_archive["sha256"],
            "checkpoint_sha256": composite_checkpoint_sha,
            "checkpoint_bytes": fastsam["bytes"] + dinov2_checkpoint["bytes"],
            "model_config_sha256": adapter_config["sha256"],
        },
        "renderer": {
            "implementation_name": "official-cnos-pyrender-dinov2-templates",
            "source_commit": cnos_identity["commit"],
            "source_tree": cnos_identity["tree"],
            "source_archive_sha256": cnos_archive["sha256"],
            "renderer_config_sha256": adapter_config["sha256"],
            "cad_render_manifest_sha256": render_manifest_asset["sha256"],
            "descriptor_model_sha256": dinov2_checkpoint["sha256"],
            "view_sampling_id": adapter_value["renderer"]["view_sampling_id"],
            "render_view_count_per_object": adapter_value["renderer"][
                "view_count_per_object"
            ],
        },
        "data": data,
        "boundary": copy.deepcopy(BOUNDARY_ZERO),
    }
    runtime_lock["runtime_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in runtime_lock.items()
            if key != "runtime_lock_sha256"
        }
    )
    validate_runtime_lock(runtime_lock, protocol, input_root=deployment_root)

    deployment = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "route_id": ROUTE_ID,
        "route_lock_sha256": route_validation["route_lock_sha256"],
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol_validation["protocol_lock_sha256"],
        "deployment_lock_sha256": "pending",
        "role": "DEVELOPMENT_ONLY_CNOS_PRODUCER",
        "state": "EXECUTION_ASSETS_HASHED_NOT_EXECUTED",
        "implementation": {
            "source_archive": implementation_archive,
            "checkout_relative_path": implementation_checkout_rel.as_posix(),
            "commit": implementation_identity["commit"],
            "tree": implementation_identity["tree"],
            "base_commit": PINNED_BASE_COMMIT,
        },
        "source": {
            "cnos_archive": cnos_archive,
            "cnos_checkout_relative_path": cnos_checkout_rel.as_posix(),
            "cnos_commit": cnos_identity["commit"],
            "cnos_tree": cnos_identity["tree"],
            "dinov2_archive": dinov2_archive,
            "dinov2_repository": dinov2_repository,
            "dinov2_checkout_relative_path": dinov2_checkout_rel.as_posix(),
            "dinov2_commit": dinov2_identity["commit"],
            "dinov2_tree": dinov2_identity["tree"],
        },
        "models": {
            "fastsam_checkpoint": fastsam,
            "dinov2_checkpoint": dinov2_checkpoint,
            "composite_checkpoint_sha256": composite_checkpoint_sha,
        },
        "adapter_config": adapter_config,
        "renderer": {
            "render_manifest": render_manifest_asset,
            "render_manifest_lock_sha256": _load_object(
                render_manifest_path, "render manifest"
            )["render_manifest_lock_sha256"],
        },
        "runtime": copy.deepcopy(parsed_request["runtime"]),
        "input_manifest_lock_sha256": data["input_manifest_lock_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "boundary": copy.deepcopy(DEPLOYMENT_BOUNDARY_ZERO),
    }
    deployment["deployment_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in deployment.items()
            if key != "deployment_lock_sha256"
        }
    )
    validate_deployment(
        deployment,
        route,
        protocol,
        runtime_lock,
        deployment_root=deployment_root,
    )
    return deployment, runtime_lock


def validate_deployment(
    deployment: Mapping[str, Any],
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    *,
    deployment_root: Path,
) -> dict[str, Any]:
    """Revalidate the frozen deployment from disk before any model import."""
    route_validation = validate_route(route)
    protocol_validation = validate_protocol(protocol)
    _exact(
        deployment,
        {
            "schema_version",
            "route_id",
            "route_lock_sha256",
            "protocol_id",
            "protocol_lock_sha256",
            "deployment_lock_sha256",
            "role",
            "state",
            "implementation",
            "source",
            "models",
            "adapter_config",
            "renderer",
            "runtime",
            "input_manifest_lock_sha256",
            "runtime_lock_sha256",
            "boundary",
        },
        "deployment",
    )
    if (
        deployment["schema_version"] != DEPLOYMENT_SCHEMA
        or deployment["route_id"] != ROUTE_ID
        or deployment["route_lock_sha256"] != route_validation["route_lock_sha256"]
        or deployment["protocol_id"] != PROTOCOL_ID
        or deployment["protocol_lock_sha256"]
        != protocol_validation["protocol_lock_sha256"]
        or deployment["role"] != "DEVELOPMENT_ONLY_CNOS_PRODUCER"
        or deployment["state"] != "EXECUTION_ASSETS_HASHED_NOT_EXECUTED"
    ):
        raise ContractError("Frozen CNOS deployment identity/state mismatch")
    if deployment["boundary"] != DEPLOYMENT_BOUNDARY_ZERO:
        raise ContractError("Frozen CNOS deployment boundary changed")
    implementation = _exact(
        deployment["implementation"],
        {
            "source_archive",
            "checkout_relative_path",
            "commit",
            "tree",
            "base_commit",
        },
        "deployment.implementation",
    )
    _validate_bound_file(
        deployment_root,
        implementation["source_archive"],
        label="deployment PoseLoop implementation archive",
    )
    implementation_checkout = _resolve(
        deployment_root,
        _relative(
            implementation["checkout_relative_path"],
            "deployment PoseLoop implementation checkout",
        ),
        "deployment PoseLoop implementation checkout",
    )
    if (
        implementation["base_commit"] != PINNED_BASE_COMMIT
        or implementation["commit"] == PINNED_BASE_COMMIT
        or git_identity(implementation_checkout)
        != {
            "commit": implementation["commit"],
            "tree": implementation["tree"],
            "clean": True,
        }
        or not git_is_ancestor(
            implementation_checkout,
            PINNED_BASE_COMMIT,
            implementation["commit"],
        )
    ):
        raise ContractError("Frozen PoseLoop implementation identity changed")
    source = _exact(
        deployment["source"],
        {
            "cnos_archive",
            "cnos_checkout_relative_path",
            "cnos_commit",
            "cnos_tree",
            "dinov2_archive",
            "dinov2_repository",
            "dinov2_checkout_relative_path",
            "dinov2_commit",
            "dinov2_tree",
        },
        "deployment.source",
    )
    _validate_bound_file(
        deployment_root, source["cnos_archive"], label="deployment CNOS archive"
    )
    if source["cnos_archive"] != route["source"]["archive"]:
        raise ContractError("Frozen CNOS archive differs from route")
    cnos_checkout = _resolve(
        deployment_root,
        _relative(source["cnos_checkout_relative_path"], "deployment CNOS checkout"),
        "deployment CNOS checkout",
    )
    if git_identity(cnos_checkout) != {
        "commit": source["cnos_commit"],
        "tree": source["cnos_tree"],
        "clean": True,
    }:
        raise ContractError("CNOS checkout changed after deployment freeze")
    for interface in route["source"]["interface_files"]:
        relative = _relative(interface["relative_path"], "CNOS interface")
        candidate = _resolve(cnos_checkout, relative, "CNOS interface")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != interface["bytes"]
            or sha256_file(candidate) != interface["sha256"]
        ):
            raise ContractError(
                f"CNOS source interface changed after freeze: {relative.as_posix()}"
            )
    _validate_bound_file(
        deployment_root, source["dinov2_archive"], label="deployment DINOv2 archive"
    )
    dinov2_checkout = _resolve(
        deployment_root,
        _relative(
            source["dinov2_checkout_relative_path"], "deployment DINOv2 checkout"
        ),
        "deployment DINOv2 checkout",
    )
    if git_identity(dinov2_checkout) != {
        "commit": source["dinov2_commit"],
        "tree": source["dinov2_tree"],
        "clean": True,
    }:
        raise ContractError("DINOv2 checkout changed after deployment freeze")
    if (
        source["dinov2_repository"] != PINNED_DINOV2_REPOSITORY
        or git_remote_repository(dinov2_checkout) != PINNED_DINOV2_REPOSITORY
    ):
        raise ContractError("Official DINOv2 repository identity changed")
    models = _exact(
        deployment["models"],
        {"fastsam_checkpoint", "dinov2_checkpoint", "composite_checkpoint_sha256"},
        "deployment.models",
    )
    fastsam = _validate_bound_file(
        deployment_root,
        models["fastsam_checkpoint"],
        label="deployment FastSAM checkpoint",
        required_root="models",
        forbidden_inputs=route["forbidden_inputs"],
    )
    dinov2_checkpoint = _validate_bound_file(
        deployment_root,
        models["dinov2_checkpoint"],
        label="deployment DINOv2 checkpoint",
        required_root="models",
        forbidden_inputs=route["forbidden_inputs"],
    )
    expected_composite = canonical_sha256(
        {"fastsam": fastsam, "dinov2": dinov2_checkpoint}
    )
    if models["composite_checkpoint_sha256"] != expected_composite:
        raise ContractError("Composite model identity changed")
    adapter = _validate_bound_file(
        deployment_root, deployment["adapter_config"], label="deployment adapter config"
    )
    adapter_path = _resolve(
        deployment_root,
        _relative(adapter["relative_path"], "deployment adapter config"),
        "deployment adapter config",
    )
    _validate_adapter_config(_load_object(adapter_path, "deployment adapter config"))
    renderer = _exact(
        deployment["renderer"],
        {"render_manifest", "render_manifest_lock_sha256"},
        "deployment.renderer",
    )
    render_asset = _validate_bound_file(
        deployment_root,
        renderer["render_manifest"],
        label="deployment render manifest",
        required_root="inputs",
        forbidden_inputs=route["forbidden_inputs"],
    )
    render_path = _resolve(
        deployment_root,
        _relative(render_asset["relative_path"], "deployment render manifest"),
        "deployment render manifest",
    )
    render_value = _load_object(render_path, "deployment render manifest")
    render_validation = _validate_render_manifest(
        render_value,
        deployment_root=deployment_root,
        protocol=protocol,
        adapter_config_sha256=adapter["sha256"],
        forbidden_inputs=route["forbidden_inputs"],
    )
    if renderer["render_manifest_lock_sha256"] != render_value[
        "render_manifest_lock_sha256"
    ]:
        raise ContractError("Render manifest lock changed")
    runtime_validation = validate_runtime_lock(
        runtime_lock, protocol, input_root=deployment_root
    )
    for label, asset in (
        ("runtime source manifest", runtime_lock["data"]["source_manifest"]),
        ("runtime workload manifest", runtime_lock["data"]["workload_manifest"]),
    ):
        path = _resolve(
            deployment_root,
            _relative(asset["relative_path"], label),
            label,
        )
        _reject_forbidden_json(
            _load_object(path, label), route["forbidden_inputs"], label
        )
    for item in runtime_lock["data"]["items"]:
        camera = item["inputs"]["camera"]
        path = _resolve(
            deployment_root,
            _relative(camera["relative_path"], f"{item['item_id']}.camera"),
            f"{item['item_id']}.camera",
        )
        _reject_forbidden_json(
            _load_object(path, f"{item['item_id']}.camera"),
            route["forbidden_inputs"],
            f"{item['item_id']}.camera",
        )
    if deployment["runtime"] != {
        "device": "cuda:0",
        "proposal_chunk_size": 16,
        "minimum_proposal_chunk_size": 1,
    }:
        raise ContractError("Frozen CNOS runtime execution settings changed")
    if (
        deployment["runtime_lock_sha256"]
        != runtime_validation["runtime_lock_sha256"]
        or deployment["input_manifest_lock_sha256"]
        != runtime_validation["input_manifest_lock_sha256"]
        or runtime_lock["backend"]["checkpoint_sha256"] != expected_composite
        or runtime_lock["renderer"]["cad_render_manifest_sha256"]
        != render_asset["sha256"]
        or runtime_lock["backend"]["source_repository"] != PINNED_CNOS_REPOSITORY
        or runtime_lock["backend"]["source_commit"] != source["cnos_commit"]
        or runtime_lock["backend"]["source_tree"] != source["cnos_tree"]
        or runtime_lock["backend"]["source_archive_sha256"]
        != source["cnos_archive"]["sha256"]
        or runtime_lock["backend"]["checkpoint_bytes"]
        != fastsam["bytes"] + dinov2_checkpoint["bytes"]
        or runtime_lock["backend"]["model_config_sha256"] != adapter["sha256"]
        or runtime_lock["renderer"]["source_commit"] != source["cnos_commit"]
        or runtime_lock["renderer"]["source_tree"] != source["cnos_tree"]
        or runtime_lock["renderer"]["source_archive_sha256"]
        != source["cnos_archive"]["sha256"]
        or runtime_lock["renderer"]["renderer_config_sha256"] != adapter["sha256"]
        or runtime_lock["renderer"]["descriptor_model_sha256"]
        != dinov2_checkpoint["sha256"]
    ):
        raise ContractError("Deployment and parent runtime locks diverged")
    deployment_lock = _require_lock(
        deployment, "deployment_lock_sha256", "deployment"
    )
    return {
        "status": "valid",
        "execution_assets_ready": True,
        "deployment_lock_sha256": deployment_lock,
        "runtime_lock_sha256": runtime_validation["runtime_lock_sha256"],
        "input_manifest_lock_sha256": runtime_validation[
            "input_manifest_lock_sha256"
        ],
        "source_commit": source["cnos_commit"],
        "source_tree": source["cnos_tree"],
        "implementation_commit": implementation["commit"],
        "implementation_tree": implementation["tree"],
        "model_identity_sha256": expected_composite,
        "object_count": render_validation["object_count"],
        "item_count": 10,
        "label_access_count": 0,
    }


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    return _load_object(path, label)
