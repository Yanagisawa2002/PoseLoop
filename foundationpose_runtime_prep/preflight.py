"""Zero-install static preflight for the future official Python runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    A_PRODUCER_OUTPUT_SCHEMA,
    PREP_PROTOCOL_ID,
    RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2,
    PrepError,
    is_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
)
from .manifest import V2_VARIANT_IDS
from .producer import load_protocol


PYTHON_RUNTIME_SCHEMA = "poseloop.r4c.prep.python-runtime-contract.v1"
PYTHON_RUNTIME_ID = "official-foundationpose-python-v1"
ISAAC_RUNTIME_SCHEMA = "poseloop.r4c.prep.isaac-ros-tensorrt-runtime-contract.v1"
ISAAC_RUNTIME_ID = "isaac-ros-foundationpose-tensorrt-cpp-v1"
BACKEND_TRANSPORT_SCHEMA = "poseloop.r4c.prep.backend-transport-contract.v2"
BACKEND_REQUEST_SCHEMA = "poseloop.r4c.prep.backend-request.v2"
BACKEND_RESPONSE_SCHEMA = "poseloop.r4c.prep.backend-response.v2"
BACKEND_REFERENCE_KEYS = {"relative_path", "schema_version", "sha256"}
BACKEND_TRANSPORT_KEYS = {
    "schema_version",
    "protocol_id",
    "formal_manifest_schema",
    "a_output_schema",
    "request_schema",
    "response_schema",
    "mask_variants",
    "request_fields",
    "response_fields_required_on_every_attempt",
    "response_fields_required_on_success",
    "a_output_fields_exact",
    "access_counter_fields_exact",
    "visualization_roles_exact",
    "top_k_count",
    "refiner_trace_state_count",
    "candidate_limit",
    "pose_hypothesis_count",
    "score_data_chunk",
    "missing_top_k_or_refiner_trace_fails",
    "candidate_top_score_may_substitute_for_top_k",
    "adapter_start_requires_contract_acknowledgement",
    "raw_gt_or_evaluator_paths_permitted",
    "nonzero_access_counter_permitted",
}
BACKEND_REQUEST_FIELDS = [
    "schema_version",
    "protocol_id",
    "producer_manifest_lock_sha256",
    "item_id",
    "item_fingerprint",
    "scene_id",
    "image_id",
    "object_id",
    "mask_variant",
    "inputs",
    "camera_intrinsics",
    "depth_scale",
    "mask_provenance",
    "inference",
]
BACKEND_EVERY_ATTEMPT_FIELDS = [
    "schema_version",
    "item_id",
    "sample_key",
    "mask_variant_id",
    "manifest_lock_sha256",
    "input_sha256",
    "implementation_commit",
    "implementation_sha256",
    "model_sha256",
    "checkpoint_sha256",
    "initial_model_to_camera_pose_m",
    "final_model_to_camera_pose_m",
    "top_k",
    "refiner_trace",
    "status",
    "attempt",
    "latency_ms",
    "failure",
    "oom",
    "failure_reason",
    "access_counters",
    "visualization_inventory",
]
BACKEND_SUCCESS_FIELDS = [
    "initial_model_to_camera_pose_m",
    "final_model_to_camera_pose_m",
    "top_k",
    "refiner_trace",
    "visualization_inventory",
    "stage_timings_ms",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
]
A_OUTPUT_FIELDS = list(BACKEND_EVERY_ATTEMPT_FIELDS)
ACCESS_COUNTER_FIELDS = [
    "label_access_count_on_gpu_c",
    "gt_path_open_count_on_gpu_c",
    "evaluator_path_open_count_on_gpu_c",
    "scorer_path_open_count_on_gpu_c",
]
VISUALIZATION_ROLES = [
    "rgb",
    "input_mask",
    "initial_pose_overlay",
    "top_k_overlay",
    "final_pose_overlay",
]


def validate_backend_transport_contract(
    runtime_contract_path: Path,
    reference: Any,
) -> dict[str, Any]:
    """Validate the exact adapter promise before either runtime may start."""

    if not isinstance(reference, dict) or set(reference) != BACKEND_REFERENCE_KEYS:
        raise PrepError("Backend transport reference fields differ")
    if reference.get("schema_version") != BACKEND_TRANSPORT_SCHEMA:
        raise PrepError("Backend transport reference schema differs")
    if not is_sha256(reference.get("sha256")):
        raise PrepError("Backend transport reference SHA-256 is invalid")
    relative = reference.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise PrepError("Backend transport relative path is invalid")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PrepError("Backend transport relative path escapes its contract root")
    root = runtime_contract_path.resolve().parent
    contract_path = (root / relative_path).resolve()
    try:
        contract_path.relative_to(root)
    except ValueError as exc:
        raise PrepError("Backend transport path escapes its contract root") from exc
    if not contract_path.is_file() or contract_path.is_symlink():
        raise PrepError("Backend transport contract is missing or is a symlink")
    if sha256_file(contract_path) != reference["sha256"]:
        raise PrepError("Backend transport contract SHA-256 differs")
    contract = read_json(contract_path)
    if not isinstance(contract, dict) or set(contract) != BACKEND_TRANSPORT_KEYS:
        raise PrepError("Backend transport contract fields differ")
    expected_identities = {
        "schema_version": BACKEND_TRANSPORT_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "formal_manifest_schema": RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2,
        "a_output_schema": A_PRODUCER_OUTPUT_SCHEMA,
        "request_schema": BACKEND_REQUEST_SCHEMA,
        "response_schema": BACKEND_RESPONSE_SCHEMA,
    }
    for field, expected in expected_identities.items():
        if contract.get(field) != expected:
            raise PrepError(f"Backend transport identity differs: {field}")
    expected_lists = {
        "mask_variants": list(V2_VARIANT_IDS),
        "request_fields": BACKEND_REQUEST_FIELDS,
        "response_fields_required_on_every_attempt": BACKEND_EVERY_ATTEMPT_FIELDS,
        "response_fields_required_on_success": BACKEND_SUCCESS_FIELDS,
        "a_output_fields_exact": A_OUTPUT_FIELDS,
        "access_counter_fields_exact": ACCESS_COUNTER_FIELDS,
        "visualization_roles_exact": VISUALIZATION_ROLES,
    }
    for field, expected in expected_lists.items():
        if contract.get(field) != expected:
            raise PrepError(f"Backend transport field promise differs: {field}")
    expected_scalars = {
        "top_k_count": 5,
        "refiner_trace_state_count": 6,
        "candidate_limit": 252,
        "pose_hypothesis_count": 252,
        "score_data_chunk": 8,
        "missing_top_k_or_refiner_trace_fails": True,
        "candidate_top_score_may_substitute_for_top_k": False,
        "adapter_start_requires_contract_acknowledgement": True,
        "raw_gt_or_evaluator_paths_permitted": False,
        "nonzero_access_counter_permitted": False,
    }
    for field, expected in expected_scalars.items():
        if contract.get(field) != expected:
            raise PrepError(f"Backend transport invariant differs: {field}")
    return contract


def validate_python_runtime_contract(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    value = read_json(resolved_path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PYTHON_RUNTIME_SCHEMA
        or value.get("runtime_id") != PYTHON_RUNTIME_ID
        or value.get("protocol_id") != PREP_PROTOCOL_ID
    ):
        raise PrepError("Python runtime contract identity is invalid")
    if value.get("auto_deploy_default") is not False:
        raise PrepError("Python runtime contract must default AUTO_DEPLOY to false")
    validate_backend_transport_contract(
        resolved_path,
        value.get("backend_transport_contract"),
    )
    source = value.get("foundationpose_source")
    if not isinstance(source, dict) or set(source) != {"repository", "commit"}:
        raise PrepError("FoundationPose source contract differs")
    if source["repository"] != "https://github.com/NVlabs/FoundationPose.git":
        raise PrepError("FoundationPose repository identity differs")
    if source["commit"] != "a1b694b83e633c2cb6115b9063d940a687759392":
        raise PrepError("FoundationPose commit identity differs")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"refiner", "scorer"}:
        raise PrepError("Python checkpoint contract differs")
    expected = {
        "refiner": (
            68220109,
            "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
        ),
        "scorer": (
            190229389,
            "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
        ),
    }
    for name, (byte_count, digest) in expected.items():
        record = checkpoints[name]
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "bytes",
            "sha256",
        }:
            raise PrepError(f"{name} checkpoint record differs")
        if record["bytes"] != byte_count or record["sha256"] != digest:
            raise PrepError(f"{name} checkpoint identity differs")
    configs = value.get("checkpoint_configs")
    if not isinstance(configs, dict) or set(configs) != {"refiner", "scorer"}:
        raise PrepError("Python checkpoint config contract differs")
    for name, digest in {
        "refiner": "28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686",
        "scorer": "a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f",
    }.items():
        if not isinstance(configs[name], dict) or not is_sha256(configs[name].get("sha256")):
            raise PrepError(f"{name} config hash is invalid")
        if configs[name]["sha256"] != digest:
            raise PrepError(f"{name} config identity differs")
    environment = value.get("environment")
    if not isinstance(environment, dict):
        raise PrepError("Python environment contract is invalid")
    if environment.get("python") != "3.12.5" or environment.get("torch") != "2.8.0+cu128":
        raise PrepError("Previously verified Python/Torch runtime identity changed")
    if value.get("production_runner_status") not in {
        "static-contract-only",
        "implemented-unverified-on-gpu",
    }:
        raise PrepError("Python production runner status is invalid")
    return value


def validate_isaac_runtime_contract(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    value = read_json(resolved_path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != ISAAC_RUNTIME_SCHEMA
        or value.get("runtime_id") != ISAAC_RUNTIME_ID
        or value.get("protocol_id") != PREP_PROTOCOL_ID
    ):
        raise PrepError("Isaac ROS/TensorRT runtime contract identity is invalid")
    if value.get("auto_deploy_default") is not False:
        raise PrepError("Isaac runtime contract must default AUTO_DEPLOY to false")
    validate_backend_transport_contract(
        resolved_path,
        value.get("backend_transport_contract"),
    )
    if value.get("lock_status") != "unresolved-no-local-or-live-runtime-inspection":
        raise PrepError("Isaac runtime lock status is invalid")
    unresolved = value.get("unresolved_required_fields")
    expected_unresolved = [
        "container_image_digest",
        "isaac_ros_release",
        "foundationpose_package_version_or_commit",
        "cuda_version",
        "tensorrt_version",
        "refiner_engine_sha256",
        "scorer_engine_sha256",
        "adapter_executable_sha256",
    ]
    if unresolved != expected_unresolved:
        raise PrepError("Isaac unresolved runtime fields differ")
    runtime_lock = value.get("runtime_lock")
    if not isinstance(runtime_lock, dict) or set(runtime_lock) != set(
        expected_unresolved
    ):
        raise PrepError("Isaac runtime lock fields differ")
    if any(runtime_lock[field] is not None for field in expected_unresolved):
        raise PrepError("Uninspected Isaac runtime identities must remain null in PREP")
    weights = value.get("source_checkpoint_identities")
    expected_weights = {
        "refiner": "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
        "scorer": "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
    }
    if not isinstance(weights, dict) or {
        name: record.get("sha256") if isinstance(record, dict) else None
        for name, record in weights.items()
    } != expected_weights:
        raise PrepError("Isaac source checkpoint identities differ")
    if value.get("production_runner_status") != "blocked-until-runtime-lock-is-frozen":
        raise PrepError("Isaac production runner status is invalid")
    return value


def validate_runtime_contract(path: Path) -> dict[str, Any]:
    value = read_json(path.resolve())
    schema = value.get("schema_version") if isinstance(value, dict) else None
    if schema == PYTHON_RUNTIME_SCHEMA:
        return validate_python_runtime_contract(path)
    if schema == ISAAC_RUNTIME_SCHEMA:
        return validate_isaac_runtime_contract(path)
    raise PrepError("Unsupported FoundationPose runtime contract schema")


def static_preflight(
    *,
    protocol_path: Path,
    runtime_contract_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    runtime = validate_runtime_contract(runtime_contract_path)
    if runtime["schema_version"] == PYTHON_RUNTIME_SCHEMA:
        blockers = [
            {
                "code": "FOUNDATIONPOSE_SOURCE_NOT_INSPECTED",
                "required": runtime["foundationpose_source"],
                "resolution": "On a future authorized GPU-C session, point preflight at an existing checkout and verify git HEAD exactly; do not clone during PREP.",
            },
            {
                "code": "CHECKPOINT_FILES_NOT_INSPECTED",
                "required": runtime["checkpoints"],
                "resolution": "On future GPU-C, hash the existing refiner/scorer files before launch; do not download during PREP.",
            },
            {
                "code": "CUDA_RUNTIME_NOT_INSPECTED",
                "required": runtime["environment"],
                "resolution": "Run the read-only live preflight only after AUTO_DEPLOY is explicitly enabled for a future session.",
            },
        ]
    else:
        blockers = [
            {
                "code": "ISAAC_RUNTIME_LOCK_UNRESOLVED",
                "required_fields": runtime["unresolved_required_fields"],
                "resolution": "In a future authorized GPU-C session, inspect the already-present image/packages/engines/executable and freeze their exact identities before launch. Do not pull or build as a fallback.",
            },
            {
                "code": "ISAAC_ADAPTER_NOT_HASH_LOCKED",
                "required": runtime["backend_transport_contract"],
                "resolution": "Provide a hash-locked adapter executable implementing the frozen request/response transport; a missing adapter is a stop condition.",
            },
        ]
    report = {
        "schema_version": "poseloop.r4c.prep.static-preflight.v1",
        "protocol_id": PREP_PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "runtime_id": runtime["runtime_id"],
        "runtime_schema": runtime["schema_version"],
        "runtime_contract_sha256": sha256_file(runtime_contract_path.resolve()),
        "backend_transport_contract_sha256": runtime[
            "backend_transport_contract"
        ]["sha256"],
        "auto_deploy": False,
        "network_accessed": False,
        "server_contacted": False,
        "downloads_or_installs": False,
        "fixture_ready": True,
        "gpu_runtime_ready": False,
        "gpu_runtime_status": "blocked-until-future-live-hash-preflight",
        "blockers": blockers,
        "frozen_inference": protocol["inference"],
        "label_access_count": 0,
        "official_scorer_run": False,
    }
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
