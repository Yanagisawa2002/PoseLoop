"""Backend-neutral producer state machine with a deterministic fixture backend."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .common import (
    CHECKPOINT_SCHEMA,
    PREP_PROTOCOL_ID,
    RESULT_SCHEMA,
    RUN_LOCK_SCHEMA,
    RUN_RECEIPT_SCHEMA,
    PrepError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .manifest import V2_VARIANT_IDS, validate_manifest


PROTOCOL_SCHEMA = "poseloop.r4c.prep.protocol.v1"
PLANNED_CRASH_EXIT = 75
FIXTURE_BACKEND_ID = "deterministic-fixture-v1"
FIXTURE_IMPLEMENTATION_COMMIT = "186efe4e72ef3ae497bff192d44b733b31b8e044"
FIXTURE_IMPLEMENTATION_SHA256 = canonical_sha256(
    {"namespace": "foundationpose_runtime_prep", "fixture_schema": "v2"}
)
FIXTURE_MODEL_SHA256 = canonical_sha256(
    {"backend": FIXTURE_BACKEND_ID, "model": "synthetic"}
)
FIXTURE_CHECKPOINTS = {
    "refiner": canonical_sha256({"fixture": "refiner"}),
    "scorer": canonical_sha256({"fixture": "scorer"}),
}
FIXTURE_CHECKPOINT_SHA256 = canonical_sha256(FIXTURE_CHECKPOINTS)
FIXED_INFERENCE = {
    "seed": 0,
    "iterations": 5,
    "candidate_limit": 252,
    "pose_hypothesis_count": 252,
    "candidate_pruning_allowed": False,
    "resource_batches": {
        "warp": 32,
        "score_data": 8,
        "score_feature": 32,
        "refine": 32,
    },
}
STAGE_NAMES = (
    "input_decode_ms",
    "hypothesis_generation_ms",
    "refine_ms",
    "score_ms",
    "selection_ms",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path.resolve())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PROTOCOL_SCHEMA
        or value.get("protocol_id") != PREP_PROTOCOL_ID
    ):
        raise PrepError("PREP protocol identity is invalid")
    mode = value.get("execution_mode")
    if not isinstance(mode, dict) or mode.get("auto_deploy_default") is not False:
        raise PrepError("PREP protocol must default AUTO_DEPLOY to false")
    if mode.get("network_allowed_during_prep") is not False:
        raise PrepError("PREP protocol unexpectedly allows network access")
    if value.get("inference") != FIXED_INFERENCE:
        raise PrepError("PREP protocol changed the frozen c252/chunk contract")
    input_contract = value.get("input_contract")
    if not isinstance(input_contract, dict) or input_contract != {
        "formal_schema_version": "poseloop.r4c.prep.runtime-isolated-manifest.v2",
        "legacy_fixture_schema_version": "poseloop.r4c.prep.label-free-manifest.v1",
        "required_roles": [
            "public_rgb",
            "public_depth",
            "input_mask",
            "public_camera",
            "public_cad",
        ],
        "mask_variant_ids": list(V2_VARIANT_IDS),
        "gt_derived_control_semantics": (
            "A-side DEVELOPMENT controls may be GT-derived, but only a hash-locked "
            "opaque mask asset crosses into GPU-C; derivation and raw paths remain "
            "unavailable to GPU-C."
        ),
        "contains_gt_derived_control_inputs_required": True,
        "contains_raw_gt_paths_required": False,
        "gpu_c_resolves_derivation_required": False,
        "label_access_count_on_gpu_c_required": 0,
        "dataset_scan_allowed": False,
        "exact_path_hash_and_byte_match_required": True,
        "variant_base_sample_sets_must_match": True,
        "required_coverage_percent": 100,
    }:
        raise PrepError("PREP formal V2 input contract differs")
    if value.get("producer_output_contract") != {
        "schema_version": "poseloop.pose-accuracy-recovery.producer-output.v1",
        "backend_transport_schema": "poseloop.r4c.prep.backend-transport-contract.v2",
        "top_k_count": 5,
        "refiner_trace_state_count": 6,
        "candidate_top_score_may_substitute_for_top_k": False,
        "self_validation_required": True,
        "coverage_percent_required": 100,
    }:
        raise PrepError("PREP GPU-A producer-output contract differs")
    profiling = value.get("profiling")
    if not isinstance(profiling, dict) or profiling.get("resolution") != {
        "width": 1280,
        "height": 720,
    }:
        raise PrepError("PREP profiling resolution is not frozen at 720p")
    if profiling.get("external_reference_target_ms") != 89.0:
        raise PrepError("PREP external reference target changed")
    if profiling.get("external_reference_comparable") is not False:
        raise PrepError("PREP external reference must not be declared comparable")
    equivalence = value.get("equivalence")
    if not isinstance(equivalence, dict) or equivalence != {
        "promotion_requires_bitwise_prediction_equality": True,
        "numeric_tolerance_is_diagnostic_only": True,
        "diagnostic_numeric_tolerance": {
            "pose_abs": 1e-7,
            "score_abs": 1e-8,
            "margin_abs": 1e-8,
        },
        "missing_extra_duplicate_or_pruned_candidate_fails": True,
    }:
        raise PrepError("PREP exact-equivalence contract differs")
    visualization = value.get("visualization")
    if not isinstance(visualization, dict) or visualization != {
        "layout": "baseline-left-improved-right",
        "layers": [
            "public_rgb",
            "input_mask",
            "public_cad_wireframe",
            "predicted_pose_axes",
        ],
        "cad_scale_metres": 0.001,
        "video": {"codec": "mp4v", "fps": 2.0},
        "pixel_equality_claim": "no-regression-only-not-accuracy",
    }:
        raise PrepError("PREP visualization contract differs")
    return value


def _checkpoint_path(root: Path, item_id: str) -> Path:
    return root.resolve() / "checkpoints" / "success" / f"{item_id}.json"


def _run_lock(
    *,
    protocol_path: Path,
    manifest_path: Path,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_LOCK_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
        "backend_id": FIXTURE_BACKEND_ID,
        "inference": FIXED_INFERENCE,
        "item_ids": [item["item_id"] for item in items],
        "item_fingerprints": [item["item_fingerprint"] for item in items],
        "label_access_count": 0,
        "official_scorer_run": False,
    }


def _binding(
    *,
    run_lock_sha256: str,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": FIXTURE_BACKEND_ID,
        "run_lock_sha256": run_lock_sha256,
        "item_id": item["item_id"],
        "item_fingerprint": item["item_fingerprint"],
        "mask_variant_id": item["mask_variant_id"],
    }


def _load_checkpoint(
    path: Path,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise PrepError(f"Invalid PREP checkpoint schema: {path}")
    if value.get("binding") != dict(binding):
        raise PrepError(f"PREP checkpoint binding differs: {path}")
    result = value.get("result")
    if not isinstance(result, dict) or result.get("status") != "success":
        raise PrepError(f"PREP checkpoint is not successful: {path}")
    if value.get("result_sha256") != canonical_sha256(result):
        raise PrepError(f"PREP checkpoint result hash differs: {path}")
    return result


def _write_checkpoint(
    path: Path,
    *,
    binding: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "binding": dict(binding),
            "result_sha256": canonical_sha256(result),
            "result": dict(result),
            "label_access_count": 0,
            "official_scorer_run": False,
        },
    )


def fixture_result(
    item: Mapping[str, Any],
    *,
    runtime_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one deterministic c252 result without reading an asset or label."""

    if runtime_lock is None:
        runtime_identity = {
            "implementation_commit": FIXTURE_IMPLEMENTATION_COMMIT,
            "implementation_sha256": FIXTURE_IMPLEMENTATION_SHA256,
            "model_sha256": FIXTURE_MODEL_SHA256,
            "checkpoint_sha256": FIXTURE_CHECKPOINT_SHA256,
        }
    else:
        checkpoints = runtime_lock.get("checkpoint_sha256")
        if not isinstance(checkpoints, dict):
            raise PrepError("Fixture runtime checkpoint lock is invalid")
        runtime_identity = {
            "implementation_commit": runtime_lock.get("implementation_commit"),
            "implementation_sha256": runtime_lock.get("implementation_sha256"),
            "model_sha256": runtime_lock.get("model_sha256"),
            "checkpoint_sha256": canonical_sha256(checkpoints),
        }

    fingerprint = str(item["item_fingerprint"])
    integer = int(fingerprint[:16], 16)
    offset = (integer % 10000) / 10_000_000.0
    final_pose = [
        [1.0, 0.0, 0.0, offset],
        [0.0, 1.0, 0.0, offset / 2.0],
        [0.0, 0.0, 1.0, 0.8 + offset],
        [0.0, 0.0, 0.0, 1.0],
    ]
    initial_pose = [list(row) for row in final_pose]
    initial_pose[0][3] = float(final_pose[0][3]) - 0.005
    top_k: list[dict[str, Any]] = []
    score = 0.5 + ((integer >> 8) % 4000) / 10000.0
    margin = 0.01 + ((integer >> 20) % 1000) / 100000.0
    for rank in range(1, 6):
        hypothesis_pose = [list(row) for row in initial_pose]
        hypothesis_pose[1][3] = float(initial_pose[1][3]) + (rank - 1) * 0.001
        hypothesis_score = score if rank == 1 else score - margin - (rank - 2) * 0.01
        top_k.append(
            {
                "rank": rank,
                "candidate_id": f"synthetic-fixture-{fingerprint[:16]}-h{rank - 1:03d}",
                "score": hypothesis_score,
                "model_to_camera_pose_m": hypothesis_pose,
            }
        )
    trace: list[dict[str, Any]] = []
    for step in range(int(FIXED_INFERENCE["iterations"]) + 1):
        if step == 0:
            state = [list(row) for row in initial_pose]
        elif step == int(FIXED_INFERENCE["iterations"]):
            state = [list(row) for row in final_pose]
        else:
            fraction = step / float(FIXED_INFERENCE["iterations"])
            state = [list(row) for row in initial_pose]
            for axis in range(3):
                state[axis][3] = float(initial_pose[axis][3]) + fraction * (
                    float(final_pose[axis][3]) - float(initial_pose[axis][3])
                )
        trace.append(
            {
                "iteration": step,
                "model_to_camera_pose_m": state,
                "objective": score - 0.01 * (int(FIXED_INFERENCE["iterations"]) - step),
            }
        )
    jitter = (integer % 17) / 100.0
    stages = {
        "input_decode_ms": 1.0 + jitter,
        "hypothesis_generation_ms": 6.0 + jitter,
        "refine_ms": 18.0 + jitter,
        "score_ms": 9.0 + jitter,
        "selection_ms": 0.5 + jitter,
    }
    total = sum(stages.values())
    sample = item["sample_key"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "record_type": "poseloop_r4c_prep_result",
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": FIXTURE_BACKEND_ID,
        "item_id": item["item_id"],
        "sample_key": dict(sample),
        "mask_variant_id": item["mask_variant_id"],
        "status": "success",
        "predicted_model_to_camera_pose_m": final_pose,
        "initial_model_to_camera_pose_m": initial_pose,
        "final_model_to_camera_pose_m": final_pose,
        "top_k": top_k,
        "refiner_trace_model_to_camera_m": trace,
        **runtime_identity,
        "foundationpose_top_score": score,
        "foundationpose_top_score_margin": margin,
        "selected_candidate_index": 0,
        "candidate_limit": 252,
        "pose_hypothesis_count": 252,
        "resource_batches": dict(FIXED_INFERENCE["resource_batches"]),
        "stage_timings_ms": stages,
        "total_latency_ms": total,
        "cuda_peak_allocated_bytes": 512 * 1024 * 1024 + integer % 4096,
        "cuda_peak_reserved_bytes": 640 * 1024 * 1024 + integer % 4096,
        "input_mask_sha256": item["inputs"]["mask"]["sha256"],
        "mask_provenance": dict(item["mask_provenance"]),
        "evaluator_label_read": False,
        "contains_gt_derived_control_input": (
            item["mask_provenance"].get("derivation_class")
            == "gt-derived-development-control"
        ),
        "uses_oracle_association": False,
        "attempt": 1,
        "failure_reason": None,
        "oom": False,
        "synthetic_backend": True,
        "producer_runtime_isolated": True,
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "scorer_path_open_count": 0,
        "official_scorer_run_count": 0,
        "official_scorer_run": False,
        "accuracy_claim": "unavailable-fixture-runtime-isolated",
    }
    result["prediction_fingerprint"] = canonical_sha256(
        {
            "pose": final_pose,
            "top_score": score,
            "margin": margin,
            "selected_candidate_index": 0,
            "candidate_count": 252,
        }
    )
    return result


def _receipt(
    *,
    status: str,
    exit_code: int,
    expected: int,
    cached: int,
    new: int,
    written: int,
    run_lock_path: Path,
    results_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": FIXTURE_BACKEND_ID,
        "created_at_utc": utc_now(),
        "status": status,
        "exit_code": exit_code,
        "expected_item_count": expected,
        "cached_success_count": cached,
        "new_success_count": new,
        "written_item_count": written,
        "run_lock_sha256": sha256_file(run_lock_path),
        "results_sha256": sha256_file(results_path) if results_path else None,
        "candidate_limit": 252,
        "pose_hypothesis_count": 252,
        "label_access_count": 0,
        "official_scorer_run": False,
        "auto_deploy": False,
    }


def run_fixture(
    *,
    protocol_path: Path,
    manifest_path: Path,
    output_root: Path,
    asset_root: Path | None = None,
    verify_assets: bool = False,
    resume: bool = False,
    crash_after_new_successes: int = 0,
) -> tuple[int, dict[str, Any]]:
    load_protocol(protocol_path)
    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
        asset_root=asset_root,
        verify_assets=verify_assets,
    )
    if crash_after_new_successes < 0:
        raise PrepError("crash-after-N must be non-negative")
    root = output_root.resolve()
    lock_path = root / "run-lock.json"
    proposed_lock = _run_lock(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        items=items,
    )
    if lock_path.exists():
        if not resume:
            raise PrepError("Output namespace already exists; --resume is required")
        if read_json(lock_path) != proposed_lock:
            raise PrepError("Resume run lock differs from the frozen fixture run")
    else:
        if resume:
            raise PrepError("Cannot resume: run lock is absent")
        write_json_atomic(lock_path, proposed_lock)
    lock_sha = sha256_file(lock_path)
    cached: dict[str, dict[str, Any]] = {}
    for item in items:
        binding = _binding(run_lock_sha256=lock_sha, item=item)
        result = _load_checkpoint(
            _checkpoint_path(root, item["item_id"]),
            binding=binding,
        )
        if result is not None:
            cached[item["item_id"]] = result
    results = dict(cached)
    new_successes = 0
    for index, item in enumerate(items):
        item_id = item["item_id"]
        if item_id in results:
            continue
        result = fixture_result(
            item,
            runtime_lock=manifest.get("producer_runtime_lock"),
        )
        binding = _binding(run_lock_sha256=lock_sha, item=item)
        _write_checkpoint(
            _checkpoint_path(root, item_id),
            binding=binding,
            result=result,
        )
        results[item_id] = result
        new_successes += 1
        remaining = any(later["item_id"] not in results for later in items[index + 1 :])
        if (
            crash_after_new_successes
            and new_successes >= crash_after_new_successes
            and remaining
        ):
            partial_path = root / "results.partial.jsonl"
            write_jsonl_atomic(
                partial_path,
                [results[row["item_id"]] for row in items if row["item_id"] in results],
            )
            receipt = _receipt(
                status="planned-crash-after-checkpoint",
                exit_code=PLANNED_CRASH_EXIT,
                expected=len(items),
                cached=len(cached),
                new=new_successes,
                written=len(results),
                run_lock_path=lock_path,
                results_path=partial_path,
            )
            write_json_atomic(root / "run-receipt.json", receipt)
            return PLANNED_CRASH_EXIT, receipt
    ordered = [results[item["item_id"]] for item in items]
    if len(ordered) != len(items):
        raise PrepError("Fixture producer did not reach 100 percent coverage")
    results_path = root / "results.jsonl"
    write_jsonl_atomic(results_path, ordered)
    receipt = _receipt(
        status="complete",
        exit_code=0,
        expected=len(items),
        cached=len(cached),
        new=new_successes,
        written=len(ordered),
        run_lock_path=lock_path,
        results_path=results_path,
    )
    write_json_atomic(root / "run-receipt.json", receipt)
    return 0, receipt
