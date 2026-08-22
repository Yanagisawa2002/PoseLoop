"""Atomic manifest-driven state machine for the official live Python adapter."""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from .a_output import validate_internal_live_success
from .common import (
    PREP_PROTOCOL_ID,
    PrepError,
    canonical_sha256,
    is_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .live_backend import OfficialFoundationPoseBackend
from .live_contract import (
    LIVE_BACKEND_ID,
    validate_live_lock_and_ack,
)
from .manifest import (
    INPUT_ROLES,
    ITEM_ID_PATTERN,
    ITEM_KEYS,
    RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2,
    V2_VARIANT_IDS,
    V2_VARIANT_INPUT_ROLES,
    V2_BOUNDARY_FIXED,
    V2_PROVENANCE_KEYS,
    _validate_v2_asset,
    _validate_v2_intrinsics,
    asset_path,
    validate_manifest,
)
from .producer import FIXED_INFERENCE, PLANNED_CRASH_EXIT, load_protocol, utc_now


LIVE_RUN_LOCK_SCHEMA = "poseloop.r4c.prep.live-run-lock.v1"
LIVE_CHECKPOINT_SCHEMA = "poseloop.r4c.prep.live-item-checkpoint.v1"
LIVE_FAILURE_SCHEMA = "poseloop.r4c.prep.live-failure-attempt.v1"
LIVE_RECEIPT_SCHEMA = "poseloop.r4c.prep.live-run-receipt.v1"
LIVE_ITEM_FAILURE_EXIT = 76
LIVE_SMOKE_REQUEST_SCHEMA = "poseloop.r4c.prep.live-backend-smoke-request.v1"
LIVE_SMOKE_RUN_LOCK_SCHEMA = "poseloop.r4c.prep.live-backend-smoke-run-lock.v1"

BackendFactory = Callable[[Mapping[str, Any], Path], Any]


def _default_backend_factory(runtime_lock: Mapping[str, Any], output_root: Path) -> Any:
    return OfficialFoundationPoseBackend(runtime_lock, output_root=output_root)


def _success_path(root: Path, item_id: str) -> Path:
    return root / "checkpoints" / "success" / f"{item_id}.json"


def _failure_path(root: Path, item_id: str) -> tuple[Path, int]:
    directory = root / "checkpoints" / "failures" / item_id
    attempts: list[int] = []
    if directory.is_dir():
        for path in directory.glob("attempt-*.json"):
            try:
                attempts.append(int(path.stem.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    attempt = max(attempts, default=0) + 1
    return directory / f"attempt-{attempt:04d}.json", attempt


def _run_lock(
    *,
    protocol_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    items: list[dict[str, Any]],
    runtime_lock_path: Path,
    backend_ack_path: Path,
    runtime_lock: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": LIVE_RUN_LOCK_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": LIVE_BACKEND_ID,
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
        "manifest_lock_sha256": manifest["manifest_lock_sha256"],
        "runtime_lock_sha256": sha256_file(runtime_lock_path.resolve()),
        "backend_ack_sha256": sha256_file(backend_ack_path.resolve()),
        "producer_runtime_lock": runtime_lock["producer_runtime_lock"],
        "gpu": runtime_lock["gpu"],
        "inference": FIXED_INFERENCE,
        "item_ids": [item["item_id"] for item in items],
        "item_fingerprints": [item["item_fingerprint"] for item in items],
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "scorer_path_open_count": 0,
        "official_scorer_run": False,
    }


def _binding(*, run_lock_sha256: str, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": LIVE_BACKEND_ID,
        "run_lock_sha256": run_lock_sha256,
        "item_id": item["item_id"],
        "item_fingerprint": item["item_fingerprint"],
        "mask_variant_id": item["mask_variant_id"],
    }


def _load_success(
    path: Path, *, binding: Mapping[str, Any], visualization_root: Path
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != LIVE_CHECKPOINT_SCHEMA
    ):
        raise PrepError(f"Invalid live success checkpoint: {path}")
    if value.get("binding") != dict(binding):
        raise PrepError(f"Live checkpoint binding differs: {path}")
    result = value.get("result")
    if not isinstance(result, dict) or value.get("result_sha256") != canonical_sha256(
        result
    ):
        raise PrepError(f"Live checkpoint result SHA differs: {path}")
    validate_internal_live_success(result, visualization_root=visualization_root)
    return result


def _write_success(
    path: Path, *, binding: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": LIVE_CHECKPOINT_SCHEMA,
            "binding": dict(binding),
            "result_sha256": canonical_sha256(result),
            "result": dict(result),
            "access_counters": {
                "label_access_count_on_gpu_c": 0,
                "gt_path_open_count_on_gpu_c": 0,
                "evaluator_path_open_count_on_gpu_c": 0,
                "scorer_path_open_count_on_gpu_c": 0,
            },
            "official_scorer_run": False,
        },
    )


def _write_failure(
    path: Path,
    *,
    binding: Mapping[str, Any],
    attempt: int,
    exc: BaseException,
) -> dict[str, Any]:
    oom = type(exc).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"} or (
        "out of memory" in str(exc).lower()
    )
    message = " ".join(str(exc).split()) or type(exc).__name__
    failure = {
        "schema_version": LIVE_FAILURE_SCHEMA,
        "binding": dict(binding),
        "attempt": attempt,
        "status": "oom" if oom else "failed",
        "error_type": type(exc).__name__,
        "error": message[:2000],
        "traceback_tail": traceback.format_exc().splitlines()[-20:],
        "retry_disposition": "retained-not-success-cache",
        "access_counters": {
            "label_access_count_on_gpu_c": 0,
            "gt_path_open_count_on_gpu_c": 0,
            "evaluator_path_open_count_on_gpu_c": 0,
            "scorer_path_open_count_on_gpu_c": 0,
        },
        "official_scorer_run": False,
    }
    write_json_atomic(path, failure)
    return failure


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
    failure_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": LIVE_RECEIPT_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": LIVE_BACKEND_ID,
        "created_at_utc": utc_now(),
        "status": status,
        "exit_code": exit_code,
        "expected_item_count": expected,
        "cached_success_count": cached,
        "new_success_count": new,
        "written_success_count": written,
        "run_lock_sha256": sha256_file(run_lock_path),
        "results_sha256": sha256_file(results_path) if results_path else None,
        "failure_attempt_sha256": sha256_file(failure_path) if failure_path else None,
        "candidate_limit": 252,
        "pose_hypothesis_count": 252,
        "access_counters": {
            "label_access_count_on_gpu_c": 0,
            "gt_path_open_count_on_gpu_c": 0,
            "evaluator_path_open_count_on_gpu_c": 0,
            "scorer_path_open_count_on_gpu_c": 0,
        },
        "official_scorer_run": False,
    }


def run_live(
    *,
    protocol_path: Path,
    runtime_contract_path: Path,
    manifest_path: Path,
    asset_root: Path,
    output_root: Path,
    runtime_lock_path: Path,
    backend_ack_path: Path,
    resume: bool = False,
    crash_after_new_successes: int = 0,
    backend_factory: BackendFactory | None = None,
    revalidate_environment: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Run full V2 coverage; fixture/synthetic manifests are a hard failure."""

    load_protocol(protocol_path)
    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
        asset_root=asset_root,
        verify_assets=True,
    )
    if manifest.get("schema_version") != RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2:
        raise PrepError("Live producer accepts only the formal V2 manifest")
    if manifest.get("input_kind") != "DEVELOPMENT_DATA":
        raise PrepError("Live producer rejects committed synthetic fixture manifests")
    if crash_after_new_successes < 0:
        raise PrepError("crash-after-N must be non-negative")
    runtime_lock, _ = validate_live_lock_and_ack(
        protocol_path=protocol_path,
        runtime_contract_path=runtime_contract_path,
        runtime_lock_path=runtime_lock_path,
        backend_ack_path=backend_ack_path,
        revalidate_environment=revalidate_environment,
    )
    if manifest.get("producer_runtime_lock") != runtime_lock["producer_runtime_lock"]:
        raise PrepError("V2 manifest producer runtime lock differs from live preflight")

    root = output_root.resolve()
    lock_path = root / "live-run-lock.json"
    proposed_lock = _run_lock(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        manifest=manifest,
        items=items,
        runtime_lock_path=runtime_lock_path,
        backend_ack_path=backend_ack_path,
        runtime_lock=runtime_lock,
    )
    if lock_path.exists():
        if not resume:
            raise PrepError("Live output namespace exists; --resume is required")
        if read_json(lock_path) != proposed_lock:
            raise PrepError("Live resume run lock differs")
    else:
        if resume:
            raise PrepError("Cannot resume live producer: run lock is absent")
        write_json_atomic(lock_path, proposed_lock)
    run_lock_sha = sha256_file(lock_path)
    cached: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for item in items:
        binding = _binding(run_lock_sha256=run_lock_sha, item=item)
        bindings[item["item_id"]] = binding
        result = _load_success(
            _success_path(root, item["item_id"]),
            binding=binding,
            visualization_root=root,
        )
        if result is not None:
            cached[item["item_id"]] = result
    results = dict(cached)
    if len(results) == len(items):
        ordered = [results[item["item_id"]] for item in items]
        results_path = root / "results.jsonl"
        write_jsonl_atomic(results_path, ordered)
        receipt = _receipt(
            status="complete-from-cache",
            exit_code=0,
            expected=len(items),
            cached=len(cached),
            new=0,
            written=len(ordered),
            run_lock_path=lock_path,
            results_path=results_path,
        )
        write_json_atomic(root / "live-run-receipt.json", receipt)
        return 0, receipt

    factory = backend_factory or _default_backend_factory
    backend = factory(runtime_lock, root)
    new_successes = 0
    variant_priority = {
        variant_id: index for index, variant_id in enumerate(V2_VARIANT_IDS)
    }
    execution_items = sorted(
        items,
        key=lambda item: (
            variant_priority[item["mask_variant_id"]],
            int(item["ordinal"]),
        ),
    )
    for index, item in enumerate(execution_items):
        item_id = item["item_id"]
        if item_id in results:
            continue
        failure_path, attempt = _failure_path(root, item_id)
        try:
            result = backend.run_item(item, asset_root=asset_root, attempt=attempt)
            validate_internal_live_success(result, visualization_root=root)
        except Exception as exc:
            retained_failure = _write_failure(
                failure_path,
                binding=bindings[item_id],
                attempt=attempt,
                exc=exc,
            )
            partial_path = root / "results.partial.jsonl"
            if results:
                write_jsonl_atomic(
                    partial_path,
                    [
                        results[row["item_id"]]
                        for row in items
                        if row["item_id"] in results
                    ],
                )
                receipt_results: Path | None = partial_path
            else:
                receipt_results = None
            receipt = _receipt(
                status=f"item-{retained_failure['status']}",
                exit_code=LIVE_ITEM_FAILURE_EXIT,
                expected=len(items),
                cached=len(cached),
                new=new_successes,
                written=len(results),
                run_lock_path=lock_path,
                results_path=receipt_results,
                failure_path=failure_path,
            )
            write_json_atomic(root / "live-run-receipt.json", receipt)
            return LIVE_ITEM_FAILURE_EXIT, receipt
        _write_success(
            _success_path(root, item_id),
            binding=bindings[item_id],
            result=result,
        )
        results[item_id] = result
        new_successes += 1
        remaining = any(
            later["item_id"] not in results for later in execution_items[index + 1 :]
        )
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
            write_json_atomic(root / "live-run-receipt.json", receipt)
            return PLANNED_CRASH_EXIT, receipt
    ordered = [results[item["item_id"]] for item in items]
    if len(ordered) != len(items):
        raise PrepError("Live producer did not reach 100 percent manifest coverage")
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
    write_json_atomic(root / "live-run-receipt.json", receipt)
    return 0, receipt


def validate_live_smoke_request(
    *, request_path: Path, source_manifest_path: Path, asset_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one predicted-mask smoke request without claiming V2 coverage."""

    request = read_json(request_path.resolve())
    expected_keys = {
        "schema_version",
        "protocol_id",
        "purpose",
        "source_manifest_sha256",
        "producer_runtime_lock",
        "item",
        "boundary",
        "request_lock_sha256",
    }
    if not isinstance(request, dict) or set(request) != expected_keys:
        raise PrepError("Live smoke request fields differ")
    if (
        request.get("schema_version") != LIVE_SMOKE_REQUEST_SCHEMA
        or request.get("protocol_id") != PREP_PROTOCOL_ID
        or request.get("purpose") != "one-item-no-gt-adapter-smoke-not-five-variant"
    ):
        raise PrepError("Live smoke request identity differs")
    source_path = source_manifest_path.resolve()
    forbidden_source_tokens = {
        token
        for part in source_path.parts
        for token in part.lower().replace("-", "_").replace(".", "_").split("_")
        if token in {"gt", "evaluator", "evaluation", "score", "sealed"}
    }
    if forbidden_source_tokens:
        raise PrepError("Live smoke source manifest path is forbidden")
    if request.get("source_manifest_sha256") != sha256_file(source_path):
        raise PrepError("Live smoke source manifest SHA differs")
    unlocked = dict(request)
    declared_lock = unlocked.pop("request_lock_sha256")
    if not is_sha256(declared_lock) or declared_lock != canonical_sha256(unlocked):
        raise PrepError("Live smoke request lock differs")
    runtime = request.get("producer_runtime_lock")
    if not isinstance(runtime, dict) or set(runtime) != {
        "implementation_commit",
        "implementation_sha256",
        "model_sha256",
        "checkpoint_sha256",
    }:
        raise PrepError("Live smoke producer runtime lock differs")
    if not isinstance(runtime.get("checkpoint_sha256"), dict) or set(
        runtime["checkpoint_sha256"]
    ) != {"refiner", "scorer"}:
        raise PrepError("Live smoke checkpoint lock differs")
    if not all(
        is_sha256(value)
        for value in (
            runtime.get("implementation_sha256"),
            runtime.get("model_sha256"),
            runtime["checkpoint_sha256"].get("refiner"),
            runtime["checkpoint_sha256"].get("scorer"),
        )
    ):
        raise PrepError("Live smoke runtime SHA is invalid")
    commit = runtime.get("implementation_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise PrepError("Live smoke implementation commit is invalid")
    boundary = request.get("boundary")
    expected_boundary = {
        "smoke_only": True,
        "formal_five_variant_coverage_claimed": False,
        "contains_gt_derived_control_inputs": False,
        **V2_BOUNDARY_FIXED,
    }
    if boundary != expected_boundary:
        raise PrepError("Live smoke request boundary differs")
    raw = request.get("item")
    if not isinstance(raw, dict) or set(raw) != ITEM_KEYS:
        raise PrepError("Live smoke item fields differ")
    item_id = raw.get("item_id")
    if not isinstance(item_id, str) or not ITEM_ID_PATTERN.fullmatch(item_id):
        raise PrepError("Live smoke item ID is invalid")
    sample = raw.get("sample_key")
    if not isinstance(sample, dict) or set(sample) != {
        "scene_id",
        "image_id",
        "object_id",
    }:
        raise PrepError("Live smoke sample key differs")
    scene_id, image_id, object_id = (
        sample["scene_id"],
        sample["image_id"],
        sample["object_id"],
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (scene_id, image_id, object_id)
        )
        or scene_id < 0
        or image_id < 0
        or object_id <= 0
    ):
        raise PrepError("Live smoke sample key is invalid")
    if raw.get("mask_variant_id") != "predicted_mask":
        raise PrepError("Live smoke accepts only the predicted_mask input role")
    expected_item_id = (
        f"s{scene_id:06d}-i{image_id:06d}-o{object_id:06d}-predicted_mask"
    )
    if item_id != expected_item_id:
        raise PrepError("Live smoke item ID is not deterministic")
    frame_size = raw.get("frame_size")
    if not isinstance(frame_size, dict) or set(frame_size) != {"width", "height"}:
        raise PrepError("Live smoke frame size differs")
    if any(
        isinstance(frame_size[name], bool)
        or not isinstance(frame_size[name], int)
        or frame_size[name] <= 0
        for name in ("width", "height")
    ):
        raise PrepError("Live smoke frame size is invalid")
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_ROLES):
        raise PrepError("Live smoke input role coverage differs")
    assets = {slot: _validate_v2_asset(inputs[slot], slot=slot) for slot in INPUT_ROLES}
    _validate_v2_intrinsics(raw.get("camera_intrinsics"), item_id=item_id)
    depth_scale = raw.get("depth_scale")
    if (
        isinstance(depth_scale, bool)
        or not isinstance(depth_scale, (int, float))
        or not math.isfinite(float(depth_scale))
        or float(depth_scale) <= 0
    ):
        raise PrepError("Live smoke depth scale is invalid")
    provenance = raw.get("mask_provenance")
    if not isinstance(provenance, dict) or set(provenance) != V2_PROVENANCE_KEYS:
        raise PrepError("Live smoke mask provenance fields differ")
    expected_provenance = {
        "plugin_id": "manifest-mask-file-v1",
        "input_role": V2_VARIANT_INPUT_ROLES["predicted_mask"],
        "source_artifact_sha256": assets["mask"]["sha256"],
        "generator_id": "r4c-frozen-no-gt-smoke-reuse",
        "generator_version": "v1",
        "derivation_class": "predicted-segmentation",
        "development_control": False,
        "source_path_disclosed_to_gpu_c": False,
        "gpu_c_resolves_derivation": False,
        "label_access_count_on_gpu_c": 0,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise PrepError(f"Live smoke mask provenance differs: {field}")
    model_sha = provenance.get("model_sha256")
    if model_sha is not None and not is_sha256(model_sha):
        raise PrepError("Live smoke mask model SHA is invalid")
    score = provenance.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0 <= float(score) <= 1
    ):
        raise PrepError("Live smoke mask score is invalid")
    coverage = provenance.get("coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "nonzero_pixels",
        "fraction",
    }:
        raise PrepError("Live smoke mask coverage fields differ")
    nonzero = coverage.get("nonzero_pixels")
    fraction = coverage.get("fraction")
    total = int(frame_size["width"]) * int(frame_size["height"])
    if (
        isinstance(nonzero, bool)
        or not isinstance(nonzero, int)
        or not 0 < nonzero <= total
        or isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not math.isclose(float(fraction), nonzero / total, abs_tol=1e-12)
    ):
        raise PrepError("Live smoke mask coverage is invalid")
    provenance = dict(provenance)
    item = {
        "ordinal": 0,
        "item_id": item_id,
        "sample_key": dict(sample),
        "mask_variant_id": "predicted_mask",
        "frame_size": dict(frame_size),
        "inputs": assets,
        "camera_intrinsics": raw["camera_intrinsics"],
        "depth_scale": float(depth_scale),
        "mask_provenance": provenance,
    }
    item["item_fingerprint"] = canonical_sha256(item)
    for asset in assets.values():
        asset_path(asset_root, asset)
    return request, item


def run_live_smoke(
    *,
    protocol_path: Path,
    runtime_contract_path: Path,
    request_path: Path,
    source_manifest_path: Path,
    asset_root: Path,
    output_root: Path,
    runtime_lock_path: Path,
    backend_ack_path: Path,
    resume: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Run exactly one no-GT predicted-mask adapter smoke, never an A export."""

    load_protocol(protocol_path)
    runtime_lock, _ = validate_live_lock_and_ack(
        protocol_path=protocol_path,
        runtime_contract_path=runtime_contract_path,
        runtime_lock_path=runtime_lock_path,
        backend_ack_path=backend_ack_path,
        revalidate_environment=True,
    )
    request, item = validate_live_smoke_request(
        request_path=request_path,
        source_manifest_path=source_manifest_path,
        asset_root=asset_root,
    )
    if request["producer_runtime_lock"] != runtime_lock["producer_runtime_lock"]:
        raise PrepError("Live smoke request runtime lock differs from preflight")
    root = output_root.resolve()
    run_lock_path = root / "live-smoke-run-lock.json"
    run_lock = {
        "schema_version": LIVE_SMOKE_RUN_LOCK_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": LIVE_BACKEND_ID,
        "request_sha256": sha256_file(request_path.resolve()),
        "request_lock_sha256": request["request_lock_sha256"],
        "source_manifest_sha256": request["source_manifest_sha256"],
        "runtime_lock_sha256": sha256_file(runtime_lock_path.resolve()),
        "backend_ack_sha256": sha256_file(backend_ack_path.resolve()),
        "item_id": item["item_id"],
        "item_fingerprint": item["item_fingerprint"],
        "scope": "one-item-smoke-only-not-formal-five-variant-output",
        "label_access_count": 0,
        "official_scorer_run": False,
    }
    if run_lock_path.exists():
        if not resume or read_json(run_lock_path) != run_lock:
            raise PrepError("Live smoke resume lock differs")
    else:
        if resume:
            raise PrepError("Cannot resume live smoke: run lock is absent")
        write_json_atomic(run_lock_path, run_lock)
    binding = _binding(
        run_lock_sha256=sha256_file(run_lock_path),
        item=item,
    )
    cached = _load_success(
        _success_path(root, item["item_id"]),
        binding=binding,
        visualization_root=root,
    )
    if cached is None:
        failure_path, attempt = _failure_path(root, item["item_id"])
        backend = OfficialFoundationPoseBackend(runtime_lock, output_root=root)
        try:
            result = backend.run_item(item, asset_root=asset_root, attempt=attempt)
            validate_internal_live_success(result, visualization_root=root)
        except Exception as exc:
            _write_failure(
                failure_path,
                binding=binding,
                attempt=attempt,
                exc=exc,
            )
            receipt = _receipt(
                status="smoke-item-failed",
                exit_code=LIVE_ITEM_FAILURE_EXIT,
                expected=1,
                cached=0,
                new=0,
                written=0,
                run_lock_path=run_lock_path,
                results_path=None,
                failure_path=failure_path,
            )
            receipt["scope"] = run_lock["scope"]
            write_json_atomic(root / "live-smoke-receipt.json", receipt)
            return LIVE_ITEM_FAILURE_EXIT, receipt
        _write_success(
            _success_path(root, item["item_id"]), binding=binding, result=result
        )
        cached_count, new_count = 0, 1
    else:
        result = cached
        cached_count, new_count = 1, 0
    results_path = root / "smoke-results.jsonl"
    write_jsonl_atomic(results_path, [result])
    receipt = _receipt(
        status="smoke-complete-not-formal-five-variant-output",
        exit_code=0,
        expected=1,
        cached=cached_count,
        new=new_count,
        written=1,
        run_lock_path=run_lock_path,
        results_path=results_path,
    )
    receipt["scope"] = run_lock["scope"]
    receipt["a_export_permitted"] = False
    write_json_atomic(root / "live-smoke-receipt.json", receipt)
    return 0, receipt
