from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from foundationpose_runtime_prep.a_output import (
    export_a_live_results,
    export_a_results,
    validate_a_results,
    validate_internal_live_success,
)
from foundationpose_runtime_prep.common import (
    PREP_PROTOCOL_ID,
    PrepError,
    canonical_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from foundationpose_runtime_prep.equivalence import compare_exact
from foundationpose_runtime_prep.fixture import build_fixture
from foundationpose_runtime_prep.manifest import (
    BOUNDARY,
    expected_coverage,
    validate_manifest,
)
from foundationpose_runtime_prep.live_backend import _pose_list
from foundationpose_runtime_prep.live_contract import (
    ACK_COMMITMENTS,
    IMPLEMENTATION_FILES,
    LIVE_BACKEND_ACK_SCHEMA,
    LIVE_BACKEND_ID,
    LIVE_RUNTIME_LOCK_SCHEMA,
    REUSED_RUNTIME_SOURCES,
)
from foundationpose_runtime_prep.live_producer import (
    LIVE_SMOKE_REQUEST_SCHEMA,
    run_live,
    validate_live_smoke_request,
)
from foundationpose_runtime_prep.preflight import static_preflight
from foundationpose_runtime_prep.producer import (
    FIXED_INFERENCE,
    PLANNED_CRASH_EXIT,
    fixture_result,
    run_fixture,
)
from foundationpose_runtime_prep.profile import fixture_samples, summarize_profile
from foundationpose_runtime_prep.visualization import (
    build_visualization_plan,
    render_visualization,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "foundationpose_runtime_prep"
    / "contracts"
    / "producer_protocol_v1.json"
)
PYTHON_RUNTIME = (
    REPO_ROOT
    / "foundationpose_runtime_prep"
    / "contracts"
    / "python_runtime_contract_v1.json"
)
ISAAC_RUNTIME = (
    REPO_ROOT
    / "foundationpose_runtime_prep"
    / "contracts"
    / "isaac_ros_tensorrt_runtime_contract_v1.json"
)


def test_pose_list_moves_cuda_like_tensor_to_cpu_before_numpy() -> None:
    import numpy as np

    calls: list[str] = []

    class CudaLikeTensor:
        def detach(self) -> CudaLikeTensor:
            calls.append("detach")
            return self

        def cpu(self) -> CudaLikeTensor:
            calls.append("cpu")
            return self

        def numpy(self) -> object:
            calls.append("numpy")
            return np.eye(4, dtype=np.float32)

    pose = _pose_list(CudaLikeTensor(), np)

    assert calls == ["detach", "cpu", "numpy"]
    assert pose == np.eye(4, dtype=np.float64).tolist()


def _refresh_v2_lock(value: dict[str, object]) -> None:
    unlocked = dict(value)
    unlocked.pop("manifest_lock_sha256", None)
    value["manifest_lock_sha256"] = canonical_sha256(unlocked)


def _asset(root: Path, relative: str, payload: bytes, role: str) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "role": role,
        "relative_path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _build_manifest(tmp_path: Path) -> tuple[Path, Path]:
    asset_root = tmp_path / "assets"
    variants = [
        {
            "mask_variant_id": "sanity",
            "input_role": "sanity-input",
            "plugin_id": "manifest-mask-file-v1",
            "source_boundary": "opaque-upstream-input-only",
        },
        {
            "mask_variant_id": "decomp-a",
            "input_role": "oracle-input",
            "plugin_id": "manifest-mask-file-v1",
            "source_boundary": "opaque-upstream-input-only",
        },
    ]
    items: list[dict[str, object]] = []
    for sample_index in range(2):
        shared = {
            "rgb": _asset(
                asset_root,
                f"rgb/{sample_index}.ppm",
                f"P6 2 2 255 rgb-{sample_index}".encode(),
                "public_rgb",
            ),
            "depth": _asset(
                asset_root,
                f"depth/{sample_index}.pgm",
                f"P5 2 2 255 depth-{sample_index}".encode(),
                "public_depth",
            ),
            "camera": _asset(
                asset_root,
                f"camera/{sample_index}.json",
                json.dumps({"K": [500, 0, 640, 0, 500, 360, 0, 0, 1]}).encode(),
                "public_camera",
            ),
            "cad": _asset(
                asset_root,
                f"cad/{sample_index}.ply",
                f"ply\ncomment fixture-{sample_index}\n".encode(),
                "public_cad",
            ),
        }
        for variant_index, variant in enumerate(variants):
            mask = _asset(
                asset_root,
                f"input_masks/v{variant_index}/s{sample_index}.pgm",
                f"P5 2 2 255 mask-{sample_index}-{variant_index}".encode(),
                "input_mask",
            )
            item = {
                "item_id": f"s{sample_index:06d}-i{sample_index:06d}-o{sample_index + 1:06d}-{variant['mask_variant_id']}",
                "sample_key": {
                    "scene_id": sample_index,
                    "image_id": sample_index,
                    "object_id": sample_index + 1,
                },
                "mask_variant_id": variant["mask_variant_id"],
                "frame_size": {"width": 1280, "height": 720},
                "inputs": {**shared, "mask": mask},
                "camera_intrinsics": [
                    [500.0, 0.0, 640.0],
                    [0.0, 500.0, 360.0],
                    [0.0, 0.0, 1.0],
                ],
                "depth_scale": 0.001,
                "mask_provenance": {
                    "plugin_id": variant["plugin_id"],
                    "input_role": variant["input_role"],
                    "source_artifact_sha256": mask["sha256"],
                    "generator_id": "synthetic-fixture",
                    "generator_version": "1",
                    "model_sha256": None,
                    "score": 0.75 + 0.1 * variant_index,
                    "coverage": {"nonzero_pixels": 2, "fraction": 0.5},
                    "label_access_count_on_gpu_c": 0,
                },
            }
            items.append(item)
    manifest = {
        "schema_version": "poseloop.r4c.prep.label-free-manifest.v1",
        "protocol_id": PREP_PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_contract": {
            "protocol_id": "poseloop-r4a-accuracy-decomposition-fixture-v1",
            "protocol_sha256": "1" * 64,
            "bundle_sha256": "2" * 64,
        },
        "mask_variants": variants,
        "coverage": expected_coverage(items),
        "items": items,
        "boundary": dict(BOUNDARY),
    }
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path, asset_root


def _build_v2_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    fixture_root = tmp_path / "fixture"
    build_fixture(protocol_path=PROTOCOL, output_root=fixture_root)
    run_root = tmp_path / "run"
    exit_code, _ = run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=fixture_root / "manifest.json",
        output_root=run_root,
        asset_root=fixture_root,
        verify_assets=True,
    )
    assert exit_code == 0
    return fixture_root / "manifest.json", fixture_root, run_root


def _export_v2_results(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest_path, _, run_root = _build_v2_run(tmp_path)
    output_path = tmp_path / "a-output.jsonl"
    validation_path = tmp_path / "a-output-validation.json"
    export_a_results(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        producer_results_path=run_root / "results.jsonl",
        output_path=output_path,
        validation_output_path=validation_path,
    )
    return manifest_path, output_path, validation_path


def _live_test_contracts(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    fixture_root = tmp_path / "live-fixture"
    build_fixture(protocol_path=PROTOCOL, output_root=fixture_root)
    manifest_path = fixture_root / "manifest.json"
    manifest = read_json(manifest_path)
    runtime_identity = {
        "implementation_commit": "a" * 40,
        "implementation_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "checkpoint_sha256": {"refiner": "d" * 64, "scorer": "e" * 64},
    }
    manifest["input_kind"] = "DEVELOPMENT_DATA"
    manifest["producer_runtime_lock"] = runtime_identity
    _refresh_v2_lock(manifest)
    write_json_atomic(manifest_path, manifest)

    runtime_contract = read_json(PYTHON_RUNTIME)
    lock = {
        "schema_version": LIVE_RUNTIME_LOCK_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": LIVE_BACKEND_ID,
        "created_at_utc": "2026-08-17T00:00:00+00:00",
        "protocol_sha256": sha256_file(PROTOCOL),
        "runtime_contract_sha256": sha256_file(PYTHON_RUNTIME),
        "backend_transport_contract_sha256": runtime_contract[
            "backend_transport_contract"
        ]["sha256"],
        "implementation_root": str(REPO_ROOT),
        "implementation_commit": runtime_identity["implementation_commit"],
        "implementation_files_sha256": {},
        "implementation_sha256": runtime_identity["implementation_sha256"],
        "poseloop_runtime_root": str(REPO_ROOT),
        "reused_runtime_sources_sha256": {},
        "foundationpose_root": str(tmp_path / "not-used-by-mock"),
        "foundationpose_source": {
            "repository": "https://github.com/NVlabs/FoundationPose.git",
            "commit": "a1b694b83e633c2cb6115b9063d940a687759392",
            "tracked_tree_clean": True,
        },
        "runtime_sources": {
            "pytorch3d": "3143b3baf8ef8b1023ed76f225af59e2e8a71e06",
            "nvdiffrast": "253ac4fcea7de5f396371124af597e6cc957bfae",
        },
        "environment": {"python": "mock", "torch": "mock", "required_imports": {}},
        "gpu": {
            "index": 0,
            "name": "mock-structural-only",
            "uuid": "GPU-mock-structural-only",
            "total_memory_bytes": 1,
            "cuda_runtime": "mock",
        },
        "checkpoints": {},
        "checkpoint_configs": {},
        "producer_runtime_lock": runtime_identity,
        "frozen_inference": FIXED_INFERENCE,
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "scorer_path_open_count": 0,
        "official_scorer_run": False,
    }
    lock_path = tmp_path / "live-runtime-lock.json"
    write_json_atomic(lock_path, lock)
    ack = {
        "schema_version": LIVE_BACKEND_ACK_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": LIVE_BACKEND_ID,
        "created_at_utc": "2026-08-17T00:00:01+00:00",
        "runtime_lock_sha256": sha256_file(lock_path),
        "backend_transport_contract_sha256": lock["backend_transport_contract_sha256"],
        "implementation_commit": runtime_identity["implementation_commit"],
        "implementation_sha256": runtime_identity["implementation_sha256"],
        "acknowledged": True,
        "frozen_inference": FIXED_INFERENCE,
        "commitments": ACK_COMMITMENTS,
        "label_access_count": 0,
        "official_scorer_run": False,
    }
    ack_path = tmp_path / "backend-transport-ack.json"
    write_json_atomic(ack_path, ack)
    return manifest_path, fixture_root, lock_path, ack_path, lock


class _StructuralLiveBackend:
    def __init__(self, runtime_lock: dict[str, object], output_root: Path):
        self.runtime_lock = runtime_lock
        self.output_root = output_root

    def run_item(
        self, item: dict[str, object], *, asset_root: Path, attempt: int
    ) -> dict[str, object]:
        result = fixture_result(
            item,
            runtime_lock=self.runtime_lock["producer_runtime_lock"],
        )
        result["backend_id"] = LIVE_BACKEND_ID
        result["synthetic_backend"] = False
        result["attempt"] = attempt
        for index, candidate in enumerate(result["top_k"]):
            candidate["candidate_id"] = f"foundationpose-c{index:03d}"
        result["top_k"][0]["model_to_camera_pose_m"] = result[
            "final_model_to_camera_pose_m"
        ]
        result["foundationpose_top_score_margin"] = float(
            result["top_k"][0]["score"]
        ) - float(result["top_k"][1]["score"])
        result["prediction_fingerprint"] = canonical_sha256(
            {
                "pose": result["predicted_model_to_camera_pose_m"],
                "top_score": result["foundationpose_top_score"],
                "margin": result["foundationpose_top_score_margin"],
                "selected_candidate_index": result["selected_candidate_index"],
                "candidate_count": result["pose_hypothesis_count"],
            }
        )
        result["wall_time_ms"] = result["total_latency_ms"] + 2.0
        result["evidence_capture_ms"] = 1.0
        result["visualization_ms"] = 1.0
        result["evidence_peak_allocated_bytes"] = result["cuda_peak_allocated_bytes"]
        result["evidence_peak_reserved_bytes"] = result["cuda_peak_reserved_bytes"]
        result["backend_evidence"] = {
            "synthetic": False,
            "primary_register_iterations": 5,
            "primary_candidate_count": 252,
            "foundationpose_candidate_scorer_calls_primary": 1,
            "trace_capture_mode": "exact-refiner-replay-plus-full-c252-scorer",
            "trace_refiner_replay_calls": 5,
            "trace_objective_scorer_calls": 6,
            "trace_final_pose_population_bitwise_equal": True,
            "trace_final_score_population_bitwise_equal": True,
            "official_evaluator_scorer_run": False,
        }
        inventory = []
        fixture_png = (asset_root / item["inputs"]["rgb"]["relative_path"]).read_bytes()
        for role in (
            "final_pose_overlay",
            "initial_pose_overlay",
            "input_mask",
            "rgb",
            "top_k_overlay",
        ):
            relative = f"visualizations/{item['item_id']}/{role}.png"
            path = self.output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fixture_png)
            inventory.append(
                {
                    "role": role,
                    "relative_path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        result["visualization_inventory"] = inventory
        validate_internal_live_success(result, visualization_root=self.output_root)
        return result


def test_live_implementation_identity_covers_preflight_and_launcher() -> None:
    assert "foundationpose_runtime_prep/preflight.py" in IMPLEMENTATION_FILES
    assert (
        "foundationpose_runtime_prep/launch_official_python.sh" in IMPLEMENTATION_FILES
    )
    assert REUSED_RUNTIME_SOURCES["scripts/run_xyzibd_batch.py"] == (
        "979622519b98f6249cb69af932180495716cd047303533943c10ceedb37c5b4a"
    )


def test_manifest_to_fixture_crash_resume_and_profile(tmp_path: Path) -> None:
    manifest_path, asset_root = _build_manifest(tmp_path)
    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=PROTOCOL,
        asset_root=asset_root,
        verify_assets=True,
    )
    assert len(items) == 4
    assert manifest["coverage"]["base_sample_count"] == 2
    assert manifest["coverage"]["per_variant_item_count"] == {
        "decomp-a": 2,
        "sanity": 2,
    }

    run_root = tmp_path / "run"
    exit_code, interrupted = run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        output_root=run_root,
        asset_root=asset_root,
        verify_assets=True,
        crash_after_new_successes=2,
    )
    assert exit_code == PLANNED_CRASH_EXIT
    assert interrupted["status"] == "planned-crash-after-checkpoint"
    assert interrupted["new_success_count"] == 2
    assert len(list((run_root / "checkpoints" / "success").glob("*.json"))) == 2

    exit_code, resumed = run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        output_root=run_root,
        asset_root=asset_root,
        verify_assets=True,
        resume=True,
    )
    assert exit_code == 0
    assert resumed["status"] == "complete"
    assert resumed["cached_success_count"] == 2
    assert resumed["new_success_count"] == 2
    results = read_jsonl(run_root / "results.jsonl")
    assert len(results) == 4
    assert all(row["candidate_limit"] == 252 for row in results)
    assert all(row["pose_hypothesis_count"] == 252 for row in results)
    assert all(row["resource_batches"]["score_data"] == 8 for row in results)

    profile_root = tmp_path / "profile"
    samples_path = profile_root / "profile-samples.jsonl"
    samples = fixture_samples(
        results_path=run_root / "results.jsonl",
        output_path=samples_path,
        warmup_repeats=2,
        steady_repeats=5,
    )
    assert len([row for row in samples if row["phase"] == "warmup"]) == 8
    assert len([row for row in samples if row["phase"] == "steady"]) == 20
    summary = summarize_profile(
        protocol_path=PROTOCOL,
        samples_path=samples_path,
        output_json=profile_root / "profile-summary.json",
        output_csv=profile_root / "profile-samples.csv",
    )
    assert summary["resolution"] == {"width": 1280, "height": 720}
    assert summary["steady"]["sample_count"] == 20
    assert summary["steady"]["p95_ms"] >= summary["steady"]["p50_ms"]
    assert (
        summary["steady"]["peak_reserved_vram_bytes"]
        >= summary["steady"]["peak_allocated_vram_bytes"]
    )
    assert summary["external_reference"] == {
        "target_ms": 89.0,
        "scope": "external-reference-target-only",
        "comparable": False,
        "speedup_or_claim_computed": False,
    }
    assert (
        (profile_root / "profile-samples.csv")
        .read_text(encoding="utf-8")
        .startswith("item_id,")
    )


def test_manifest_rejects_gt_path_and_variant_coverage_gap(tmp_path: Path) -> None:
    manifest_path, _ = _build_manifest(tmp_path)
    value = read_json(manifest_path)
    value["items"][0]["inputs"]["rgb"]["relative_path"] = "scene_gt/000001.json"
    write_json_atomic(manifest_path, value)
    with pytest.raises(PrepError, match="Forbidden GT/evaluator path"):
        validate_manifest(manifest_path, protocol_path=PROTOCOL)

    manifest_path, _ = _build_manifest(tmp_path / "second")
    value = read_json(manifest_path)
    value["items"].pop()
    write_json_atomic(manifest_path, value)
    with pytest.raises(PrepError, match="do not cover the same base sample set"):
        validate_manifest(manifest_path, protocol_path=PROTOCOL)


def test_resume_rejects_tampered_checkpoint_binding(tmp_path: Path) -> None:
    manifest_path, _ = _build_manifest(tmp_path)
    run_root = tmp_path / "run"
    exit_code, _ = run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        output_root=run_root,
        crash_after_new_successes=1,
    )
    assert exit_code == PLANNED_CRASH_EXIT
    checkpoint = next((run_root / "checkpoints" / "success").glob("*.json"))
    value = read_json(checkpoint)
    value["binding"]["item_fingerprint"] = "0" * 64
    write_json_atomic(checkpoint, value)
    with pytest.raises(PrepError, match="checkpoint binding differs"):
        run_fixture(
            protocol_path=PROTOCOL,
            manifest_path=manifest_path,
            output_root=run_root,
            resume=True,
        )


def test_static_preflight_is_explicitly_blocked_without_gpu_or_downloads(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight.json"
    report = static_preflight(
        protocol_path=PROTOCOL,
        runtime_contract_path=PYTHON_RUNTIME,
        output_path=output,
    )
    assert report["fixture_ready"] is True
    assert report["gpu_runtime_ready"] is False
    assert report["auto_deploy"] is False
    assert report["network_accessed"] is False
    assert report["downloads_or_installs"] is False
    assert (
        report["backend_transport_contract_sha256"]
        == read_json(PYTHON_RUNTIME)["backend_transport_contract"]["sha256"]
    )
    assert {row["code"] for row in report["blockers"]} == {
        "FOUNDATIONPOSE_SOURCE_NOT_INSPECTED",
        "CHECKPOINT_FILES_NOT_INSPECTED",
        "CUDA_RUNTIME_NOT_INSPECTED",
    }
    assert sha256_file(output) == hashlib.sha256(output.read_bytes()).hexdigest()


def test_isaac_static_preflight_preserves_unknown_runtime_as_blocker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "isaac-preflight.json"
    report = static_preflight(
        protocol_path=PROTOCOL,
        runtime_contract_path=ISAAC_RUNTIME,
        output_path=output,
    )
    assert report["fixture_ready"] is True
    assert report["gpu_runtime_ready"] is False
    assert report["runtime_schema"].endswith("isaac-ros-tensorrt-runtime-contract.v1")
    assert (
        report["backend_transport_contract_sha256"]
        == read_json(ISAAC_RUNTIME)["backend_transport_contract"]["sha256"]
    )
    assert {row["code"] for row in report["blockers"]} == {
        "ISAAC_RUNTIME_LOCK_UNRESOLVED",
        "ISAAC_ADAPTER_NOT_HASH_LOCKED",
    }
    contract = read_json(ISAAC_RUNTIME)
    assert all(
        contract["runtime_lock"][field] is None
        for field in contract["unresolved_required_fields"]
    )


def test_mask_provenance_hash_and_score_are_strict(tmp_path: Path) -> None:
    manifest_path, _ = _build_manifest(tmp_path)
    value = read_json(manifest_path)
    value["items"][0]["mask_provenance"]["source_artifact_sha256"] = "f" * 64
    write_json_atomic(manifest_path, value)
    with pytest.raises(PrepError, match="asset and provenance source hashes differ"):
        validate_manifest(manifest_path, protocol_path=PROTOCOL)

    manifest_path, _ = _build_manifest(tmp_path / "score")
    value = read_json(manifest_path)
    value["items"][0]["mask_provenance"]["score"] = 1.1
    write_json_atomic(manifest_path, value)
    with pytest.raises(PrepError, match="score is invalid"):
        validate_manifest(manifest_path, protocol_path=PROTOCOL)


def test_committed_cli_fixture_builder_is_asset_verified(tmp_path: Path) -> None:
    receipt = build_fixture(protocol_path=PROTOCOL, output_root=tmp_path / "fixture")
    assert receipt["item_count"] == 10
    assert receipt["frame_size"] == {"width": 1280, "height": 720}
    manifest_path = tmp_path / "fixture" / "manifest.json"
    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=PROTOCOL,
        asset_root=tmp_path / "fixture",
        verify_assets=True,
    )
    assert len(items) == 10
    assert manifest["coverage"]["required_percent"] == 100
    assert manifest["schema_version"] == (
        "poseloop.r4c.prep.runtime-isolated-manifest.v2"
    )
    assert manifest["coverage"]["base_sample_count"] == 2
    assert set(manifest["coverage"]["per_variant_item_count"].values()) == {2}


def test_v2_gt_derived_controls_are_truthful_opaque_inputs(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    build_fixture(protocol_path=PROTOCOL, output_root=fixture_root)
    manifest_path = fixture_root / "manifest.json"
    value = read_json(manifest_path)
    for item in value["items"]:
        if item["mask_variant_id"] in {
            "official_known_sample_sanity",
            "oracle_mask_control",
        }:
            item["mask_provenance"]["derivation_class"] = (
                "gt-derived-development-control"
            )
    value["boundary"]["contains_gt_derived_control_inputs"] = True
    _refresh_v2_lock(value)
    write_json_atomic(manifest_path, value)
    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=PROTOCOL,
        asset_root=fixture_root,
        verify_assets=True,
    )
    assert len(items) == 10
    assert manifest["boundary"] == {
        "contains_gt_derived_control_inputs": True,
        "contains_raw_gt_paths": False,
        "gpu_c_resolves_derivation": False,
        "label_access_count_on_gpu_c": 0,
        "gt_path_open_count_on_gpu_c": 0,
        "evaluator_path_open_count_on_gpu_c": 0,
        "official_scorer_run": False,
        "contains_evaluator_output": False,
        "contains_sealed_data": False,
        "accuracy_claim_permitted": False,
    }

    value["boundary"]["contains_gt_derived_control_inputs"] = False
    _refresh_v2_lock(value)
    write_json_atomic(manifest_path, value)
    with pytest.raises(PrepError, match="GT-derived control boundary is not truthful"):
        validate_manifest(manifest_path, protocol_path=PROTOCOL)


def test_v2_forbids_raw_gt_paths_even_with_a_valid_manifest_lock(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixture"
    build_fixture(protocol_path=PROTOCOL, output_root=fixture_root)
    manifest_path = fixture_root / "manifest.json"
    value = read_json(manifest_path)
    value["items"][0]["inputs"]["rgb"]["relative_path"] = "scene_gt/000001.json"
    _refresh_v2_lock(value)
    write_json_atomic(manifest_path, value)
    with pytest.raises(PrepError, match="Forbidden GT/evaluator path"):
        validate_manifest(manifest_path, protocol_path=PROTOCOL)


def test_v2_requires_frozen_a_source_protocol_sha(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    build_fixture(protocol_path=PROTOCOL, output_root=fixture_root)
    manifest_path = fixture_root / "manifest.json"
    value = read_json(manifest_path)
    value["source_contract"]["protocol_sha256"] = "f" * 64
    _refresh_v2_lock(value)
    write_json_atomic(manifest_path, value)
    with pytest.raises(PrepError, match="protocol SHA-256 differs from frozen A"):
        validate_manifest(manifest_path, protocol_path=PROTOCOL)


def test_a_producer_output_export_is_exact_and_self_validated(tmp_path: Path) -> None:
    manifest_path, output_path, validation_path = _export_v2_results(tmp_path)
    rows = read_jsonl(output_path)
    report = read_json(validation_path)
    assert len(rows) == 10
    assert report["status"] == "valid"
    assert report["coverage_percent"] == 100.0
    assert report["synthetic_fixture_row_count"] == 10
    assert report["results_sha256"] == sha256_file(output_path)
    assert all(len(row["top_k"]) == 5 for row in rows)
    assert all(len(row["refiner_trace"]) == 6 for row in rows)
    assert all(row["failure"] is False for row in rows)
    assert all(
        set(row["access_counters"])
        == {
            "label_access_count_on_gpu_c",
            "gt_path_open_count_on_gpu_c",
            "evaluator_path_open_count_on_gpu_c",
            "scorer_path_open_count_on_gpu_c",
        }
        for row in rows
    )
    assert sum(len(row["visualization_inventory"]) for row in rows) == 50
    assert (
        validate_a_results(
            protocol_path=PROTOCOL,
            manifest_path=manifest_path,
            results_path=output_path,
        )["status"]
        == "valid"
    )


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("missing-top-k", "row fields differ"),
        ("missing-trace", "row fields differ"),
        ("hash-drift", "input SHA drift"),
        ("coverage-missing", "coverage is incomplete"),
        ("label-access", "access counter is nonzero"),
        ("gt-path-open", "access counter is nonzero"),
        ("evaluator-path-open", "access counter is nonzero"),
        ("scorer-path-open", "access counter is nonzero"),
    ],
)
def test_a_producer_output_rejects_incomplete_or_unsafe_rows(
    tmp_path: Path,
    case: str,
    expected_error: str,
) -> None:
    manifest_path, output_path, _ = _export_v2_results(tmp_path)
    rows = read_jsonl(output_path)
    if case == "missing-top-k":
        rows[0].pop("top_k")
    elif case == "missing-trace":
        rows[0].pop("refiner_trace")
    elif case == "hash-drift":
        rows[0]["input_sha256"]["mask"] = "f" * 64
    elif case == "coverage-missing":
        rows.pop()
    else:
        field = {
            "label-access": "label_access_count_on_gpu_c",
            "gt-path-open": "gt_path_open_count_on_gpu_c",
            "evaluator-path-open": "evaluator_path_open_count_on_gpu_c",
            "scorer-path-open": "scorer_path_open_count_on_gpu_c",
        }[case]
        rows[0]["access_counters"][field] = 1
    invalid_path = tmp_path / f"invalid-{case}.jsonl"
    write_jsonl_atomic(invalid_path, rows)
    with pytest.raises(PrepError, match=expected_error):
        validate_a_results(
            protocol_path=PROTOCOL,
            manifest_path=manifest_path,
            results_path=invalid_path,
        )


@pytest.mark.parametrize(
    "missing_field",
    ["top_k", "refiner_trace_model_to_camera_m"],
)
def test_export_rejects_backend_rows_missing_real_evidence(
    tmp_path: Path,
    missing_field: str,
) -> None:
    manifest_path, _, run_root = _build_v2_run(tmp_path)
    rows = read_jsonl(run_root / "results.jsonl")
    rows[0].pop(missing_field)
    backend_path = tmp_path / "backend-missing-evidence.jsonl"
    write_jsonl_atomic(backend_path, rows)
    with pytest.raises(PrepError, match="top_k|refiner trace"):
        export_a_results(
            protocol_path=PROTOCOL,
            manifest_path=manifest_path,
            producer_results_path=backend_path,
            output_path=tmp_path / "must-not-exist.jsonl",
            validation_output_path=tmp_path / "must-not-exist-validation.json",
        )


def test_exact_equivalence_tolerance_never_promotes_drift(tmp_path: Path) -> None:
    manifest_path, _ = _build_manifest(tmp_path)
    run_root = tmp_path / "run"
    exit_code, _ = run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        output_root=run_root,
    )
    assert exit_code == 0
    results_path = run_root / "results.jsonl"
    exact = compare_exact(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        baseline_path=results_path,
        candidate_path=results_path,
        output_path=tmp_path / "exact.json",
    )
    assert exact["passed_exact_guard"] is True
    assert exact["all_predictions_within_numeric_tolerance"] is True

    drifted_rows = read_jsonl(results_path)
    drifted = drifted_rows[0]
    drifted["foundationpose_top_score"] += 0.5e-8
    drifted["prediction_fingerprint"] = canonical_sha256(
        {
            "pose": drifted["predicted_model_to_camera_pose_m"],
            "top_score": drifted["foundationpose_top_score"],
            "margin": drifted["foundationpose_top_score_margin"],
            "selected_candidate_index": drifted["selected_candidate_index"],
            "candidate_count": drifted["pose_hypothesis_count"],
        }
    )
    candidate_path = tmp_path / "candidate.jsonl"
    write_jsonl_atomic(candidate_path, drifted_rows)
    diagnostic = compare_exact(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        baseline_path=results_path,
        candidate_path=candidate_path,
        output_path=tmp_path / "drift.json",
    )
    assert diagnostic["all_predictions_within_numeric_tolerance"] is True
    assert diagnostic["all_predictions_bitwise_equal"] is False
    assert diagnostic["passed_exact_guard"] is False

    write_jsonl_atomic(candidate_path, drifted_rows[:-1])
    missing = compare_exact(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        baseline_path=results_path,
        candidate_path=candidate_path,
        output_path=tmp_path / "missing.json",
    )
    assert missing["coverage"]["exact"] is False
    assert missing["passed_exact_guard"] is False


def test_exact_equivalence_rejects_candidate_pruning(tmp_path: Path) -> None:
    manifest_path, _ = _build_manifest(tmp_path)
    run_root = tmp_path / "run"
    run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        output_root=run_root,
    )
    rows = read_jsonl(run_root / "results.jsonl")
    rows[0]["candidate_limit"] = 32
    candidate_path = tmp_path / "pruned.jsonl"
    write_jsonl_atomic(candidate_path, rows)
    with pytest.raises(PrepError, match="candidate limit is not c252"):
        compare_exact(
            protocol_path=PROTOCOL,
            manifest_path=manifest_path,
            baseline_path=run_root / "results.jsonl",
            candidate_path=candidate_path,
            output_path=tmp_path / "pruned-report.json",
        )


def test_visualization_plan_is_dependency_free_and_claim_limited(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixture"
    build_fixture(protocol_path=PROTOCOL, output_root=fixture_root)
    run_root = tmp_path / "run"
    run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=fixture_root / "manifest.json",
        output_root=run_root,
    )
    output = tmp_path / "visualization-plan.json"
    plan = build_visualization_plan(
        protocol_path=PROTOCOL,
        manifest_path=fixture_root / "manifest.json",
        baseline_path=run_root / "results.jsonl",
        improved_path=run_root / "results.jsonl",
        output_path=output,
    )
    assert plan["coverage"] == {
        "expected": 10,
        "planned": 10,
        "percent": 100.0,
        "missing": [],
        "duplicate": [],
    }
    assert plan["render_dependencies_required_for_plan"] is False
    assert plan["pixel_equality_claim"] == "no-regression-only-not-accuracy"
    assert all(row["prediction_projection_exact_equal"] for row in plan["frames"])


def test_visualization_renderer_writes_png_and_verified_mp4(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    pytest.importorskip("trimesh")
    fixture_root = tmp_path / "fixture"
    build_fixture(protocol_path=PROTOCOL, output_root=fixture_root)
    run_root = tmp_path / "run"
    run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=fixture_root / "manifest.json",
        output_root=run_root,
    )
    output_root = tmp_path / "visualization"
    receipt = render_visualization(
        protocol_path=PROTOCOL,
        manifest_path=fixture_root / "manifest.json",
        asset_root=fixture_root,
        baseline_path=run_root / "results.jsonl",
        improved_path=run_root / "results.jsonl",
        output_root=output_root,
    )
    assert receipt["coverage"]["rendered"] == 10
    assert receipt["video"]["frame_count"] == 10
    assert receipt["video"]["canonical_equivalence_artifact"] is False
    assert all(row["overlay_content_pixel_equal"] for row in receipt["frames"])
    assert not any(
        row["panel_pixels_equal_including_labels"] for row in receipt["frames"]
    )
    assert all(
        (output_root / row["relative_png_path"]).is_file() for row in receipt["frames"]
    )
    assert (output_root / receipt["video"]["relative_path"]).is_file()


def test_live_path_is_separate_and_crash_resume_is_atomic(tmp_path: Path) -> None:
    manifest_path, asset_root, lock_path, ack_path, _ = _live_test_contracts(tmp_path)
    output_root = tmp_path / "live-run"

    def factory(runtime_lock: dict[str, object], root: Path) -> _StructuralLiveBackend:
        return _StructuralLiveBackend(runtime_lock, root)

    exit_code, interrupted = run_live(
        protocol_path=PROTOCOL,
        runtime_contract_path=PYTHON_RUNTIME,
        manifest_path=manifest_path,
        asset_root=asset_root,
        output_root=output_root,
        runtime_lock_path=lock_path,
        backend_ack_path=ack_path,
        crash_after_new_successes=2,
        backend_factory=factory,
        revalidate_environment=False,
    )
    assert exit_code == PLANNED_CRASH_EXIT
    assert interrupted["status"] == "planned-crash-after-checkpoint"
    assert interrupted["new_success_count"] == 2
    assert len(list((output_root / "checkpoints" / "success").glob("*.json"))) == 2

    exit_code, resumed = run_live(
        protocol_path=PROTOCOL,
        runtime_contract_path=PYTHON_RUNTIME,
        manifest_path=manifest_path,
        asset_root=asset_root,
        output_root=output_root,
        runtime_lock_path=lock_path,
        backend_ack_path=ack_path,
        resume=True,
        backend_factory=factory,
        revalidate_environment=False,
    )
    assert exit_code == 0
    assert resumed["status"] == "complete"
    assert resumed["cached_success_count"] == 2
    rows = read_jsonl(output_root / "results.jsonl")
    assert len(rows) == 10
    assert all(row["synthetic_backend"] is False for row in rows)
    assert all(row["backend_id"] == LIVE_BACKEND_ID for row in rows)

    a_output = tmp_path / "live-export" / "a-producer-output.jsonl"
    a_validation = tmp_path / "live-export" / "a-validation.json"
    exported, report = export_a_live_results(
        protocol_path=PROTOCOL,
        manifest_path=manifest_path,
        producer_results_path=output_root / "results.jsonl",
        output_path=a_output,
        validation_output_path=a_validation,
    )
    assert len(exported) == 10
    assert report["status"] == "valid"
    assert report["synthetic_fixture_row_count"] == 0
    assert (
        validate_a_results(
            protocol_path=PROTOCOL,
            manifest_path=manifest_path,
            results_path=a_output,
        )["status"]
        == "valid"
    )
    with pytest.raises(PrepError, match="only explicit synthetic fixture rows"):
        export_a_results(
            protocol_path=PROTOCOL,
            manifest_path=manifest_path,
            producer_results_path=output_root / "results.jsonl",
            output_path=tmp_path / "must-not-export-live-through-fixture.jsonl",
            validation_output_path=tmp_path / "must-not-validate.json",
        )


def test_live_contract_rejects_synthetic_or_incomplete_backend_rows(
    tmp_path: Path,
) -> None:
    manifest_path, asset_root, lock_path, ack_path, _ = _live_test_contracts(tmp_path)
    output_root = tmp_path / "live-run"
    backend = _StructuralLiveBackend(read_json(lock_path), output_root)
    _, items = validate_manifest(
        manifest_path,
        protocol_path=PROTOCOL,
        asset_root=asset_root,
        verify_assets=True,
    )
    row = backend.run_item(items[0], asset_root=asset_root, attempt=1)
    missing_trace = dict(row)
    missing_trace.pop("refiner_trace_model_to_camera_m")
    with pytest.raises(PrepError, match="six states"):
        validate_internal_live_success(missing_trace, visualization_root=output_root)
    nonzero_access = dict(row)
    nonzero_access["gt_path_open_count"] = 1
    with pytest.raises(PrepError, match="access counter is nonzero"):
        validate_internal_live_success(nonzero_access, visualization_root=output_root)

    fixture_root = tmp_path / "synthetic"
    build_fixture(protocol_path=PROTOCOL, output_root=fixture_root)
    fixture_run = tmp_path / "synthetic-run"
    run_fixture(
        protocol_path=PROTOCOL,
        manifest_path=fixture_root / "manifest.json",
        output_root=fixture_run,
    )
    with pytest.raises(PrepError, match="Live export rejects synthetic backend rows"):
        export_a_live_results(
            protocol_path=PROTOCOL,
            manifest_path=fixture_root / "manifest.json",
            producer_results_path=fixture_run / "results.jsonl",
            output_path=tmp_path / "must-not-export-synthetic-live.jsonl",
            validation_output_path=tmp_path / "must-not-validate-synthetic-live.json",
        )

    # The acknowledgement is a hard input even for a mocked structural path.
    ack = read_json(ack_path)
    ack["acknowledged"] = False
    write_json_atomic(ack_path, ack)
    with pytest.raises(PrepError, match="acknowledgement identity differs"):
        run_live(
            protocol_path=PROTOCOL,
            runtime_contract_path=PYTHON_RUNTIME,
            manifest_path=manifest_path,
            asset_root=asset_root,
            output_root=tmp_path / "bad-ack-run",
            runtime_lock_path=lock_path,
            backend_ack_path=ack_path,
            backend_factory=lambda runtime, root: _StructuralLiveBackend(runtime, root),
            revalidate_environment=False,
        )


def test_live_smoke_request_is_predicted_only_and_never_formal_coverage(
    tmp_path: Path,
) -> None:
    manifest_path, asset_root, _, _, runtime_lock = _live_test_contracts(tmp_path)
    manifest = read_json(manifest_path)
    raw = next(
        item
        for item in manifest["items"]
        if item["mask_variant_id"] == "predicted_mask"
    )
    raw["mask_provenance"]["generator_id"] = "r4c-frozen-no-gt-smoke-reuse"
    raw["mask_provenance"]["generator_version"] = "v1"
    request = {
        "schema_version": LIVE_SMOKE_REQUEST_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "purpose": "one-item-no-gt-adapter-smoke-not-five-variant",
        "source_manifest_sha256": sha256_file(manifest_path),
        "producer_runtime_lock": runtime_lock["producer_runtime_lock"],
        "item": raw,
        "boundary": {
            "smoke_only": True,
            "formal_five_variant_coverage_claimed": False,
            "contains_gt_derived_control_inputs": False,
            "contains_raw_gt_paths": False,
            "gpu_c_resolves_derivation": False,
            "label_access_count_on_gpu_c": 0,
            "gt_path_open_count_on_gpu_c": 0,
            "evaluator_path_open_count_on_gpu_c": 0,
            "official_scorer_run": False,
            "contains_evaluator_output": False,
            "contains_sealed_data": False,
            "accuracy_claim_permitted": False,
        },
    }
    request["request_lock_sha256"] = canonical_sha256(request)
    request_path = tmp_path / "live-smoke-request.json"
    write_json_atomic(request_path, request)
    _, normalized = validate_live_smoke_request(
        request_path=request_path,
        source_manifest_path=manifest_path,
        asset_root=asset_root,
    )
    assert normalized["mask_variant_id"] == "predicted_mask"
    assert normalized["mask_provenance"]["development_control"] is False

    request["item"]["mask_variant_id"] = "oracle_mask_control"
    request["request_lock_sha256"] = canonical_sha256(
        {key: value for key, value in request.items() if key != "request_lock_sha256"}
    )
    write_json_atomic(request_path, request)
    with pytest.raises(PrepError, match="only the predicted_mask"):
        validate_live_smoke_request(
            request_path=request_path,
            source_manifest_path=manifest_path,
            asset_root=asset_root,
        )
