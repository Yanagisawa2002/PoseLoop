from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pose_accuracy_recovery_prep.core import ContractError, load_and_validate_manifest
from pose_accuracy_recovery_prep.diagnostics import (
    evaluate_prepared_diagnostics,
    evaluate_refiner,
    evaluate_scorer,
    generate_perturbation_grid,
    load_protocol,
    prepare_diagnostics,
    synthetic_outputs,
)
from pose_accuracy_recovery_prep.reporting import (
    evaluate_prediction_rows,
    official_metric_status,
    synthetic_prediction_rows,
    write_report,
)
from pose_accuracy_recovery_prep.se3 import (
    as_pose,
    load_cad_asset,
    load_pose_asset,
    pose_errors,
)
from pose_accuracy_recovery_prep.visualization import build_visualization_plan

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "fixtures" / "pose_accuracy_recovery_prep"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
PROTOCOL_PATH = ROOT / "protocols" / "poseloop_pose_accuracy_recovery_prep_v1.json"


def inputs():
    protocol = load_protocol(PROTOCOL_PATH)
    manifest, _ = load_and_validate_manifest(MANIFEST_PATH, data_root=DATA_ROOT)
    sample = manifest["samples"][0]
    gt_pose = load_pose_asset(DATA_ROOT, sample["evaluator_only"]["gt_pose"])
    points, _ = load_cad_asset(DATA_ROOT, sample["producer_inputs"]["cad"])
    return protocol, manifest, sample, gt_pose, points


def test_frozen_perturbation_grid_is_deterministic_and_contains_gt() -> None:
    protocol, _, _, gt_pose, points = inputs()
    first = generate_perturbation_grid(
        gt_pose, protocol["diagnostics"]["perturbation_grid"]
    )
    second = generate_perturbation_grid(
        gt_pose, protocol["diagnostics"]["perturbation_grid"]
    )
    assert first == second
    assert len(first) == 169
    assert first[0]["candidate_id"] == "perturb-0000"
    assert pose_errors(
        points, first[0]["model_to_camera_pose_m"], gt_pose, symmetric=False
    ) == pytest.approx(
        {"add_s_mm": 0.0, "rotation_error_degrees": 0.0, "translation_error_mm": 0.0},
        abs=1e-10,
    )


def test_gt_grid_cannot_be_prepared_in_producer_namespace() -> None:
    protocol, manifest, _, _, _ = inputs()
    with pytest.raises(ContractError, match="EVALUATOR_ONLY"):
        prepare_diagnostics(
            protocol, manifest, data_root=DATA_ROOT, namespace_role="producer"
        )


def test_fixture_scorer_ranks_near_gt_and_refiner_is_monotonic() -> None:
    protocol, _, sample, gt_pose, points = inputs()
    candidates = generate_perturbation_grid(
        gt_pose, protocol["diagnostics"]["perturbation_grid"]
    )
    scores, traces = synthetic_outputs(
        candidates,
        gt_pose=gt_pose,
        points=points,
        symmetric=False,
        trace_count=8,
        refiner_scales=[1.0, 0.5, 0.0],
        selection_algorithm=protocol["diagnostics"]["refiner_trace_selection"],
    )
    scorer, _ = evaluate_scorer(
        candidates,
        scores,
        gt_pose=gt_pose,
        points=points,
        symmetric=sample["symmetric_object"],
        top_k=5,
        minimum_rank_correlation=0.95,
    )
    refiner, _ = evaluate_refiner(
        traces,
        gt_pose=gt_pose,
        points=points,
        symmetric=False,
        tolerance=1e-9,
    )
    assert scorer["passed"] is True
    assert scorer["nearest_candidate_rank"] == 1
    assert refiner == {
        "trace_count": 8,
        "passed_trace_count": 8,
        "failed_trace_count": 0,
        "passed": True,
        "tolerance": 1e-9,
    }


def test_bad_scorer_and_nonmonotonic_refiner_fail_without_retry() -> None:
    protocol, _, _, gt_pose, points = inputs()
    candidates = generate_perturbation_grid(
        gt_pose, protocol["diagnostics"]["perturbation_grid"]
    )
    scores = {
        row["candidate_id"]: pose_errors(
            points, row["model_to_camera_pose_m"], gt_pose, symmetric=False
        )["add_s_mm"]
        for row in candidates
    }
    scorer, _ = evaluate_scorer(
        candidates,
        scores,
        gt_pose=gt_pose,
        points=points,
        symmetric=False,
        top_k=5,
        minimum_rank_correlation=0.95,
    )
    trace = {
        "candidate_id": "bad-trace",
        "poses_model_to_camera_m": [
            candidates[0]["model_to_camera_pose_m"],
            candidates[-1]["model_to_camera_pose_m"],
        ],
    }
    refiner, _ = evaluate_refiner(
        [trace], gt_pose=gt_pose, points=points, symmetric=False, tolerance=1e-9
    )
    assert scorer["passed"] is False
    assert refiner["passed"] is False


def test_prepared_output_evaluator_requires_exact_grid_and_coverage(
    tmp_path: Path,
) -> None:
    protocol, manifest, sample, gt_pose, points = inputs()
    grid = prepare_diagnostics(
        protocol, manifest, data_root=DATA_ROOT, namespace_role="EVALUATOR_ONLY"
    )
    scores, traces = synthetic_outputs(
        grid,
        gt_pose=gt_pose,
        points=points,
        symmetric=sample["symmetric_object"],
        trace_count=8,
        refiner_scales=[1.0, 0.5, 0.0],
        selection_algorithm=protocol["diagnostics"]["refiner_trace_selection"],
    )
    key = sample["key"]
    scorer_rows = [
        {
            **key,
            "candidate_id": candidate_id,
            "score": score,
            "namespace_role": "EVALUATOR_ONLY",
        }
        for candidate_id, score in scores.items()
    ]
    refiner_rows = [
        {**key, **trace, "namespace_role": "EVALUATOR_ONLY"} for trace in traces
    ]
    summary = evaluate_prepared_diagnostics(
        protocol,
        manifest,
        data_root=DATA_ROOT,
        grid_rows=grid,
        scorer_rows=scorer_rows,
        refiner_traces=refiner_rows,
        namespace_role="EVALUATOR_ONLY",
        output_root=tmp_path / "valid",
    )
    assert summary["scorer_passed"] is True
    assert summary["refiner_passed"] is True
    assert (tmp_path / "valid" / "diagnostic-summary.csv").is_file()
    with pytest.raises(ContractError, match="grid differs"):
        evaluate_prepared_diagnostics(
            protocol,
            manifest,
            data_root=DATA_ROOT,
            grid_rows=grid[:-1],
            scorer_rows=scorer_rows,
            refiner_traces=refiner_rows,
            namespace_role="EVALUATOR_ONLY",
            output_root=tmp_path / "rejected",
        )
    bad_traces = [
        {**row, "poses_model_to_camera_m": list(row["poses_model_to_camera_m"])}
        for row in refiner_rows
    ]
    bad_traces[0]["poses_model_to_camera_m"][0] = np.eye(4).tolist()
    with pytest.raises(ContractError, match="does not start"):
        evaluate_prepared_diagnostics(
            protocol,
            manifest,
            data_root=DATA_ROOT,
            grid_rows=grid,
            scorer_rows=scorer_rows,
            refiner_traces=bad_traces,
            namespace_role="EVALUATOR_ONLY",
            output_root=tmp_path / "bad-start",
        )


def test_illegal_rotation_is_rejected() -> None:
    pose = np.eye(4)
    pose[0, 0] = 2.0
    with pytest.raises(ContractError, match=r"SO\(3\)"):
        as_pose(pose)


def test_official_capabilities_are_explicit_and_ar_requires_vsd() -> None:
    protocol, _, _, _, _ = inputs()
    status = official_metric_status(protocol, None, fixture_mode=True)
    assert {name: row["status"] for name, row in status.items()} == {
        "AR": "not_run_fixture",
        "MSSD": "not_run_fixture",
        "MSPD": "not_run_fixture",
        "VSD": "not_run_fixture",
    }
    payload = {
        "metrics": {
            "AR": {"status": "available", "value": 0.5, "reason": None},
            "MSSD": {"status": "available", "value": 0.5, "reason": None},
            "MSPD": {"status": "available", "value": 0.5, "reason": None},
            "VSD": {
                "status": "unavailable",
                "value": None,
                "reason": "dataset has no VSD assets",
            },
        }
    }
    with pytest.raises(ContractError, match="AR cannot be available"):
        official_metric_status(protocol, payload, fixture_mode=False)


def test_internal_report_has_groups_failures_and_paired_deltas(tmp_path: Path) -> None:
    protocol, manifest, _, _, _ = inputs()
    predictions = synthetic_prediction_rows(protocol, manifest, data_root=DATA_ROOT)
    predictions[-1] = {**predictions[-1], "status": "failed"}
    predictions[-1].pop("model_to_camera_pose_m")
    normalized, grouped, paired = evaluate_prediction_rows(
        predictions, manifest, data_root=DATA_ROOT
    )
    assert len(normalized) == 10
    assert any(row["failure_count"] == 1 for row in grouped if row["grouping"] == "all")
    assert any(
        row["comparison"] == "oracle_mask_control-minus-predicted_mask"
        for row in paired
    )
    report = write_report(
        protocol,
        manifest,
        predictions,
        data_root=DATA_ROOT,
        output_root=tmp_path,
        fixture_mode=True,
    )
    assert report["accuracy_claim_permitted"] is False
    for name in (
        "metrics.json",
        "internal-metrics.csv",
        "grouped-metrics.csv",
        "paired-deltas.csv",
        "official-metrics.csv",
    ):
        assert (tmp_path / name).is_file()


def test_visualization_plan_separates_gt_and_prepares_video_entrypoint() -> None:
    _, manifest, _, _, _ = inputs()
    plan = build_visualization_plan(manifest)
    assert plan["execution_state"] == "PREP_ONLY_NOT_RENDERED"
    assert len(plan["producer"]) == 3
    assert len(plan["evaluator_only"]) == 5
    assert all("gt_overlay" not in row["layers"] for row in plan["producer"])
    assert all("gt_overlay" in row["layers"] for row in plan["evaluator_only"])
    assert plan["comparison_video"]["layout"] == "baseline_left_improved_right"
