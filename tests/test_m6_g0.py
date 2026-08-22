from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m6_g0_labels as labels  # noqa: E402
import m6_g0_features as features  # noqa: E402
import m6_g0_analysis as analysis  # noqa: E402
import m6_g0_report as report  # noqa: E402
import build_m6_g0_manifests as manifests  # noqa: E402
import validate_m6_g0 as validator  # noqa: E402


def _feature_row(
    target_id: str = "m2-target-1",
    *,
    object_id: int = 1,
    instance_id: str = "m2-instance-1",
) -> dict:
    return {
        "features": {"raw_selected_score": 1.0},
        "metadata": {
            "target_id": target_id,
            "object_id": object_id,
            "physical_instance_id": instance_id,
            "selected_sample_id": f"{target_id}-selected",
        },
    }


def _metric_row(
    target_id: str = "m2-target-1",
    *,
    joint: bool = True,
    normalized_mssd: float = 0.05,
    mspd_px: float = 10.0,
) -> dict:
    return {
        "record_type": "method_result",
        "method": "symmetry_aware_medoid",
        "requested_view_budget": 5,
        "target_sample_id": target_id,
        "object_id": 1,
        "selected_sample_id": f"{target_id}-selected",
        "diagnostic_success": {"joint": joint},
        "finite_pose": True,
        "normalized_mssd": normalized_mssd,
        "mspd_px": mspd_px,
        "mspd_scale_r": 2.0,
    }


def test_m3_label_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="M3 paths are forbidden"):
        labels.assert_m2_label_path(REPO_ROOT / "artifacts" / "m3" / "metrics.jsonl")


def test_target_aggregation_creates_one_row_per_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = [
        {"group_id": "group-b", "target_sample_id": "target-b", "object_id": 1},
        {"group_id": "group-a", "target_sample_id": "target-a", "object_id": 1},
    ]

    def fake_extract(group: dict, *_args: object) -> dict:
        return {
            "features": {"raw_selected_score": 1.0},
            "metadata": {"target_sample_id": group["target_sample_id"]},
        }

    monkeypatch.setattr(features, "extract_group_feature_row", fake_extract)
    rows = features.extract_feature_rows(groups, {}, {}, {1: {}})
    assert [row["metadata"]["target_sample_id"] for row in rows] == [
        "target-a",
        "target-b",
    ]
    with pytest.raises(ValueError, match="unique|exactly one group"):
        features.extract_feature_rows([groups[0], dict(groups[0])], {}, {}, {1: {}})


def test_label_builder_keeps_evaluator_fields_out_of_features() -> None:
    feature_rows = [_feature_row()]
    evaluator = labels.build_evaluator_rows([_metric_row()], feature_rows)
    assert evaluator[0]["correct"] is True
    assert evaluator[0]["y_failure"] == 0
    assert set(feature_rows[0]["features"]) == {"raw_selected_score"}
    assert "normalized_mssd" not in feature_rows[0]["features"]


def test_label_builder_reconstructs_inclusive_joint_threshold() -> None:
    metric = _metric_row(
        joint=True,
        normalized_mssd=labels.MSSD_THRESHOLD_DIAMETERS,
        mspd_px=labels.MSPD_THRESHOLD_R * 2.0,
    )
    row = labels.build_evaluator_rows([metric], [_feature_row()])[0]
    assert row["correct"] is True


def test_label_builder_detects_frozen_selection_mismatch() -> None:
    metric = _metric_row()
    metric["selected_sample_id"] = "different-candidate"
    with pytest.raises(ValueError, match="medoid selection mismatch"):
        labels.build_evaluator_rows([metric], [_feature_row()])


def test_label_builder_rejects_asymmetric_missing_output_pose() -> None:
    feature_row = _feature_row()
    feature_row["metadata"]["output_pose_m"] = np.eye(4).tolist()
    with pytest.raises(ValueError, match="output-pose mismatch"):
        labels.build_evaluator_rows([_metric_row()], [feature_row])


def test_label_support_summary_applies_all_minimums() -> None:
    rows = []
    for index in range(60):
        rows.append(
            {
                "target_id": f"t{index}",
                "object_id": index % 5,
                "physical_instance_id": f"g{index}",
                "y_failure": int(index < 30),
            }
        )
    summary = labels.label_support_summary(rows)
    assert summary["failure_count"] == 30
    assert summary["success_count"] == 30
    assert summary["physical_instances_with_failures"] == 30
    assert summary["objects_with_failures"] == 5
    assert summary["passed"] is True


def _object_data_with_half_turn_symmetry() -> dict:
    half_turn = np.diag([-1.0, -1.0, 1.0])
    return {
        "points_mm": np.asarray(
            [[-10.0, -5.0, 0.0], [10.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            dtype=np.float64,
        ),
        "diameter_mm": 25.0,
        "symmetries": (
            {"R": np.eye(3), "t": np.zeros((3, 1))},
            {"R": half_turn, "t": np.zeros((3, 1))},
        ),
    }


def test_symmetry_aware_disagreement_is_invariant_to_equivalent_pose() -> None:
    object_data = _object_data_with_half_turn_symmetry()
    pose = np.eye(4)
    pose[:3, 3] = [0.02, -0.01, 0.7]
    diagnostics = features.equivalent_symmetry_invariance(
        pose,
        object_data["symmetries"][1],
        object_data,
    )
    assert diagnostics["rotation_degrees"] == pytest.approx(0.0, abs=1e-7)
    assert diagnostics["normalized_mssd"] == pytest.approx(0.0, abs=1e-12)


def test_translation_and_rotation_dispersion_units() -> None:
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.012
    angle = np.deg2rad(90.0)
    second[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    identity_symmetry = ({"R": np.eye(3), "t": np.zeros((3, 1))},)
    assert features.translation_distance_mm(first, second) == pytest.approx(12.0)
    assert features.symmetry_aware_rotation_degrees(
        first, second, identity_symmetry
    ) == pytest.approx(90.0)


def test_invalid_candidate_handling_is_deterministic() -> None:
    invalid = {
        "sample_id": "candidate-1",
        "status": "inference_error",
        "foundationpose_top_score": None,
        "gt_model_to_camera_pose_m": [[999]],
        "normalized_mssd": -999,
    }
    first = features.sanitize_prediction(invalid, "candidate-1")
    second = features.sanitize_prediction(invalid, "candidate-1")
    assert first == second
    assert first == (
        "inference_error",
        None,
        None,
        "prediction_status_inference_error",
    )


def test_feature_sanitizer_drops_ground_truth_and_errors() -> None:
    raw = {
        "sample_id": "candidate-1",
        "status": "success",
        "predicted_model_to_camera_pose_m": np.eye(4).tolist(),
        "foundationpose_top_score": 3.0,
        "foundationpose_score_semantics": features.FOUNDATIONPOSE_SCORE_SEMANTICS,
        "gt_model_to_camera_pose_m": [[999]],
        "normalized_mssd": -999,
        "diagnostic_success": {"joint": True},
    }
    sanitized = features.sanitize_prediction_row(raw)
    assert "gt_model_to_camera_pose_m" not in sanitized
    assert "normalized_mssd" not in sanitized
    assert "diagnostic_success" not in sanitized


def test_no_prohibited_identifier_enters_feature_schema() -> None:
    row = {
        "features": dict.fromkeys(features.FEATURE_SPEC, None),
        "metadata": {"target_id": "metadata-only"},
    }
    for name, specification in features.FEATURE_SPEC.items():
        if not specification["nullable"]:
            row["features"][name] = 0
    assert features.validate_feature_schema(row) is True
    row["features"]["object_id"] = 1
    with pytest.raises(ValueError, match="schema mismatch|Prohibited"):
        features.validate_feature_schema(row)


def test_m3_feature_paths_and_generic_roots_are_rejected() -> None:
    with pytest.raises(ValueError, match="M3 paths are prohibited"):
        features.assert_safe_input_path(REPO_ROOT / "artifacts" / "m3" / "x.jsonl")
    with pytest.raises(ValueError, match="Generic artifact"):
        features.assert_safe_input_path(REPO_ROOT / "artifacts")


@pytest.mark.parametrize(
    "payload",
    [
        {"selected_sample_id": "m3-secret-target"},
        {"sample_id": "m3-secret-target"},
        {"candidate_sample_ids": ["m3-secret-target"]},
        {"provenance": "artifacts/m3/raw.jsonl"},
        {"medoid_selection_scores": {"xyzibd-m3-secret": 1.0}},
    ],
)
def test_formal_m3_scan_rejects_all_id_and_path_bypasses(payload: dict) -> None:
    with pytest.raises(ValueError, match="M3 path|Identifier|dataset identifier"):
        validator._validate_no_m3_inputs_or_ids(  # noqa: SLF001
            REPO_ROOT,
            {"synthetic": payload},
        )


def test_formal_m3_scan_accepts_protected_m1_m2_identifier_universes() -> None:
    universes = validator._m2_identifier_universes(REPO_ROOT)  # noqa: SLF001
    target_id = sorted(universes["target"])[0]
    sample_ids = sorted(universes["sample"])
    instance_id = sorted(universes["instance"])[0]
    evidence = validator._validate_no_m3_inputs_or_ids(  # noqa: SLF001
        REPO_ROOT,
        {
            "synthetic": {
                "target_id": target_id,
                "group_id": target_id,
                "selected_sample_id": sample_ids[0],
                "candidate_sample_ids": sample_ids[:2],
                "physical_instance_id": instance_id,
                "medoid_selection_scores": {sample_ids[0]: 0.1},
            }
        },
    )
    assert evidence["non_m1_m2_identifiers_absent"] is True
    assert evidence["checked_output_id_occurrences"] == 6


def test_tracked_access_disclosure_strictly_matches_frozen_source_audit() -> None:
    tracked = json.loads(
        (REPO_ROOT / validator.ACCESS_BOUNDARY_DISCLOSURE_PATH).read_text(
            encoding="utf-8"
        )
    )
    source_audit = json.loads(
        (REPO_ROOT / validator.FROZEN_DIR / "source_audit.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = validator._validate_access_boundary_disclosure(  # noqa: SLF001
        tracked,
        source_audit["m3_access_disclosure"],
    )
    assert evidence["passed"] is True

    drifted = dict(tracked)
    drifted["m3_candidate_values_accessed"] = True
    with pytest.raises(ValueError, match="tracked M3 access-boundary disclosure"):
        validator._validate_access_boundary_disclosure(  # noqa: SLF001
            drifted,
            source_audit["m3_access_disclosure"],
        )


def test_maximum_mutual_agreement_count_is_exact() -> None:
    matrix = np.asarray(
        [[0.0, 0.01, 0.20], [0.01, 0.0, 0.15], [0.20, 0.15, 0.0]],
        dtype=np.float64,
    )
    assert features.maximum_mutual_agreement_count(matrix, threshold=0.10) == 2


def test_raw_score_rank_aurc_hand_example_is_exact() -> None:
    assert analysis.aurc([0, 1], [0.0, 1.0], tie_breaker=["a", "b"]) == pytest.approx(
        0.25
    )


def test_risk_coverage_curve_hand_example_is_exact() -> None:
    curve = analysis.risk_coverage_curve(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.3, 0.4],
        tie_breaker=["a", "b", "c", "d"],
    )
    assert curve["coverage"] == [0.25, 0.5, 0.75, 1.0]
    assert curve["empirical_failure_risk"] == pytest.approx([0.0, 0.0, 1.0 / 3.0, 0.5])
    assert curve["aurc"] == pytest.approx(5.0 / 24.0)


def test_tied_isotonic_risk_curve_is_identifier_invariant() -> None:
    labels = [1, 0, 0, 1]
    tied_risk = [0.2, 0.2, 0.8, 0.8]
    first = analysis.risk_coverage_curve(
        labels,
        tied_risk,
        tie_breaker=["object-z", "object-a", "target-z", "target-a"],
    )
    second = analysis.risk_coverage_curve(
        labels,
        tied_risk,
        tie_breaker=["target-a", "target-z", "object-a", "object-z"],
    )
    assert first == second
    assert first["empirical_failure_risk"] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert first["aurc"] == pytest.approx(0.5)
    assert (
        first["equal_risk_tie_policy"] == "expected_over_all_within_block_permutations"
    )


def test_report_per_object_aurc_is_invariant_within_risk_ties() -> None:
    rows = [
        {"target_id": "object-z", "risk_score": 0.2, "y_failure": 1},
        {"target_id": "object-a", "risk_score": 0.2, "y_failure": 0},
        {"target_id": "target-z", "risk_score": 0.8, "y_failure": 0},
        {"target_id": "target-a", "risk_score": 0.8, "y_failure": 1},
    ]
    permuted = [
        {**rows[3], "target_id": "aaa"},
        {**rows[1], "target_id": "zzz"},
        {**rows[2], "target_id": "bbb"},
        {**rows[0], "target_id": "yyy"},
    ]

    expected = analysis.aurc(
        [int(row["y_failure"]) for row in rows],
        [float(row["risk_score"]) for row in rows],
    )
    assert report._aurc_for_rows(rows) == pytest.approx(expected)
    assert report._aurc_for_rows(permuted) == pytest.approx(expected)


def test_report_labels_positive_aurc_gain_as_isotonic_minus_nested() -> None:
    rendered = report._robustness_tables(
        {
            "object_jackknife": {
                "aggregate_absolute_aurc_gain": 0.2,
                "object_driven": False,
                "object_driven_flags": {},
            }
        }
    )
    assert "Aggregate isotonic-minus-nested absolute AURC gain" in rendered
    assert "Aggregate nested-minus-isotonic" not in rendered


def test_report_object_presentations_use_numeric_then_text_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    for object_id in (10, 2, 1, "other"):
        for method in ("NESTED_MULTIFEATURE", "SCORE_ISOTONIC"):
            rows.append(
                {
                    "method": method,
                    "object_id": object_id,
                    "target_id": f"{object_id}-{method}",
                    "risk_score": 0.5,
                    "y_failure": 0,
                }
            )

    differences = report._per_object_aurc_differences({"oof_predictions": rows})
    assert [object_id for object_id, _ in differences] == ["1", "2", "10", "other"]

    removed = report._sorted_object_records(
        [{"removed_object_id": value} for value in (10, 2, 1, "other")],
        "removed_object_id",
    )
    assert [row["removed_object_id"] for row in removed] == [1, 2, 10, "other"]

    captured: dict[str, list[str]] = {}

    def capture_labels(fig: object, _path: Path) -> None:
        axes = getattr(fig, "axes")
        captured["labels"] = [tick.get_text() for tick in axes[0].get_xticklabels()]
        report.plt.close(fig)

    monkeypatch.setattr(report, "_save_figure", capture_labels)
    report._failure_prevalence_plot(
        tmp_path / "prevalence.png",
        [],
        {
            "failure_prevalence": 0.0,
            "by_object": [
                {
                    "object_id": value,
                    "target_count": 1,
                    "failure_count": 0,
                    "failure_prevalence": 0.0,
                }
                for value in (10, 2, 1, "other")
            ],
        },
    )
    assert captured["labels"] == ["1", "2", "10", "other"]


def test_report_source_hashes_require_and_include_access_disclosure(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.md"
    plot_path = tmp_path / "plot.png"
    report_path.write_text("report", encoding="utf-8")
    plot_path.write_bytes(b"plot")

    with pytest.raises(FileNotFoundError, match="access-boundary disclosure"):
        report._source_hashes(tmp_path, report_path, [plot_path])

    disclosure = tmp_path / "precomputed" / "m6_g0" / "access_boundary_disclosure.json"
    disclosure.parent.mkdir(parents=True)
    disclosure.write_text("{}\n", encoding="utf-8")
    hashes = report._source_hashes(tmp_path, report_path, [plot_path])
    relative = disclosure.relative_to(tmp_path).as_posix()
    assert hashes[relative] == report._sha256_file(disclosure)


def test_equal_frequency_ece_hand_example_is_exact() -> None:
    calibration = analysis.equal_frequency_calibration_error(
        [0, 1, 0, 1],
        [0.1, 0.2, 0.8, 0.9],
        n_bins=2,
        tie_breaker=["a", "b", "c", "d"],
    )
    assert calibration["ece"] == pytest.approx(0.35)
    assert calibration["mce"] == pytest.approx(0.35)


def test_equal_frequency_ece_is_invariant_within_probability_ties() -> None:
    probability = [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8]
    first = analysis.equal_frequency_calibration_error(
        [1, 1, 0, 0, 1, 0, 1, 0],
        probability,
        n_bins=4,
        tie_breaker=[f"first-{index}" for index in range(8)],
    )
    second = analysis.equal_frequency_calibration_error(
        [0, 1, 0, 1, 0, 1, 0, 1],
        probability,
        n_bins=4,
        tie_breaker=[f"second-{7 - index}" for index in range(8)],
    )
    assert first == second
    assert first["ece"] == pytest.approx(0.3)
    assert first["mce"] == pytest.approx(0.3)
    assert (
        first["equal_probability_tie_policy"]
        == "expected_failure_allocation_over_within_tie_permutations"
    )


def test_grouped_bootstrap_resamples_whole_instances() -> None:
    samples = analysis.grouped_bootstrap_indices(
        ["a", "a", "b", "c", "c", "c"],
        n_resamples=25,
        seed=123,
    )
    for indices in samples:
        counts = np.bincount(indices, minlength=6)
        assert counts[0] == counts[1]
        assert counts[3] == counts[4] == counts[5]


def test_grouped_fold_manifest_is_deterministic_and_nested() -> None:
    evaluator_rows = [
        {
            "target_id": f"target-{index:03d}",
            "physical_instance_id": f"instance-{index:03d}",
            "object_id": index % 5,
            "y_failure": (index // 5) % 2,
        }
        for index in range(50)
    ]
    first = analysis.make_fold_manifest(evaluator_rows)
    second = analysis.make_fold_manifest(evaluator_rows)
    assert analysis.fold_manifest_hash(first) == analysis.fold_manifest_hash(second)
    validation = analysis.validate_fold_manifest(first, evaluator_rows)
    assert validation["passed"] is True
    assert validation["checks"]["physical_instances_do_not_cross_outer_folds"] is True
    assert validation["checks"]["inner_partitions_inside_outer_train"] is True


def test_feature_ablation_manifest_is_deterministic_and_complete() -> None:
    first = analysis.build_ablation_feature_families(
        features.FEATURE_FAMILIES,
        raw_score_feature="raw_selected_score",
    )
    second = analysis.build_ablation_feature_families(
        features.FEATURE_FAMILIES,
        raw_score_feature="raw_selected_score",
    )
    assert first == second
    assert first["score_only"]
    assert first["disagreement_only"]
    assert set(first["score_disagreement"]) == set(first["score_only"]) | set(
        first["disagreement_only"]
    )
    assert set(first["score_disagreement"]) <= set(first["all"])


def test_validator_reconstructs_every_ablation_metric_from_persisted_oof() -> None:
    target_ids = [f"target-{index:03d}" for index in range(8)]
    y_failure = [0, 1, 0, 1, 0, 1, 0, 1]
    outer_folds = [0, 0, 0, 0, 1, 1, 1, 1]
    evaluator_by_target = {
        target_id: {"target_id": target_id, "y_failure": y_failure[index]}
        for index, target_id in enumerate(target_ids)
    }
    outer_assignment = dict(zip(target_ids, outer_folds, strict=True))

    def rows_for(
        method: str,
        risk_scores: list[float],
        probabilities: list[float] | None,
        *,
        ablation: str | None = None,
    ) -> list[dict]:
        rows = []
        for index, target_id in enumerate(target_ids):
            row = {
                "target_id": target_id,
                "y_failure": y_failure[index],
                "outer_fold": outer_folds[index],
                "method": method,
                "risk_score": risk_scores[index],
                "failure_probability": (
                    None if probabilities is None else probabilities[index]
                ),
            }
            if ablation is not None:
                row = {"ablation": ablation, **row}
            rows.append(row)
        return rows

    raw_risk = [0.20, 0.80, 0.30, 0.70, 0.25, 0.75, 0.35, 0.65]
    isotonic_risk = [0.22, 0.78, 0.32, 0.68, 0.27, 0.73, 0.37, 0.63]
    formal_by_method = {
        "RAW_SCORE_RANK": rows_for("RAW_SCORE_RANK", raw_risk, None),
        "SCORE_ISOTONIC": rows_for("SCORE_ISOTONIC", isotonic_risk, isotonic_risk),
    }
    raw_aurc = analysis.compute_prediction_metrics(
        y_failure, raw_risk, failure_probability=None, tie_breaker=target_ids
    )["aurc"]
    isotonic_aurc = analysis.compute_prediction_metrics(
        y_failure,
        isotonic_risk,
        failure_probability=isotonic_risk,
        tie_breaker=target_ids,
    )["aurc"]
    risk_by_ablation = {
        "score_only": [0.18, 0.82, 0.28, 0.72, 0.23, 0.77, 0.33, 0.67],
        "disagreement_only": [0.12, 0.88, 0.24, 0.76, 0.16, 0.84, 0.36, 0.64],
        "score_disagreement": [0.10, 0.90, 0.20, 0.80, 0.15, 0.85, 0.30, 0.70],
        "all": [0.08, 0.92, 0.22, 0.78, 0.14, 0.86, 0.34, 0.66],
    }
    formal_by_method["NESTED_MULTIFEATURE"] = rows_for(
        "NESTED_MULTIFEATURE",
        risk_by_ablation["all"],
        risk_by_ablation["all"],
    )
    ablation_rows: list[dict] = []
    feature_ablations = {}
    for ablation in analysis.ABLATION_NAMES:
        risks = risk_by_ablation[ablation]
        rows = rows_for("NESTED_MULTIFEATURE", risks, risks, ablation=ablation)
        ablation_rows.extend(rows)
        aggregate = analysis.compute_prediction_metrics(
            y_failure, risks, failure_probability=risks, tie_breaker=target_ids
        )
        aggregate["relative_aurc_improvement_vs_raw_score_rank"] = (
            analysis.relative_aurc_improvement(raw_aurc, aggregate["aurc"])
        )
        aggregate["relative_aurc_improvement_vs_score_isotonic"] = (
            analysis.relative_aurc_improvement(isotonic_aurc, aggregate["aurc"])
        )
        per_fold = []
        for fold in (0, 1):
            selected = [
                index for index, value in enumerate(outer_folds) if value == fold
            ]
            per_fold.append(
                {
                    "outer_fold": fold,
                    "method": "NESTED_MULTIFEATURE",
                    "metrics": analysis.compute_prediction_metrics(
                        [y_failure[index] for index in selected],
                        [risks[index] for index in selected],
                        failure_probability=[risks[index] for index in selected],
                        tie_breaker=[target_ids[index] for index in selected],
                    ),
                }
            )
        feature_ablations[ablation] = {
            "aggregate_metrics": aggregate,
            "per_fold_metrics": per_fold,
        }

    evidence = validator._validate_ablation_metric_reconstruction(  # noqa: SLF001
        feature_ablations,
        ablation_rows,
        evaluator_by_target,
        outer_assignment,
        formal_by_method,
    )
    assert evidence == {
        "ablation_count": 4,
        "prediction_row_count": 32,
        "per_fold_metric_row_count": 8,
        "each_target_once_per_ablation": True,
        "all_ablation_matches_formal_nested_oof": True,
        "pooled_and_fold_metrics_exact": True,
    }

    tampered_rows = [dict(row) for row in ablation_rows]
    tampered_rows[0]["risk_score"] = 0.99
    with pytest.raises(ValueError, match="ablation pooled metric reconstruction"):
        validator._validate_ablation_metric_reconstruction(  # noqa: SLF001
            feature_ablations,
            tampered_rows,
            evaluator_by_target,
            outer_assignment,
            formal_by_method,
        )


def test_oof_prediction_validation_requires_exactly_one_row_per_method() -> None:
    target_ids = ["target-a", "target-b"]
    rows = [
        {"method": method, "target_id": target_id}
        for method in analysis.FORMAL_METHODS
        for target_id in target_ids
    ]
    assert analysis.validate_oof_predictions(rows, target_ids)["passed"] is True
    duplicated = [*rows, dict(rows[0])]
    invalid = analysis.validate_oof_predictions(duplicated, target_ids)
    assert invalid["passed"] is False
    assert any("duplicate" in error for error in invalid["errors"])


def _formal_prediction_inputs() -> tuple[list[dict], dict, dict, dict]:
    prediction_rows = validator._strict_jsonl(  # noqa: SLF001
        REPO_ROOT / validator.JSONL_ARTIFACTS["oof_predictions"]
    )
    feature_rows = validator._strict_jsonl(  # noqa: SLF001
        REPO_ROOT / validator.JSONL_ARTIFACTS["features"]
    )
    evaluator_rows = validator._strict_jsonl(  # noqa: SLF001
        REPO_ROOT / validator.JSONL_ARTIFACTS["evaluator_rows"]
    )
    fold_manifest = validator._strict_json(  # noqa: SLF001
        REPO_ROOT / validator.FROZEN_DIR / "fold_manifest.json"
    )
    feature_by_target = {
        validator._target_id_from_feature(row): row  # noqa: SLF001
        for row in feature_rows
    }
    evaluator_by_target = {
        validator._target_id_from_evaluator(row): row  # noqa: SLF001
        for row in evaluator_rows
    }
    outer_assignment = {
        str(row["target_id"]): int(row["outer_fold"])
        for row in fold_manifest["outer_assignments"]
    }
    return (
        prediction_rows,
        feature_by_target,
        evaluator_by_target,
        outer_assignment,
    )


def test_formal_prediction_semantics_reject_raw_orientation_tamper() -> None:
    rows, features_by_target, evaluators_by_target, outer_assignment = (
        _formal_prediction_inputs()
    )
    tampered = copy.deepcopy(rows)
    raw_row = next(row for row in tampered if row["method"] == "RAW_SCORE_RANK")
    raw_row["risk_score"] = float(raw_row["risk_score"]) + 1.0
    with pytest.raises(ValueError, match="negative frozen confidence score"):
        validator._validate_oof_predictions(  # noqa: SLF001
            tampered,
            features_by_target,
            evaluators_by_target,
            outer_assignment,
        )


def test_formal_prediction_semantics_reject_probability_tamper() -> None:
    rows, features_by_target, evaluators_by_target, outer_assignment = (
        _formal_prediction_inputs()
    )
    tampered = copy.deepcopy(rows)
    calibrated = next(
        row for row in tampered if row["method"] == "LOGISTIC_MULTIFEATURE"
    )
    calibrated["risk_score"] = float(calibrated["failure_probability"]) / 2.0
    with pytest.raises(ValueError, match="risk differs from failure probability"):
        validator._validate_oof_predictions(  # noqa: SLF001
            tampered,
            features_by_target,
            evaluators_by_target,
            outer_assignment,
        )


def test_nested_oof_rows_link_exactly_to_fold_selected_fixed_family() -> None:
    rows, features_by_target, evaluators_by_target, outer_assignment = (
        _formal_prediction_inputs()
    )
    by_method, score_evidence = validator._validate_oof_predictions(  # noqa: SLF001
        rows,
        features_by_target,
        evaluators_by_target,
        outer_assignment,
    )
    model_selections = validator._strict_json(  # noqa: SLF001
        REPO_ROOT / validator.JSON_ARTIFACTS["model_selections"]
    )
    linkage = validator._validate_nested_prediction_linkage(  # noqa: SLF001
        by_method,
        model_selections,
    )
    assert score_evidence["raw_score_rank_rows_with_frozen_negative_orientation"] == 300
    assert linkage["nested_rows_linked_to_selected_fixed_family"] == 300

    tampered_rows = copy.deepcopy(rows)
    nested = next(
        row for row in tampered_rows if row["method"] == "NESTED_MULTIFEATURE"
    )
    changed = 0.0 if float(nested["failure_probability"]) > 0.5 else 1.0
    nested["risk_score"] = changed
    nested["failure_probability"] = changed
    tampered_by_method, _ = validator._validate_oof_predictions(  # noqa: SLF001
        tampered_rows,
        features_by_target,
        evaluators_by_target,
        outer_assignment,
    )
    with pytest.raises(ValueError, match="nested risk linkage"):
        validator._validate_nested_prediction_linkage(  # noqa: SLF001
            tampered_by_method,
            model_selections,
        )


def test_m3_boundary_forces_no_go_and_explicitly_denies_holdout() -> None:
    aggregate = {
        "RAW_SCORE_RANK": {"aurc": 0.20},
        "SCORE_ISOTONIC": {"aurc": 0.15, "brier": 0.10},
        "NESTED_MULTIFEATURE": {
            "auroc": 0.80,
            "aurc": 0.10,
            "brier": 0.10,
            "ece": 0.02,
            "predicted_risk_le_0_05_coverage": 0.50,
            "predicted_risk_le_0_05_empirical_failure_rate": 0.04,
            "predicted_risk_le_0_05_count": 40,
        },
    }
    bootstrap = {
        "paired_differences": {
            "aurc_improvement_vs_score_isotonic": {"lower": 0.01, "upper": 0.09}
        }
    }
    robustness = {
        "leave_one_object_out": [{"improvement_remains_non_negative": True}],
        "object_driven": False,
    }
    decision = analysis.frozen_signal_decision(
        aggregate,
        bootstrap,
        robustness,
        feature_ablations={},
        stable_disagreement_benefit=True,
        leakage_checks_passed=True,
        label_support_passed=True,
        oof_complete=True,
        m3_access_boundary_passed=False,
    )
    assert decision["classification"] == "NO-GO"
    assert decision["m3_holdout_authorized"] is False


def _decision_test_inputs() -> tuple[dict, dict, dict]:
    aggregate = {
        "RAW_SCORE_RANK": {"aurc": 0.20},
        "SCORE_ISOTONIC": {"aurc": 0.20, "brier": 0.10},
        "NESTED_MULTIFEATURE": {
            "auroc": 0.75,
            "aurc": 0.18,
            "brier": 0.102,
            "ece": 0.05,
            "predicted_risk_le_0_05_coverage": 0.40,
            "predicted_risk_le_0_05_empirical_failure_rate": 0.05,
            "predicted_risk_le_0_05_count": 30,
        },
    }
    bootstrap = {
        "paired_differences": {
            "aurc_improvement_vs_score_isotonic": {
                "lower": 1e-12,
                "upper": 0.10,
            }
        }
    }
    robustness = {
        "leave_one_object_out": [{"improvement_remains_non_negative": True}],
        "object_driven": False,
    }
    return aggregate, bootstrap, robustness


def test_validator_independently_enforces_frozen_decision_boundaries() -> None:
    aggregate, bootstrap, robustness = _decision_test_inputs()
    go = validator._reconstruct_frozen_decision(  # noqa: SLF001
        aggregate,
        bootstrap,
        robustness,
        stable_disagreement_benefit=True,
        leakage_checks_passed=True,
        label_support_passed=True,
        oof_complete=True,
        m3_access_boundary_passed=True,
    )
    assert go["classification"] == "SIGNAL GO"
    assert all(go["go_conditions"].values())

    weak_metrics = copy.deepcopy(aggregate)
    weak_metrics["NESTED_MULTIFEATURE"]["auroc"] = 0.70
    weak_metrics["NESTED_MULTIFEATURE"]["aurc"] = 0.19
    weak = validator._reconstruct_frozen_decision(  # noqa: SLF001
        weak_metrics,
        bootstrap,
        robustness,
        stable_disagreement_benefit=True,
        leakage_checks_passed=True,
        label_support_passed=True,
        oof_complete=True,
        m3_access_boundary_passed=True,
    )
    assert weak["classification"] == "WEAK SIGNAL / HOLDOUT NOT AUTHORIZED"
    assert all(weak["weak_conditions"].values())

    below_weak = copy.deepcopy(weak_metrics)
    below_weak["NESTED_MULTIFEATURE"]["auroc"] = np.nextafter(0.70, 0.0)
    no_go = validator._reconstruct_frozen_decision(  # noqa: SLF001
        below_weak,
        bootstrap,
        robustness,
        stable_disagreement_benefit=True,
        leakage_checks_passed=True,
        label_support_passed=True,
        oof_complete=True,
        m3_access_boundary_passed=True,
    )
    assert no_go["classification"] == "NO-GO"
    assert no_go["weak_conditions"]["auroc_at_least_0_70"] is False


def test_formal_decision_validation_does_not_call_analysis_decision_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate, bootstrap, robustness = _decision_test_inputs()
    ablations = {
        "score_only": {
            "aggregate_metrics": {"aurc": 0.20},
            "per_fold_metrics": [
                {"outer_fold": fold, "metrics": {"aurc": 0.20}} for fold in range(5)
            ],
        },
        "score_disagreement": {
            "aggregate_metrics": {"aurc": 0.19},
            "per_fold_metrics": [
                {"outer_fold": fold, "metrics": {"aurc": 0.19}} for fold in range(5)
            ],
        },
    }
    decision = validator._reconstruct_frozen_decision(  # noqa: SLF001
        aggregate,
        bootstrap,
        robustness,
        stable_disagreement_benefit=True,
        leakage_checks_passed=True,
        label_support_passed=True,
        oof_complete=True,
        m3_access_boundary_passed=False,
    )

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("production decision helper must not be called")

    monkeypatch.setattr(analysis, "frozen_signal_decision", forbidden)
    evidence = validator._validate_decision(  # noqa: SLF001
        decision,
        {
            "stable_disagreement_benefit": True,
            "leakage_checks_passed": True,
            "label_support_passed": True,
            "oof_complete": True,
        },
        aggregate,
        bootstrap,
        robustness,
        ablations,
    )
    assert evidence["decision_exactly_reconstructed"] is True
    assert evidence["classification"] == "NO-GO"


def test_stable_disagreement_gate_does_not_credit_availability_features() -> None:
    fold_rows = lambda values: [  # noqa: E731
        {"outer_fold": fold, "metrics": {"aurc": value}}
        for fold, value in enumerate(values)
    ]
    ablations = {
        "score_only": {
            "aggregate_metrics": {"aurc": 0.20},
            "per_fold_metrics": fold_rows([0.20] * 5),
        },
        "score_disagreement": {
            "aggregate_metrics": {"aurc": 0.20},
            "per_fold_metrics": fold_rows([0.20] * 5),
        },
        "all": {
            "aggregate_metrics": {"aurc": 0.10},
            "per_fold_metrics": fold_rows([0.10] * 5),
        },
    }
    assert analysis._stable_disagreement_benefit(ablations) is False  # noqa: SLF001


def test_isotonic_and_nested_selection_ignore_outer_test_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "LOGISTIC_C_GRID", (1.0,))
    monkeypatch.setattr(
        analysis,
        "TREE_PARAM_GRID",
        ({"max_depth": 2, "n_estimators": 5, "min_samples_leaf": 2},),
    )
    target_ids = [f"target-{index:02d}" for index in range(40)]
    outer_folds = np.asarray([index % 2 for index in range(40)], dtype=int)
    labels_original = np.asarray([(index // 2) % 2 for index in range(40)], dtype=int)
    labels_mutated = labels_original.copy()
    labels_mutated[outer_folds == 0] = 1 - labels_mutated[outer_folds == 0]
    inner_folds = {
        outer_fold: {
            analysis._id_key(target_ids[index]): (index // 2) % 4  # noqa: SLF001
            for index in range(40)
            if outer_folds[index] != outer_fold
        }
        for outer_fold in (0, 1)
    }
    x = np.column_stack([np.linspace(-1.0, 1.0, 40), np.cos(np.linspace(0.0, 3.0, 40))])
    raw_score = np.linspace(10.0, 20.0, 40)

    def evaluate(y: np.ndarray) -> dict:
        return analysis._cross_validated_models(  # noqa: SLF001
            x,
            y,
            raw_score,
            target_ids,
            outer_folds,
            inner_folds,
            ["raw_selected_score", "pair_normalized_mssd_median"],
            raw_score_higher_is_confident=True,
            collect_contributions=False,
        )

    original = evaluate(labels_original)
    mutated = evaluate(labels_mutated)
    fold_zero = outer_folds == 0
    assert original["probabilities"]["SCORE_ISOTONIC"][fold_zero] == pytest.approx(
        mutated["probabilities"]["SCORE_ISOTONIC"][fold_zero]
    )
    assert original["model_selections"][0] == mutated["model_selections"][0]
    assert (
        original["model_selections"][0]["outer_test_labels_used_for_selection"] is False
    )


def test_tree_inner_selection_uses_raw_oof_ranking_before_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = {"max_depth": 2, "n_estimators": 5, "min_samples_leaf": 2}
    monkeypatch.setattr(analysis, "TREE_PARAM_GRID", (parameters,))
    labels_inner = np.tile([0, 1], 4)
    inner_folds = np.repeat(np.arange(4), 2)
    raw_oof = np.tile([0.1, 0.9], 4)
    same_label_fit_calibrated = 1.0 - raw_oof

    def fake_inner_oof_tree(*_args: object, **_kwargs: object) -> tuple:
        return raw_oof.copy(), same_label_fit_calibrated.copy(), object()

    monkeypatch.setattr(analysis, "_inner_oof_tree", fake_inner_oof_tree)
    selected, audit = analysis._select_tree(  # noqa: SLF001
        np.zeros((8, 1), dtype=float),
        labels_inner,
        inner_folds,
        [f"target-{index}" for index in range(8)],
        seed=123,
    )
    raw_mean, _ = analysis._mean_fold_aurc(  # noqa: SLF001
        labels_inner,
        raw_oof,
        inner_folds,
        [f"target-{index}" for index in range(8)],
    )
    calibrated_mean, _ = analysis._mean_fold_aurc(  # noqa: SLF001
        labels_inner,
        same_label_fit_calibrated,
        inner_folds,
        [f"target-{index}" for index in range(8)],
    )
    assert selected == parameters
    assert audit[0]["inner_mean_aurc"] == pytest.approx(raw_mean)
    assert audit[0]["inner_mean_aurc"] != pytest.approx(calibrated_mean)


def test_protected_m1_m5_receipt_remains_valid() -> None:
    receipt = manifests.validate_protected_baseline(REPO_ROOT)
    assert receipt["passed"] is True
    assert receipt["entry_count"] == 182
    assert receipt["m3_hashes_unchanged"] is True
    assert receipt["all_m1_m5_evidence_unchanged"] is True


def test_outer_test_values_do_not_change_missing_value_imputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "LOGISTIC_C_GRID", (1.0,))
    monkeypatch.setattr(
        analysis,
        "TREE_PARAM_GRID",
        ({"max_depth": 2, "n_estimators": 5, "min_samples_leaf": 2},),
    )
    target_ids = [f"target-{index:02d}" for index in range(40)]
    outer_folds = np.asarray([index % 2 for index in range(40)], dtype=int)
    y_failure = np.asarray([(index // 2) % 2 for index in range(40)], dtype=int)
    inner_folds = {
        outer_fold: {
            analysis._id_key(target_ids[index]): (index // 2) % 4  # noqa: SLF001
            for index in range(40)
            if outer_folds[index] != outer_fold
        }
        for outer_fold in (0, 1)
    }
    raw_score = np.linspace(10.0, 20.0, 40)
    first_feature = np.linspace(-1.0, 1.0, 40)
    second_feature = np.linspace(1.0, 4.0, 40)
    sentinel = int(np.flatnonzero(outer_folds == 0)[0])

    def evaluate(held_out_fill: float) -> dict:
        x = np.column_stack([first_feature, second_feature])
        x[outer_folds == 0, 1] = held_out_fill
        x[sentinel, 1] = np.nan
        return analysis._cross_validated_models(  # noqa: SLF001
            x,
            y_failure,
            raw_score,
            target_ids,
            outer_folds,
            inner_folds,
            ["raw_selected_score", "pair_normalized_mssd_median"],
            raw_score_higher_is_confident=True,
            collect_contributions=False,
        )

    high = evaluate(1_000_000.0)
    low = evaluate(-1_000_000.0)
    for method in (
        "LOGISTIC_MULTIFEATURE",
        "SHALLOW_TREE_MULTIFEATURE",
        "NESTED_MULTIFEATURE",
    ):
        assert high["probabilities"][method][sentinel] == pytest.approx(
            low["probabilities"][method][sentinel],
            abs=1e-12,
        )
    assert high["model_selections"][0] == low["model_selections"][0]


def test_object_jackknife_metrics_are_reconstructible() -> None:
    y = [0, 1, 0, 1, 0, 1]
    candidate = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
    isotonic = [0.2, 0.6, 0.3, 0.5, 0.4, 0.7]
    result = analysis.object_robustness_analysis(
        y,
        candidate,
        isotonic,
        [1, 1, 2, 2, 3, 3],
        ["a", "b", "c", "d", "e", "f"],
        ["t0", "t1", "t2", "t3", "t4", "t5"],
    )
    expected_gain = analysis.aurc(
        y, isotonic, tie_breaker=[f"t{i}" for i in range(6)]
    ) - analysis.aurc(y, candidate, tie_breaker=[f"t{i}" for i in range(6)])
    assert result["aggregate_absolute_aurc_gain"] == pytest.approx(expected_gain)
    assert len(result["leave_one_object_out"]) == 3


def test_additive_aurc_gain_contributions_sum_and_define_group_shares() -> None:
    y_failure = [1, 0, 1, 0]
    isotonic_tie = [0.5, 0.5, 0.5, 0.5]
    candidate = [0.9, 0.1, 0.8, 0.2]
    contributions = analysis._per_target_aurc_gain_contributions(  # noqa: SLF001
        y_failure,
        candidate,
        isotonic_tie,
    )
    expected_gain = analysis.aurc(y_failure, isotonic_tie) - analysis.aurc(
        y_failure, candidate
    )
    assert contributions == pytest.approx([3.0 / 16.0, 0.0, 5.0 / 48.0, 0.0])
    assert float(np.sum(contributions)) == pytest.approx(expected_gain)

    result = analysis.object_robustness_analysis(
        y_failure,
        candidate,
        isotonic_tie,
        [1, 1, 2, 2],
        ["instance-a", "instance-a", "instance-b", "instance-b"],
        ["target-a", "target-b", "target-c", "target-d"],
    )
    object_rows = result["leave_one_object_out"]
    instance_rows = result["physical_instance_gain_contributions"]
    assert sum(
        row["additive_aurc_gain_contribution"] for row in object_rows
    ) == pytest.approx(expected_gain)
    assert sum(
        row["aggregate_gain_contribution"] for row in instance_rows
    ) == pytest.approx(expected_gain)
    assert sum(
        row["fraction_of_aggregate_gain"] for row in object_rows
    ) == pytest.approx(1.0)
    assert sum(
        row["fraction_of_aggregate_gain"] for row in instance_rows
    ) == pytest.approx(1.0)
    assert result["object_driven_flags"]["one_object_exceeds_40_percent_of_gain"]
    assert result["object_driven_flags"]["one_instance_exceeds_20_percent_of_gain"]


def test_object_deferred_expectation_is_invariant_to_boundary_tie_permutation() -> None:
    y_failure = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    candidate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8]
    isotonic = [0.5] * 10
    object_ids = ["safe"] * 7 + ["alpha", "beta", "alpha"]
    instance_ids = [f"instance-{index}" for index in range(10)]
    target_ids = [f"target-{index}" for index in range(10)]

    def run(permutation: list[int]) -> dict:
        return analysis.object_robustness_analysis(
            np.asarray(y_failure)[permutation].tolist(),
            np.asarray(candidate)[permutation].tolist(),
            np.asarray(isotonic)[permutation].tolist(),
            np.asarray(object_ids)[permutation].tolist(),
            np.asarray(instance_ids)[permutation].tolist(),
            np.asarray(target_ids)[permutation].tolist(),
        )

    original = run(list(range(10)))
    permuted = run([0, 1, 2, 3, 4, 5, 6, 9, 7, 8])

    def object_summary(result: dict) -> dict[str, tuple[float, float | None]]:
        return {
            row["object_id"]: (
                row["expected_correctly_deferred_failure_count"],
                row["fraction_of_expected_correctly_deferred_failures"],
            )
            for row in result["object_correctly_deferred_failure_contributions"]
        }

    original_summary = object_summary(original)
    permuted_summary = object_summary(permuted)
    assert original_summary.keys() == permuted_summary.keys()
    for object_id in original_summary:
        assert permuted_summary[object_id] == pytest.approx(original_summary[object_id])
    assert original_summary["alpha"] == pytest.approx((2.0 / 3.0, 0.5))
    assert original_summary["beta"] == pytest.approx((2.0 / 3.0, 0.5))
    assert original_summary["safe"] == pytest.approx((0.0, 0.0))

    metadata = original["object_correctly_deferred_failure_contributions_metadata"]
    assert metadata["cutoff_splits_equal_risk_block"] is True
    assert metadata["identifiers_used_as_secondary_ranking_inputs"] is False
    assert metadata["equal_risk_boundary_policy"] == (
        "uniform_expected_allocation_within_boundary_block"
    )
    assert metadata["total_expected_correctly_deferred_failure_count"] == (
        pytest.approx(4.0 / 3.0)
    )
    assert (
        metadata == permuted["object_correctly_deferred_failure_contributions_metadata"]
    )


def test_object_deferred_expected_counts_reconstruct_and_preserve_integers() -> None:
    result = analysis.object_robustness_analysis(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.5] * 10,
        ["safe"] * 8 + ["alpha", "beta"],
        [f"instance-{index}" for index in range(10)],
        [f"target-{index}" for index in range(10)],
    )
    rows = result["object_correctly_deferred_failure_contributions"]
    metadata = result["object_correctly_deferred_failure_contributions_metadata"]
    expected_counts = [row["expected_correctly_deferred_failure_count"] for row in rows]
    expected_fractions = [
        row["fraction_of_expected_correctly_deferred_failures"] for row in rows
    ]

    assert metadata["cutoff_splits_equal_risk_block"] is False
    assert metadata["total_expected_correctly_deferred_failure_count"] == 2
    assert isinstance(metadata["total_expected_correctly_deferred_failure_count"], int)
    assert all(isinstance(count, int) for count in expected_counts)
    assert sum(expected_counts) == metadata["object_expected_count_sum"] == 2
    assert sum(expected_fractions) == pytest.approx(1.0)
    assert metadata["object_expected_fraction_sum"] == pytest.approx(1.0)
    assert metadata["object_counts_reconstruct_total"] is True
    assert all(
        row["correctly_deferred_failure_count"]
        == row["expected_correctly_deferred_failure_count"]
        for row in rows
    )
