#!/usr/bin/env python3
"""Leakage-safe inference-time feature extraction for PoseLoop M6-G0.

The module deliberately consumes only M1/M2 candidate predictions, calibrated
M2 group geometry, and declared ``models_eval`` object metadata.  It never
loads evaluator metrics or labels.  Every public row keeps numeric model
features and audit metadata in separate dictionaries.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m1_common import assert_pose, load_jsonl  # noqa: E402
from m2_common import transform_model_pose_to_target  # noqa: E402


M6_G0_FEATURE_SCHEMA_VERSION = 1
FROZEN_METHOD = "symmetry_aware_medoid"
FROZEN_VIEW_BUDGET = 5
AGREEMENT_THRESHOLD_NORMALIZED_MSSD = 0.10
FOUNDATIONPOSE_SCORE_SEMANTICS = "raw_upstream_score_not_calibrated_confidence"
FOUNDATIONPOSE_SCORE_DIRECTION = "higher_is_better"


def _feature_spec(
    family: str,
    unit: str,
    description: str,
    *,
    nullable: bool = False,
) -> dict[str, str | bool]:
    return {
        "family": family,
        "unit": unit,
        "description": description,
        "nullable": nullable,
    }


# Dict insertion order is the frozen feature-column order.
FEATURE_SPEC: dict[str, dict[str, str | bool]] = {
    "raw_selected_score": _feature_spec(
        "score",
        "raw_score",
        "FoundationPose score of the frozen medoid candidate; higher is better.",
        nullable=True,
    ),
    "raw_selected_score_missing": _feature_spec(
        "score", "indicator", "One when the selected candidate score is missing."
    ),
    "score_top1": _feature_spec(
        "score", "raw_score", "Largest finite candidate score.", nullable=True
    ),
    "score_top2": _feature_spec(
        "score", "raw_score", "Second-largest finite candidate score.", nullable=True
    ),
    "score_top1_minus_top2": _feature_spec(
        "score", "raw_score", "Top-one minus top-two score margin.", nullable=True
    ),
    "score_mean": _feature_spec(
        "score", "raw_score", "Mean finite candidate score.", nullable=True
    ),
    "score_std": _feature_spec(
        "score",
        "raw_score",
        "Population standard deviation of finite scores.",
        nullable=True,
    ),
    "score_range": _feature_spec(
        "score", "raw_score", "Range of finite candidate scores.", nullable=True
    ),
    "score_valid_count": _feature_spec(
        "score", "count", "Number of candidates with a finite score."
    ),
    "score_missing_count": _feature_spec(
        "score", "count", "Number of candidates without a finite score."
    ),
    "score_valid_fraction": _feature_spec(
        "score", "fraction", "Fraction of candidates with a finite score."
    ),
    "score_any_missing": _feature_spec(
        "score", "indicator", "One when at least one candidate score is missing."
    ),
    "score_all_missing": _feature_spec(
        "score", "indicator", "One when all candidate scores are missing."
    ),
    "score_top2_missing": _feature_spec(
        "score", "indicator", "One when fewer than two finite scores exist."
    ),
    "candidate_count_total": _feature_spec(
        "availability", "count", "Number of acquired candidates at frozen k=5."
    ),
    "candidate_count_valid": _feature_spec(
        "availability", "count", "Number of finite target-camera candidate poses."
    ),
    "candidate_count_invalid": _feature_spec(
        "availability", "count", "Number of missing or invalid candidate poses."
    ),
    "candidate_any_valid": _feature_spec(
        "availability", "indicator", "One when at least one candidate pose is valid."
    ),
    "candidate_all_invalid": _feature_spec(
        "availability", "indicator", "One when no candidate pose is valid."
    ),
    "distinct_view_count": _feature_spec(
        "availability", "count", "Number of distinct explicit scene/image views."
    ),
    "valid_view_fraction": _feature_spec(
        "availability", "fraction", "Fraction of acquired views with a valid pose."
    ),
    "pairwise_valid_pair_count": _feature_spec(
        "disagreement", "count", "Number of valid unordered candidate pairs."
    ),
    "pair_translation_mm_median": _feature_spec(
        "disagreement", "mm", "Median pairwise translation disagreement.", nullable=True
    ),
    "pair_translation_mm_p90": _feature_spec(
        "disagreement",
        "mm",
        "90th-percentile pairwise translation disagreement.",
        nullable=True,
    ),
    "pair_translation_mm_max": _feature_spec(
        "disagreement",
        "mm",
        "Maximum pairwise translation disagreement.",
        nullable=True,
    ),
    "pair_rotation_deg_median": _feature_spec(
        "disagreement",
        "degree",
        "Median symmetry-aware rotation disagreement.",
        nullable=True,
    ),
    "pair_rotation_deg_p90": _feature_spec(
        "disagreement",
        "degree",
        "90th-percentile symmetry-aware rotation disagreement.",
        nullable=True,
    ),
    "pair_rotation_deg_max": _feature_spec(
        "disagreement",
        "degree",
        "Maximum symmetry-aware rotation disagreement.",
        nullable=True,
    ),
    "pair_normalized_mssd_median": _feature_spec(
        "disagreement",
        "diameter_fraction",
        "Median bidirectional normalized MSSD.",
        nullable=True,
    ),
    "pair_normalized_mssd_p90": _feature_spec(
        "disagreement",
        "diameter_fraction",
        "90th-percentile bidirectional normalized MSSD.",
        nullable=True,
    ),
    "pair_normalized_mssd_max": _feature_spec(
        "disagreement",
        "diameter_fraction",
        "Maximum bidirectional normalized MSSD.",
        nullable=True,
    ),
    "candidate_to_output_translation_mm_median": _feature_spec(
        "agreement",
        "mm",
        "Median candidate-to-output translation distance.",
        nullable=True,
    ),
    "candidate_to_output_translation_mm_max": _feature_spec(
        "agreement",
        "mm",
        "Maximum candidate-to-output translation distance.",
        nullable=True,
    ),
    "candidate_to_output_rotation_deg_median": _feature_spec(
        "agreement",
        "degree",
        "Median candidate-to-output symmetry-aware rotation distance.",
        nullable=True,
    ),
    "candidate_to_output_rotation_deg_max": _feature_spec(
        "agreement",
        "degree",
        "Maximum candidate-to-output symmetry-aware rotation distance.",
        nullable=True,
    ),
    "candidate_to_output_normalized_mssd_median": _feature_spec(
        "agreement",
        "diameter_fraction",
        "Median candidate-to-output bidirectional normalized MSSD.",
        nullable=True,
    ),
    "candidate_to_output_normalized_mssd_max": _feature_spec(
        "agreement",
        "diameter_fraction",
        "Maximum candidate-to-output bidirectional normalized MSSD.",
        nullable=True,
    ),
    "agreement_fraction_0_10d": _feature_spec(
        "agreement",
        "fraction",
        "Fraction of valid candidates within the frozen 0.10d output threshold.",
    ),
    "agreeing_view_count_0_10d": _feature_spec(
        "agreement", "count", "Valid candidates within 0.10d of the output."
    ),
    "top_score_to_output_translation_mm": _feature_spec(
        "agreement",
        "mm",
        "Top-score candidate to output translation distance.",
        nullable=True,
    ),
    "top_score_to_output_rotation_deg": _feature_spec(
        "agreement",
        "degree",
        "Top-score candidate to output symmetry-aware rotation distance.",
        nullable=True,
    ),
    "top_score_to_output_normalized_mssd": _feature_spec(
        "agreement",
        "diameter_fraction",
        "Top-score candidate to output bidirectional normalized MSSD.",
        nullable=True,
    ),
    "top_score_to_output_missing": _feature_spec(
        "agreement",
        "indicator",
        "One when top-score-to-output distance is unavailable.",
    ),
    "medoid_to_output_translation_mm": _feature_spec(
        "agreement",
        "mm",
        "Recomputed medoid to frozen output translation distance.",
        nullable=True,
    ),
    "medoid_to_output_rotation_deg": _feature_spec(
        "agreement",
        "degree",
        "Recomputed medoid to frozen output rotation distance.",
        nullable=True,
    ),
    "medoid_to_output_normalized_mssd": _feature_spec(
        "agreement",
        "diameter_fraction",
        "Recomputed medoid to frozen output normalized MSSD.",
        nullable=True,
    ),
    "medoid_to_output_missing": _feature_spec(
        "agreement", "indicator", "One when no frozen output exists."
    ),
    "mutually_agreeing_view_count_0_10d": _feature_spec(
        "view", "count", "Largest all-pairs-agreeing view subset at 0.10d."
    ),
    "strongest_view_vs_remaining_normalized_mssd_median": _feature_spec(
        "view",
        "diameter_fraction",
        "Median top-score-view disagreement to other valid views.",
        nullable=True,
    ),
    "strongest_view_vs_remaining_normalized_mssd_max": _feature_spec(
        "view",
        "diameter_fraction",
        "Maximum top-score-view disagreement to other valid views.",
        nullable=True,
    ),
    "strongest_view_vs_remaining_missing": _feature_spec(
        "view",
        "indicator",
        "One when strongest-versus-remaining disagreement is unavailable.",
    ),
    "view_score_std": _feature_spec(
        "view",
        "raw_score",
        "Across-view population score standard deviation.",
        nullable=True,
    ),
    "view_score_range": _feature_spec(
        "view", "raw_score", "Across-view finite score range.", nullable=True
    ),
    "output_supported_by_multiple_views_0_10d": _feature_spec(
        "view", "indicator", "One when at least two views support the output at 0.10d."
    ),
    "strongest_view_supported_by_multiple_views_0_10d": _feature_spec(
        "view",
        "indicator",
        "One when another view agrees with the top-score view at 0.10d.",
    ),
}

FEATURE_FAMILIES: dict[str, str] = {
    name: str(specification["family"]) for name, specification in FEATURE_SPEC.items()
}
# Compatibility aliases kept explicit for validators and ablation code.
FEATURE_FAMILY = FEATURE_FAMILIES
FAMILY = FEATURE_FAMILIES
FAMILY_FEATURES: dict[str, tuple[str, ...]] = {
    family: tuple(
        name
        for name, assigned_family in FEATURE_FAMILIES.items()
        if assigned_family == family
    )
    for family in ("score", "availability", "disagreement", "agreement", "view")
}

PROHIBITED_FEATURE_TOKENS = frozenset(
    {
        "ground_truth",
        "gt_pose",
        "target_error",
        "pose_error",
        "evaluator",
        "correct",
        "success",
        "failure",
        "oracle",
        "object_id",
        "physical_instance",
        "scene_id",
        "target_id",
        "sample_id",
        "split_id",
        "m3",
    }
)


@dataclass(frozen=True, slots=True)
class PreparedCandidate:
    """Sanitized inference-only candidate in the target camera frame."""

    sample_id: str
    prediction_source: str
    acquisition_rank: int
    scene_id: int | None
    image_id: int | None
    status: str
    score: float | None
    transformed_pose_m: np.ndarray | None
    invalid_reason: str | None

    @property
    def finite_pose(self) -> bool:
        return self.transformed_pose_m is not None


@dataclass(frozen=True, slots=True)
class LoadedFeatureInputs:
    groups: tuple[dict[str, Any], ...]
    m1_predictions: dict[str, dict[str, Any]]
    m2_predictions: dict[str, dict[str, Any]]
    m1_metadata: dict[str, Any]
    m2_metadata: dict[str, Any]


def finite_number_or_none(value: Any) -> float | None:
    """Return a finite float, otherwise the deterministic missing value ``None``."""

    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def coerce_pose_or_none(value: Any) -> np.ndarray | None:
    """Return a validated pose copy, treating malformed poses as unavailable."""

    try:
        pose = np.asarray(value, dtype=np.float64)
        assert_pose(pose, "inference-time pose")
    except (AssertionError, TypeError, ValueError):
        return None
    return pose.copy()


def _normalized_path(path: Path | str) -> str:
    normalized = re.sub(r"/+", "/", str(path).replace("\\", "/").lower())
    return normalized.rstrip("/")


def assert_safe_input_path(
    path: Path | str,
    *,
    allowed_artifact_stage: str | None = None,
) -> Path:
    """Reject M3/report inputs and ambiguous artifact-root inputs.

    When ``allowed_artifact_stage`` is supplied, the path must identify a file
    below that exact ``artifacts/<stage>/`` subtree.  This prevents callers from
    passing a generic artifact root and letting feature code discover inputs.
    """

    raw_normalized = _normalized_path(path)
    resolved_normalized = _normalized_path(Path(path).resolve())
    normalized_variants = (raw_normalized, resolved_normalized)
    forbidden_fragments = ("artifacts/m3", "reports/m3")
    if any(
        normalized == fragment or f"/{fragment}" in f"/{normalized}"
        for normalized in normalized_variants
        for fragment in forbidden_fragments
    ):
        raise ValueError(f"M3 paths are prohibited for M6-G0 feature inputs: {path}")
    basenames = {
        normalized.rsplit("/", maxsplit=1)[-1] for normalized in normalized_variants
    }
    if basenames & {"artifacts", "reports"}:
        raise ValueError(f"Generic artifact/report roots are not valid inputs: {path}")
    if allowed_artifact_stage is not None:
        stage = str(allowed_artifact_stage).strip().lower()
        prefix = f"artifacts/{stage}/"
        if prefix not in f"/{resolved_normalized}":
            raise ValueError(f"Expected an explicit file under {prefix}, got: {path}")
    return Path(path)


def sanitize_prediction_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a stored prediction onto the inference-time feature allowlist."""

    return {
        key: row[key]
        for key in (
            "sample_id",
            "status",
            "predicted_model_to_camera_pose_m",
            "foundationpose_top_score",
            "foundationpose_score_semantics",
        )
        if key in row
    }


def sanitize_group_record(group: Mapping[str, Any]) -> dict[str, Any]:
    """Project an M2 group onto calibration and grouping fields only."""

    association = group.get("oracle_association")
    physical_instance_id = group.get("physical_instance_id") or group.get("track_id")
    if not physical_instance_id and isinstance(association, Mapping):
        physical_instance_id = association.get("track_id")
    raw_views = group.get("views")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, (str, bytes)):
        raise ValueError("M2 group views must be a sequence")
    views = [
        {
            key: view[key]
            for key in (
                "sample_id",
                "prediction_source",
                "acquisition_rank",
                "scene_id",
                "image_id",
                "object_id",
                "camera_world_to_camera_pose_m",
            )
            if key in view
        }
        for view in raw_views
        if isinstance(view, Mapping)
    ]
    if len(views) != len(raw_views):
        raise ValueError("M2 group contains a non-object view")
    projected = {
        key: group[key]
        for key in ("group_id", "target_sample_id", "scene_id", "object_id")
        if key in group
    }
    projected["physical_instance_id"] = (
        str(physical_instance_id) if physical_instance_id else None
    )
    projected["views"] = views
    return projected


def _load_prediction_index(
    path: Path | str,
    *,
    artifact_stage: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    safe_path = assert_safe_input_path(path, allowed_artifact_stage=artifact_stage)
    rows = load_jsonl(safe_path)
    metadata_rows = [row for row in rows if row.get("record_type") == "metadata"]
    if len(metadata_rows) != 1:
        raise ValueError(f"Prediction input needs one metadata row: {safe_path}")
    metadata = dict(metadata_rows[0])
    declared_semantics = metadata.get("foundationpose", {}).get("score_semantics")
    if declared_semantics not in {None, FOUNDATIONPOSE_SCORE_SEMANTICS}:
        raise ValueError(
            f"Unexpected FoundationPose score semantics: {declared_semantics}"
        )

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("record_type") == "metadata":
            continue
        if row.get("record_type") != "prediction":
            raise ValueError(f"Unexpected prediction record type in {safe_path}")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in indexed:
            raise ValueError(
                f"Missing or duplicate prediction sample_id in {safe_path}"
            )
        indexed[sample_id] = sanitize_prediction_row(row)
    return metadata, indexed


def load_feature_inputs(
    groups_path: Path | str,
    m1_predictions_path: Path | str,
    m2_predictions_path: Path | str,
) -> LoadedFeatureInputs:
    """Load only the three explicit M1/M2 inference-time input files."""

    safe_groups = assert_safe_input_path(groups_path, allowed_artifact_stage="m2")
    groups = tuple(sanitize_group_record(row) for row in load_jsonl(safe_groups))
    m1_metadata, m1_predictions = _load_prediction_index(
        m1_predictions_path, artifact_stage="m1"
    )
    m2_metadata, m2_predictions = _load_prediction_index(
        m2_predictions_path, artifact_stage="m2"
    )
    return LoadedFeatureInputs(
        groups=groups,
        m1_predictions=m1_predictions,
        m2_predictions=m2_predictions,
        m1_metadata=m1_metadata,
        m2_metadata=m2_metadata,
    )


def _validate_object_data(object_data: Mapping[str, Any]) -> dict[str, Any]:
    if (
        isinstance(object_data, dict)
        and object_data.get("_m6_g0_object_data_validated") is True
    ):
        return object_data
    points_mm = np.asarray(object_data["points_mm"], dtype=np.float64)
    if (
        points_mm.ndim != 2
        or points_mm.shape[1] != 3
        or not np.isfinite(points_mm).all()
    ):
        raise ValueError("models_eval points must be a finite Nx3 array")
    diameter_mm = finite_number_or_none(object_data.get("diameter_mm"))
    if diameter_mm is None or diameter_mm <= 0.0:
        raise ValueError("models_eval diameter must be finite and positive")
    raw_symmetries = object_data.get("symmetries")
    if not isinstance(raw_symmetries, Sequence) or not raw_symmetries:
        raise ValueError("models_eval must declare at least the identity symmetry")
    symmetries: list[dict[str, np.ndarray]] = []
    for index, raw in enumerate(raw_symmetries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid symmetry record {index}")
        rotation = np.asarray(raw.get("R"), dtype=np.float64)
        translation = np.asarray(raw.get("t"), dtype=np.float64).reshape(-1)
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
        ):
            raise ValueError(f"Invalid symmetry transform {index}")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-5):
            raise ValueError(f"Non-orthonormal symmetry rotation {index}")
        symmetries.append({"R": rotation.copy(), "t": translation.reshape(3, 1)})
    normalized = dict(object_data)
    normalized.update(
        {
            "points_mm": points_mm.copy(),
            "diameter_mm": float(diameter_mm),
            "symmetries": tuple(symmetries),
            "_m6_g0_object_data_validated": True,
        }
    )
    return normalized


def load_declared_object_data(
    dataset_root: Path | str,
    object_ids: Iterable[int],
    toolkit_root: Path | str | None = None,
) -> dict[int, dict[str, Any]]:
    """Load BOP ``models_eval`` points/symmetries using the frozen M1 helper."""

    safe_dataset_root = assert_safe_input_path(dataset_root)
    selected_toolkit_root = Path(
        toolkit_root or REPO_ROOT / "third_party" / "bop_toolkit"
    )
    assert_safe_input_path(selected_toolkit_root)
    if str(selected_toolkit_root) not in sys.path:
        sys.path.insert(0, str(selected_toolkit_root))

    from evaluate_m1 import (  # noqa: PLC0415
        load_object_evaluation_data,
        load_official_models,
    )

    model_params, model_info = load_official_models(safe_dataset_root)
    loaded: dict[int, dict[str, Any]] = {}
    for object_id in sorted({int(value) for value in object_ids}):
        object_data = load_object_evaluation_data(object_id, model_params, model_info)
        model_path = Path(str(object_data["model_path"]))
        if model_path.parent.name != "models_eval":
            raise RuntimeError(
                f"Object {object_id} did not use models_eval: {model_path}"
            )
        loaded[object_id] = _validate_object_data(object_data)
    return loaded


def transform_prediction_to_target(
    predicted_model_to_source_camera_pose_m: np.ndarray,
    source_world_to_camera_pose_m: np.ndarray,
    target_world_to_camera_pose_m: np.ndarray,
) -> np.ndarray:
    """Apply the frozen M2 convention to express a prediction at the target."""

    return transform_model_pose_to_target(
        np.asarray(target_world_to_camera_pose_m, dtype=np.float64),
        np.asarray(source_world_to_camera_pose_m, dtype=np.float64),
        np.asarray(predicted_model_to_source_camera_pose_m, dtype=np.float64),
    )


def translation_distance_mm(
    first_pose_m: np.ndarray, second_pose_m: np.ndarray
) -> float:
    first = np.asarray(first_pose_m, dtype=np.float64)
    second = np.asarray(second_pose_m, dtype=np.float64)
    assert_pose(first, "first disagreement pose")
    assert_pose(second, "second disagreement pose")
    return float(1000.0 * np.linalg.norm(first[:3, 3] - second[:3, 3]))


def _rotation_angle_degrees(
    first_rotation: np.ndarray, second_rotation: np.ndarray
) -> float:
    cosine = float(
        np.clip(
            (np.trace(first_rotation @ second_rotation.T) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cosine)))


def one_way_symmetry_aware_rotation_degrees(
    first_pose_m: np.ndarray,
    second_pose_m: np.ndarray,
    symmetries: Sequence[Mapping[str, Any]],
) -> float:
    """Minimum rotation from ``first`` to a declared equivalent of ``second``."""

    first = np.asarray(first_pose_m, dtype=np.float64)
    second = np.asarray(second_pose_m, dtype=np.float64)
    assert_pose(first, "first rotation pose")
    assert_pose(second, "second rotation pose")
    if not symmetries:
        raise ValueError("At least one symmetry is required")
    values = [
        _rotation_angle_degrees(
            first[:3, :3],
            second[:3, :3] @ np.asarray(symmetry["R"], dtype=np.float64),
        )
        for symmetry in symmetries
    ]
    result = min(values)
    if not math.isfinite(result):
        raise ValueError("Symmetry-aware rotation distance is non-finite")
    return float(result)


def symmetry_aware_rotation_degrees(
    first_pose_m: np.ndarray,
    second_pose_m: np.ndarray,
    object_data_or_symmetries: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> float:
    """Bidirectionally symmetrized rotation distance in degrees."""

    symmetries = (
        object_data_or_symmetries["symmetries"]
        if isinstance(object_data_or_symmetries, Mapping)
        else object_data_or_symmetries
    )
    forward = one_way_symmetry_aware_rotation_degrees(
        first_pose_m, second_pose_m, symmetries
    )
    reverse = one_way_symmetry_aware_rotation_degrees(
        second_pose_m, first_pose_m, symmetries
    )
    return float((forward + reverse) / 2.0)


def _transform_points_mm(
    points_mm: np.ndarray, rotation: np.ndarray, translation_mm: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(rotation, dtype=np.float64)
        @ np.asarray(points_mm, dtype=np.float64).T
        + np.asarray(translation_mm, dtype=np.float64).reshape(3, 1)
    ).T


def one_way_mssd_mm(
    first_pose_m: np.ndarray,
    second_pose_m: np.ndarray,
    object_data: Mapping[str, Any],
) -> float:
    """Official BOP MSSD formula from ``first`` to symmetric ``second``."""

    normalized = _validate_object_data(object_data)
    first = np.asarray(first_pose_m, dtype=np.float64)
    second = np.asarray(second_pose_m, dtype=np.float64)
    assert_pose(first, "first MSSD pose")
    assert_pose(second, "second MSSD pose")
    points_mm = normalized["points_mm"]
    first_points = _transform_points_mm(points_mm, first[:3, :3], first[:3, 3] * 1000.0)
    errors: list[float] = []
    for symmetry in normalized["symmetries"]:
        symmetry_rotation = np.asarray(symmetry["R"], dtype=np.float64)
        symmetry_translation = np.asarray(symmetry["t"], dtype=np.float64).reshape(3)
        second_rotation = second[:3, :3] @ symmetry_rotation
        second_translation_mm = (
            second[:3, :3] @ symmetry_translation + second[:3, 3] * 1000.0
        )
        second_points = _transform_points_mm(
            points_mm, second_rotation, second_translation_mm
        )
        errors.append(float(np.linalg.norm(first_points - second_points, axis=1).max()))
    result = min(errors)
    if not math.isfinite(result):
        raise ValueError("MSSD returned a non-finite distance")
    return float(result)


def bidirectional_normalized_symmetric_mssd(
    first_pose_m: np.ndarray,
    second_pose_m: np.ndarray,
    object_data: Mapping[str, Any],
) -> float:
    """Frozen M2 pair metric: mean forward/reverse MSSD divided by diameter."""

    normalized = _validate_object_data(object_data)
    forward_mm = one_way_mssd_mm(first_pose_m, second_pose_m, normalized)
    reverse_mm = one_way_mssd_mm(second_pose_m, first_pose_m, normalized)
    result = (forward_mm + reverse_mm) / (2.0 * normalized["diameter_mm"])
    if not math.isfinite(result):
        raise ValueError("Bidirectional normalized MSSD is non-finite")
    return float(result)


# Frozen evaluator terminology compatibility.
symmetric_normalized_mssd = bidirectional_normalized_symmetric_mssd


def apply_symmetry_to_pose(
    pose_m: np.ndarray, symmetry: Mapping[str, Any]
) -> np.ndarray:
    """Right-apply a BOP object symmetry, converting its mm shift to metres."""

    pose = np.asarray(pose_m, dtype=np.float64)
    assert_pose(pose, "pose receiving object symmetry")
    rotation = np.asarray(symmetry["R"], dtype=np.float64).reshape(3, 3)
    translation_m = np.asarray(symmetry["t"], dtype=np.float64).reshape(3) * 0.001
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_m
    result = pose @ transform
    assert_pose(result, "symmetry-equivalent pose")
    return result


def equivalent_symmetry_invariance(
    pose_m: np.ndarray,
    symmetry: Mapping[str, Any],
    object_data: Mapping[str, Any],
) -> dict[str, float]:
    """Return pure invariance diagnostics for a declared equivalent pose."""

    equivalent = apply_symmetry_to_pose(pose_m, symmetry)
    return {
        "rotation_degrees": symmetry_aware_rotation_degrees(
            pose_m, equivalent, object_data
        ),
        "normalized_mssd": bidirectional_normalized_symmetric_mssd(
            pose_m, equivalent, object_data
        ),
    }


def pose_disagreement(
    first_pose_m: np.ndarray,
    second_pose_m: np.ndarray,
    object_data: Mapping[str, Any],
) -> tuple[float, float, float]:
    return (
        translation_distance_mm(first_pose_m, second_pose_m),
        symmetry_aware_rotation_degrees(first_pose_m, second_pose_m, object_data),
        bidirectional_normalized_symmetric_mssd(
            first_pose_m, second_pose_m, object_data
        ),
    )


def pairwise_pose_disagreements(
    poses_m: Sequence[np.ndarray], object_data: Mapping[str, Any]
) -> dict[str, list[float]]:
    """Compute each unordered pair exactly once in deterministic input order."""

    result = {
        "translation_mm": [],
        "rotation_deg": [],
        "normalized_mssd": [],
    }
    for first_index, second_index in combinations(range(len(poses_m)), 2):
        translation_mm, rotation_deg, normalized_mssd = pose_disagreement(
            poses_m[first_index], poses_m[second_index], object_data
        )
        result["translation_mm"].append(translation_mm)
        result["rotation_deg"].append(rotation_deg)
        result["normalized_mssd"].append(normalized_mssd)
    return result


def _candidate_pose(candidate: Any) -> np.ndarray | None:
    if isinstance(candidate, PreparedCandidate):
        return candidate.transformed_pose_m
    if not isinstance(candidate, Mapping):
        return None
    pose = candidate.get("transformed_pose_m")
    if pose is None:
        pose = candidate.get("pose_m")
    return coerce_pose_or_none(pose)


def _candidate_id(candidate: Any) -> str:
    if isinstance(candidate, PreparedCandidate):
        return candidate.sample_id
    if isinstance(candidate, Mapping):
        if candidate.get("sample_id") is not None:
            return str(candidate["sample_id"])
        view = candidate.get("view")
        if isinstance(view, Mapping):
            return str(view.get("sample_id", ""))
    return ""


def _candidate_rank(candidate: Any) -> int:
    if isinstance(candidate, PreparedCandidate):
        return candidate.acquisition_rank
    if isinstance(candidate, Mapping):
        if candidate.get("acquisition_rank") is not None:
            return int(candidate["acquisition_rank"])
        view = candidate.get("view")
        if isinstance(view, Mapping):
            return int(view.get("acquisition_rank", 0))
    return 0


def choose_frozen_medoid(
    candidates: Sequence[Any],
    object_data: Mapping[str, Any],
) -> tuple[Any | None, dict[str, float]]:
    """Recompute the frozen k=5 M2 medoid and exact tie-break contract."""

    successful = [
        candidate for candidate in candidates if _candidate_pose(candidate) is not None
    ]
    if not successful:
        return None, {}
    if len(successful) == 1:
        return successful[0], {_candidate_id(successful[0]): 0.0}

    scores: dict[str, float] = {}
    for candidate in successful:
        candidate_pose = _candidate_pose(candidate)
        assert candidate_pose is not None
        disagreements = [
            bidirectional_normalized_symmetric_mssd(
                candidate_pose,
                other_pose,
                object_data,
            )
            for other in successful
            if other is not candidate
            for other_pose in [_candidate_pose(other)]
            if other_pose is not None
        ]
        scores[_candidate_id(candidate)] = float(np.mean(disagreements))
    selected = min(
        successful,
        key=lambda candidate: (
            scores[_candidate_id(candidate)],
            _candidate_rank(candidate),
            _candidate_id(candidate),
        ),
    )
    return selected, scores


def maximum_mutual_agreement_count(
    normalized_mssd_matrix: np.ndarray,
    threshold: float = AGREEMENT_THRESHOLD_NORMALIZED_MSSD,
) -> int:
    """Return the largest clique under a symmetric pairwise threshold."""

    matrix = np.asarray(normalized_mssd_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Pairwise disagreement matrix must be square")
    if matrix.size == 0:
        return 0
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Pairwise disagreement matrix must be finite and symmetric")
    count = matrix.shape[0]
    for subset_size in range(count, 0, -1):
        for subset in combinations(range(count), subset_size):
            if all(
                matrix[first, second] <= threshold
                for first, second in combinations(subset, 2)
            ):
                return subset_size
    return 0


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _median_max(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)), float(np.max(array))


def _prediction_for_view(
    view: Mapping[str, Any],
    m1_predictions: Mapping[str, Mapping[str, Any]],
    m2_predictions: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    sample_id = str(view.get("sample_id", ""))
    source = str(view.get("prediction_source", ""))
    if source == "m1":
        return m1_predictions.get(
            sample_id, {"sample_id": sample_id, "status": "missing_prediction"}
        )
    if source == "m2":
        return m2_predictions.get(
            sample_id, {"sample_id": sample_id, "status": "missing_prediction"}
        )
    return {"sample_id": sample_id, "status": "unknown_prediction_source"}


def sanitize_prediction(
    prediction: Mapping[str, Any], expected_sample_id: str
) -> tuple[str, float | None, np.ndarray | None, str | None]:
    """Select inference-time fields only; evaluator/GT fields are never read."""

    actual_sample_id = str(prediction.get("sample_id", expected_sample_id))
    if actual_sample_id != expected_sample_id:
        return "identity_mismatch", None, None, "prediction_sample_id_mismatch"
    status = str(prediction.get("status", "missing_prediction"))
    score_semantics = prediction.get("foundationpose_score_semantics")
    if score_semantics not in {None, FOUNDATIONPOSE_SCORE_SEMANTICS}:
        raise ValueError(
            f"Unknown score semantics for {expected_sample_id}: {score_semantics}"
        )
    score = finite_number_or_none(prediction.get("foundationpose_top_score"))
    if status != "success":
        return status, score, None, f"prediction_status_{status}"
    pose = coerce_pose_or_none(prediction.get("predicted_model_to_camera_pose_m"))
    if pose is None:
        return status, score, None, "missing_or_invalid_prediction_pose"
    return status, score, pose, None


def prepare_group_candidates(
    group: Mapping[str, Any],
    m1_predictions: Mapping[str, Mapping[str, Any]],
    m2_predictions: Mapping[str, Mapping[str, Any]],
) -> list[PreparedCandidate]:
    """Join group views to M1/M2 predictions by ``sample_id`` and transform."""

    raw_views = group.get("views")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, (str, bytes)):
        raise ValueError("M2 group views must be a sequence")
    views = sorted(
        (dict(view) for view in raw_views),
        key=lambda view: (
            int(view.get("acquisition_rank", 0)),
            str(view.get("sample_id", "")),
        ),
    )[:FROZEN_VIEW_BUDGET]
    sample_ids = [str(view.get("sample_id", "")) for view in views]
    if any(not sample_id for sample_id in sample_ids) or len(sample_ids) != len(
        set(sample_ids)
    ):
        raise ValueError("M2 group contains missing or duplicate candidate sample IDs")
    ranks = [int(view.get("acquisition_rank", -1)) for view in views]
    if ranks != list(range(len(views))):
        raise ValueError("M2 group acquisition ranks are not consecutive from zero")

    target_world_to_camera = (
        coerce_pose_or_none(views[0].get("camera_world_to_camera_pose_m"))
        if views
        else None
    )
    prepared: list[PreparedCandidate] = []
    for view in views:
        sample_id = str(view["sample_id"])
        source = str(view.get("prediction_source", ""))
        prediction = _prediction_for_view(view, m1_predictions, m2_predictions)
        status, score, source_pose, invalid_reason = sanitize_prediction(
            prediction, sample_id
        )
        transformed_pose: np.ndarray | None = None
        source_world_to_camera = coerce_pose_or_none(
            view.get("camera_world_to_camera_pose_m")
        )
        if source_pose is not None and target_world_to_camera is None:
            invalid_reason = "missing_or_invalid_target_camera_pose"
        elif source_pose is not None and source_world_to_camera is None:
            invalid_reason = "missing_or_invalid_source_camera_pose"
        elif source_pose is not None:
            try:
                transformed_pose = transform_prediction_to_target(
                    source_pose,
                    source_world_to_camera,
                    target_world_to_camera,
                )
            except (AssertionError, TypeError, ValueError, np.linalg.LinAlgError):
                invalid_reason = "target_camera_transform_failed"
        prepared.append(
            PreparedCandidate(
                sample_id=sample_id,
                prediction_source=source,
                acquisition_rank=int(view["acquisition_rank"]),
                scene_id=(
                    int(view["scene_id"]) if view.get("scene_id") is not None else None
                ),
                image_id=(
                    int(view["image_id"]) if view.get("image_id") is not None else None
                ),
                status=status,
                score=score,
                transformed_pose_m=transformed_pose,
                invalid_reason=None if transformed_pose is not None else invalid_reason,
            )
        )
    return prepared


def _physical_instance_id(
    group: Mapping[str, Any], candidates: Sequence[PreparedCandidate]
) -> str:
    for key in ("physical_instance_id", "track_id"):
        value = group.get(key)
        if value:
            return str(value)
    association = group.get("oracle_association")
    if isinstance(association, Mapping) and association.get("track_id"):
        return str(association["track_id"])
    target = candidates[0] if candidates else None
    return (
        f"scene-{target.scene_id}-object-{group.get('object_id')}-target-{target.sample_id}"
        if target is not None
        else f"group-{group.get('group_id', '')}"
    )


def _top_scored_valid_candidate(
    candidates: Sequence[PreparedCandidate],
) -> PreparedCandidate | None:
    scored = [
        candidate
        for candidate in candidates
        if candidate.finite_pose and candidate.score is not None
    ]
    if not scored:
        return None
    return min(
        scored,
        key=lambda candidate: (
            -float(candidate.score),
            candidate.acquisition_rank,
            candidate.sample_id,
        ),
    )


def extract_group_feature_row(
    group: Mapping[str, Any],
    m1_predictions: Mapping[str, Mapping[str, Any]],
    m2_predictions: Mapping[str, Mapping[str, Any]],
    object_data: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Construct exactly one leakage-safe feature row for one M2 group."""

    normalized_object_data = _validate_object_data(object_data)
    candidates = prepare_group_candidates(group, m1_predictions, m2_predictions)
    valid_candidates = [candidate for candidate in candidates if candidate.finite_pose]
    valid_poses = [candidate.transformed_pose_m for candidate in valid_candidates]
    assert all(pose is not None for pose in valid_poses)

    selected, medoid_scores = choose_frozen_medoid(candidates, normalized_object_data)
    output_candidate = selected if isinstance(selected, PreparedCandidate) else None
    output_pose = output_candidate.transformed_pose_m if output_candidate else None
    top_score_candidate = _top_scored_valid_candidate(candidates)

    all_scores = [
        candidate.score for candidate in candidates if candidate.score is not None
    ]
    sorted_scores = sorted((float(score) for score in all_scores), reverse=True)
    score_top1 = sorted_scores[0] if sorted_scores else None
    score_top2 = sorted_scores[1] if len(sorted_scores) >= 2 else None
    total_count = len(candidates)
    valid_count = len(valid_candidates)
    score_count = len(sorted_scores)

    pairwise = pairwise_pose_disagreements(
        [np.asarray(pose) for pose in valid_poses], normalized_object_data
    )
    translation_summary = _distribution(pairwise["translation_mm"])
    rotation_summary = _distribution(pairwise["rotation_deg"])
    mssd_summary = _distribution(pairwise["normalized_mssd"])

    output_distances = {
        "translation_mm": [],
        "rotation_deg": [],
        "normalized_mssd": [],
    }
    if output_pose is not None:
        for candidate in valid_candidates:
            candidate_pose = candidate.transformed_pose_m
            assert candidate_pose is not None
            translation_mm, rotation_deg, normalized_mssd = pose_disagreement(
                candidate_pose, output_pose, normalized_object_data
            )
            output_distances["translation_mm"].append(translation_mm)
            output_distances["rotation_deg"].append(rotation_deg)
            output_distances["normalized_mssd"].append(normalized_mssd)
    output_translation_median, output_translation_max = _median_max(
        output_distances["translation_mm"]
    )
    output_rotation_median, output_rotation_max = _median_max(
        output_distances["rotation_deg"]
    )
    output_mssd_median, output_mssd_max = _median_max(
        output_distances["normalized_mssd"]
    )
    agreeing_count = sum(
        value <= AGREEMENT_THRESHOLD_NORMALIZED_MSSD
        for value in output_distances["normalized_mssd"]
    )

    top_score_distance: tuple[float, float, float] | None = None
    if top_score_candidate is not None and output_pose is not None:
        assert top_score_candidate.transformed_pose_m is not None
        top_score_distance = pose_disagreement(
            top_score_candidate.transformed_pose_m,
            output_pose,
            normalized_object_data,
        )
    medoid_distance = (
        pose_disagreement(output_pose, output_pose, normalized_object_data)
        if output_pose is not None
        else None
    )

    pair_matrix = np.zeros((valid_count, valid_count), dtype=np.float64)
    for first_index, second_index in combinations(range(valid_count), 2):
        first_pose = valid_candidates[first_index].transformed_pose_m
        second_pose = valid_candidates[second_index].transformed_pose_m
        assert first_pose is not None and second_pose is not None
        value = bidirectional_normalized_symmetric_mssd(
            first_pose, second_pose, normalized_object_data
        )
        pair_matrix[first_index, second_index] = value
        pair_matrix[second_index, first_index] = value
    mutual_count = maximum_mutual_agreement_count(pair_matrix)

    strongest_remaining: list[float] = []
    if top_score_candidate is not None:
        strongest_pose = top_score_candidate.transformed_pose_m
        assert strongest_pose is not None
        strongest_remaining = [
            bidirectional_normalized_symmetric_mssd(
                strongest_pose,
                candidate.transformed_pose_m,
                normalized_object_data,
            )
            for candidate in valid_candidates
            if candidate is not top_score_candidate
            and candidate.transformed_pose_m is not None
        ]
    strongest_median, strongest_max = _median_max(strongest_remaining)
    strongest_support_count = sum(
        value <= AGREEMENT_THRESHOLD_NORMALIZED_MSSD for value in strongest_remaining
    )

    distinct_views = {
        (
            candidate.scene_id,
            candidate.image_id
            if candidate.image_id is not None
            else candidate.sample_id,
        )
        for candidate in candidates
    }
    features: dict[str, int | float | bool | None] = {
        "raw_selected_score": output_candidate.score if output_candidate else None,
        "raw_selected_score_missing": int(
            output_candidate is None or output_candidate.score is None
        ),
        "score_top1": score_top1,
        "score_top2": score_top2,
        "score_top1_minus_top2": (
            score_top1 - score_top2
            if score_top1 is not None and score_top2 is not None
            else None
        ),
        "score_mean": float(np.mean(sorted_scores)) if sorted_scores else None,
        "score_std": float(np.std(sorted_scores)) if sorted_scores else None,
        "score_range": (
            float(max(sorted_scores) - min(sorted_scores)) if sorted_scores else None
        ),
        "score_valid_count": score_count,
        "score_missing_count": total_count - score_count,
        "score_valid_fraction": score_count / total_count if total_count else 0.0,
        "score_any_missing": int(score_count < total_count),
        "score_all_missing": int(score_count == 0),
        "score_top2_missing": int(score_count < 2),
        "candidate_count_total": total_count,
        "candidate_count_valid": valid_count,
        "candidate_count_invalid": total_count - valid_count,
        "candidate_any_valid": int(valid_count > 0),
        "candidate_all_invalid": int(valid_count == 0),
        "distinct_view_count": len(distinct_views),
        "valid_view_fraction": valid_count / total_count if total_count else 0.0,
        "pairwise_valid_pair_count": len(pairwise["normalized_mssd"]),
        "pair_translation_mm_median": translation_summary["median"],
        "pair_translation_mm_p90": translation_summary["p90"],
        "pair_translation_mm_max": translation_summary["max"],
        "pair_rotation_deg_median": rotation_summary["median"],
        "pair_rotation_deg_p90": rotation_summary["p90"],
        "pair_rotation_deg_max": rotation_summary["max"],
        "pair_normalized_mssd_median": mssd_summary["median"],
        "pair_normalized_mssd_p90": mssd_summary["p90"],
        "pair_normalized_mssd_max": mssd_summary["max"],
        "candidate_to_output_translation_mm_median": output_translation_median,
        "candidate_to_output_translation_mm_max": output_translation_max,
        "candidate_to_output_rotation_deg_median": output_rotation_median,
        "candidate_to_output_rotation_deg_max": output_rotation_max,
        "candidate_to_output_normalized_mssd_median": output_mssd_median,
        "candidate_to_output_normalized_mssd_max": output_mssd_max,
        "agreement_fraction_0_10d": agreeing_count / valid_count
        if valid_count
        else 0.0,
        "agreeing_view_count_0_10d": agreeing_count,
        "top_score_to_output_translation_mm": (
            top_score_distance[0] if top_score_distance else None
        ),
        "top_score_to_output_rotation_deg": (
            top_score_distance[1] if top_score_distance else None
        ),
        "top_score_to_output_normalized_mssd": (
            top_score_distance[2] if top_score_distance else None
        ),
        "top_score_to_output_missing": int(top_score_distance is None),
        "medoid_to_output_translation_mm": (
            medoid_distance[0] if medoid_distance else None
        ),
        "medoid_to_output_rotation_deg": (
            medoid_distance[1] if medoid_distance else None
        ),
        "medoid_to_output_normalized_mssd": (
            medoid_distance[2] if medoid_distance else None
        ),
        "medoid_to_output_missing": int(medoid_distance is None),
        "mutually_agreeing_view_count_0_10d": mutual_count,
        "strongest_view_vs_remaining_normalized_mssd_median": strongest_median,
        "strongest_view_vs_remaining_normalized_mssd_max": strongest_max,
        "strongest_view_vs_remaining_missing": int(not strongest_remaining),
        "view_score_std": float(np.std(sorted_scores)) if sorted_scores else None,
        "view_score_range": (
            float(max(sorted_scores) - min(sorted_scores)) if sorted_scores else None
        ),
        "output_supported_by_multiple_views_0_10d": int(agreeing_count >= 2),
        "strongest_view_supported_by_multiple_views_0_10d": int(
            strongest_support_count >= 1
        ),
    }

    candidate_audit = [
        {
            "sample_id": candidate.sample_id,
            "prediction_source": candidate.prediction_source,
            "acquisition_rank": candidate.acquisition_rank,
            "scene_id": candidate.scene_id,
            "image_id": candidate.image_id,
            "status": candidate.status,
            "finite_pose": candidate.finite_pose,
            "invalid_reason": candidate.invalid_reason,
            "foundationpose_top_score": candidate.score,
            "transformed_model_to_target_camera_pose_m": (
                candidate.transformed_pose_m.tolist()
                if candidate.transformed_pose_m is not None
                else None
            ),
        }
        for candidate in candidates
    ]
    metadata: dict[str, Any] = {
        "schema_version": M6_G0_FEATURE_SCHEMA_VERSION,
        "group_id": str(group.get("group_id", "")),
        "target_sample_id": str(group.get("target_sample_id", "")),
        "scene_id": int(group["scene_id"])
        if group.get("scene_id") is not None
        else None,
        "object_id": int(group["object_id"]),
        "physical_instance_id": _physical_instance_id(group, candidates),
        "score_semantics": FOUNDATIONPOSE_SCORE_SEMANTICS,
        "score_direction": FOUNDATIONPOSE_SCORE_DIRECTION,
        "frozen_method": FROZEN_METHOD,
        "view_budget": FROZEN_VIEW_BUDGET,
        "selected_sample_id": output_candidate.sample_id if output_candidate else None,
        "selected_acquisition_rank": (
            output_candidate.acquisition_rank if output_candidate else None
        ),
        "output_status": output_candidate.status
        if output_candidate
        else "no_usable_prediction",
        "output_finite_pose": output_pose is not None,
        "output_pose_m": output_pose.tolist() if output_pose is not None else None,
        "top_score_sample_id": (
            top_score_candidate.sample_id if top_score_candidate else None
        ),
        "top_score_acquisition_rank": (
            top_score_candidate.acquisition_rank if top_score_candidate else None
        ),
        "medoid_selection_scores": dict(sorted(medoid_scores.items())),
        "candidate_sample_ids": [candidate.sample_id for candidate in candidates],
        "candidate_audit": candidate_audit,
        "models_eval": {
            "model_path": normalized_object_data.get("model_path"),
            "diameter_mm": normalized_object_data["diameter_mm"],
            "point_count": int(normalized_object_data["points_mm"].shape[0]),
            "symmetry_transform_count": len(normalized_object_data["symmetries"]),
        },
    }
    row = {"features": features, "metadata": metadata}
    validate_feature_schema(row)
    return row


def extract_feature_rows(
    groups: Sequence[Mapping[str, Any]],
    m1_predictions: Mapping[str, Mapping[str, Any]],
    m2_predictions: Mapping[str, Mapping[str, Any]],
    object_data_by_id: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    """Create one deterministically ordered row per unique M2 group."""

    indexed_groups: dict[str, Mapping[str, Any]] = {}
    target_ids: set[str] = set()
    for group in groups:
        group_id = str(group.get("group_id", ""))
        target_id = str(group.get("target_sample_id", ""))
        if not group_id or group_id in indexed_groups:
            raise ValueError("M2 groups need unique non-empty group IDs")
        if not target_id or target_id in target_ids:
            raise ValueError("Each M2 target must appear in exactly one group")
        indexed_groups[group_id] = group
        target_ids.add(target_id)

    rows: list[dict[str, dict[str, Any]]] = []
    for group_id in sorted(indexed_groups):
        group = indexed_groups[group_id]
        object_id = int(group["object_id"])
        if object_id not in object_data_by_id:
            raise KeyError(f"No declared models_eval data for object {object_id}")
        rows.append(
            extract_group_feature_row(
                group,
                m1_predictions,
                m2_predictions,
                object_data_by_id[object_id],
            )
        )
    if len(rows) != len(groups):
        raise RuntimeError("One-row-per-target feature contract failed")
    return rows


def build_feature_rows(
    *,
    groups_path: Path | str,
    m1_predictions_path: Path | str,
    m2_predictions_path: Path | str,
    dataset_root: Path | str,
    toolkit_root: Path | str | None = None,
) -> list[dict[str, dict[str, Any]]]:
    """Load explicit inputs, declared object data, and build all feature rows."""

    loaded = load_feature_inputs(groups_path, m1_predictions_path, m2_predictions_path)
    object_ids = {int(group["object_id"]) for group in loaded.groups}
    object_data = load_declared_object_data(
        dataset_root, object_ids, toolkit_root=toolkit_root
    )
    return extract_feature_rows(
        loaded.groups,
        loaded.m1_predictions,
        loaded.m2_predictions,
        object_data,
    )


# Descriptive alias used by orchestration code.
load_m6_g0_feature_rows = build_feature_rows


def _contains_prohibited_token(name: str) -> str | None:
    lowered = name.lower()
    for token in sorted(PROHIBITED_FEATURE_TOKENS):
        if token in lowered:
            return token
    # Short identity/GT tokens require component boundaries to avoid e.g. "valid".
    components = set(filter(None, re.split(r"[^a-z0-9]+|_", lowered)))
    for token in ("id", "gt"):
        if token in components:
            return token
    return None


def validate_feature_schema(
    row_or_features: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> bool:
    """Prove feature keys/values are numeric-only and leakage-separated."""

    if "features" in row_or_features:
        if set(row_or_features) != {"features", "metadata"}:
            raise ValueError("Feature rows must contain only features and metadata")
        features = row_or_features["features"]
        metadata = row_or_features["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("Feature-row metadata must be a mapping")
    else:
        features = row_or_features
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping")

    expected = set(FEATURE_SPEC)
    observed = set(features)
    if require_complete and observed != expected:
        raise ValueError(
            "Feature schema mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if not require_complete and not observed <= expected:
        raise ValueError(f"Unknown feature keys: {sorted(observed - expected)}")

    for name, value in features.items():
        prohibited = _contains_prohibited_token(str(name))
        if prohibited is not None:
            raise ValueError(f"Prohibited token {prohibited!r} in feature {name!r}")
        if value is None:
            if not bool(FEATURE_SPEC[str(name)]["nullable"]):
                raise ValueError(f"Non-nullable feature {name!r} is missing")
            continue
        if not isinstance(value, (bool, int, float, np.integer, np.floating)):
            raise TypeError(f"Feature {name!r} is not scalar numeric: {type(value)}")
        if not math.isfinite(float(value)):
            raise ValueError(f"Feature {name!r} is non-finite")
    return True


__all__ = [
    "AGREEMENT_THRESHOLD_NORMALIZED_MSSD",
    "FAMILY",
    "FAMILY_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_FAMILIES",
    "FEATURE_SPEC",
    "FOUNDATIONPOSE_SCORE_DIRECTION",
    "FOUNDATIONPOSE_SCORE_SEMANTICS",
    "FROZEN_METHOD",
    "FROZEN_VIEW_BUDGET",
    "LoadedFeatureInputs",
    "M6_G0_FEATURE_SCHEMA_VERSION",
    "PROHIBITED_FEATURE_TOKENS",
    "PreparedCandidate",
    "apply_symmetry_to_pose",
    "assert_safe_input_path",
    "bidirectional_normalized_symmetric_mssd",
    "build_feature_rows",
    "choose_frozen_medoid",
    "coerce_pose_or_none",
    "equivalent_symmetry_invariance",
    "extract_feature_rows",
    "extract_group_feature_row",
    "finite_number_or_none",
    "load_declared_object_data",
    "load_feature_inputs",
    "load_m6_g0_feature_rows",
    "maximum_mutual_agreement_count",
    "one_way_mssd_mm",
    "one_way_symmetry_aware_rotation_degrees",
    "pairwise_pose_disagreements",
    "pose_disagreement",
    "prepare_group_candidates",
    "sanitize_group_record",
    "sanitize_prediction",
    "sanitize_prediction_row",
    "symmetric_normalized_mssd",
    "symmetry_aware_rotation_degrees",
    "transform_prediction_to_target",
    "translation_distance_mm",
    "validate_feature_schema",
]
