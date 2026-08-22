"""Strict prediction projection shared by equivalence and visualization gates."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .common import (
    PREP_PROTOCOL_ID,
    RESULT_SCHEMA,
    PrepError,
    canonical_sha256,
    is_sha256,
)
from .producer import FIXED_INFERENCE


PREDICTION_FIELDS = (
    "predicted_model_to_camera_pose_m",
    "foundationpose_top_score",
    "foundationpose_top_score_margin",
    "selected_candidate_index",
    "candidate_limit",
    "pose_hypothesis_count",
)


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrepError(f"Producer result {field} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PrepError(f"Producer result {field} is not finite")
    return number


def pose_values(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise PrepError("Producer result pose is not 4x4")
    flattened: list[float] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise PrepError("Producer result pose is not 4x4")
        flattened.extend(
            _finite_number(scalar, field="predicted_model_to_camera_pose_m")
            for scalar in row
        )
    if flattened[12:] != [0.0, 0.0, 0.0, 1.0]:
        raise PrepError("Producer result pose homogeneous row differs")
    return flattened


def prediction_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != RESULT_SCHEMA:
        raise PrepError("Producer result schema is invalid")
    if value.get("protocol_id") != PREP_PROTOCOL_ID:
        raise PrepError("Producer result protocol differs")
    if value.get("status") != "success":
        raise PrepError("Producer result is not successful")
    item_id = value.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise PrepError("Producer result item ID is invalid")
    sample = value.get("sample_key")
    if not isinstance(sample, dict) or set(sample) != {
        "scene_id",
        "image_id",
        "object_id",
    }:
        raise PrepError(f"Producer result sample key differs: {item_id}")
    pose_values(value.get("predicted_model_to_camera_pose_m"))
    _finite_number(value.get("foundationpose_top_score"), field="top score")
    margin = _finite_number(
        value.get("foundationpose_top_score_margin"), field="top score margin"
    )
    if margin < 0:
        raise PrepError("Producer result score margin is negative")
    if value.get("candidate_limit") != 252:
        raise PrepError("Producer result candidate limit is not c252")
    if value.get("pose_hypothesis_count") != 252:
        raise PrepError("Producer result hypothesis count is not c252")
    selected = value.get("selected_candidate_index")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or not 0 <= selected < 252
    ):
        raise PrepError("Producer result selected candidate is invalid")
    if value.get("resource_batches") != FIXED_INFERENCE["resource_batches"]:
        raise PrepError("Producer result resource batches differ")
    if not is_sha256(value.get("input_mask_sha256")):
        raise PrepError("Producer result input mask hash is invalid")
    if value.get("evaluator_label_read") is not False:
        raise PrepError("Producer result declares evaluator label access")
    if value.get("official_scorer_run") is not False:
        raise PrepError("Producer result declares official scorer execution")
    projection = {field: value[field] for field in PREDICTION_FIELDS}
    if value.get("prediction_fingerprint") != canonical_sha256(
        {
            "pose": projection["predicted_model_to_camera_pose_m"],
            "top_score": projection["foundationpose_top_score"],
            "margin": projection["foundationpose_top_score_margin"],
            "selected_candidate_index": projection["selected_candidate_index"],
            "candidate_count": projection["pose_hypothesis_count"],
        }
    ):
        raise PrepError(f"Producer result prediction fingerprint differs: {item_id}")
    return projection
