#!/usr/bin/env python3
"""Causal static-scene estimator for the PoseLoop M5-R3 sealed replay."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

import m5_g0_core as legacy


@dataclass(frozen=True, slots=True)
class StaticQuotientMedoidConfig:
    """Frozen metric and optional measurement-history limit.

    ``history_limit=None`` means all causally available measurements.  The
    evaluator canonicalizes arbitrary continuous axes before calling this
    estimator because the legacy representative-alignment helper uses z.
    """

    history_limit: int | None = None
    translation_scale_m: float = 0.01
    rotation_scale_deg: float = 5.0

    def __post_init__(self) -> None:
        if self.history_limit is not None and self.history_limit < 1:
            raise ValueError("history_limit must be positive or None")
        if not math.isfinite(self.translation_scale_m) or self.translation_scale_m <= 0:
            raise ValueError("translation_scale_m must be finite and positive")
        if not math.isfinite(self.rotation_scale_deg) or self.rotation_scale_deg <= 0:
            raise ValueError("rotation_scale_deg must be finite and positive")


def normalized_quotient_distance(
    first_pose: np.ndarray,
    second_pose: np.ndarray,
    symmetry: legacy.SymmetrySpec,
    config: StaticQuotientMedoidConfig,
) -> float:
    translation_m, rotation_deg = legacy.quotient_pose_error(
        first_pose, second_pose, symmetry
    )
    return float(
        math.hypot(
            translation_m / config.translation_scale_m,
            rotation_deg / config.rotation_scale_deg,
        )
    )


def quotient_medoid_index(
    poses: list[np.ndarray],
    symmetry: legacy.SymmetrySpec,
    config: StaticQuotientMedoidConfig,
) -> int:
    """Return the earliest minimum-sum medoid under the frozen quotient loss."""

    if not poses:
        raise ValueError("Cannot select a medoid from an empty history")
    costs = np.zeros(len(poses), dtype=np.float64)
    for first_index in range(len(poses)):
        for second_index in range(first_index):
            distance = normalized_quotient_distance(
                poses[first_index], poses[second_index], symmetry, config
            )
            costs[first_index] += distance
            costs[second_index] += distance
    return int(np.argmin(costs))


class StaticQuotientMedoidEstimator:
    """Hold an all-history quotient medoid for a physically static object."""

    def __init__(self, config: StaticQuotientMedoidConfig) -> None:
        self.config = config
        self.history: list[np.ndarray] = []
        self.pose: np.ndarray | None = None
        self.timestamp_s: float | None = None

    def step(self, item: legacy.EstimatorInput) -> legacy.EstimatorOutput:
        started = time.perf_counter_ns()
        if self.timestamp_s is not None and item.timestamp_s <= self.timestamp_s:
            raise ValueError("timestamps must increase strictly")
        selected_index: int | None = None
        selected_gauge: float | None = None
        if item.measurement_pose is not None:
            measurement = np.asarray(item.measurement_pose, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(measurement).all():
                raise ValueError("measurement pose is non-finite")
            self.history.append(measurement.copy())
            if self.config.history_limit is not None:
                self.history = self.history[-self.config.history_limit :]
            medoid_index = quotient_medoid_index(self.history, item.symmetry, self.config)
            medoid = self.history[medoid_index]
            if self.pose is None:
                self.pose = medoid.copy()
                selected_index = 0 if not item.symmetry.continuous else None
                selected_gauge = 0.0 if item.symmetry.continuous else None
                reason = "initialized_from_measurement"
            else:
                _, representative, selected_index, selected_gauge = legacy.quotient_residual(
                    self.pose,
                    medoid,
                    item.symmetry,
                    translation_scale_m=self.config.translation_scale_m,
                    rotation_scale_rad=math.radians(self.config.rotation_scale_deg),
                )
                self.pose = representative
                reason = "static_quotient_medoid_updated"
            accepted = True
        elif self.pose is not None:
            reason = "static_dropout_hold"
            accepted = False
        else:
            raise RuntimeError("Static quotient medoid cannot initialize from missing data")
        self.timestamp_s = float(item.timestamp_s)
        assert self.pose is not None
        if not np.isfinite(self.pose).all():
            raise FloatingPointError("Static quotient medoid state became non-finite")
        return legacy.EstimatorOutput(
            timestamp_s=float(item.timestamp_s),
            pose=self.pose.copy(),
            body_twist=np.zeros(6, dtype=np.float64),
            uncertainty_diag=np.zeros(6, dtype=np.float64),
            accepted=accepted,
            residual=None,
            residual_norm=None,
            reason=reason,
            selected_symmetry_index=selected_index,
            selected_gauge_rad=selected_gauge,
            runtime_s=(time.perf_counter_ns() - started) * 1e-9,
        )


def run_static_quotient_medoid(
    sequence: legacy.CorruptedSequence,
    config: StaticQuotientMedoidConfig,
) -> list[legacy.EstimatorOutput]:
    estimator = StaticQuotientMedoidEstimator(config)
    return [estimator.step(item) for item in sequence.inputs]


__all__ = [
    "StaticQuotientMedoidConfig",
    "StaticQuotientMedoidEstimator",
    "normalized_quotient_distance",
    "quotient_medoid_index",
    "run_static_quotient_medoid",
]
