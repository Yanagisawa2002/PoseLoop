#!/usr/bin/env python3
"""Causal quotient constant-velocity Kalman filter for PoseLoop M5-R1."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from m5_g0_core import (
    CorruptedSequence,
    EstimatorInput,
    EstimatorOutput,
    quotient_residual,
    rigid_inverse,
    se3_exp,
    se3_log,
)


_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class QuotientCVKalmanConfig:
    measurement_translation_sigma_m: float = 0.002
    measurement_rotation_sigma_deg: float = 1.0
    acceleration_translation_sigma_m_s2: float = 0.08
    acceleration_rotation_sigma_deg_s2: float = 35.0
    initial_velocity_translation_sigma_m_s: float = 0.08
    initial_velocity_rotation_sigma_deg_s: float = 30.0
    innovation_gate_sigma: float = 5.5
    reacquire_after: int = 4


class QuotientCVKalmanEstimator:
    """A causal 12-state pose/twist filter on the symmetry quotient.

    The state is a model-to-camera pose and body twist.  The covariance keeps
    pose/twist cross terms, unlike the frozen M5-G0 nearest-representative
    heuristic.  Measurements are symmetry-aligned to the predicted pose before
    the Kalman innovation is formed.
    """

    def __init__(self, config: QuotientCVKalmanConfig) -> None:
        self.config = config
        self.pose: np.ndarray | None = None
        self.twist = np.zeros(6, dtype=float)
        self.covariance = np.zeros((12, 12), dtype=float)
        self.timestamp: float | None = None
        self.rejection_streak = 0

    def _measurement_variance(self) -> np.ndarray:
        return np.asarray(
            [
                *([self.config.measurement_translation_sigma_m**2] * 3),
                *([math.radians(self.config.measurement_rotation_sigma_deg) ** 2] * 3),
            ],
            dtype=float,
        )

    def _velocity_variance(self) -> np.ndarray:
        return np.asarray(
            [
                *([self.config.initial_velocity_translation_sigma_m_s**2] * 3),
                *(
                    [
                        math.radians(
                            self.config.initial_velocity_rotation_sigma_deg_s
                        )
                        ** 2
                    ]
                    * 3
                ),
            ],
            dtype=float,
        )

    def _acceleration_variance(self) -> np.ndarray:
        return np.asarray(
            [
                *([self.config.acceleration_translation_sigma_m_s2**2] * 3),
                *(
                    [
                        math.radians(
                            self.config.acceleration_rotation_sigma_deg_s2
                        )
                        ** 2
                    ]
                    * 3
                ),
            ],
            dtype=float,
        )

    def _reset_covariance(self) -> None:
        self.covariance.fill(0.0)
        self.covariance[:6, :6] = np.diag(self._measurement_variance())
        self.covariance[6:, 6:] = np.diag(self._velocity_variance())

    def _initialize(self, item: EstimatorInput) -> EstimatorOutput:
        if item.measurement_pose is None:
            raise RuntimeError("M5-R1 filter cannot initialize from a missing measurement")
        self.pose = np.asarray(item.measurement_pose, dtype=float).copy()
        self.twist.fill(0.0)
        if item.symmetry.continuous:
            self.twist[5] = 0.0
        self.timestamp = float(item.timestamp_s)
        self.rejection_streak = 0
        self._reset_covariance()
        return self._output(
            item,
            accepted=True,
            residual=np.zeros(6),
            residual_norm=0.0,
            reason="initialized_from_measurement",
            selected_index=0,
            selected_gauge=0.0 if item.symmetry.continuous else None,
        )

    def _predict_covariance(self, dt: float) -> None:
        identity = np.eye(6)
        transition = np.block(
            [[identity, dt * identity], [np.zeros((6, 6)), identity]]
        )
        acceleration = self._acceleration_variance()
        process = np.zeros((12, 12), dtype=float)
        process[:6, :6] = np.diag(acceleration * dt**3 / 3.0)
        process[:6, 6:] = np.diag(acceleration * dt**2 / 2.0)
        process[6:, :6] = np.diag(acceleration * dt**2 / 2.0)
        process[6:, 6:] = np.diag(acceleration * dt)
        self.covariance = transition @ self.covariance @ transition.T + process
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def step(self, item: EstimatorInput) -> EstimatorOutput:
        if self.pose is None:
            return self._initialize(item)
        assert self.timestamp is not None
        dt = float(item.timestamp_s - self.timestamp)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"timestamp must increase, observed dt={dt}")
        predicted = self.pose @ se3_exp(self.twist * dt)
        self._predict_covariance(dt)
        self.timestamp = float(item.timestamp_s)

        if item.measurement_pose is None:
            self.pose = predicted
            return self._output(
                item,
                accepted=False,
                residual=None,
                residual_norm=None,
                reason="dropout_prediction",
                selected_index=None,
                selected_gauge=None,
            )

        residual, representative, selected_index, selected_gauge = quotient_residual(
            predicted,
            np.asarray(item.measurement_pose, dtype=float),
            item.symmetry,
            translation_scale_m=self.config.measurement_translation_sigma_m,
            rotation_scale_rad=math.radians(
                self.config.measurement_rotation_sigma_deg
            ),
        )
        observable = np.arange(5 if item.symmetry.continuous else 6)
        if item.symmetry.continuous:
            residual = residual.copy()
            residual[5] = 0.0
        measurement_variance = self._measurement_variance()
        # A rejected measurement must not make the next measurement easier to
        # accept merely because prediction covariance grew.  Cap the pose
        # contribution used by the robust gate, while retaining the full
        # covariance for the Kalman update below.
        gate_variance = measurement_variance[observable] + np.minimum(
            np.diag(self.covariance)[observable],
            4.0 * measurement_variance[observable],
        )
        innovation = residual[observable]
        normalized_squared = float(
            np.sum(np.square(innovation) / np.maximum(gate_variance, _EPS))
        )
        residual_norm = math.sqrt(max(normalized_squared, 0.0) / len(observable))

        if residual_norm > self.config.innovation_gate_sigma:
            self.rejection_streak += 1
            if self.rejection_streak < self.config.reacquire_after:
                self.pose = predicted
                return self._output(
                    item,
                    accepted=False,
                    residual=residual,
                    residual_norm=residual_norm,
                    reason="innovation_rejected",
                    selected_index=selected_index,
                    selected_gauge=selected_gauge,
                )
            self.rejection_streak = 0
            self.pose = representative.copy()
            if item.symmetry.continuous:
                self.pose = predicted @ se3_exp(residual)
            self.twist.fill(0.0)
            if item.symmetry.continuous:
                self.twist[5] = 0.0
            self._reset_covariance()
            return self._output(
                item,
                accepted=True,
                residual=residual,
                residual_norm=residual_norm,
                reason="persistent_innovation_reacquired",
                selected_index=selected_index,
                selected_gauge=selected_gauge,
            )

        self.rejection_streak = 0
        measurement_matrix = np.zeros((len(observable), 12), dtype=float)
        for row_index, state_index in enumerate(observable):
            measurement_matrix[row_index, state_index] = 1.0
        measurement_covariance = np.diag(measurement_variance[observable])
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + measurement_covariance
        )
        gain = (
            self.covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        correction = gain @ innovation
        if item.symmetry.continuous:
            correction[5] = 0.0
            correction[11] = 0.0
        self.pose = predicted @ se3_exp(correction[:6])
        self.twist = self.twist + correction[6:]
        if item.symmetry.continuous:
            self.twist[5] = 0.0
        identity = np.eye(12)
        joseph = identity - gain @ measurement_matrix
        self.covariance = (
            joseph @ self.covariance @ joseph.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return self._output(
            item,
            accepted=True,
            residual=residual,
            residual_norm=residual_norm,
            reason="quotient_kalman_update",
            selected_index=selected_index,
            selected_gauge=selected_gauge,
        )

    def _output(
        self,
        item: EstimatorInput,
        *,
        accepted: bool,
        residual: np.ndarray | None,
        residual_norm: float | None,
        reason: str,
        selected_index: int | None,
        selected_gauge: float | None,
    ) -> EstimatorOutput:
        assert self.pose is not None
        values = (self.pose, self.twist, self.covariance)
        if any(not np.all(np.isfinite(value)) for value in values):
            raise FloatingPointError("M5-R1 filter state became non-finite")
        uncertainty = np.maximum(np.diag(self.covariance)[:6], 1e-16)
        return EstimatorOutput(
            timestamp_s=float(item.timestamp_s),
            pose=self.pose.copy(),
            body_twist=self.twist.copy(),
            uncertainty_diag=uncertainty,
            accepted=accepted,
            residual=None if residual is None else residual.copy(),
            residual_norm=residual_norm,
            reason=reason,
            selected_symmetry_index=selected_index,
            selected_gauge_rad=selected_gauge,
        )


def run_quotient_cv_kalman(
    sequence: CorruptedSequence, config: QuotientCVKalmanConfig
) -> list[EstimatorOutput]:
    estimator = QuotientCVKalmanEstimator(config)
    outputs: list[EstimatorOutput] = []
    for item in sequence.inputs:
        started = time.perf_counter_ns()
        output = estimator.step(item)
        runtime_s = (time.perf_counter_ns() - started) * 1e-9
        outputs.append(
            EstimatorOutput(
                timestamp_s=output.timestamp_s,
                pose=output.pose,
                body_twist=output.body_twist,
                uncertainty_diag=output.uncertainty_diag,
                accepted=output.accepted,
                residual=output.residual,
                residual_norm=output.residual_norm,
                reason=output.reason,
                selected_symmetry_index=output.selected_symmetry_index,
                selected_gauge_rad=output.selected_gauge_rad,
                runtime_s=runtime_s,
            )
        )
    return outputs


def output_arrays(outputs: Iterable[EstimatorOutput]) -> dict[str, np.ndarray]:
    rows = list(outputs)
    return {
        "poses": np.stack([row.pose for row in rows]),
        "twists": np.stack([row.body_twist for row in rows]),
        "uncertainty": np.stack([row.uncertainty_diag for row in rows]),
        "accepted": np.asarray([row.accepted for row in rows], dtype=bool),
        "runtime_s": np.asarray([row.runtime_s for row in rows], dtype=float),
    }
