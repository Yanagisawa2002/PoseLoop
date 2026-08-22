#!/usr/bin/env python3
"""Object-local adaptive confidence estimator for PoseLoop M5-R4."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import m5_g0_core as legacy


_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class AdaptiveConfidenceConfig:
    """Frozen estimator parameters shared by one M5-R4 candidate."""

    translation_scale_m: float = 0.01
    rotation_scale_deg: float = 5.0
    pose_gain_min: float = 0.2
    pose_gain_max: float = 0.85
    velocity_update_gain: float = 0.2
    innovation_history_min: int = 4
    innovation_gate_multiplier: float = 4.0
    reacquire_after: int = 3
    max_predict_gap_frames: int = 4
    dropout_decay_frames: float = 3.0
    quality_floor: float = 0.05

    def __post_init__(self) -> None:
        positive = {
            "translation_scale_m": self.translation_scale_m,
            "rotation_scale_deg": self.rotation_scale_deg,
            "innovation_gate_multiplier": self.innovation_gate_multiplier,
            "dropout_decay_frames": self.dropout_decay_frames,
            "quality_floor": self.quality_floor,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.pose_gain_min <= self.pose_gain_max <= 1:
            raise ValueError("pose gains must satisfy 0 < min <= max <= 1")
        if not 0 <= self.velocity_update_gain <= 1:
            raise ValueError("velocity_update_gain must be in [0, 1]")
        if self.innovation_history_min < 1:
            raise ValueError("innovation_history_min must be positive")
        if self.reacquire_after < 1:
            raise ValueError("reacquire_after must be positive")
        if self.max_predict_gap_frames < 0:
            raise ValueError("max_predict_gap_frames must be nonnegative")
        if self.quality_floor > 1:
            raise ValueError("quality_floor must not exceed one")


@dataclass(frozen=True, slots=True)
class AdaptiveObservation:
    """Label-blind temporal input plus input-support diagnostics."""

    timestamp_s: float
    measurement_pose: np.ndarray | None
    symmetry: legacy.SymmetrySpec
    mask_area_fraction: float | None
    valid_depth_ratio: float | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        missing = self.measurement_pose is None
        if missing:
            if self.mask_area_fraction is not None or self.valid_depth_ratio is not None:
                raise ValueError("missing observations cannot carry support diagnostics")
            return
        pose = np.asarray(self.measurement_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("measurement_pose must be a finite 4x4 pose")
        if self.mask_area_fraction is None or self.valid_depth_ratio is None:
            raise ValueError("available observations require support diagnostics")
        if not 0 < self.mask_area_fraction <= 1:
            raise ValueError("mask_area_fraction must be in (0, 1]")
        if not 0 < self.valid_depth_ratio <= 1:
            raise ValueError("valid_depth_ratio must be in (0, 1]")

    @property
    def missing(self) -> bool:
        return self.measurement_pose is None


@dataclass(frozen=True, slots=True)
class AdaptiveConfidenceOutput:
    timestamp_s: float
    pose: np.ndarray
    body_twist: np.ndarray
    uncertainty_diag: np.ndarray
    accepted: bool
    residual: np.ndarray | None
    residual_norm: float | None
    reason: str
    selected_symmetry_index: int | None
    selected_gauge_rad: float | None
    confidence: float
    support_quality: float | None
    innovation_scale: float
    gap_frames: int
    runtime_s: float = 0.0

    @property
    def uncertainty(self) -> float:
        return float(np.sqrt(np.sum(self.uncertainty_diag)))


def _robust_scale(values: list[float], minimum: float = 1.0) -> float:
    if not values:
        return float(minimum)
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return float(max(minimum, median + 1.4826 * mad))


def _causal_support_ratio(value: float, history: list[float], *, square_root: bool) -> float:
    if not history:
        return 1.0
    reference = float(np.median(np.asarray(history, dtype=np.float64)))
    ratio = float(np.clip(value / max(reference, _EPS), 0.0, 1.0))
    return math.sqrt(ratio) if square_root else ratio


def _normalized_twist_step(
    twist: np.ndarray, dt: float, config: AdaptiveConfidenceConfig
) -> float:
    step = np.asarray(twist, dtype=np.float64).reshape(6) * float(dt)
    translation = float(np.linalg.norm(step[:3])) / config.translation_scale_m
    rotation = float(np.linalg.norm(step[3:])) / math.radians(config.rotation_scale_deg)
    return float(math.hypot(translation, rotation))


class ObjectLocalAdaptiveConfidenceEstimator:
    """Causal robust constant-twist estimator with track-local calibration."""

    def __init__(self, config: AdaptiveConfidenceConfig) -> None:
        self.config = config
        self.pose: np.ndarray | None = None
        self.twist = np.zeros(6, dtype=np.float64)
        self.timestamp_s: float | None = None
        self.last_accepted_pose: np.ndarray | None = None
        self.mask_history: list[float] = []
        self.depth_history: list[float] = []
        self.innovation_history: list[float] = []
        self.rejection_streak = 0
        self.gap_frames = 0
        self.motion_confidence = 0.0
        self.last_measurement_confidence = 1.0

    def _uncertainty(self, confidence: float, innovation_scale: float) -> np.ndarray:
        effective = max(float(confidence), self.config.quality_floor)
        gap_factor = 1.0 + 0.5 * self.gap_frames
        translation_sigma = (
            self.config.translation_scale_m
            * max(innovation_scale, 1.0)
            * gap_factor
            / effective
        )
        rotation_sigma = (
            math.radians(self.config.rotation_scale_deg)
            * max(innovation_scale, 1.0)
            * gap_factor
            / effective
        )
        return np.asarray(
            [*([translation_sigma**2] * 3), *([rotation_sigma**2] * 3)],
            dtype=np.float64,
        )

    def _output(
        self,
        observation: AdaptiveObservation,
        *,
        accepted: bool,
        residual: np.ndarray | None,
        residual_norm: float | None,
        reason: str,
        selected_index: int | None,
        selected_gauge: float | None,
        confidence: float,
        support_quality: float | None,
        innovation_scale: float,
    ) -> AdaptiveConfidenceOutput:
        assert self.pose is not None
        values = (self.pose, self.twist)
        if any(not np.isfinite(value).all() for value in values):
            raise FloatingPointError("M5-R4 estimator state became non-finite")
        clipped_confidence = float(np.clip(confidence, 0.0, 1.0))
        uncertainty = self._uncertainty(clipped_confidence, innovation_scale)
        if not np.isfinite(uncertainty).all():
            raise FloatingPointError("M5-R4 uncertainty became non-finite")
        return AdaptiveConfidenceOutput(
            timestamp_s=float(observation.timestamp_s),
            pose=self.pose.copy(),
            body_twist=self.twist.copy(),
            uncertainty_diag=uncertainty,
            accepted=accepted,
            residual=None if residual is None else residual.copy(),
            residual_norm=residual_norm,
            reason=reason,
            selected_symmetry_index=selected_index,
            selected_gauge_rad=selected_gauge,
            confidence=clipped_confidence,
            support_quality=support_quality,
            innovation_scale=float(innovation_scale),
            gap_frames=int(self.gap_frames),
        )

    def _initialize(self, observation: AdaptiveObservation) -> AdaptiveConfidenceOutput:
        if observation.measurement_pose is None:
            raise RuntimeError("M5-R4 cannot initialize from missing input")
        self.pose = np.asarray(observation.measurement_pose, dtype=np.float64).copy()
        self.last_accepted_pose = self.pose.copy()
        self.timestamp_s = float(observation.timestamp_s)
        self.twist.fill(0.0)
        self.mask_history.append(float(observation.mask_area_fraction))
        self.depth_history.append(float(observation.valid_depth_ratio))
        self.innovation_history.append(0.0)
        self.rejection_streak = 0
        self.gap_frames = 0
        self.motion_confidence = 0.0
        self.last_measurement_confidence = 1.0
        return self._output(
            observation,
            accepted=True,
            residual=np.zeros(6, dtype=np.float64),
            residual_norm=0.0,
            reason="initialized_from_measurement",
            selected_index=0,
            selected_gauge=0.0 if observation.symmetry.continuous else None,
            confidence=1.0,
            support_quality=1.0,
            innovation_scale=1.0,
        )

    def step(self, observation: AdaptiveObservation) -> AdaptiveConfidenceOutput:
        if self.pose is None:
            return self._initialize(observation)
        assert self.timestamp_s is not None
        dt = float(observation.timestamp_s - self.timestamp_s)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"timestamp must increase, observed dt={dt}")
        previous_pose = self.pose.copy()
        self.timestamp_s = float(observation.timestamp_s)
        innovation_scale = _robust_scale(self.innovation_history)

        if observation.measurement_pose is None:
            self.gap_frames += 1
            confidence = self.last_measurement_confidence * math.exp(
                -self.gap_frames / self.config.dropout_decay_frames
            )
            can_predict = (
                self.gap_frames <= self.config.max_predict_gap_frames
                and self.motion_confidence >= 0.25
            )
            if can_predict:
                motion_weight = self.motion_confidence * math.exp(
                    -(self.gap_frames - 1) / self.config.dropout_decay_frames
                )
                self.pose = self.pose @ legacy.se3_exp(self.twist * dt * motion_weight)
                reason = "adaptive_dropout_prediction"
            else:
                reason = "adaptive_dropout_hold"
            self.twist *= math.exp(-1.0 / self.config.dropout_decay_frames)
            return self._output(
                observation,
                accepted=False,
                residual=None,
                residual_norm=None,
                reason=reason,
                selected_index=None,
                selected_gauge=None,
                confidence=confidence,
                support_quality=None,
                innovation_scale=innovation_scale,
            )

        predicted = self.pose @ legacy.se3_exp(self.twist * dt)
        residual, representative, selected_index, selected_gauge = legacy.quotient_residual(
            predicted,
            np.asarray(observation.measurement_pose, dtype=np.float64),
            observation.symmetry,
            translation_scale_m=self.config.translation_scale_m,
            rotation_scale_rad=math.radians(self.config.rotation_scale_deg),
        )
        if observation.symmetry.continuous:
            residual = residual.copy()
            residual[5] = 0.0
        translation_m, rotation_deg = legacy.quotient_pose_error(
            predicted, representative, observation.symmetry
        )
        innovation_norm = float(
            math.hypot(
                translation_m / self.config.translation_scale_m,
                rotation_deg / self.config.rotation_scale_deg,
            )
        )
        mask_quality = _causal_support_ratio(
            float(observation.mask_area_fraction), self.mask_history, square_root=True
        )
        depth_quality = _causal_support_ratio(
            float(observation.valid_depth_ratio), self.depth_history, square_root=False
        )
        support_quality = math.sqrt(max(mask_quality * depth_quality, 0.0))
        innovation_quality = 1.0 / (
            1.0 + (innovation_norm / max(innovation_scale, _EPS)) ** 2
        )
        confidence = float(
            np.clip(
                math.sqrt(max(support_quality * innovation_quality, 0.0)),
                self.config.quality_floor,
                1.0,
            )
        )
        self.mask_history.append(float(observation.mask_area_fraction))
        self.depth_history.append(float(observation.valid_depth_ratio))

        warmed_up = len(self.innovation_history) >= self.config.innovation_history_min
        rejected = warmed_up and (
            innovation_norm > self.config.innovation_gate_multiplier * innovation_scale
        )
        if rejected:
            self.rejection_streak += 1
            self.gap_frames += 1
            if self.rejection_streak < self.config.reacquire_after:
                self.pose = predicted
                self.last_measurement_confidence = confidence
                self.motion_confidence *= 0.5
                return self._output(
                    observation,
                    accepted=False,
                    residual=residual,
                    residual_norm=innovation_norm,
                    reason="adaptive_innovation_rejected",
                    selected_index=selected_index,
                    selected_gauge=selected_gauge,
                    confidence=confidence,
                    support_quality=support_quality,
                    innovation_scale=innovation_scale,
                )
            self.rejection_streak = 0
            self.gap_frames = 0
            self.pose = representative.copy()
            self.last_accepted_pose = self.pose.copy()
            self.twist.fill(0.0)
            self.innovation_history = [min(innovation_norm, innovation_scale)]
            self.motion_confidence = 0.0
            self.last_measurement_confidence = confidence
            return self._output(
                observation,
                accepted=True,
                residual=residual,
                residual_norm=innovation_norm,
                reason="adaptive_persistent_innovation_reacquired",
                selected_index=selected_index,
                selected_gauge=selected_gauge,
                confidence=confidence,
                support_quality=support_quality,
                innovation_scale=innovation_scale,
            )

        self.rejection_streak = 0
        self.gap_frames = 0
        pose_gain = self.config.pose_gain_min + confidence * (
            self.config.pose_gain_max - self.config.pose_gain_min
        )
        corrected = predicted @ legacy.se3_exp(pose_gain * residual)
        observed_twist = legacy.se3_log(legacy.rigid_inverse(previous_pose) @ corrected) / dt
        if observation.symmetry.continuous:
            observed_twist[5] = 0.0
        twist_delta = observed_twist - self.twist
        motion_disagreement = _normalized_twist_step(twist_delta, dt, self.config)
        motion_consistency = 1.0 / (1.0 + motion_disagreement**2)
        if len(self.innovation_history) < self.config.innovation_history_min:
            motion_consistency = 1.0
        velocity_gain = (
            self.config.velocity_update_gain
            * confidence
            * (0.5 + 0.5 * motion_consistency)
        )
        self.twist = (1.0 - velocity_gain) * self.twist + velocity_gain * observed_twist
        if observation.symmetry.continuous:
            self.twist[5] = 0.0
        self.pose = corrected
        self.last_accepted_pose = corrected.copy()
        self.innovation_history.append(innovation_norm)
        self.motion_confidence = float(
            np.clip(
                0.8 * self.motion_confidence
                + 0.2 * confidence * motion_consistency,
                0.0,
                1.0,
            )
        )
        self.last_measurement_confidence = confidence
        return self._output(
            observation,
            accepted=True,
            residual=residual,
            residual_norm=innovation_norm,
            reason="adaptive_confidence_update",
            selected_index=selected_index,
            selected_gauge=selected_gauge,
            confidence=confidence,
            support_quality=support_quality,
            innovation_scale=innovation_scale,
        )


def run_object_local_adaptive_confidence(
    observations: Iterable[AdaptiveObservation], config: AdaptiveConfidenceConfig
) -> list[AdaptiveConfidenceOutput]:
    estimator = ObjectLocalAdaptiveConfidenceEstimator(config)
    outputs: list[AdaptiveConfidenceOutput] = []
    for observation in observations:
        started = time.perf_counter_ns()
        output = estimator.step(observation)
        outputs.append(
            AdaptiveConfidenceOutput(
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
                confidence=output.confidence,
                support_quality=output.support_quality,
                innovation_scale=output.innovation_scale,
                gap_frames=output.gap_frames,
                runtime_s=(time.perf_counter_ns() - started) * 1e-9,
            )
        )
    return outputs


__all__ = [
    "AdaptiveConfidenceConfig",
    "AdaptiveConfidenceOutput",
    "AdaptiveObservation",
    "ObjectLocalAdaptiveConfidenceEstimator",
    "run_object_local_adaptive_confidence",
]
