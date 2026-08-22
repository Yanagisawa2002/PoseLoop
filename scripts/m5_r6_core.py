#!/usr/bin/env python3
"""Measurement-first, nearest-fallback confidence gate for PoseLoop M5-R6."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import m5_g0_core as legacy


_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class MeasurementFirstConfig:
    translation_scale_m: float = 0.01
    rotation_scale_deg: float = 5.0
    blend_gain: float = 1.0
    minimum_measurement_confidence: float = 0.35
    dropout_decay_frames: float = 3.0
    quality_floor: float = 0.05
    anchor_staleness_scale: float = 5.0
    motion_innovation_scale_floor: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "translation_scale_m": self.translation_scale_m,
            "rotation_scale_deg": self.rotation_scale_deg,
            "dropout_decay_frames": self.dropout_decay_frames,
            "quality_floor": self.quality_floor,
            "anchor_staleness_scale": self.anchor_staleness_scale,
            "motion_innovation_scale_floor": self.motion_innovation_scale_floor,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.blend_gain <= 1:
            raise ValueError("blend_gain must be in [0, 1]")
        if not 0 <= self.minimum_measurement_confidence <= 1:
            raise ValueError("minimum_measurement_confidence must be in [0, 1]")
        if self.quality_floor > 1:
            raise ValueError("quality_floor must not exceed one")


@dataclass(frozen=True, slots=True)
class MeasurementFirstObservation:
    timestamp_s: float
    anchor_pose: np.ndarray
    measurement_pose: np.ndarray | None
    symmetry: legacy.SymmetrySpec
    mask_area_fraction: float | None
    valid_depth_ratio: float | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        anchor = np.asarray(self.anchor_pose, dtype=np.float64)
        if anchor.shape != (4, 4) or not np.isfinite(anchor).all():
            raise ValueError("anchor_pose must be a finite 4x4 pose")
        if self.measurement_pose is None:
            if self.mask_area_fraction is not None or self.valid_depth_ratio is not None:
                raise ValueError("missing observations cannot carry support diagnostics")
            return
        measurement = np.asarray(self.measurement_pose, dtype=np.float64)
        if measurement.shape != (4, 4) or not np.isfinite(measurement).all():
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
class MeasurementFirstOutput:
    timestamp_s: float
    pose: np.ndarray
    confidence: float
    uncertainty_diag: np.ndarray
    correction_applied: bool
    correction_gain: float
    anchor_measurement_disagreement: float | None
    support_quality: float | None
    innovation_scale: float
    gap_frames: int
    reason: str
    runtime_s: float = 0.0

    @property
    def uncertainty(self) -> float:
        return float(np.sqrt(np.sum(self.uncertainty_diag)))


def _robust_scale(values: list[float], minimum: float) -> float:
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


def _normalized_pose_disagreement(
    first: np.ndarray,
    second: np.ndarray,
    symmetry: legacy.SymmetrySpec,
    config: MeasurementFirstConfig,
) -> float:
    translation_m, rotation_deg = legacy.quotient_pose_error(first, second, symmetry)
    return float(
        math.hypot(
            translation_m / config.translation_scale_m,
            rotation_deg / config.rotation_scale_deg,
        )
    )


class MeasurementFirstConfidenceGate:
    """Causal object-local measurement gate that never owns persistent pose state."""

    def __init__(self, config: MeasurementFirstConfig) -> None:
        self.config = config
        self.timestamp_s: float | None = None
        self.mask_history: list[float] = []
        self.depth_history: list[float] = []
        self.measurement_poses: list[np.ndarray] = []
        self.measurement_timestamps: list[float] = []
        self.motion_innovation_history: list[float] = []
        self.last_available_confidence = 1.0
        self.last_anchor_disagreement = 0.0
        self.gap_frames = 0

    def _motion_quality(
        self,
        measurement: np.ndarray,
        timestamp_s: float,
        symmetry: legacy.SymmetrySpec,
    ) -> tuple[float, float, float]:
        innovation_scale = _robust_scale(
            self.motion_innovation_history,
            self.config.motion_innovation_scale_floor,
        )
        if len(self.measurement_poses) < 2:
            return 1.0, 0.0, innovation_scale
        previous_previous = self.measurement_poses[-2]
        previous = self.measurement_poses[-1]
        previous_dt = self.measurement_timestamps[-1] - self.measurement_timestamps[-2]
        current_dt = timestamp_s - self.measurement_timestamps[-1]
        if previous_dt <= 0 or current_dt <= 0:
            raise ValueError("measurement timestamps must increase")
        previous_step, _, _, _ = legacy.quotient_residual(
            previous_previous,
            previous,
            symmetry,
            translation_scale_m=self.config.translation_scale_m,
            rotation_scale_rad=math.radians(self.config.rotation_scale_deg),
        )
        predicted = previous @ legacy.se3_exp((current_dt / previous_dt) * previous_step)
        motion_innovation = _normalized_pose_disagreement(
            predicted, measurement, symmetry, self.config
        )
        motion_quality = 1.0 / (
            1.0 + (motion_innovation / max(innovation_scale, _EPS)) ** 2
        )
        return float(motion_quality), motion_innovation, innovation_scale

    def _uncertainty(
        self, confidence: float, innovation_scale: float, anchor_disagreement: float
    ) -> np.ndarray:
        effective = max(float(confidence), self.config.quality_floor)
        gap_factor = 1.0 + 0.5 * self.gap_frames
        staleness_factor = 1.0 + anchor_disagreement / self.config.anchor_staleness_scale
        risk_scale = max(1.0, innovation_scale, staleness_factor)
        translation_sigma = (
            self.config.translation_scale_m * risk_scale * gap_factor / effective
        )
        rotation_sigma = (
            math.radians(self.config.rotation_scale_deg)
            * risk_scale
            * gap_factor
            / effective
        )
        return np.asarray(
            [*([translation_sigma**2] * 3), *([rotation_sigma**2] * 3)],
            dtype=np.float64,
        )

    def step(self, observation: MeasurementFirstObservation) -> MeasurementFirstOutput:
        if self.timestamp_s is not None:
            dt = float(observation.timestamp_s - self.timestamp_s)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError(f"timestamp must increase, observed dt={dt}")
        elif observation.missing:
            raise RuntimeError("M5-R6 cannot initialize from missing input")
        self.timestamp_s = float(observation.timestamp_s)
        anchor = np.asarray(observation.anchor_pose, dtype=np.float64).copy()
        innovation_scale = _robust_scale(
            self.motion_innovation_history,
            self.config.motion_innovation_scale_floor,
        )

        if observation.missing:
            self.gap_frames += 1
            staleness_factor = (
                1.0
                + self.last_anchor_disagreement / self.config.anchor_staleness_scale
            )
            confidence = (
                self.last_available_confidence
                * math.exp(-self.gap_frames / self.config.dropout_decay_frames)
                / staleness_factor
            )
            return self._output(
                observation,
                pose=anchor,
                confidence=confidence,
                correction_applied=False,
                correction_gain=0.0,
                disagreement=None,
                support_quality=None,
                innovation_scale=innovation_scale,
                uncertainty_anchor_disagreement=self.last_anchor_disagreement,
                reason="missing_anchor_passthrough",
            )

        measurement = np.asarray(observation.measurement_pose, dtype=np.float64)
        mask_quality = _causal_support_ratio(
            float(observation.mask_area_fraction), self.mask_history, square_root=True
        )
        depth_quality = _causal_support_ratio(
            float(observation.valid_depth_ratio), self.depth_history, square_root=False
        )
        support_quality = math.sqrt(max(mask_quality * depth_quality, 0.0))
        motion_quality, motion_innovation, innovation_scale = self._motion_quality(
            measurement, float(observation.timestamp_s), observation.symmetry
        )
        confidence = float(
            np.clip(
                math.sqrt(max(support_quality * motion_quality, 0.0)),
                self.config.quality_floor,
                1.0,
            )
        )
        disagreement = _normalized_pose_disagreement(
            anchor, measurement, observation.symmetry, self.config
        )
        correction_applied = bool(
            self.config.blend_gain > 0
            and confidence >= self.config.minimum_measurement_confidence
        )
        correction_gain = self.config.blend_gain if correction_applied else 0.0
        if correction_applied:
            residual, _, _, _ = legacy.quotient_residual(
                anchor,
                measurement,
                observation.symmetry,
                translation_scale_m=self.config.translation_scale_m,
                rotation_scale_rad=math.radians(self.config.rotation_scale_deg),
            )
            if observation.symmetry.continuous:
                residual = residual.copy()
                residual[5] = 0.0
            pose = anchor @ legacy.se3_exp(correction_gain * residual)
        else:
            pose = anchor

        self.mask_history.append(float(observation.mask_area_fraction))
        self.depth_history.append(float(observation.valid_depth_ratio))
        self.measurement_poses.append(measurement.copy())
        self.measurement_timestamps.append(float(observation.timestamp_s))
        if len(self.measurement_poses) > 2:
            self.motion_innovation_history.append(motion_innovation)
        self.last_available_confidence = confidence
        self.last_anchor_disagreement = disagreement
        self.gap_frames = 0
        return self._output(
            observation,
            pose=pose,
            confidence=confidence,
            correction_applied=correction_applied,
            correction_gain=correction_gain,
            disagreement=disagreement,
            support_quality=support_quality,
            innovation_scale=innovation_scale,
            uncertainty_anchor_disagreement=disagreement,
            reason=(
                "measurement_first_correction"
                if correction_applied
                else "measurement_confidence_anchor_fallback"
            ),
        )

    def _output(
        self,
        observation: MeasurementFirstObservation,
        *,
        pose: np.ndarray,
        confidence: float,
        correction_applied: bool,
        correction_gain: float,
        disagreement: float | None,
        support_quality: float | None,
        innovation_scale: float,
        uncertainty_anchor_disagreement: float,
        reason: str,
    ) -> MeasurementFirstOutput:
        pose_array = np.asarray(pose, dtype=np.float64)
        clipped_confidence = float(np.clip(confidence, 0.0, 1.0))
        uncertainty = self._uncertainty(
            clipped_confidence, innovation_scale, uncertainty_anchor_disagreement
        )
        if not np.isfinite(pose_array).all() or not np.isfinite(uncertainty).all():
            raise FloatingPointError("M5-R6 output became non-finite")
        return MeasurementFirstOutput(
            timestamp_s=float(observation.timestamp_s),
            pose=pose_array.copy(),
            confidence=clipped_confidence,
            uncertainty_diag=uncertainty,
            correction_applied=correction_applied,
            correction_gain=float(correction_gain),
            anchor_measurement_disagreement=disagreement,
            support_quality=support_quality,
            innovation_scale=float(innovation_scale),
            gap_frames=int(self.gap_frames),
            reason=reason,
        )


def run_measurement_first_confidence_gate(
    observations: Iterable[MeasurementFirstObservation],
    config: MeasurementFirstConfig,
) -> list[MeasurementFirstOutput]:
    estimator = MeasurementFirstConfidenceGate(config)
    outputs: list[MeasurementFirstOutput] = []
    for observation in observations:
        started = time.perf_counter_ns()
        output = estimator.step(observation)
        outputs.append(
            MeasurementFirstOutput(
                timestamp_s=output.timestamp_s,
                pose=output.pose,
                confidence=output.confidence,
                uncertainty_diag=output.uncertainty_diag,
                correction_applied=output.correction_applied,
                correction_gain=output.correction_gain,
                anchor_measurement_disagreement=output.anchor_measurement_disagreement,
                support_quality=output.support_quality,
                innovation_scale=output.innovation_scale,
                gap_frames=output.gap_frames,
                reason=output.reason,
                runtime_s=(time.perf_counter_ns() - started) * 1e-9,
            )
        )
    return outputs


__all__ = [
    "MeasurementFirstConfig",
    "MeasurementFirstObservation",
    "MeasurementFirstOutput",
    "MeasurementFirstConfidenceGate",
    "run_measurement_first_confidence_gate",
]
