#!/usr/bin/env python3
"""Nearest-anchor object-local confidence gate for PoseLoop M5-R5."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import m5_g0_core as legacy


_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class AnchoredGateConfig:
    translation_scale_m: float = 0.01
    rotation_scale_deg: float = 5.0
    blend_gain: float = 0.5
    minimum_correction_confidence: float = 0.55
    maximum_anchor_measurement_disagreement: float = 2.0
    dropout_decay_frames: float = 3.0
    quality_floor: float = 0.05

    def __post_init__(self) -> None:
        positive = {
            "translation_scale_m": self.translation_scale_m,
            "rotation_scale_deg": self.rotation_scale_deg,
            "dropout_decay_frames": self.dropout_decay_frames,
            "quality_floor": self.quality_floor,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.blend_gain <= 1:
            raise ValueError("blend_gain must be in [0, 1]")
        if not 0 <= self.minimum_correction_confidence <= 1:
            raise ValueError("minimum_correction_confidence must be in [0, 1]")
        if not math.isfinite(self.maximum_anchor_measurement_disagreement) or (
            self.maximum_anchor_measurement_disagreement < 0
        ):
            raise ValueError("maximum disagreement must be finite and nonnegative")
        if self.quality_floor > 1:
            raise ValueError("quality_floor must not exceed one")


@dataclass(frozen=True, slots=True)
class AnchoredGateObservation:
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
class AnchoredGateOutput:
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


class NearestAnchoredConfidenceGate:
    """Causal confidence calibrator that never owns persistent pose state."""

    def __init__(self, config: AnchoredGateConfig) -> None:
        self.config = config
        self.timestamp_s: float | None = None
        self.mask_history: list[float] = []
        self.depth_history: list[float] = []
        self.innovation_history: list[float] = []
        self.last_available_confidence = 1.0
        self.gap_frames = 0

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

    def step(self, observation: AnchoredGateObservation) -> AnchoredGateOutput:
        if self.timestamp_s is not None:
            dt = float(observation.timestamp_s - self.timestamp_s)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError(f"timestamp must increase, observed dt={dt}")
        elif observation.missing:
            raise RuntimeError("M5-R5 cannot initialize from missing input")
        self.timestamp_s = float(observation.timestamp_s)
        anchor = np.asarray(observation.anchor_pose, dtype=np.float64).copy()
        innovation_scale = _robust_scale(self.innovation_history)

        if observation.missing:
            self.gap_frames += 1
            confidence = self.last_available_confidence * math.exp(
                -self.gap_frames / self.config.dropout_decay_frames
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
                reason="missing_anchor_passthrough",
            )

        measurement = np.asarray(observation.measurement_pose, dtype=np.float64)
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
        translation_m, rotation_deg = legacy.quotient_pose_error(
            anchor, measurement, observation.symmetry
        )
        disagreement = float(
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
            1.0 + (disagreement / max(innovation_scale, _EPS)) ** 2
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
        self.innovation_history.append(disagreement)
        self.last_available_confidence = confidence
        self.gap_frames = 0
        correction_applied = bool(
            self.config.blend_gain > 0
            and confidence >= self.config.minimum_correction_confidence
            and disagreement <= self.config.maximum_anchor_measurement_disagreement
        )
        correction_gain = (
            self.config.blend_gain * confidence if correction_applied else 0.0
        )
        pose = anchor @ legacy.se3_exp(correction_gain * residual)
        return self._output(
            observation,
            pose=pose,
            confidence=confidence,
            correction_applied=correction_applied,
            correction_gain=correction_gain,
            disagreement=disagreement,
            support_quality=support_quality,
            innovation_scale=innovation_scale,
            reason=(
                "bounded_anchor_correction"
                if correction_applied
                else "confidence_anchor_fallback"
            ),
        )

    def _output(
        self,
        observation: AnchoredGateObservation,
        *,
        pose: np.ndarray,
        confidence: float,
        correction_applied: bool,
        correction_gain: float,
        disagreement: float | None,
        support_quality: float | None,
        innovation_scale: float,
        reason: str,
    ) -> AnchoredGateOutput:
        pose_array = np.asarray(pose, dtype=np.float64)
        clipped_confidence = float(np.clip(confidence, 0.0, 1.0))
        uncertainty = self._uncertainty(clipped_confidence, innovation_scale)
        if not np.isfinite(pose_array).all() or not np.isfinite(uncertainty).all():
            raise FloatingPointError("M5-R5 output became non-finite")
        return AnchoredGateOutput(
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


def run_nearest_anchored_confidence_gate(
    observations: Iterable[AnchoredGateObservation], config: AnchoredGateConfig
) -> list[AnchoredGateOutput]:
    estimator = NearestAnchoredConfidenceGate(config)
    outputs: list[AnchoredGateOutput] = []
    for observation in observations:
        started = time.perf_counter_ns()
        output = estimator.step(observation)
        outputs.append(
            AnchoredGateOutput(
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
    "AnchoredGateConfig",
    "AnchoredGateObservation",
    "AnchoredGateOutput",
    "NearestAnchoredConfidenceGate",
    "run_nearest_anchored_confidence_gate",
]
