#!/usr/bin/env python3
"""Synthetic SE(3), symmetry, corruption, and estimator kernel for M5-G0."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from m1_common import assert_pose


FRAME_COUNT = 120
CONTROL_HZ = 20.0
DT_SECONDS = 1.0 / CONTROL_HZ
TRANSLATION_NOISE_SIGMA_M = 0.002
ROTATION_NOISE_SIGMA_DEG = 1.0
_EPS = 1e-12


class SymmetryClass(str, Enum):
    ASYMMETRIC = "ASYMMETRIC"
    C2 = "C2"
    C4 = "C4"
    CONTINUOUS_AXIAL = "CONTINUOUS_AXIAL"


class MotionFamily(str, Enum):
    STATIC = "STATIC"
    CONSTANT_TWIST = "CONSTANT_TWIST"
    CURVED_ACCELERATING = "CURVED_ACCELERATING"
    DIRECTION_REVERSAL = "DIRECTION_REVERSAL"


class StressFamily(str, Enum):
    NOMINAL = "NOMINAL"
    REPRESENTATION_SWITCH = "REPRESENTATION_SWITCH"
    DROPOUT_5 = "DROPOUT_5"
    DROPOUT_10 = "DROPOUT_10"
    OUTLIER_BURST = "OUTLIER_BURST"
    COMBINED = "COMBINED"


class EstimatorKind(str, Enum):
    RAW_HOLD = "RAW_HOLD"
    STANDARD_SE3_CT = "STANDARD_SE3_CT"
    NEAREST_REPRESENTATIVE_CT = "NEAREST_REPRESENTATIVE_CT"
    SYMQUOT_CT = "SYMQUOT_CT"


def _coerce_enum(value: Any, enum_type: type[Enum]) -> Any:
    return value if isinstance(value, enum_type) else enum_type(str(value))


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _so3_left_jacobian(rotvec: np.ndarray) -> np.ndarray:
    omega = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    matrix = skew(omega)
    if theta < 1e-7:
        return np.eye(3) + 0.5 * matrix + (1.0 / 6.0) * matrix @ matrix
    a = (1.0 - math.cos(theta)) / (theta * theta)
    b = (theta - math.sin(theta)) / (theta**3)
    return np.eye(3) + a * matrix + b * matrix @ matrix


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()


def so3_log(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return Rotation.from_matrix(matrix).as_rotvec()


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """Exponential of ``[translation, rotation]`` body coordinates."""

    coordinates = np.asarray(twist, dtype=np.float64).reshape(6)
    translation = coordinates[:3]
    rotvec = coordinates[3:]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = so3_exp(rotvec)
    transform[:3, 3] = _so3_left_jacobian(rotvec) @ translation
    return transform


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Stable logarithm of a finite homogeneous transform."""

    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(value).all():
        raise ValueError("SE(3) logarithm input is non-finite")
    rotvec = so3_log(value[:3, :3])
    jacobian = _so3_left_jacobian(rotvec)
    translation = np.linalg.solve(jacobian, value[:3, 3])
    return np.concatenate([translation, rotvec])


def rigid_inverse(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = value[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ value[:3, 3]
    return inverse


def rotation_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).reshape(3, 3).T @ np.asarray(second).reshape(3, 3)
    return float(np.degrees(np.linalg.norm(so3_log(relative))))


def z_rotation_transform(angle_rad: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("z", angle_rad).as_matrix()
    return transform


@dataclass(frozen=True, slots=True)
class SymmetrySpec:
    symmetry_class: SymmetryClass
    finite_transforms: tuple[np.ndarray, ...]
    axis_object: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

    @property
    def name(self) -> str:
        return self.symmetry_class.value

    @property
    def continuous(self) -> bool:
        return self.symmetry_class is SymmetryClass.CONTINUOUS_AXIAL


def symmetry_spec(value: SymmetryClass | str) -> SymmetrySpec:
    kind = _coerce_enum(value, SymmetryClass)
    count = {
        SymmetryClass.ASYMMETRIC: 1,
        SymmetryClass.C2: 2,
        SymmetryClass.C4: 4,
        SymmetryClass.CONTINUOUS_AXIAL: 1,
    }[kind]
    transforms = tuple(
        z_rotation_transform(2.0 * math.pi * index / count) for index in range(count)
    )
    return SymmetrySpec(kind, transforms)


make_symmetry_spec = symmetry_spec


def align_continuous_gauge(
    measurement_pose: np.ndarray, reference_pose: np.ndarray
) -> tuple[np.ndarray, float]:
    """Choose the right axial gauge closest to a reference pose."""

    measurement = np.asarray(measurement_pose, dtype=np.float64).reshape(4, 4)
    reference = np.asarray(reference_pose, dtype=np.float64).reshape(4, 4)
    relative = reference[:3, :3].T @ measurement[:3, :3]
    angle = math.atan2(relative[0, 1] - relative[1, 0], relative[0, 0] + relative[1, 1])
    aligned = measurement @ z_rotation_transform(angle)
    return aligned, float(angle)


def quotient_residual(
    reference_pose: np.ndarray,
    measured_pose: np.ndarray,
    symmetry: SymmetrySpec,
    *,
    translation_scale_m: float = 1.0,
    rotation_scale_rad: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int | None, float]:
    """Return the minimum right-quotient residual and selected representative."""

    reference = np.asarray(reference_pose, dtype=np.float64).reshape(4, 4)
    measurement = np.asarray(measured_pose, dtype=np.float64).reshape(4, 4)
    if symmetry.continuous:
        representative, gauge = align_continuous_gauge(measurement, reference)
        residual = se3_log(rigid_inverse(reference) @ representative)
        residual[5] = 0.0
        return residual, representative, None, gauge

    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for index, element in enumerate(symmetry.finite_transforms):
        representative = measurement @ element
        residual = se3_log(rigid_inverse(reference) @ representative)
        score = float(
            np.sum(np.square(residual[:3] / translation_scale_m))
            + np.sum(np.square(residual[3:] / rotation_scale_rad))
        )
        candidates.append((score, index, residual, representative))
    score, index, residual, representative = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return residual.copy(), representative.copy(), index, float(score)


def quotient_rotation_error_degrees(
    first_pose: np.ndarray, second_pose: np.ndarray, symmetry: SymmetrySpec
) -> float:
    first = np.asarray(first_pose, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second_pose, dtype=np.float64).reshape(4, 4)
    if symmetry.continuous:
        first_axis = first[:3, :3] @ symmetry.axis_object
        second_axis = second[:3, :3] @ symmetry.axis_object
        cosine = float(np.clip(np.dot(first_axis, second_axis), -1.0, 1.0))
        return float(np.degrees(math.acos(cosine)))
    return min(
        rotation_angle_degrees(first[:3, :3], (second @ element)[:3, :3])
        for element in symmetry.finite_transforms
    )


def quotient_pose_error(
    first_pose: np.ndarray, second_pose: np.ndarray, symmetry: SymmetrySpec
) -> tuple[float, float]:
    translation = float(
        np.linalg.norm(np.asarray(first_pose)[:3, 3] - np.asarray(second_pose)[:3, 3])
    )
    rotation = quotient_rotation_error_degrees(first_pose, second_pose, symmetry)
    return translation, rotation


@dataclass(frozen=True, slots=True)
class GroundTruthSequence:
    timestamps: np.ndarray
    poses: np.ndarray
    symmetry: SymmetrySpec
    motion_family: MotionFamily
    seed: int
    reversal_frame: int | None
    translation_direction_camera: np.ndarray
    observable_rotation_axis_object: np.ndarray


def _initial_pose(rng: np.random.Generator) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(rng.normal(0.0, 0.18, size=3)).as_matrix()
    pose[:3, 3] = np.array([0.05, -0.03, 0.70]) + rng.normal(0.0, 0.015, size=3)
    return pose


def generate_ground_truth(
    symmetry: SymmetrySpec | SymmetryClass | str,
    motion_family: MotionFamily | str,
    seed: int,
    *,
    frame_count: int = FRAME_COUNT,
    control_hz: float = CONTROL_HZ,
) -> GroundTruthSequence:
    """Generate a deterministic, bounded synthetic object-to-camera trajectory."""

    spec = symmetry if isinstance(symmetry, SymmetrySpec) else symmetry_spec(symmetry)
    motion = _coerce_enum(motion_family, MotionFamily)
    rng = np.random.default_rng(int(seed))
    timestamps = np.arange(frame_count, dtype=np.float64) / float(control_hz)
    initial = _initial_pose(rng)
    translation_direction = rng.normal(size=3)
    translation_direction[2] *= 0.25
    translation_direction /= np.linalg.norm(translation_direction)
    rotation_axis = rng.normal(size=3)
    if spec.continuous:
        rotation_axis[2] = 0.0
    if np.linalg.norm(rotation_axis) < 1e-8:
        rotation_axis = np.array([1.0, 0.0, 0.0])
    rotation_axis /= np.linalg.norm(rotation_axis)
    speed = 0.025 * (0.85 + 0.30 * rng.random())
    angular_speed = math.radians(12.0 * (0.85 + 0.30 * rng.random()))
    poses = np.repeat(initial[None, :, :], frame_count, axis=0)
    reversal_frame: int | None = None

    for index, timestamp in enumerate(timestamps):
        pose = initial.copy()
        if motion is MotionFamily.STATIC:
            displacement = np.zeros(3)
            rotvec = np.zeros(3)
        elif motion is MotionFamily.CONSTANT_TWIST:
            displacement = translation_direction * speed * timestamp
            rotvec = rotation_axis * angular_speed * timestamp
        elif motion is MotionFamily.CURVED_ACCELERATING:
            displacement = translation_direction * (
                speed * timestamp + 0.0025 * timestamp**2
            ) + np.array(
                [
                    0.008 * math.sin(1.1 * timestamp),
                    0.006 * (1.0 - math.cos(0.9 * timestamp)),
                    0.004 * math.sin(0.7 * timestamp),
                ]
            )
            angle = angular_speed * timestamp + math.radians(1.2) * timestamp**2
            secondary = np.array([0.08 * math.sin(0.8 * timestamp), 0.0, 0.0])
            if spec.continuous:
                secondary[2] = 0.0
            rotvec = rotation_axis * angle + secondary
        else:
            reversal_frame = frame_count // 2
            midpoint = timestamps[reversal_frame]
            tau = 0.35
            scalar = (
                speed
                * tau
                * (
                    math.log(math.cosh(midpoint / tau))
                    - math.log(math.cosh((midpoint - timestamp) / tau))
                )
            )
            angle = (
                angular_speed
                * tau
                * (
                    math.log(math.cosh(midpoint / tau))
                    - math.log(math.cosh((midpoint - timestamp) / tau))
                )
            )
            displacement = translation_direction * scalar
            rotvec = rotation_axis * angle
        pose[:3, 3] = initial[:3, 3] + displacement
        pose[:3, :3] = initial[:3, :3] @ so3_exp(rotvec)
        poses[index] = pose

    assert_pose(poses[0], "synthetic truth first frame", rotation_atol=1e-7)
    assert_pose(poses[-1], "synthetic truth final frame", rotation_atol=1e-7)

    return GroundTruthSequence(
        timestamps=timestamps,
        poses=poses,
        symmetry=spec,
        motion_family=motion,
        seed=int(seed),
        reversal_frame=reversal_frame,
        translation_direction_camera=translation_direction,
        observable_rotation_axis_object=rotation_axis,
    )


@dataclass(frozen=True, slots=True)
class EstimatorInput:
    timestamp_s: float
    measurement_pose: np.ndarray | None
    missing: bool
    symmetry: SymmetrySpec

    def __post_init__(self) -> None:
        if self.missing != (self.measurement_pose is None):
            raise ValueError("missing must exactly match absence of measurement_pose")


@dataclass(frozen=True, slots=True)
class EvaluatorFrame:
    ground_truth_pose: np.ndarray
    noisy_canonical_measurement_pose: np.ndarray
    representative_index: int | None
    axial_gauge_rad: float | None
    dropout: bool
    outlier: bool


@dataclass(frozen=True, slots=True)
class CorruptedSequence:
    inputs: tuple[EstimatorInput, ...]
    evaluator_frames: tuple[EvaluatorFrame, ...]
    ground_truth: GroundTruthSequence
    stress_family: StressFamily
    corruption_seed: int
    dropout_intervals: tuple[tuple[int, int], ...]
    outlier_intervals: tuple[tuple[int, int], ...]

    @property
    def timestamps(self) -> np.ndarray:
        return self.ground_truth.timestamps


def _intervals(
    stress: StressFamily, seed: int
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    dropout: tuple[int, int] | None = None
    outlier: tuple[int, int] | None = None
    if stress is StressFamily.DROPOUT_5:
        start = 38 + seed % 17
        dropout = (start, start + 5)
    elif stress is StressFamily.DROPOUT_10:
        start = 38 + seed % 17
        dropout = (start, start + 10)
    elif stress is StressFamily.OUTLIER_BURST:
        start = 72 + seed % 13
        outlier = (start, start + 3)
    elif stress is StressFamily.COMBINED:
        dropout_start = 40 + seed % 9
        outlier_start = 76 + seed % 11
        dropout = (dropout_start, dropout_start + 10)
        outlier = (outlier_start, outlier_start + 3)
    return dropout, outlier


def _in_interval(index: int, interval: tuple[int, int] | None) -> bool:
    return interval is not None and interval[0] <= index < interval[1]


def corrupt_measurements(
    ground_truth: GroundTruthSequence,
    stress_family: StressFamily | str,
    corruption_seed: int,
) -> CorruptedSequence:
    """Apply deterministic noise, representation changes, dropout, and outliers."""

    stress = _coerce_enum(stress_family, StressFamily)
    rng = np.random.default_rng(int(corruption_seed))
    dropout_interval, outlier_interval = _intervals(stress, int(corruption_seed))
    switching = stress in {StressFamily.REPRESENTATION_SWITCH, StressFamily.COMBINED}
    representative = 0
    dwell = 5
    gauge = float(rng.uniform(-math.pi, math.pi))
    gauge_velocity = float(rng.normal(math.radians(15.0), math.radians(3.0)))
    inputs: list[EstimatorInput] = []
    evaluator: list[EvaluatorFrame] = []

    for index, (timestamp, truth) in enumerate(
        zip(ground_truth.timestamps, ground_truth.poses)
    ):
        noise = np.concatenate(
            [
                rng.normal(0.0, TRANSLATION_NOISE_SIGMA_M, size=3),
                rng.normal(0.0, math.radians(ROTATION_NOISE_SIGMA_DEG), size=3),
            ]
        )
        canonical = truth @ se3_exp(noise)
        outlier = _in_interval(index, outlier_interval)
        if outlier:
            direction = np.array([0.73, -0.51, 0.45], dtype=np.float64)
            direction /= np.linalg.norm(direction)
            axis = np.array([0.76, 0.65, 0.0], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            perturbation = np.concatenate(
                [0.030 * direction, math.radians(20.0) * axis]
            )
            canonical = canonical @ se3_exp(perturbation)

        representative_index: int | None = 0
        axial_gauge: float | None = None
        measurement = canonical
        if switching and not ground_truth.symmetry.continuous:
            dwell += 1
            if dwell >= 5 and rng.random() > 0.92:
                choices = [
                    item
                    for item in range(len(ground_truth.symmetry.finite_transforms))
                    if item != representative
                ]
                if choices:
                    representative = int(rng.choice(choices))
                    dwell = 0
            representative_index = representative
            measurement = (
                canonical @ ground_truth.symmetry.finite_transforms[representative]
            )
        elif switching and ground_truth.symmetry.continuous:
            dwell += 1
            gauge += gauge_velocity * DT_SECONDS
            if dwell >= 5 and rng.random() > 0.92:
                gauge += float(rng.choice([-1.0, 1.0])) * math.radians(
                    float(rng.uniform(70.0, 130.0))
                )
                gauge_velocity = float(
                    rng.normal(math.radians(15.0), math.radians(3.0))
                )
                dwell = 0
            gauge = float((gauge + math.pi) % (2.0 * math.pi) - math.pi)
            representative_index = None
            axial_gauge = gauge
            measurement = canonical @ z_rotation_transform(gauge)

        dropout = _in_interval(index, dropout_interval)
        input_record = EstimatorInput(
            timestamp_s=float(timestamp),
            measurement_pose=None if dropout else measurement.copy(),
            missing=dropout,
            symmetry=ground_truth.symmetry,
        )
        inputs.append(input_record)
        evaluator.append(
            EvaluatorFrame(
                ground_truth_pose=truth.copy(),
                noisy_canonical_measurement_pose=canonical.copy(),
                representative_index=representative_index,
                axial_gauge_rad=axial_gauge,
                dropout=dropout,
                outlier=outlier,
            )
        )

    return CorruptedSequence(
        inputs=tuple(inputs),
        evaluator_frames=tuple(evaluator),
        ground_truth=ground_truth,
        stress_family=stress,
        corruption_seed=int(corruption_seed),
        dropout_intervals=() if dropout_interval is None else (dropout_interval,),
        outlier_intervals=() if outlier_interval is None else (outlier_interval,),
    )


@dataclass(frozen=True, slots=True)
class EstimatorConfig:
    process_translation_sigma_m: float = 0.002
    process_rotation_sigma_deg: float = 1.0
    velocity_process_sigma: float = 0.04
    measurement_translation_sigma_m: float = 0.002
    measurement_rotation_sigma_deg: float = 1.0
    innovation_gate_sigma: float = 5.5
    velocity_update_gain: float = 0.35
    reacquire_after: int = 4


@dataclass(frozen=True, slots=True)
class EstimatorOutput:
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
    runtime_s: float = 0.0

    @property
    def uncertainty(self) -> float:
        return float(np.sqrt(np.sum(self.uncertainty_diag)))


class RawHoldEstimator:
    def __init__(self) -> None:
        self.pose: np.ndarray | None = None
        self.timestamp: float | None = None

    def step(self, item: EstimatorInput) -> EstimatorOutput:
        if item.measurement_pose is not None:
            self.pose = np.asarray(item.measurement_pose, dtype=np.float64).copy()
            reason = "raw_measurement"
            accepted = True
        elif self.pose is not None:
            reason = "dropout_hold"
            accepted = False
        else:
            raise RuntimeError("RAW_HOLD cannot initialize from a missing measurement")
        self.timestamp = item.timestamp_s
        return EstimatorOutput(
            timestamp_s=item.timestamp_s,
            pose=self.pose.copy(),
            body_twist=np.zeros(6),
            uncertainty_diag=np.zeros(6),
            accepted=accepted,
            residual=None,
            residual_norm=None,
            reason=reason,
            selected_symmetry_index=None,
            selected_gauge_rad=None,
        )


class ConstantTwistEstimator:
    def __init__(self, kind: EstimatorKind, config: EstimatorConfig) -> None:
        if kind not in {
            EstimatorKind.STANDARD_SE3_CT,
            EstimatorKind.NEAREST_REPRESENTATIVE_CT,
            EstimatorKind.SYMQUOT_CT,
        }:
            raise ValueError(f"Unsupported temporal estimator: {kind}")
        self.kind = kind
        self.config = config
        self.pose: np.ndarray | None = None
        self.twist = np.zeros(6, dtype=np.float64)
        self.pose_variance = np.zeros(6, dtype=np.float64)
        self.twist_variance = np.zeros(6, dtype=np.float64)
        self.timestamp: float | None = None
        self.last_accepted_pose: np.ndarray | None = None
        self.rejection_streak = 0

    def _measurement_variance(self) -> np.ndarray:
        return np.array(
            [
                *([self.config.measurement_translation_sigma_m**2] * 3),
                *([math.radians(self.config.measurement_rotation_sigma_deg) ** 2] * 3),
            ],
            dtype=np.float64,
        )

    def _process_variance(self, dt: float) -> np.ndarray:
        return dt * np.array(
            [
                *([self.config.process_translation_sigma_m**2] * 3),
                *([math.radians(self.config.process_rotation_sigma_deg) ** 2] * 3),
            ],
            dtype=np.float64,
        )

    def _initialize(self, item: EstimatorInput) -> EstimatorOutput:
        if item.measurement_pose is None:
            raise RuntimeError("Temporal estimator cannot initialize from missing data")
        self.pose = np.asarray(item.measurement_pose, dtype=np.float64).copy()
        self.timestamp = item.timestamp_s
        self.last_accepted_pose = self.pose.copy()
        self.pose_variance = self._measurement_variance()
        self.twist_variance.fill(self.config.velocity_process_sigma**2)
        if self.kind is EstimatorKind.SYMQUOT_CT and item.symmetry.continuous:
            self.twist[5] = 0.0
        return EstimatorOutput(
            timestamp_s=item.timestamp_s,
            pose=self.pose.copy(),
            body_twist=self.twist.copy(),
            uncertainty_diag=self.pose_variance.copy(),
            accepted=True,
            residual=np.zeros(6),
            residual_norm=0.0,
            reason="initialized_from_measurement",
            selected_symmetry_index=0,
            selected_gauge_rad=0.0 if item.symmetry.continuous else None,
        )

    def step(self, item: EstimatorInput) -> EstimatorOutput:
        if self.pose is None:
            return self._initialize(item)
        assert self.timestamp is not None
        dt = float(item.timestamp_s - self.timestamp)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"timestamp must increase, observed dt={dt}")
        previous_pose = self.pose.copy()
        predicted = self.pose @ se3_exp(self.twist * dt)
        self.pose_variance = (
            self.pose_variance
            + self._process_variance(dt)
            + self.twist_variance * dt * dt
        )
        self.twist_variance = self.twist_variance + (
            self.config.velocity_process_sigma**2 * dt
        )
        self.timestamp = item.timestamp_s

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

        measurement = np.asarray(item.measurement_pose, dtype=np.float64)
        selected_index: int | None = None
        selected_gauge: float | None = None
        if self.kind is EstimatorKind.STANDARD_SE3_CT:
            representative = measurement
            residual = se3_log(rigid_inverse(predicted) @ representative)
        elif self.kind is EstimatorKind.NEAREST_REPRESENTATIVE_CT:
            assert self.last_accepted_pose is not None
            _, representative, selected_index, selected_gauge = quotient_residual(
                self.last_accepted_pose,
                measurement,
                item.symmetry,
                translation_scale_m=self.config.measurement_translation_sigma_m,
                rotation_scale_rad=math.radians(
                    self.config.measurement_rotation_sigma_deg
                ),
            )
            residual = se3_log(rigid_inverse(predicted) @ representative)
        else:
            residual, representative, selected_index, selected_gauge = (
                quotient_residual(
                    predicted,
                    measurement,
                    item.symmetry,
                    translation_scale_m=self.config.measurement_translation_sigma_m,
                    rotation_scale_rad=math.radians(
                        self.config.measurement_rotation_sigma_deg
                    ),
                )
            )
            if item.symmetry.continuous:
                residual[5] = 0.0

        innovation_variance = self.pose_variance + self._measurement_variance()
        normalized = residual / np.sqrt(np.maximum(innovation_variance, _EPS))
        if self.kind is EstimatorKind.SYMQUOT_CT and item.symmetry.continuous:
            normalized = normalized.copy()
            normalized[5] = 0.0
            divisor = 5.0
        else:
            divisor = 6.0
        residual_norm = float(math.sqrt(np.sum(normalized**2) / divisor))
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
            if self.kind is EstimatorKind.SYMQUOT_CT and item.symmetry.continuous:
                self.pose = predicted @ se3_exp(residual)
            else:
                self.pose = representative.copy()
            self.twist.fill(0.0)
            if self.kind is EstimatorKind.SYMQUOT_CT and item.symmetry.continuous:
                self.twist[5] = 0.0
            self.pose_variance = self._measurement_variance()
            self.last_accepted_pose = self.pose.copy()
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
        measurement_variance = self._measurement_variance()
        gain = self.pose_variance / np.maximum(
            self.pose_variance + measurement_variance, _EPS
        )
        if self.kind is EstimatorKind.SYMQUOT_CT and item.symmetry.continuous:
            gain[5] = 0.0
        corrected = predicted @ se3_exp(gain * residual)
        observed_twist = se3_log(rigid_inverse(previous_pose) @ corrected) / dt
        if self.kind is EstimatorKind.SYMQUOT_CT and item.symmetry.continuous:
            observed_twist[5] = 0.0
        velocity_gain = self.config.velocity_update_gain
        self.twist = (1.0 - velocity_gain) * self.twist + velocity_gain * observed_twist
        if self.kind is EstimatorKind.SYMQUOT_CT and item.symmetry.continuous:
            self.twist[5] = 0.0
        self.pose = corrected
        self.pose_variance = np.maximum(
            (1.0 - gain) * self.pose_variance,
            1e-16,
        )
        self.twist_variance *= 1.0 - 0.25 * velocity_gain
        self.last_accepted_pose = self.pose.copy()
        return self._output(
            item,
            accepted=True,
            residual=residual,
            residual_norm=residual_norm,
            reason="measurement_updated",
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
        if (
            not np.isfinite(self.pose).all()
            or not np.isfinite(self.twist).all()
            or not np.isfinite(self.pose_variance).all()
            or not np.isfinite(self.twist_variance).all()
        ):
            raise FloatingPointError("Estimator state became non-finite")
        return EstimatorOutput(
            timestamp_s=item.timestamp_s,
            pose=self.pose.copy(),
            body_twist=self.twist.copy(),
            uncertainty_diag=self.pose_variance.copy(),
            accepted=accepted,
            residual=None if residual is None else residual.copy(),
            residual_norm=residual_norm,
            reason=reason,
            selected_symmetry_index=selected_index,
            selected_gauge_rad=selected_gauge,
        )


def create_estimator(
    kind: EstimatorKind | str, config: EstimatorConfig | None = None
) -> RawHoldEstimator | ConstantTwistEstimator:
    estimator_kind = _coerce_enum(kind, EstimatorKind)
    if estimator_kind is EstimatorKind.RAW_HOLD:
        return RawHoldEstimator()
    return ConstantTwistEstimator(estimator_kind, config or EstimatorConfig())


def run_estimator(
    sequence: CorruptedSequence,
    kind: EstimatorKind | str,
    config: EstimatorConfig | None = None,
) -> list[EstimatorOutput]:
    estimator = create_estimator(kind, config)
    outputs: list[EstimatorOutput] = []
    for item in sequence.inputs:
        started = time.perf_counter_ns()
        output = estimator.step(item)
        runtime = (time.perf_counter_ns() - started) * 1e-9
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
                runtime_s=runtime,
            )
        )
    return outputs


def measurement_stream(sequence: CorruptedSequence) -> np.ndarray:
    values = np.full((len(sequence.inputs), 4, 4), np.nan, dtype=np.float64)
    for index, item in enumerate(sequence.inputs):
        if item.measurement_pose is not None:
            values[index] = item.measurement_pose
    return values


def output_arrays(outputs: Iterable[EstimatorOutput]) -> dict[str, np.ndarray]:
    rows = list(outputs)
    return {
        "poses": np.stack([row.pose for row in rows]),
        "twists": np.stack([row.body_twist for row in rows]),
        "uncertainty": np.stack([row.uncertainty_diag for row in rows]),
        "accepted": np.asarray([row.accepted for row in rows], dtype=bool),
        "residual_norm": np.asarray(
            [
                np.nan if row.residual_norm is None else row.residual_norm
                for row in rows
            ],
            dtype=np.float64,
        ),
        "runtime_s": np.asarray([row.runtime_s for row in rows], dtype=np.float64),
    }


__all__ = [
    "CONTROL_HZ",
    "CorruptedSequence",
    "DT_SECONDS",
    "EstimatorConfig",
    "EstimatorInput",
    "EstimatorKind",
    "EstimatorOutput",
    "EvaluatorFrame",
    "FRAME_COUNT",
    "GroundTruthSequence",
    "MotionFamily",
    "StressFamily",
    "SymmetryClass",
    "SymmetrySpec",
    "align_continuous_gauge",
    "corrupt_measurements",
    "create_estimator",
    "generate_ground_truth",
    "make_symmetry_spec",
    "measurement_stream",
    "output_arrays",
    "quotient_pose_error",
    "quotient_residual",
    "quotient_rotation_error_degrees",
    "rigid_inverse",
    "rotation_angle_degrees",
    "run_estimator",
    "se3_exp",
    "se3_log",
    "so3_exp",
    "so3_log",
    "symmetry_spec",
    "z_rotation_transform",
]
