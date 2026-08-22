"""A-R4 hash-locked CNOS renderer repair namespace."""

from __future__ import annotations

ROUTE_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-runtime-route.v1r4"
ROUTE_ID = "poseloop.pose-accuracy-recovery.development.cnos-runtime-prep.v1r4"
REQUEST_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-render-request.v1r4"
ATTEMPT_START_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-render-attempt-start.v1r4"
ATTEMPT_FAILURE_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-render-attempt-failure.v1r4"
)
ATTEMPT_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-render-attempt-receipt.v1r4"
)

OBJECT_IDS = (1, 2, 4, 5, 6)
VIEW_COUNT = 42
GPU_OVERRIDE = "0"
ROLE = "DEVELOPMENT_ONLY_DESCRIPTOR_RENDERER"

__all__ = [
    "ATTEMPT_FAILURE_SCHEMA",
    "ATTEMPT_RECEIPT_SCHEMA",
    "ATTEMPT_START_SCHEMA",
    "GPU_OVERRIDE",
    "OBJECT_IDS",
    "REQUEST_SCHEMA",
    "ROLE",
    "ROUTE_ID",
    "ROUTE_SCHEMA",
    "VIEW_COUNT",
]
