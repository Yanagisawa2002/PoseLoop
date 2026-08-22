"""A-R3 hash-locked thin wrapper for the pinned CNOS CAD renderer."""

from __future__ import annotations

ROUTE_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-runtime-route.v1r3"
ROUTE_ID = "poseloop.pose-accuracy-recovery.development.cnos-runtime-prep.v1r3"
REQUEST_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-render-request.v1r3"
ATTEMPT_START_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-render-attempt-start.v1r3"
)
ATTEMPT_FAILURE_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-render-attempt-failure.v1r3"
)
ATTEMPT_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-render-attempt-receipt.v1r3"
)

OBJECT_IDS = (1, 2, 4, 5, 6)
VIEW_COUNT = 42
GPU_OVERRIDE = "0"

__all__ = [
    "ATTEMPT_FAILURE_SCHEMA",
    "ATTEMPT_RECEIPT_SCHEMA",
    "ATTEMPT_START_SCHEMA",
    "GPU_OVERRIDE",
    "OBJECT_IDS",
    "REQUEST_SCHEMA",
    "ROUTE_ID",
    "ROUTE_SCHEMA",
    "VIEW_COUNT",
]
