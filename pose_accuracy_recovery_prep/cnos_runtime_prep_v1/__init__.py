"""Pinned official-CNOS runtime preparation and producer namespace."""

from __future__ import annotations

ROUTE_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-runtime-route.v1"
ROUTE_ID = "poseloop.pose-accuracy-recovery.development.cnos-runtime-prep.v1"
DEPLOYMENT_REQUEST_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-deployment-request.v1"
)
DEPLOYMENT_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-deployment.v1"
RENDER_MANIFEST_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-render-manifest.v1"
RUN_STATE_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-run-state.v1"
RUN_RECEIPT_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-run-receipt.v1"
SCORE_TRACE_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-candidate-score-trace.v1"

__all__ = [
    "DEPLOYMENT_REQUEST_SCHEMA",
    "DEPLOYMENT_SCHEMA",
    "RENDER_MANIFEST_SCHEMA",
    "ROUTE_ID",
    "ROUTE_SCHEMA",
    "RUN_RECEIPT_SCHEMA",
    "RUN_STATE_SCHEMA",
    "SCORE_TRACE_SCHEMA",
]
