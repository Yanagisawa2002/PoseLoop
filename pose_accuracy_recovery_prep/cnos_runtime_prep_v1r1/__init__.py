"""R1 compatibility route for the frozen official-CNOS runtime prep."""

from __future__ import annotations

ROUTE_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-runtime-route.v1r1"
ROUTE_ID = "poseloop.pose-accuracy-recovery.development.cnos-runtime-prep.v1r1"
DEPLOYMENT_REQUEST_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-deployment-request.v1r1"
)
DEPLOYMENT_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-deployment.v1r1"
RUNTIME_LOCK_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-runtime-lock.v1r1"
FREEZE_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-deployment-freeze-receipt.v1r1"
)
WORKLOAD_AUDIT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-workload-provenance-audit.v1r1"
)
RUN_RECEIPT_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-run-receipt.v1r1"

__all__ = [
    "DEPLOYMENT_REQUEST_SCHEMA",
    "DEPLOYMENT_SCHEMA",
    "FREEZE_RECEIPT_SCHEMA",
    "ROUTE_ID",
    "ROUTE_SCHEMA",
    "RUN_RECEIPT_SCHEMA",
    "RUNTIME_LOCK_SCHEMA",
    "WORKLOAD_AUDIT_SCHEMA",
]
