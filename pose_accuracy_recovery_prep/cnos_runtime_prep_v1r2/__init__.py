"""A-R2 camera-isolated compatibility route for the official CNOS producer."""

from __future__ import annotations

ROUTE_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-runtime-route.v1r2"
ROUTE_ID = "poseloop.pose-accuracy-recovery.development.cnos-runtime-prep.v1r2"
CAMERA_PROVENANCE_MANIFEST_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-camera-provenance-manifest.v1r2"
)
DERIVED_CAMERA_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-derived-runtime-camera.v1r2"
)
CAMERA_DERIVATION_MANIFEST_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-camera-derivation-manifest.v1r2"
)
CAMERA_DERIVATION_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-camera-derivation-receipt.v1r2"
)
DEPLOYMENT_REQUEST_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-deployment-request.v1r2"
)
DEPLOYMENT_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-deployment.v1r2"
RUNTIME_LOCK_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-runtime-lock.v1r2"
FREEZE_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-deployment-freeze-receipt.v1r2"
)
RUN_RECEIPT_SCHEMA = "poseloop.pose-accuracy-recovery.cnos-run-receipt.v1r2"

__all__ = [
    "CAMERA_DERIVATION_MANIFEST_SCHEMA",
    "CAMERA_DERIVATION_RECEIPT_SCHEMA",
    "CAMERA_PROVENANCE_MANIFEST_SCHEMA",
    "DEPLOYMENT_REQUEST_SCHEMA",
    "DEPLOYMENT_SCHEMA",
    "DERIVED_CAMERA_SCHEMA",
    "FREEZE_RECEIPT_SCHEMA",
    "ROUTE_ID",
    "ROUTE_SCHEMA",
    "RUN_RECEIPT_SCHEMA",
    "RUNTIME_LOCK_SCHEMA",
]
