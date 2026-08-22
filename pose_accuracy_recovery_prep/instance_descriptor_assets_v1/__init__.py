"""A-R5-P2 create-only CNOS descriptor asset namespace."""

from __future__ import annotations

PROTOCOL_SCHEMA = (
    "poseloop.pose-accuracy-recovery.instance-descriptor-assets-protocol.v1"
)
PROTOCOL_ID = (
    "poseloop.pose-accuracy-recovery.development.instance-descriptor-assets.v1"
)
REQUEST_SCHEMA = "poseloop.pose-accuracy-recovery.instance-descriptor-assets-request.v1"
TEMPLATE_IMPORT_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.a-r4-template-import-receipt.v1"
)
OBJECT_SIDECAR_SCHEMA = (
    "poseloop.pose-accuracy-recovery.instance-descriptor-object-sidecar.v1"
)
PLANNED_STOP_SCHEMA = (
    "poseloop.pose-accuracy-recovery.instance-descriptor-planned-stop.v1"
)
FINAL_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.instance-descriptor-generation-receipt.v1"
)
COMPATIBILITY_BUNDLE_SCHEMA = (
    "poseloop.pose-accuracy-recovery.instance-descriptor-a-r5-bundle.v1"
)
VALIDATION_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.instance-descriptor-validation-receipt.v1"
)

OBJECT_IDS = (1, 2, 4, 5, 6)
VIEW_COUNT = 42
FEATURE_DIMENSION = 1024

__all__ = [
    "COMPATIBILITY_BUNDLE_SCHEMA",
    "FEATURE_DIMENSION",
    "FINAL_RECEIPT_SCHEMA",
    "OBJECT_IDS",
    "OBJECT_SIDECAR_SCHEMA",
    "PLANNED_STOP_SCHEMA",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "REQUEST_SCHEMA",
    "TEMPLATE_IMPORT_RECEIPT_SCHEMA",
    "VALIDATION_RECEIPT_SCHEMA",
    "VIEW_COUNT",
]
