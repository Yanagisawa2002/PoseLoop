"""A-R5 label-blind frame-level CNOS instance-proposal namespace."""

from __future__ import annotations

PROTOCOL_SCHEMA = "poseloop.pose-accuracy-recovery.instance-proposal-protocol.v1r5"
PROTOCOL_ID = "poseloop.pose-accuracy-recovery.development.instance-proposal.v1r5"
FRAME_MANIFEST_SCHEMA = "poseloop.pose-accuracy-recovery.instance-frame-manifest.v1r5"
RUNTIME_REQUEST_SCHEMA = "poseloop.pose-accuracy-recovery.instance-runtime-request.v1r5"
OUTPUT_BUNDLE_SCHEMA = "poseloop.pose-accuracy-recovery.instance-producer-output.v1r5"
RUN_STATE_SCHEMA = "poseloop.pose-accuracy-recovery.instance-run-state.v1r5"
RUN_RECEIPT_SCHEMA = "poseloop.pose-accuracy-recovery.instance-run-receipt.v1r5"
ITEM_RECEIPT_SCHEMA = "poseloop.pose-accuracy-recovery.instance-item-receipt.v1r5"
SCORE_TRACE_SCHEMA = "poseloop.pose-accuracy-recovery.instance-score-trace.v1r5"

__all__ = [
    "FRAME_MANIFEST_SCHEMA",
    "ITEM_RECEIPT_SCHEMA",
    "OUTPUT_BUNDLE_SCHEMA",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "RUN_RECEIPT_SCHEMA",
    "RUN_STATE_SCHEMA",
    "RUNTIME_REQUEST_SCHEMA",
    "SCORE_TRACE_SCHEMA",
]
