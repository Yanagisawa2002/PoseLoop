"""Independent CAD-conditioned instance-proposal preparation contracts."""

from __future__ import annotations

PROTOCOL_SCHEMA = "poseloop.pose-accuracy-recovery.instance-proposal-protocol.v1"
PROTOCOL_ID = "poseloop.pose-accuracy-recovery.development.instance-proposal.v1"
RUNTIME_LOCK_SCHEMA = "poseloop.pose-accuracy-recovery.instance-proposal-runtime-lock.v1"
INPUT_MANIFEST_SCHEMA = (
    "poseloop.pose-accuracy-recovery.instance-proposal-input-manifest.v1"
)
PROPOSAL_SCHEMA = "poseloop.pose-accuracy-recovery.instance-proposal-output.v1"
CONTENT_GATE_SCHEMA = "poseloop.pose-accuracy-recovery.instance-content-gate.v1"

__all__ = [
    "CONTENT_GATE_SCHEMA",
    "INPUT_MANIFEST_SCHEMA",
    "PROPOSAL_SCHEMA",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "RUNTIME_LOCK_SCHEMA",
]
