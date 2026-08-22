"""Local-only preparation for PoseLoop pose-accuracy recovery diagnostics."""

from __future__ import annotations

PROTOCOL_ID = "poseloop.pose-accuracy-recovery.development-prep.v1"
MANIFEST_SCHEMA = "poseloop.pose-accuracy-recovery.manifest.v1"
PRODUCER_MANIFEST_SCHEMA = "poseloop.pose-accuracy-recovery.producer-manifest.v1"

__all__ = ["MANIFEST_SCHEMA", "PRODUCER_MANIFEST_SCHEMA", "PROTOCOL_ID"]
