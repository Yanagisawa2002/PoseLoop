"""Evaluator-only finalization for the frozen R4-A v3 handoff."""

PROTOCOL_ID = "poseloop.r4a.xyzibd-train-pbr.development-final-evaluation.v1"
PROTOCOL_SCHEMA = "poseloop.r4a.development-final-evaluation.protocol.v1"
PROTOCOL_ID_V2 = "poseloop.r4a.xyzibd-train-pbr.development-final-evaluation.v2"
PROTOCOL_SCHEMA_V2 = "poseloop.r4a.development-final-evaluation.protocol.v2"
PROTOCOL_ID_V3 = "poseloop.r4a.xyzibd-train-pbr.development-final-evaluation.v3"
PROTOCOL_SCHEMA_V3 = "poseloop.r4a.development-final-evaluation.protocol.v3"
PROTOCOL_ID_V4 = "poseloop.r4a.xyzibd-train-pbr.development-final-evaluation.v4"
PROTOCOL_SCHEMA_V4 = "poseloop.r4a.development-final-evaluation.protocol.v4"
SUPPORTED_PROTOCOLS = {
    PROTOCOL_ID: PROTOCOL_SCHEMA,
    PROTOCOL_ID_V2: PROTOCOL_SCHEMA_V2,
    PROTOCOL_ID_V3: PROTOCOL_SCHEMA_V3,
    PROTOCOL_ID_V4: PROTOCOL_SCHEMA_V4,
}

__all__ = [
    "PROTOCOL_ID",
    "PROTOCOL_ID_V2",
    "PROTOCOL_ID_V3",
    "PROTOCOL_ID_V4",
    "PROTOCOL_SCHEMA",
    "PROTOCOL_SCHEMA_V2",
    "PROTOCOL_SCHEMA_V3",
    "PROTOCOL_SCHEMA_V4",
    "SUPPORTED_PROTOCOLS",
]
