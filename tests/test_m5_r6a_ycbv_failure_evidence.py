from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "r2"
    / "m5_r6a_ycbv_sealed_failure"
    / "input_failure_audit.json"
)
REPORT_PATH = REPO_ROOT / "reports" / "r2" / "m5_r6a_ycbv_sealed_input_failure.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_ycbv_sealed_failure_evidence_is_internally_consistent() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    by_object = audit["by_object"]

    assert audit["status"] == "FAIL_M5_R6A_SEALED_INPUT"
    assert audit["failed_gates"] == ["missing_represented_object_count"]
    assert audit["represented_object_count"] == len(by_object) == 21
    assert audit["missing_represented_object_count"] == sum(
        int(row["natural_missing_frame_count"] > 0) for row in by_object.values()
    ) == 7
    assert audit["natural_missing_frame_count"] == sum(
        row["natural_missing_frame_count"] for row in by_object.values()
    ) == 357

    gap_counts: Counter[str] = Counter()
    for row in by_object.values():
        gap_counts.update(row["missing_gap_frame_counts"])
    assert dict(sorted(gap_counts.items())) == audit["missing_gap_frame_counts"] == {
        "long_gap_8_plus": 121,
        "medium_gap_3_7": 131,
        "short_gap_1_2": 105,
    }

    checks = audit["gate_checks"]
    assert checks["missing_represented_object_count"] == {
        "observed": 7,
        "passed": False,
        "required_minimum": 12,
    }
    assert all(
        row["passed"]
        for name, row in checks.items()
        if name != "missing_represented_object_count"
    )
    assert audit["foundationpose_inference_invocation_count"] == 0
    assert audit["sealed_evaluation_invocation_count"] == 0
    assert audit["prediction_or_pose_error_read"] is False
    assert audit["protocol_relaxed_or_rerun"] is False

    provenance_paths = {
        "protocol": REPO_ROOT / "protocols" / "poseloop_m5_r6a_ycbv_sealed_protocol.json",
        "frozen_builder": REPO_ROOT / "scripts" / "build_m5_r6a_ycbv_sealed.py",
        "failure_auditor": REPO_ROOT / "scripts" / "audit_m5_r6a_ycbv_sealed_failure.py",
    }
    for key, path in provenance_paths.items():
        provenance = audit["provenance"][key]
        assert _sha256(path) == provenance["sha256"]

    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "| Objects with natural missing frames | 12 | 7 | FAIL |" in report
    assert "| Natural missing frames | 96 | 357 | PASS |" in report
    assert "| Short-gap frames | 8 | 105 | PASS |" in report
    assert "| Medium-gap frames | 16 | 131 | PASS |" in report
    assert "| Long-gap frames | 8 | 121 | PASS |" in report
