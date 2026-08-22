from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_m3_r2_scheduler as scheduler  # noqa: E402


def test_sealed_scheduler_quotas_keep_future_prefix_nested() -> None:
    q1 = scheduler.rounded_quota(150, {"numerator": 144, "denominator": 300})
    q2 = scheduler.rounded_quota(150, {"numerator": 33, "denominator": 300})
    assert (q1, q2) == (72, 17)
    assert q2 <= q1
    assert (150 + 2 * q1 + 2 * q2) / 150 <= 3.0
