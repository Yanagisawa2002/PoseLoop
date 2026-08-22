"""CLI for the frozen R4-A development readiness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .core import audit_readiness


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--extraction-result", type=Path, required=True)
    parser.add_argument("--prefreeze-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--label-access", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_readiness(
        protocol_path=args.protocol.resolve(),
        plan_path=args.plan.resolve(),
        extraction_result_path=args.extraction_result.resolve(),
        prefreeze_receipt_path=args.prefreeze_receipt.resolve(),
        data_root=args.data_root.resolve(),
        models_root=args.models_root.resolve(),
        output_path=args.output.resolve(),
        target_manifest_path=args.target_manifest.resolve(),
        label_access_path=args.label_access.resolve(),
    )
    print(json.dumps({"status": result["status"], "gate_pass": result["gate_pass"], "target_count": result["target_count"]}, sort_keys=True))
    return 0
