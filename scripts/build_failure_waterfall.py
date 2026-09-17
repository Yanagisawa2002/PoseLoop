#!/usr/bin/env python3
"""Render the v1.1.0 end-to-end failure budget from tracked release results.

This script is intentionally standard-library only. It does not recompute model
metrics; it derives an auditable stage-level accounting from the frozen compact
release result bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "release/v1.1.0/results.json"


def _percent(part: int, whole: int) -> str:
    if whole <= 0:
        raise ValueError("percentage denominator must be positive")
    return f"{100.0 * part / whole:.1f}%"


def load_release_results(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "poseloop.release-result.v1":
        raise ValueError("unexpected PoseLoop release result schema")
    return value


def validate_accounting(result: dict[str, Any]) -> dict[str, int]:
    dataset = result["dataset"]
    detector = result["detector"]
    pose = result["pose_pipeline"]

    gt = int(dataset["ground_truth_instance_count"])
    predictions = int(detector["prediction_count"])
    mask_matches = int(pose["mask_iou50_matches"])
    joint_successes = int(pose["joint_pose_successes"])
    tp = int(detector["tp_iou50"])
    fp = int(detector["fp_iou50"])
    fn = int(detector["fn_iou50"])
    runtime_completed = int(pose["runtime_completed"])
    runtime_expected = int(pose["runtime_expected"])

    if gt <= 0 or predictions <= 0:
        raise ValueError("release counts must be positive")
    if tp != mask_matches:
        raise ValueError("detector TP and pose mask-match counts disagree")
    if tp + fn != gt:
        raise ValueError("detector TP + FN does not equal GT count")
    if tp + fp != predictions:
        raise ValueError("detector TP + FP does not equal prediction count")
    if not 0 <= joint_successes <= mask_matches:
        raise ValueError("joint pose successes must be a subset of mask matches")
    if runtime_completed != runtime_expected or runtime_completed != predictions:
        raise ValueError("runtime completion does not cover all frozen predictions")

    return {
        "gt": gt,
        "predictions": predictions,
        "mask_matches": mask_matches,
        "joint_successes": joint_successes,
        "mask_stage_loss": gt - mask_matches,
        "pose_stage_loss": mask_matches - joint_successes,
        "false_positives": fp,
    }


def render_markdown(result: dict[str, Any]) -> str:
    counts = validate_accounting(result)
    pose = result["pose_pipeline"]
    detector = result["detector"]

    gt = counts["gt"]
    matches = counts["mask_matches"]
    successes = counts["joint_successes"]
    mask_loss = counts["mask_stage_loss"]
    pose_loss = counts["pose_stage_loss"]
    predictions = counts["predictions"]
    fp = counts["false_positives"]

    scene_rows = []
    recalls = pose.get("per_scene_joint_recall", {})
    for scene_id in sorted(recalls, key=lambda value: int(value)):
        scene_rows.append(f"| {scene_id} | {float(recalls[scene_id]):.4f} |")

    return "\n".join(
        [
            "# PoseLoop v1.1.0 failure waterfall",
            "",
            "This page is generated from `release/v1.1.0/results.json`. It does not introduce",
            "new evaluation data or recompute model metrics; it exposes the frozen end-to-end",
            "result as a stage-level failure budget.",
            "",
            "```mermaid",
            "flowchart LR",
            f'    GT["{gt} GT instances"] -->|"{mask_loss} not matched at mask IoU >= 0.50"| MM["{matches} mask-IoU50 matches"]',
            f'    MM -->|"{pose_loss} fail joint pose criteria"| JS["{successes} joint pose successes"]',
            "```",
            "",
            "| Stage | Count | Share of GT | Interpretation |",
            "| --- | ---: | ---: | --- |",
            f"| Ground-truth instances | {gt} | 100.0% | Fixed development evaluation population |",
            f"| Mask-IoU50 matches | {matches} | {_percent(matches, gt)} | Detection/instance-mask handoff reached the pose stage |",
            f"| Not matched at mask IoU >= 0.50 | {mask_loss} | {_percent(mask_loss, gt)} | Lost before the pose-correctness gate; this includes misses and masks that do not form an IoU50 match |",
            f"| Joint pose successes | {successes} | {_percent(successes, gt)} | Mask match plus normalized MSSD < 0.10 and MSPD < 10 px |",
            f"| Pose-stage losses after mask match | {pose_loss} | {_percent(pose_loss, gt)} | {_percent(pose_loss, matches)} of mask-matched GT instances fail the joint pose gate |",
            "",
            f"The detector emitted {predictions} predictions, including {fp} IoU50 false positives",
            f"({_percent(fp, predictions)} of emitted predictions). Runtime completed all",
            f"{int(pose['runtime_completed'])}/{int(pose['runtime_expected'])} frozen pose registrations.",
            "",
            "## What this says about the next optimization",
            "",
            "The largest absolute GT loss in v1.1.0 is still before joint pose scoring:",
            f"{mask_loss} GT instances do not obtain a mask-IoU50 match, versus {pose_loss} additional",
            "losses after a mask match exists. This does **not** prove every upstream loss is a",
            "detector miss: the IoU50 bucket also contains boundary/instance-formation failures.",
            f"The detector AP75 is only {float(detector['ap75']):.3f}, so high-IoU mask quality remains",
            "a concrete diagnostic target. The next experiment should therefore separate outright",
            "misses, over/under-segmentation, mask-boundary errors, and pose-registration failures",
            "before changing either model family.",
            "",
            "## Per-scene joint recall",
            "",
            "| Scene | Joint recall |",
            "| ---: | ---: |",
            *scene_rows,
            "",
            f"The weakest tracked scene is 25 at {float(recalls['25']):.4f} joint recall. Use it as a",
            "failure-analysis slice, not as a new tuning target for the already-consumed frozen",
            "evaluation split.",
            "",
            "## Regenerate or check",
            "",
            "```bash",
            "python -B scripts/build_failure_waterfall.py --output docs/failure-waterfall.md",
            "python -B scripts/build_failure_waterfall.py --check docs/failure-waterfall.md",
            "```",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()

    rendered = render_markdown(load_release_results(args.results))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        return 0
    if args.check is not None:
        if not args.check.is_file():
            raise SystemExit(f"missing generated waterfall: {args.check}")
        current = args.check.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit(
                "failure-waterfall output is stale; regenerate with "
                "scripts/build_failure_waterfall.py --output"
            )
        return 0

    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
