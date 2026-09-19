"""Compare frozen A9 per-item semantics and timing without loading labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def predictions(root: Path) -> list[dict]:
    return [row for line in (root / "predictions.jsonl").read_text().splitlines()
            if (row := json.loads(line)).get("record_type") == "prediction"]


def summarize(root: Path) -> dict:
    rows = predictions(root)
    seconds = np.asarray([r["registration_seconds"] for r in rows], dtype=float)
    return {
        "root": str(root), "completion": json.loads((root / "completion-receipt.json").read_text()),
        "latency_seconds": {"median": float(np.median(seconds)), "p95": float(np.quantile(seconds, .95)),
                            "mean": float(seconds.mean()), "max": float(seconds.max())},
        "peak_allocated_bytes": max(r["cuda_peak_allocated_bytes"] for r in rows),
    }


def compare(baseline: Path, candidate: Path, *, safety: bool = False) -> dict:
    old = predictions(baseline)
    new = predictions(candidate)
    by_id = {r["item_id"]: r for r in old}
    ids = [r["item_id"] for r in new]
    if len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
        raise ValueError("Duplicate or unexpected candidate item IDs")
    if not safety and ids != [r["item_id"] for r in old]:
        raise ValueError("Full ordered population changed")
    pose_deltas = []; score_deltas = []; margin_deltas = []; records = []
    status_ok = True; shape_ok = True; finite = True; exact_pose = 0
    for row in new:
        ref = by_id[row["item_id"]]
        status_ok &= row["status"] == ref["status"] == "success" and row.get("candidate_count") == ref.get("candidate_count") == 252
        pose = np.asarray(row.get("predicted_model_to_camera_pose_m", []), dtype=float)
        original = np.asarray(ref["predicted_model_to_camera_pose_m"], dtype=float)
        shape_ok &= pose.shape == original.shape == (4, 4)
        if not shape_ok or not status_ok:
            raise ValueError("Failed registration, candidate-count change or pose-shape change")
        delta = np.abs(pose-original); pose_deltas.append(delta)
        score = abs(row["foundationpose_top_score"] - ref["foundationpose_top_score"])
        margin = abs(row["foundationpose_top_score_margin"] - ref["foundationpose_top_score_margin"])
        score_deltas.append(score); margin_deltas.append(margin)
        finite &= bool(np.isfinite(pose).all() and np.isfinite(score) and np.isfinite(margin))
        exact_pose += int(np.array_equal(pose, original))
        records.append({"item_id": row["item_id"], "pose_max_abs": float(delta.max()),
                        "rotation_element_max_abs": float(delta[:3, :3].max()),
                        "translation_max_abs_m": float(delta[:3, 3].max()),
                        "top_score_abs": score, "score_margin_abs": margin})
    deltas = np.asarray(pose_deltas)
    summary = {
        "item_count": len(new), "exact_pose_match_count": exact_pose, "non_exact_pose_count": len(new)-exact_pose,
        "max_pose_element_absolute_difference": float(deltas.max()),
        "median_pose_element_absolute_difference": float(np.median(deltas)),
        "max_rotation_element_absolute_difference": float(deltas[:, :3, :3].max()),
        "max_translation_absolute_difference_m": float(deltas[:, :3, 3].max()),
        "exact_top_score_match_count": sum(x == 0 for x in score_deltas), "max_top_score_absolute_difference": max(score_deltas),
        "exact_score_margin_match_count": sum(x == 0 for x in margin_deltas), "max_score_margin_absolute_difference": max(margin_deltas),
        "finite": finite, "status_and_252_candidates_unchanged": status_ok, "pose_shapes_4x4": shape_ok,
    }
    timing = summarize(candidate)
    # Predeclared safety screen, not an evaluation-equivalence tolerance.
    limits = {"rotation_element_max": .01, "translation_max_m": .001, "top_score_max": 1., "margin_max": 1., "peak_allocated_gib_max": 24.}
    safe = (len(new) == 5 and finite and summary["max_rotation_element_absolute_difference"] <= .01
            and summary["max_translation_absolute_difference_m"] <= .001
            and max(score_deltas) <= 1 and max(margin_deltas) <= 1
            and timing["peak_allocated_bytes"] < 24*1024**3) if safety else None
    return {"comparison": summary, "timing": timing, "per_item": records,
            "safety_screen_limits": limits if safety else None, "safety_pass": safe}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--safety", action="store_true")
    a = p.parse_args(); result = compare(a.baseline_root, a.candidate_root, safety=a.safety)
    with a.output.open("x", encoding="utf-8") as f: json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != "per_item"}, indent=2))
    return 0 if not a.safety or result["safety_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
