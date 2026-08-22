"""Factories for committed/temporary PREP-only producer result fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pose_accuracy_recovery_prep.c_results import C_RESULTS_SCHEMA
from pose_accuracy_recovery_prep.core import canonical_sha256, sha256_file, write_jsonl

INITIAL_POSE = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
]
FINAL_POSE = [
    [1.0, 0.0, 0.0, 0.001],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
]
VISUAL_ROLES = (
    "rgb",
    "input_mask",
    "initial_pose_overlay",
    "top_k_overlay",
    "final_pose_overlay",
)


def _translated_pose(x: float, y: float = 0.0) -> list[list[float]]:
    pose = [list(row) for row in INITIAL_POSE]
    pose[0][3] = x
    pose[1][3] = y
    return pose


def write_valid_c_result_fixture(
    handoff: Mapping[str, Any], result_root: Path
) -> Path:
    """Write schema-complete fake producer structure, never accuracy evidence."""
    result_root.mkdir(parents=True, exist_ok=True)
    runtime = handoff["producer_runtime_lock"]
    rows: list[dict[str, Any]] = []
    for item in handoff["items"]:
        members = []
        for role in VISUAL_ROLES:
            relative = f"visualizations/{item['item_id']}/{role}.json"
            path = result_root / Path(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "fixture_only": True,
                        "item_id": item["item_id"],
                        "role": role,
                        "accuracy_claim_permitted": False,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            members.append(
                {
                    "role": role,
                    "relative_path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        top_k = [
            {
                "rank": rank,
                "candidate_id": f"fixture-{item['item_id']}-h{rank - 1:03d}",
                "score": 1.0 - rank * 0.1,
                "model_to_camera_pose_m": _translated_pose(
                    0.0, (rank - 1) * 0.001
                ),
            }
            for rank in range(1, 6)
        ]
        trace = [
            {
                "iteration": step,
                "model_to_camera_pose_m": _translated_pose(step * 0.0002),
                "objective": 1.0 - step * 0.1,
            }
            for step in range(6)
        ]
        rows.append(
            {
                "schema_version": C_RESULTS_SCHEMA,
                "item_id": item["item_id"],
                "sample_key": dict(item["sample_key"]),
                "mask_variant_id": item["mask_variant_id"],
                "manifest_lock_sha256": handoff["manifest_lock_sha256"],
                "input_sha256": {
                    name: item["inputs"][name]["sha256"]
                    for name in ("rgb", "depth", "mask", "camera", "cad")
                },
                "implementation_commit": runtime["implementation_commit"],
                "implementation_sha256": runtime["implementation_sha256"],
                "model_sha256": runtime["model_sha256"],
                "checkpoint_sha256": canonical_sha256(
                    runtime["checkpoint_sha256"]
                ),
                "initial_model_to_camera_pose_m": INITIAL_POSE,
                "final_model_to_camera_pose_m": FINAL_POSE,
                "top_k": top_k,
                "refiner_trace": trace,
                "status": "success",
                "attempt": 1,
                "latency_ms": 12.5,
                "failure": False,
                "oom": False,
                "failure_reason": None,
                "access_counters": {
                    "label_access_count_on_gpu_c": 0,
                    "gt_path_open_count_on_gpu_c": 0,
                    "evaluator_path_open_count_on_gpu_c": 0,
                    "scorer_path_open_count_on_gpu_c": 0,
                },
                "visualization_inventory": members,
            }
        )
    results = result_root / "results.jsonl"
    write_jsonl(results, rows)
    return results
