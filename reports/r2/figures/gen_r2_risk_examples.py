#!/usr/bin/env python3
"""Render low- and high-risk real Photoneo outputs for M6-R2."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from paper_plot_style import COLORS, save_figure
from r2_visual_common import (
    OUTPUT_DIR,
    SEALED_ROOT,
    index_rows,
    load_json,
    load_jsonl,
    load_mesh_m,
    overlay_pose_silhouettes,
    projected_silhouette,
    read_mask,
    read_rgb,
    write_json,
)


def distinct_objects(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    chosen = []
    objects: set[int] = set()
    for row in rows:
        object_id = int(row["object_id"])
        if object_id in objects:
            continue
        chosen.append(row)
        objects.add(object_id)
        if len(chosen) == count:
            return chosen
    if len(chosen) != count:
        raise RuntimeError("Insufficient distinct-object M6 examples")
    return chosen


def main() -> None:
    result = load_json(SEALED_ROOT / "sealed_result.json")
    outcomes = load_jsonl(SEALED_ROOT / "sealed_evaluated_outcomes.jsonl")
    decisions = index_rows(
        load_jsonl(SEALED_ROOT / "label_blind_decisions.jsonl"), "group_id"
    )
    groups = index_rows(load_jsonl(SEALED_ROOT / "groups_evaluator.jsonl"), "group_id")
    targets = index_rows(load_jsonl(SEALED_ROOT / "target_manifest.jsonl"), "sample_id")

    successes = sorted(
        (row for row in outcomes if bool(row["joint_success"])),
        key=lambda row: (row["learned_failure_risk"], row["object_id"], row["group_id"]),
    )
    failures = sorted(
        (row for row in outcomes if not bool(row["joint_success"])),
        key=lambda row: (-row["learned_failure_risk"], row["object_id"], row["group_id"]),
    )
    selected = distinct_objects(successes, 2) + distinct_objects(failures, 2)
    write_json(
        OUTPUT_DIR / "r2_risk_example_selection.json",
        {
            "schema_version": 1,
            "protocol_id": result["protocol_id"],
            "selection_timing": "post_evaluation_visualization_only",
            "used_for_model_policy_threshold_or_gate": False,
            "rule": (
                "Select the two lowest learned-risk successes and two highest learned-risk "
                "failures, requiring distinct objects within each category; stable ties use "
                "object ID then group ID."
            ),
            "selected": [
                {
                    "group_id": row["group_id"],
                    "object_id": row["object_id"],
                    "learned_failure_risk": row["learned_failure_risk"],
                    "joint_success": row["joint_success"],
                    "normalized_mssd": row["normalized_mssd"],
                    "mspd_px": row["mspd_px"],
                }
                for row in selected
            ],
        },
    )

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.75))
    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, row in enumerate(selected):
        group_id = str(row["group_id"])
        group = groups[group_id]
        decision = decisions[group_id]
        target_id = str(group["target_sample_id"])
        target = targets[target_id]
        mesh_path = str(target["model_path"])
        if mesh_path not in mesh_cache:
            mesh_cache[mesh_path] = load_mesh_m(mesh_path)
        vertices, faces = mesh_cache[mesh_path]
        rgb = read_rgb(target["rgb_path"])
        visible = read_mask(target["mask_path"])
        camera = np.asarray(target["camera_intrinsics_row_major"], dtype=np.float64)
        height, width = rgb.shape[:2]
        gt = projected_silhouette(
            vertices,
            faces,
            np.asarray(group["views"][0]["gt_model_to_camera_pose_m"], dtype=np.float64),
            camera,
            height,
            width,
        )
        predicted = projected_silhouette(
            vertices,
            faces,
            np.asarray(decision["final"]["predicted_model_to_target_camera_pose_m"], dtype=np.float64),
            camera,
            height,
            width,
        )
        overlay, _ = overlay_pose_silhouettes(rgb, gt, predicted, visible)
        axes[index].imshow(overlay)
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        for spine in axes[index].spines.values():
            spine.set_visible(False)
        status = "success" if row["joint_success"] else "failure"
        axes[index].set_xlabel(
            f"obj {row['object_id']} | risk {row['learned_failure_risk']:.3f}\n"
            f"{status} | MSSD/d {row['normalized_mssd']:.3f}",
            color=COLORS["green"] if row["joint_success"] else COLORS["vermillion"],
            labelpad=3,
        )
    fig.text(0.27, 0.98, "Lowest-risk successful outputs", ha="center", va="top")
    fig.text(0.75, 0.98, "Highest-risk failed outputs", ha="center", va="top")
    legend = [
        Line2D([0], [0], color=COLORS["green"], linewidth=3, label="GT CAD silhouette"),
        Line2D([0], [0], color=COLORS["magenta"], linewidth=3, label="Predicted CAD silhouette"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.91), w_pad=0.65)
    save_figure(fig, OUTPUT_DIR, "r2_risk_examples")
    plt.close(fig)


if __name__ == "__main__":
    main()
