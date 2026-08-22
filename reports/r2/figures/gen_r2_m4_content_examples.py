#!/usr/bin/env python3
"""Render real Photoneo before/after examples for the frozen M4-R2 policy."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from paper_plot_style import COLORS, save_figure
from r2_visual_common import (
    OUTPUT_DIR,
    REPO_ROOT,
    SEALED_ROOT,
    acquisition_view,
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


SCRIPTS = REPO_ROOT / "scripts"
TOOLKIT = REPO_ROOT / "third_party" / "bop_toolkit"
for path in (SCRIPTS, TOOLKIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_m2  # noqa: E402
import fit_m4_voi  # noqa: E402
from evaluate_m1 import load_official_models  # noqa: E402


def prediction_index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return index_rows(
        (row for row in rows if row.get("record_type") == "r1_sealed_prediction"),
        "sample_id",
    )


def pair_selected_view(prepared: Mapping[str, Any], slot: int) -> dict[str, Any]:
    return max(
        (prepared["views"][0], prepared["views"][slot]),
        key=lambda view: (
            int(view["view"]["visible_mask_pixel_count"]),
            -int(view["view"]["acquisition_rank"]),
        ),
    )


def choose_distinct(
    rows: Sequence[dict[str, Any]], count: int, excluded: set[str]
) -> list[dict[str, Any]]:
    chosen = []
    objects: set[int] = set()
    for row in rows:
        if row["group_id"] in excluded or int(row["object_id"]) in objects:
            continue
        chosen.append(row)
        excluded.add(str(row["group_id"]))
        objects.add(int(row["object_id"]))
        if len(chosen) == count:
            return chosen
    for row in rows:
        if row["group_id"] in excluded:
            continue
        chosen.append(row)
        excluded.add(str(row["group_id"]))
        if len(chosen) == count:
            break
    return chosen


def main() -> None:
    result = load_json(SEALED_ROOT / "sealed_result.json")
    contract = load_json(SEALED_ROOT / "sealed_contract.json")
    target_manifest = index_rows(load_jsonl(SEALED_ROOT / "target_manifest.jsonl"), "sample_id")
    inference_manifest = index_rows(
        load_jsonl(SEALED_ROOT / "inference_manifest.jsonl"), "sample_id"
    )
    labels = index_rows(load_jsonl(SEALED_ROOT / "evaluator_labels.jsonl"), "sample_id")
    groups = load_jsonl(SEALED_ROOT / "groups_evaluator.jsonl")
    decisions = index_rows(
        load_jsonl(SEALED_ROOT / "label_blind_decisions.jsonl"), "group_id"
    )
    predictions = prediction_index(load_jsonl(SEALED_ROOT / "predictions.jsonl"))
    evaluator_manifest = {
        sample_id: {**row, **labels[sample_id]} for sample_id, row in target_manifest.items()
    }

    dataset_root = Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd"))
    model_params, model_info = load_official_models(dataset_root)
    prepared, _ = evaluate_m2.prepare_groups(
        groups,
        evaluator_manifest,
        predictions,
        predictions,
        model_params,
        model_info,
        float(result["gt_transform_audit_limits"]["normalized_mssd_max"]),
        float(result["gt_transform_audit_limits"]["mspd_px_max"]),
    )
    prepared_index = {str(row["group"]["group_id"]): row for row in prepared}

    comparisons = []
    for group_id, decision in decisions.items():
        active_slot = decision["final"]["m4_selected_slot"]
        if active_slot is None:
            continue
        fixed = fit_m4_voi._pair_outcome(prepared_index[group_id], 2)
        active = fit_m4_voi._pair_outcome(prepared_index[group_id], int(active_slot))
        comparisons.append(
            {
                "group_id": group_id,
                "object_id": int(decision["object_id"]),
                "active_slot": int(active_slot),
                "fixed_utility": float(fixed["actual_utility"]),
                "active_utility": float(active["actual_utility"]),
                "delta_utility": float(active["actual_utility"] - fixed["actual_utility"]),
                "fixed_success": bool(fixed["joint_success"]),
                "active_success": bool(active["joint_success"]),
            }
        )

    changed_comparisons = [row for row in comparisons if int(row["active_slot"]) != 2]

    rescues = sorted(
        (
            row
            for row in changed_comparisons
            if not row["fixed_success"] and row["active_success"]
        ),
        key=lambda row: (-row["delta_utility"], row["object_id"], row["group_id"]),
    )
    regressions = sorted(
        (
            row
            for row in changed_comparisons
            if row["fixed_success"] and not row["active_success"]
        ),
        key=lambda row: (row["delta_utility"], row["object_id"], row["group_id"]),
    )
    excluded: set[str] = set()
    selected = choose_distinct(rescues, 2, excluded)
    if regressions:
        selected += choose_distinct(regressions, 1, excluded)
    if len(selected) < 3:
        remainder = sorted(
            changed_comparisons,
            key=lambda row: (-abs(row["delta_utility"]), row["object_id"], row["group_id"]),
        )
        selected += choose_distinct(remainder, 3 - len(selected), excluded)
    if len(selected) != 3:
        raise RuntimeError("Could not select three deterministic M4 content examples")

    selection_payload = {
        "schema_version": 1,
        "protocol_id": contract["protocol_id"],
        "selection_timing": "post_evaluation_visualization_only",
        "used_for_model_policy_threshold_or_gate": False,
        "rule": (
            "Select the two largest fixed-slot-2 failure to R2 success rescues with "
            "distinct objects when available, then the largest R2 regression; fill any "
            "missing category by largest absolute utility change. Stable ties use object "
            "ID then group ID."
        ),
        "diagnostic_summary": {
            "continue_target_count": len(comparisons),
            "r2_deviated_from_fixed_slot_2_count": len(changed_comparisons),
            "micro_active_success_count": sum(row["active_success"] for row in comparisons),
            "micro_fixed_success_count": sum(row["fixed_success"] for row in comparisons),
            "rescue_count": sum(
                (not row["fixed_success"]) and row["active_success"]
                for row in comparisons
            ),
            "regression_count": sum(
                row["fixed_success"] and (not row["active_success"])
                for row in comparisons
            ),
            "utility_improved_count": sum(row["delta_utility"] > 0 for row in comparisons),
            "utility_regressed_count": sum(row["delta_utility"] < 0 for row in comparisons),
            "utility_tied_count": sum(row["delta_utility"] == 0 for row in comparisons),
            "micro_mean_utility_delta": float(
                np.mean([row["delta_utility"] for row in comparisons])
            ),
        },
        "selected": selected,
    }
    write_json(OUTPUT_DIR / "r2_m4_content_selection.json", selection_payload)

    fig, axes = plt.subplots(3, 4, figsize=(10.4, 7.7))
    column_labels = [
        "Fixed baseline: acquired slot 2",
        "Fixed baseline: pose on target",
        "R2: selected acquisition",
        "R2: final pose on target",
    ]
    for index, label in enumerate(column_labels):
        fig.text(0.145 + index * 0.237, 0.985, label, ha="center", va="top", fontsize=10)

    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row_index, comparison in enumerate(selected):
        group_id = str(comparison["group_id"])
        prepared_group = prepared_index[group_id]
        group = prepared_group["group"]
        decision = decisions[group_id]
        active_slot = int(comparison["active_slot"])
        target_id = str(group["target_sample_id"])
        target_info = target_manifest[target_id]
        fixed_view_info = group["views"][2]
        active_view_info = group["views"][active_slot]
        fixed_source = pair_selected_view(prepared_group, 2)
        active_source = pair_selected_view(prepared_group, active_slot)

        mesh_path = str(target_info["model_path"])
        if mesh_path not in mesh_cache:
            mesh_cache[mesh_path] = load_mesh_m(mesh_path)
        vertices, faces = mesh_cache[mesh_path]
        target_rgb = read_rgb(target_info["rgb_path"])
        target_visible = read_mask(target_info["mask_path"])
        camera = np.asarray(target_info["camera_intrinsics_row_major"], dtype=np.float64)
        height, width = target_rgb.shape[:2]
        gt_mask = projected_silhouette(
            vertices,
            faces,
            np.asarray(prepared_group["target_gt_pose_m"], dtype=np.float64),
            camera,
            height,
            width,
        )

        fixed_prediction_mask = projected_silhouette(
            vertices,
            faces,
            np.asarray(fixed_source["transformed_pose_m"], dtype=np.float64),
            camera,
            height,
            width,
        )
        active_prediction_mask = projected_silhouette(
            vertices,
            faces,
            np.asarray(active_source["transformed_pose_m"], dtype=np.float64),
            camera,
            height,
            width,
        )
        fixed_overlay, _ = overlay_pose_silhouettes(
            target_rgb, gt_mask, fixed_prediction_mask, target_visible
        )
        active_overlay, _ = overlay_pose_silhouettes(
            target_rgb, gt_mask, active_prediction_mask, target_visible
        )

        fixed_sample = inference_manifest[str(fixed_view_info["sample_id"])]
        active_sample = inference_manifest[str(active_view_info["sample_id"])]
        fixed_acquisition, _ = acquisition_view(
            read_rgb(fixed_sample["rgb_path"]), read_mask(fixed_sample["mask_path"])
        )
        active_acquisition, _ = acquisition_view(
            read_rgb(active_sample["rgb_path"]), read_mask(active_sample["mask_path"])
        )

        images = [fixed_acquisition, fixed_overlay, active_acquisition, active_overlay]
        for column, image in enumerate(images):
            axes[row_index, column].imshow(image)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values():
                spine.set_visible(False)

        fixed_pose_rank = int(fixed_source["view"]["acquisition_rank"])
        active_pose_rank = int(active_source["view"]["acquisition_rank"])
        axes[row_index, 0].set_xlabel("slot 2 observation", labelpad=2)
        axes[row_index, 1].set_xlabel(
            f"U={comparison['fixed_utility']:.2f}\n"
            f"{'success' if comparison['fixed_success'] else 'failure'} | pose rank {fixed_pose_rank}",
            color=COLORS["green"] if comparison["fixed_success"] else COLORS["vermillion"],
            labelpad=2,
            fontsize=8,
        )
        axes[row_index, 2].set_xlabel(f"slot {active_slot} observation", labelpad=2)
        axes[row_index, 3].set_xlabel(
            f"U={comparison['active_utility']:.2f}\n"
            f"{'success' if comparison['active_success'] else 'failure'} | pose rank {active_pose_rank}",
            color=COLORS["green"] if comparison["active_success"] else COLORS["vermillion"],
            labelpad=2,
            fontsize=8,
        )
        if not comparison["fixed_success"] and comparison["active_success"]:
            category = "rescue"
        elif comparison["fixed_success"] and not comparison["active_success"]:
            category = "regression"
        else:
            category = "largest change"
        axes[row_index, 0].text(
            -0.11,
            0.5,
            f"{category}\nobj {comparison['object_id']}\nΔU {comparison['delta_utility']:+.2f}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )

    legend = [
        Line2D([0], [0], color=COLORS["green"], linewidth=3, label="GT CAD silhouette"),
        Line2D([0], [0], color=COLORS["magenta"], linewidth=3, label="Predicted CAD silhouette"),
        Line2D([0], [0], color=COLORS["sky"], linewidth=3, label="Visible object mask"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.005))
    fig.tight_layout(rect=(0.08, 0.045, 1.0, 0.96), h_pad=1.25, w_pad=0.55)
    save_figure(fig, OUTPUT_DIR, "r2_m4_content_examples")
    plt.close(fig)


if __name__ == "__main__":
    main()
