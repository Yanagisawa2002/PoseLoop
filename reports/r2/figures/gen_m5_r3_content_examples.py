#!/usr/bin/env python3
"""Render one M5-R3 missing-frame rescue and one regression."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from paper_plot_style import COLORS, save_figure
from r2_visual_common import (
    OUTPUT_DIR,
    REPO_ROOT,
    color_rgb,
    crop_image,
    index_rows,
    load_json,
    load_jsonl,
    load_mesh_m,
    projected_silhouette,
    read_mask,
    read_rgb,
    square_crop_box,
    write_json,
)


M5_ROOT = REPO_ROOT / "artifacts" / "r2" / "m5_r3"


def missing_relative(row: Mapping[str, Any]) -> float:
    baseline = float(row["baseline_natural_missing_loss"])
    proposed = float(row["proposed_natural_missing_loss"])
    return (baseline - proposed) / baseline


def choose_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if int(row["natural_missing_frame_count"]) >= 10
        and float(row["baseline_natural_missing_loss"]) <= 2.0
    ]
    if len(eligible) < 2:
        raise RuntimeError("Too few interpretable M5-R3 content candidates")
    rescue = max(eligible, key=lambda row: (missing_relative(row), str(row["track_id"])))
    regression = min(
        (row for row in eligible if row["track_id"] != rescue["track_id"]),
        key=lambda row: (missing_relative(row), str(row["track_id"])),
    )
    return [rescue, regression]


def choose_frame(row: Mapping[str, Any], rescue: bool) -> dict[str, Any]:
    missing = [frame for frame in row["frames"] if bool(frame["natural_input_missing"])]
    if not missing:
        raise RuntimeError(f"Track lacks missing frames: {row['track_id']}")
    key = lambda frame: (
        float(frame["baseline_normalized_pose_error"])
        - float(frame["proposed_normalized_pose_error"]),
        -int(frame["replay_frame_index"]),
    )
    return dict(max(missing, key=key) if rescue else min(missing, key=key))


def overlay_full(
    rgb: np.ndarray, gt_mask: np.ndarray, predicted_mask: np.ndarray
) -> np.ndarray:
    canvas = rgb.astype(np.float32)
    gt_color = color_rgb("green")
    pred_color = color_rgb("magenta")
    for mask, color in ((gt_mask, gt_color), (predicted_mask, pred_color)):
        selected = mask > 0
        canvas[selected] = 0.82 * canvas[selected] + 0.18 * color
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    for mask, color, halo in (
        (gt_mask, gt_color, color_rgb("black")),
        (predicted_mask, pred_color, color_rgb("white")),
    ):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in halo), 8, cv2.LINE_AA)
        cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in color), 5, cv2.LINE_AA)
    return canvas


def input_full(rgb: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    canvas = rgb.copy()
    contours, _ = cv2.findContours(
        visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        canvas,
        contours,
        -1,
        tuple(int(v) for v in color_rgb("sky")),
        5,
        cv2.LINE_AA,
    )
    return canvas


def missing_spans(frames: list[dict[str, Any]]) -> list[tuple[int, int]]:
    indices = [
        int(frame["replay_frame_index"])
        for frame in frames
        if bool(frame["natural_input_missing"])
    ]
    spans: list[tuple[int, int]] = []
    for index in indices:
        if not spans or index > spans[-1][1] + 1:
            spans.append((index, index))
        else:
            spans[-1] = (spans[-1][0], index)
    return spans


def main() -> None:
    result = load_json(M5_ROOT / "result.json")
    track_results = load_jsonl(M5_ROOT / "track_results.jsonl")
    track_manifest = index_rows(load_jsonl(M5_ROOT / "track_manifest.jsonl"), "track_id")
    selected_tracks = choose_examples(track_results)
    selected_frames = [
        choose_frame(row, rescue=index == 0) for index, row in enumerate(selected_tracks)
    ]
    selected_error_values = [
        float(frame[field])
        for track in selected_tracks
        for frame in track["frames"]
        for field in (
            "baseline_normalized_pose_error",
            "proposed_normalized_pose_error",
        )
    ]
    shared_y_min = max(0.0, min(selected_error_values) - 0.05)
    shared_y_max = max(selected_error_values) + 0.05
    selection = {
        "schema_version": 1,
        "protocol_id": result["protocol_id"],
        "selection_timing": "post_evaluation_visualization_only",
        "used_for_method_configuration_threshold_metric_or_gate": False,
        "rule": (
            "Among tracks with at least 10 natural missing frames and baseline missing-frame "
            "loss at most 2.0, choose the greatest and least missing-frame relative "
            "improvement. Within each, show the missing frame with the greatest rescue or "
            "greatest regression; stable ties use the earliest replay frame."
        ),
        "selected": [
            {
                "role": "rescue" if index == 0 else "regression",
                "track_id": str(track["track_id"]),
                "object_id": int(track["object_id"]),
                "natural_missing_frame_count": int(track["natural_missing_frame_count"]),
                "track_missing_relative_improvement": missing_relative(track),
                "replay_frame_index": int(frame["replay_frame_index"]),
                "sample_id": str(frame["sample_id"]),
                "baseline_normalized_pose_error": float(
                    frame["baseline_normalized_pose_error"]
                ),
                "proposed_normalized_pose_error": float(
                    frame["proposed_normalized_pose_error"]
                ),
            }
            for index, (track, frame) in enumerate(zip(selected_tracks, selected_frames))
        ],
    }
    write_json(OUTPUT_DIR / "m5_r3_content_selection.json", selection)

    dataset_root = Path(
        os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")
    )
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.35))
    column_titles = [
        "Recorded frame: input unavailable",
        "Nearest baseline",
        "Static quotient medoid",
        "Track error through time",
    ]
    for index, title in enumerate(column_titles):
        fig.text(0.19 + index * 0.24, 0.985, title, ha="center", va="top", fontsize=10)

    for row_index, (track, selected_frame) in enumerate(
        zip(selected_tracks, selected_frames)
    ):
        track_id = str(track["track_id"])
        manifest = track_manifest[track_id]
        manifest_frame = next(
            frame
            for frame in manifest["frames"]
            if int(frame["replay_frame_index"])
            == int(selected_frame["replay_frame_index"])
        )
        scene_id = int(manifest_frame["scene_id"])
        image_id = int(manifest_frame["image_id"])
        gt_index = int(manifest_frame["gt_instance_index"])
        object_id = int(track["object_id"])
        scene = dataset_root / "val" / f"{scene_id:06d}"
        rgb = read_rgb(scene / "gray_photoneo" / f"{image_id:06d}.png")
        visible = read_mask(
            scene / "mask_visib_photoneo" / f"{image_id:06d}_{gt_index:06d}.png"
        )
        camera_rows = load_json(scene / "scene_camera_photoneo.json")
        camera = np.asarray(camera_rows[str(image_id)]["cam_K"], dtype=np.float64).reshape(
            3, 3
        )
        world_to_camera = np.asarray(
            manifest_frame["camera_world_to_camera_pose_m"], dtype=np.float64
        )
        gt_camera = world_to_camera @ np.asarray(
            selected_frame["gt_model_to_world_pose_m"], dtype=np.float64
        )
        baseline_camera = world_to_camera @ np.asarray(
            selected_frame["baseline_output_pose_m"], dtype=np.float64
        )
        proposed_camera = world_to_camera @ np.asarray(
            selected_frame["proposed_output_pose_m"], dtype=np.float64
        )
        vertices, faces = load_mesh_m(
            dataset_root / "models" / f"obj_{object_id:06d}.ply"
        )
        height, width = rgb.shape[:2]
        gt_mask = projected_silhouette(vertices, faces, gt_camera, camera, height, width)
        baseline_mask = projected_silhouette(
            vertices, faces, baseline_camera, camera, height, width
        )
        proposed_mask = projected_silhouette(
            vertices, faces, proposed_camera, camera, height, width
        )
        crop = square_crop_box(
            [visible, gt_mask, baseline_mask, proposed_mask], rgb.shape[:2], pad_fraction=0.32
        )
        input_view = crop_image(input_full(rgb, visible), crop)
        baseline_view = crop_image(overlay_full(rgb, gt_mask, baseline_mask), crop)
        proposed_view = crop_image(overlay_full(rgb, gt_mask, proposed_mask), crop)

        for column, image in enumerate((input_view, baseline_view, proposed_view)):
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
        mask_pct = 100 * float(manifest_frame["input_mask_area_fraction"])
        depth_pct = 100 * float(manifest_frame["valid_depth_ratio_inside_mask"])
        axes[row_index, 0].set_title(
            f"mask {mask_pct:.3f}% · valid depth {depth_pct:.1f}%", fontsize=9
        )
        baseline_error = float(selected_frame["baseline_normalized_pose_error"])
        proposed_error = float(selected_frame["proposed_normalized_pose_error"])
        axes[row_index, 1].set_title(f"normalized error {baseline_error:.3f}", fontsize=9)
        axes[row_index, 2].set_title(f"normalized error {proposed_error:.3f}", fontsize=9)

        frames = track["frames"]
        frame_ids = np.asarray([int(frame["replay_frame_index"]) for frame in frames])
        baseline_errors = np.asarray(
            [float(frame["baseline_normalized_pose_error"]) for frame in frames]
        )
        proposed_errors = np.asarray(
            [float(frame["proposed_normalized_pose_error"]) for frame in frames]
        )
        axis = axes[row_index, 3]
        for start, end in missing_spans(frames):
            axis.axvspan(
                start - 0.5,
                end + 0.5,
                color=COLORS["light_gray"],
                alpha=0.75,
                linewidth=0,
            )
        axis.plot(frame_ids, baseline_errors, color=COLORS["orange"], linewidth=1.7)
        axis.plot(frame_ids, proposed_errors, color=COLORS["blue"], linewidth=1.7)
        chosen_index = int(selected_frame["replay_frame_index"])
        axis.axvline(chosen_index, color=COLORS["vermillion"], linestyle="--", linewidth=1.1)
        axis.scatter(
            [chosen_index, chosen_index],
            [baseline_error, proposed_error],
            color=[COLORS["orange"], COLORS["blue"]],
            edgecolor=COLORS["black"],
            linewidth=0.5,
            zorder=5,
        )
        axis.set_xlim(frame_ids.min(), frame_ids.max())
        axis.set_ylim(shared_y_min, shared_y_max)
        axis.set_xlabel("Replay frame")
        axis.set_ylabel("Normalized pose error")
        role = "Rescue" if row_index == 0 else "Regression"
        track_delta = missing_relative(track)
        fig.text(
            0.008,
            0.725 - row_index * 0.445,
            f"{role}\nobj {object_id:02d}\nmissing {track_delta:+.1%}",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    legend = [
        Line2D([0], [0], color=COLORS["green"], linewidth=3, label="GT CAD silhouette"),
        Line2D([0], [0], color=COLORS["magenta"], linewidth=3, label="Estimated CAD silhouette"),
        Line2D([0], [0], color=COLORS["orange"], linewidth=2, label="Nearest baseline error"),
        Line2D([0], [0], color=COLORS["blue"], linewidth=2, label="Static medoid error"),
        Line2D([0], [0], color=COLORS["gray"], linewidth=7, alpha=0.45, label="Natural missing span"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle(
        "M5-R3 recorded Photoneo behavior at natural missing frames (post-evaluation examples)",
        y=1.025,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.095, 0.065, 1, 0.955), h_pad=2.1, w_pad=1.3)
    save_figure(fig, OUTPUT_DIR, "m5_r3_content_examples")
    plt.close(fig)


if __name__ == "__main__":
    main()
