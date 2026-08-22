#!/usr/bin/env python3
"""Render honest post-evaluation content evidence for M5-R6A RU-APC replay."""

from __future__ import annotations

import hashlib
import json
import os
import re
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


M5_ROOT = REPO_ROOT / "artifacts" / "r2" / "m5_r6a"
REPORT_PATH = REPO_ROOT / "reports" / "r2" / "m5_r6a_content_visualization.md"
SAMPLE_PATTERN = re.compile(
    r"^ruapc-test-s(?P<scene>\d+)-i(?P<image>\d+)-g(?P<gt>\d+)-o(?P<object>\d+)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measurement_first_tracks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["candidate_id"] == "measurement_first"]
    if len(selected) != 14 or len({row["track_id"] for row in selected}) != 14:
        raise RuntimeError("Expected exactly 14 M5-R6A measurement-first tracks")
    return selected


def choose_range_tracks(
    tracks: list[dict[str, Any]], global_improvement: float
) -> list[tuple[str, dict[str, Any]]]:
    weakest = min(tracks, key=lambda row: (float(row["relative_improvement"]), row["track_id"]))
    strongest = max(
        tracks, key=lambda row: (float(row["relative_improvement"]), row["track_id"])
    )
    typical = min(
        tracks,
        key=lambda row: (
            abs(float(row["relative_improvement"]) - global_improvement),
            str(row["track_id"]),
        ),
    )
    if len({row["track_id"] for row in (weakest, typical, strongest)}) != 3:
        raise RuntimeError("Weakest, typical, and strongest visualization tracks overlap")
    return [("weakest", weakest), ("typical", typical), ("strongest", strongest)]


def choose_rescue_frame(track: Mapping[str, Any]) -> dict[str, Any]:
    eligible = [
        frame
        for frame in track["frames"]
        if not bool(frame["natural_input_missing"])
        and bool(frame["proposed_correction_applied"])
        and frame["prediction_status"] == "success"
    ]
    if not eligible:
        raise RuntimeError(f"No available correction frame: {track['track_id']}")
    return dict(
        max(
            eligible,
            key=lambda frame: (
                float(frame["baseline_normalized_pose_error"])
                - float(frame["proposed_normalized_pose_error"]),
                -int(frame["replay_frame_index"]),
            ),
        )
    )


def choose_missing_storyboard(
    tracks: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        (track, frame)
        for track in tracks
        for frame in track["frames"]
        if bool(frame["natural_input_missing"])
    ]
    track, terminal = max(
        candidates,
        key=lambda item: (
            int(item[1]["proposed_gap_frames"]),
            float(item[1]["proposed_uncertainty"]),
            str(item[0]["track_id"]),
            -int(item[1]["replay_frame_index"]),
        ),
    )
    frames = track["frames"]
    terminal_index = int(terminal["replay_frame_index"])
    start = terminal_index
    while start > 0 and bool(frames[start - 1]["natural_input_missing"]):
        start -= 1
    end = terminal_index
    while end + 1 < len(frames) and bool(frames[end + 1]["natural_input_missing"]):
        end += 1
    if start == 0 or end + 1 >= len(frames):
        raise RuntimeError("Selected missing span lacks available boundary frames")
    maximum_gap = int(terminal["proposed_gap_frames"])
    target_gaps = sorted({1, min(6, maximum_gap), min(18, maximum_gap), maximum_gap})
    selected = [dict(frames[start - 1])]
    for gap in target_gaps:
        matches = [
            frame
            for frame in frames[start : end + 1]
            if int(frame["proposed_gap_frames"]) == gap
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Missing gap {gap} is not unique in selected span")
        selected.append(dict(matches[0]))
    selected.append(dict(frames[end + 1]))
    for frame in selected:
        if bool(frame["natural_input_missing"]):
            np.testing.assert_allclose(
                np.asarray(frame["baseline_output_pose_m"], dtype=np.float64),
                np.asarray(frame["proposed_output_pose_m"], dtype=np.float64),
                rtol=0,
                atol=1e-12,
            )
            if not np.isclose(
                float(frame["baseline_normalized_pose_error"]),
                float(frame["proposed_normalized_pose_error"]),
                rtol=0,
                atol=1e-12,
            ):
                raise RuntimeError("Missing-frame exact fallback error mismatch")
    return track, selected


def missing_spans(frames: list[dict[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for frame in frames:
        if not bool(frame["natural_input_missing"]):
            continue
        index = int(frame["replay_frame_index"])
        if not spans or index > spans[-1][1] + 1:
            spans.append((index, index))
        else:
            spans[-1] = (spans[-1][0], index)
    return spans


def overlay_full(
    rgb: np.ndarray, gt_mask: np.ndarray, predicted_mask: np.ndarray
) -> np.ndarray:
    canvas = rgb.astype(np.float32)
    gt_color = color_rgb("green")
    predicted_color = color_rgb("magenta")
    for mask, color in ((gt_mask, gt_color), (predicted_mask, predicted_color)):
        selected = mask > 0
        canvas[selected] = 0.82 * canvas[selected] + 0.18 * color
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    for mask, color, halo in (
        (gt_mask, gt_color, color_rgb("black")),
        (predicted_mask, predicted_color, color_rgb("white")),
    ):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in halo), 7, cv2.LINE_AA)
        cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in color), 4, cv2.LINE_AA)
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
        4,
        cv2.LINE_AA,
    )
    return canvas


class ReplaySource:
    def __init__(
        self,
        dataset_root: Path,
        inference_by_sample: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.dataset_root = dataset_root
        self.inference_by_sample = inference_by_sample
        self.camera_cache: dict[int, dict[str, Any]] = {}
        self.mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def paths_and_camera(
        self, frame: Mapping[str, Any]
    ) -> tuple[Path, Path, Path, np.ndarray, int, int, int]:
        sample_id = str(frame["sample_id"])
        match = SAMPLE_PATTERN.match(sample_id)
        if match is None:
            raise ValueError(f"Unexpected RU-APC sample ID: {sample_id}")
        scene_id = int(match.group("scene"))
        image_id = int(match.group("image"))
        gt_index = int(match.group("gt"))
        object_id = int(match.group("object"))
        inference = self.inference_by_sample.get(sample_id)
        if inference is not None:
            camera = np.asarray(
                inference["camera_intrinsics_row_major"], dtype=np.float64
            ).reshape(3, 3)
            return (
                Path(inference["rgb_path"]),
                Path(inference["mask_path"]),
                Path(inference["model_path"]),
                camera,
                scene_id,
                image_id,
                gt_index,
            )
        scene = self.dataset_root / "test" / f"{scene_id:06d}"
        if scene_id not in self.camera_cache:
            self.camera_cache[scene_id] = load_json(scene / "scene_camera.json")
        camera = np.asarray(
            self.camera_cache[scene_id][str(image_id)]["cam_K"], dtype=np.float64
        ).reshape(3, 3)
        return (
            scene / "rgb" / f"{image_id:06d}.png",
            scene / "mask_visib" / f"{image_id:06d}_{gt_index:06d}.png",
            self.dataset_root / "models" / f"obj_{object_id:06d}.ply",
            camera,
            scene_id,
            image_id,
            gt_index,
        )

    def render_masks(
        self, frame: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rgb_path, mask_path, model_path, camera, _, _, _ = self.paths_and_camera(frame)
        rgb = read_rgb(rgb_path)
        visible = read_mask(mask_path)
        object_id = int(SAMPLE_PATTERN.match(str(frame["sample_id"])).group("object"))
        if object_id not in self.mesh_cache:
            self.mesh_cache[object_id] = load_mesh_m(model_path)
        vertices, faces = self.mesh_cache[object_id]
        height, width = rgb.shape[:2]
        gt_mask = projected_silhouette(
            vertices,
            faces,
            np.asarray(frame["gt_model_to_camera_pose_m"], dtype=np.float64),
            camera,
            height,
            width,
        )
        baseline_mask = projected_silhouette(
            vertices,
            faces,
            np.asarray(frame["baseline_output_pose_m"], dtype=np.float64),
            camera,
            height,
            width,
        )
        proposed_mask = projected_silhouette(
            vertices,
            faces,
            np.asarray(frame["proposed_output_pose_m"], dtype=np.float64),
            camera,
            height,
            width,
        )
        return rgb, visible, gt_mask, baseline_mask, proposed_mask


def render_range_figure(
    source: ReplaySource,
    selected: list[tuple[str, dict[str, Any]]],
    selected_frames: list[dict[str, Any]],
) -> None:
    errors = [
        float(frame[field])
        for _, track in selected
        for frame in track["frames"]
        for field in ("baseline_normalized_pose_error", "proposed_normalized_pose_error")
    ]
    shared_y_max = max(errors) * 1.04
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.15))
    titles = ["Recorded RGB-D support", "Nearest baseline", "Measurement-first", "Full 128-frame replay"]
    for column, title in enumerate(titles):
        fig.text(0.19 + column * 0.24, 0.987, title, ha="center", va="top", fontsize=10)

    for row_index, ((role, track), frame) in enumerate(zip(selected, selected_frames)):
        rgb, visible, gt_mask, baseline_mask, proposed_mask = source.render_masks(frame)
        crop = square_crop_box(
            [visible, gt_mask, baseline_mask, proposed_mask], rgb.shape[:2], pad_fraction=0.34
        )
        views = (
            crop_image(input_full(rgb, visible), crop),
            crop_image(overlay_full(rgb, gt_mask, baseline_mask), crop),
            crop_image(overlay_full(rgb, gt_mask, proposed_mask), crop),
        )
        for column, image in enumerate(views):
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
        mask_percent = 100 * float(frame["input_mask_area_fraction"])
        depth_percent = 100 * float(frame["valid_depth_ratio_inside_mask"])
        axes[row_index, 0].set_title(
            f"mask {mask_percent:.2f}% · valid depth {depth_percent:.1f}%", fontsize=8.5
        )
        baseline_error = float(frame["baseline_normalized_pose_error"])
        proposed_error = float(frame["proposed_normalized_pose_error"])
        axes[row_index, 1].set_title(f"error {baseline_error:.2f}", fontsize=9)
        axes[row_index, 2].set_title(
            f"error {proposed_error:.2f} · frame rescue {baseline_error - proposed_error:+.2f}",
            fontsize=8.5,
        )

        frames = track["frames"]
        frame_ids = np.asarray([int(item["replay_frame_index"]) for item in frames])
        baseline = np.asarray(
            [float(item["baseline_normalized_pose_error"]) for item in frames]
        )
        proposed = np.asarray(
            [float(item["proposed_normalized_pose_error"]) for item in frames]
        )
        axis = axes[row_index, 3]
        for start, end in missing_spans(frames):
            axis.axvspan(start - 0.5, end + 0.5, color=COLORS["light_gray"], alpha=0.7, linewidth=0)
        axis.plot(frame_ids, baseline, color=COLORS["orange"], linewidth=1.5)
        axis.plot(frame_ids, proposed, color=COLORS["blue"], linewidth=1.5)
        chosen = int(frame["replay_frame_index"])
        axis.axvline(chosen, color=COLORS["vermillion"], linestyle="--", linewidth=1.0)
        axis.scatter(
            [chosen, chosen],
            [baseline_error, proposed_error],
            color=[COLORS["orange"], COLORS["blue"]],
            edgecolor=COLORS["black"],
            linewidth=0.45,
            zorder=5,
        )
        axis.set_xlim(0, 127)
        axis.set_ylim(0, shared_y_max)
        axis.set_xlabel("Replay frame")
        axis.set_ylabel("Normalized pose error")
        fig.text(
            0.008,
            0.80 - row_index * 0.304,
            f"{role.capitalize()}\nobj {int(track['object_id']):02d}\ntrack {float(track['relative_improvement']):+.1%}",
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="bold",
        )

    legend = [
        Line2D([0], [0], color=COLORS["green"], linewidth=3, label="GT CAD silhouette"),
        Line2D([0], [0], color=COLORS["magenta"], linewidth=3, label="Estimated CAD silhouette"),
        Line2D([0], [0], color=COLORS["orange"], linewidth=2, label="Nearest error"),
        Line2D([0], [0], color=COLORS["blue"], linewidth=2, label="Measurement-first error"),
        Line2D([0], [0], color=COLORS["gray"], linewidth=7, alpha=0.45, label="Natural missing span"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=8.2)
    fig.suptitle(
        "M5-R6A RU-APC recorded replay: weak, typical, and strong object outcomes",
        y=1.015,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.09, 0.055, 1, 0.96), h_pad=2.0, w_pad=1.15)
    save_figure(fig, OUTPUT_DIR, "m5_r6a_content_examples")
    plt.close(fig)


def render_missing_storyboard(
    source: ReplaySource,
    track: Mapping[str, Any],
    frames: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 6.6))
    for axis, frame in zip(axes.flat, frames):
        rgb, visible, gt_mask, baseline_mask, proposed_mask = source.render_masks(frame)
        if bool(frame["natural_input_missing"]):
            np.testing.assert_array_equal(baseline_mask, proposed_mask)
        crop = square_crop_box(
            [visible, gt_mask, baseline_mask], rgb.shape[:2], pad_fraction=0.4
        )
        view = crop_image(overlay_full(rgb, gt_mask, proposed_mask), crop)
        axis.imshow(view)
        axis.axis("off")
        if bool(frame["natural_input_missing"]):
            title = f"gap {int(frame['proposed_gap_frames'])}: confidence {float(frame['proposed_confidence']):.3f}"
            subtitle = f"uncertainty {float(frame['proposed_uncertainty']):.1f} · exact nearest fallback"
        elif int(frame["replay_frame_index"]) < int(frames[1]["replay_frame_index"]):
            title = f"last available: confidence {float(frame['proposed_confidence']):.3f}"
            subtitle = f"uncertainty {float(frame['proposed_uncertainty']):.1f}"
        else:
            title = f"reacquired: confidence {float(frame['proposed_confidence']):.3f}"
            subtitle = f"uncertainty {float(frame['proposed_uncertainty']):.1f}"
        axis.set_title(f"{title}\n{subtitle}", fontsize=8.7)
    fig.legend(
        handles=[
            Line2D([0], [0], color=COLORS["green"], linewidth=3, label="GT CAD silhouette"),
            Line2D([0], [0], color=COLORS["magenta"], linewidth=3, label="Held/reacquired output"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"M5-R6A natural support loss: object {int(track['object_id']):02d}, 36-frame gap",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.045,
        "During missing input, nearest and measurement-first poses are exactly identical; only confidence and uncertainty change.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95), h_pad=2.0, w_pad=1.0)
    save_figure(fig, OUTPUT_DIR, "m5_r6a_missing_confidence")
    plt.close(fig)


def resize_square(image: np.ndarray, side: int = 320) -> np.ndarray:
    return cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)


def draw_video_timeline(
    track: Mapping[str, Any], current_index: int, width: int, height: int
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 55, width - 18, 12, height - 20
    frames = track["frames"]
    baseline = np.asarray(
        [float(frame["baseline_normalized_pose_error"]) for frame in frames]
    )
    proposed = np.asarray(
        [float(frame["proposed_normalized_pose_error"]) for frame in frames]
    )
    maximum = max(float(np.max(baseline)), float(np.max(proposed)), 1e-6)

    def point(index: int, value: float) -> tuple[int, int]:
        x = int(round(left + index * (right - left) / 127.0))
        y = int(round(bottom - value * (bottom - top) / maximum))
        return x, y

    for start, end in missing_spans(frames):
        x0 = point(start, 0)[0]
        x1 = point(end, 0)[0]
        cv2.rectangle(canvas, (x0, top), (x1, bottom), (222, 222, 222), -1)
    cv2.line(canvas, (left, bottom), (right, bottom), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.polylines(
        canvas,
        [np.asarray([point(i, value) for i, value in enumerate(baseline)], dtype=np.int32)],
        False,
        tuple(int(value) for value in color_rgb("orange")),
        2,
        cv2.LINE_AA,
    )
    cv2.polylines(
        canvas,
        [np.asarray([point(i, value) for i, value in enumerate(proposed)], dtype=np.int32)],
        False,
        tuple(int(value) for value in color_rgb("blue")),
        2,
        cv2.LINE_AA,
    )
    current_x = point(current_index, 0)[0]
    cv2.line(canvas, (current_x, top), (current_x, bottom), tuple(int(value) for value in color_rgb("vermillion")), 2)
    cv2.putText(canvas, "normalized pose error", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, "frame 0", (left, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(canvas, "127", (right - 22, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55, 55, 55), 1, cv2.LINE_AA)
    return canvas


def render_video(
    source: ReplaySource, track: Mapping[str, Any], output_path: Path
) -> None:
    width, height = 1080, 458
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")
    try:
        for frame in track["frames"]:
            rgb, visible, gt_mask, baseline_mask, proposed_mask = source.render_masks(frame)
            crop = square_crop_box(
                [visible, gt_mask, baseline_mask, proposed_mask],
                rgb.shape[:2],
                pad_fraction=0.36,
            )
            panels = [
                resize_square(crop_image(input_full(rgb, visible), crop)),
                resize_square(crop_image(overlay_full(rgb, gt_mask, baseline_mask), crop)),
                resize_square(crop_image(overlay_full(rgb, gt_mask, proposed_mask), crop)),
            ]
            canvas = np.full((height, width, 3), 250, dtype=np.uint8)
            x_positions = (30, 380, 730)
            for x, panel in zip(x_positions, panels):
                canvas[58:378, x : x + 320] = panel
            labels = ("Recorded support", "Nearest baseline", "Measurement-first")
            for x, label in zip(x_positions, labels):
                cv2.putText(canvas, label, (x, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
            status = "MISSING" if bool(frame["natural_input_missing"]) else "AVAILABLE"
            header = (
                f"RU-APC obj {int(track['object_id']):02d} | frame {int(frame['replay_frame_index']):03d}/127 | "
                f"{status} | confidence {float(frame['proposed_confidence']):.3f} | uncertainty {float(frame['proposed_uncertainty']):.2f}"
            )
            cv2.putText(canvas, header, (30, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
            baseline_error = float(frame["baseline_normalized_pose_error"])
            proposed_error = float(frame["proposed_normalized_pose_error"])
            cv2.putText(canvas, f"error {baseline_error:.2f}", (380, 401), cv2.FONT_HERSHEY_SIMPLEX, 0.52, tuple(int(value) for value in color_rgb("orange")), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"error {proposed_error:.2f}", (730, 401), cv2.FONT_HERSHEY_SIMPLEX, 0.52, tuple(int(value) for value in color_rgb("blue")), 1, cv2.LINE_AA)
            timeline = draw_video_timeline(track, int(frame["replay_frame_index"]), 1020, 48)
            canvas[410:458, 30:1050] = timeline
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(output_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    readable, _ = capture.read()
    capture.release()
    if not readable or frame_count != 128:
        raise RuntimeError(f"Generated replay video is invalid: frames={frame_count}")


def render_report(selection: Mapping[str, Any]) -> str:
    rows = selection["range_examples"]
    missing = selection["missing_confidence_storyboard"]
    lines = [
        "# PoseLoop M5-R6A content visualization",
        "",
        "Status: **RU-APC DEVELOPMENT REPLAY VISUALIZED**",
        "",
        "These figures and the replay video show actual recorded RU-APC RGB-D frames with official CAD silhouettes. Green is GT; magenta is the estimator output. The examples were selected only after the development result was frozen and were not used to alter any method, threshold, metric, or gate.",
        "",
        "| Role | Object | Track improvement | Replay frame | Nearest error | Measurement-first error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | {row['object_id']} | {row['track_relative_improvement']:+.2%} | "
            f"{row['replay_frame_index']} | {row['baseline_normalized_pose_error']:.3f} | "
            f"{row['proposed_normalized_pose_error']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The missing-support storyboard uses object {missing['object_id']} and its longest/highest-uncertainty natural gap ({missing['maximum_gap_frames']} frames). On every missing frame, the nearest and measurement-first pose matrices are exactly equal. Confidence falls from {missing['last_available_confidence']:.3f} to {missing['terminal_missing_confidence']:.3f}, while uncertainty rises from {missing['last_available_uncertainty']:.2f} to {missing['terminal_missing_uncertainty']:.2f}; the first available frame after the gap reacquires a measurement.",
            "",
            "This is recorded cluttered-shelf development replay with nominal timing, oracle target association, and GT visible masks. It is not a simulator, hardware deployment, deployable association result, or unseen-sensor validation. The separate YCB-V sealed protocol stopped at its frozen input-feasibility gate before inference.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    dataset_root = Path(os.environ.get("POSELOOP_RUAPC_ROOT", "/home/cgliu/datasets/ruapc"))
    result = load_json(M5_ROOT / "development_result.json")
    tracks = measurement_first_tracks(
        load_jsonl(M5_ROOT / "development_track_results.jsonl")
    )
    inference = index_rows(load_jsonl(M5_ROOT / "inference_manifest.jsonl"), "sample_id")
    global_improvement = float(
        result["development_gate"]["observed"][
            "cross_fitted_macro_object_relative_improvement"
        ]
    )
    selected = choose_range_tracks(tracks, global_improvement)
    selected_frames = [choose_rescue_frame(track) for _, track in selected]
    missing_track, missing_frames = choose_missing_storyboard(tracks)
    source = ReplaySource(dataset_root, inference)

    render_range_figure(source, selected, selected_frames)
    render_missing_storyboard(source, missing_track, missing_frames)
    video_path = OUTPUT_DIR / "m5_r6a_typical_replay.mp4"
    render_video(source, selected[1][1], video_path)

    selection: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": result["protocol_id"],
        "stage_id": result["stage_id"],
        "status": "POST_EVALUATION_VISUALIZATION_ONLY",
        "selection_timing": "after_frozen_development_evaluation",
        "used_for_method_configuration_threshold_metric_or_gate": False,
        "global_cross_fitted_macro_object_relative_improvement": global_improvement,
        "range_rule": (
            "Among the 14 fixed measurement-first object tracks, select the lowest "
            "relative improvement, the relative improvement closest to the global macro "
            "improvement, and the highest relative improvement. Within each selected "
            "track, show the available corrected frame with largest baseline-minus-proposed "
            "normalized error; stable ties use the earliest replay frame."
        ),
        "range_examples": [],
        "missing_rule": (
            "Select the natural-missing frame with greatest causal gap length, then highest "
            "uncertainty, then stable track/frame identifiers. Show its immediately preceding "
            "available frame, gap lengths 1, 6, 18, and maximum, and the next available frame."
        ),
        "missing_confidence_storyboard": {
            "track_id": missing_track["track_id"],
            "object_id": int(missing_track["object_id"]),
            "maximum_gap_frames": max(
                int(frame["proposed_gap_frames"]) for frame in missing_frames
            ),
            "last_available_confidence": float(missing_frames[0]["proposed_confidence"]),
            "last_available_uncertainty": float(missing_frames[0]["proposed_uncertainty"]),
            "terminal_missing_confidence": float(missing_frames[-2]["proposed_confidence"]),
            "terminal_missing_uncertainty": float(missing_frames[-2]["proposed_uncertainty"]),
            "post_gap_confidence": float(missing_frames[-1]["proposed_confidence"]),
            "post_gap_uncertainty": float(missing_frames[-1]["proposed_uncertainty"]),
            "frames": [
                {
                    "sample_id": frame["sample_id"],
                    "replay_frame_index": int(frame["replay_frame_index"]),
                    "natural_input_missing": bool(frame["natural_input_missing"]),
                    "gap_frames": int(frame["proposed_gap_frames"]),
                    "confidence": float(frame["proposed_confidence"]),
                    "uncertainty": float(frame["proposed_uncertainty"]),
                    "baseline_equals_proposed_pose": bool(
                        np.allclose(
                            np.asarray(frame["baseline_output_pose_m"], dtype=np.float64),
                            np.asarray(frame["proposed_output_pose_m"], dtype=np.float64),
                            rtol=0,
                            atol=1e-12,
                        )
                    ),
                }
                for frame in missing_frames
            ],
        },
        "typical_full_replay_video": {
            "track_id": selected[1][1]["track_id"],
            "object_id": int(selected[1][1]["object_id"]),
            "frame_count": 128,
            "nominal_source_rate_hz": 30.0,
            "render_rate_fps": 24.0,
            "path": str(video_path),
            "sha256": sha256_file(video_path),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "contract": M5_ROOT / "contract.json",
                "development_result": M5_ROOT / "development_result.json",
                "development_track_results": M5_ROOT / "development_track_results.jsonl",
                "inference_manifest": M5_ROOT / "inference_manifest.jsonl",
            }.items()
        },
        "outputs": {},
        "claim_boundary": (
            "Recorded RU-APC cluttered-shelf development replay with nominal timing, oracle "
            "target association, and GT visible masks; not simulation, hardware deployment, "
            "deployable association, or unseen-sensor validation."
        ),
    }
    for (role, track), frame in zip(selected, selected_frames):
        selection["range_examples"].append(
            {
                "role": role,
                "track_id": track["track_id"],
                "object_id": int(track["object_id"]),
                "track_relative_improvement": float(track["relative_improvement"]),
                "replay_frame_index": int(frame["replay_frame_index"]),
                "sample_id": frame["sample_id"],
                "baseline_normalized_pose_error": float(
                    frame["baseline_normalized_pose_error"]
                ),
                "proposed_normalized_pose_error": float(
                    frame["proposed_normalized_pose_error"]
                ),
                "frame_error_reduction": float(
                    frame["baseline_normalized_pose_error"]
                    - frame["proposed_normalized_pose_error"]
                ),
                "confidence": float(frame["proposed_confidence"]),
                "uncertainty": float(frame["proposed_uncertainty"]),
            }
        )
    for name in (
        "m5_r6a_content_examples.pdf",
        "m5_r6a_content_examples.png",
        "m5_r6a_missing_confidence.pdf",
        "m5_r6a_missing_confidence.png",
        "m5_r6a_typical_replay.mp4",
    ):
        path = OUTPUT_DIR / name
        selection["outputs"][name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(OUTPUT_DIR / "m5_r6a_content_selection.json", selection)
    REPORT_PATH.write_text(render_report(selection), encoding="utf-8", newline="\n")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
