#!/usr/bin/env python3
"""Generate quantitative diagnostics for the failed M5-R3 sealed gate."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_figure
from r2_visual_common import OUTPUT_DIR, REPO_ROOT, load_json


M5_ROOT = REPO_ROOT / "artifacts" / "r2" / "m5_r3"


def main() -> None:
    result = load_json(M5_ROOT / "result.json")
    gate = result["sealed_gate"]
    observed = gate["observed"]
    thresholds = gate["thresholds"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.65))

    values = 100 * np.asarray(
        [
            observed["macro_object_relative_improvement"],
            observed["natural_missing_macro_object_relative_improvement"],
            observed["one_sided_90pct_lower_relative_improvement"],
        ],
        dtype=np.float64,
    )
    required = 100 * np.asarray(
        [
            thresholds["macro_object_relative_trajectory_loss_improvement_min"],
            thresholds["natural_missing_frame_macro_object_relative_improvement_min"],
            thresholds["one_sided_hierarchical_bootstrap_90pct_lower_improvement_min"],
        ],
        dtype=np.float64,
    )
    labels = ["All frames", "Natural\nmissing", "Bootstrap\n10th pct."]
    x = np.arange(len(values))
    bars = axes[0].bar(
        x,
        values,
        color=[COLORS["orange"], COLORS["vermillion"], COLORS["magenta"]],
        width=0.62,
    )
    axes[0].axhline(0.0, color=COLORS["black"], linewidth=0.8)
    for index, (bar, value, threshold) in enumerate(zip(bars, values, required)):
        axes[0].plot(
            [index - 0.39, index + 0.39],
            [threshold, threshold],
            color=COLORS["black"],
            linewidth=2.0,
        )
        label_y = value - 0.55 if value >= 0 else value - 0.45
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:+.1f} pp",
            ha="center",
            va="top",
            fontsize=9,
        )
        axes[0].text(
            index,
            threshold + 0.48,
            f"gate {threshold:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["black"],
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Relative loss improvement (pp)")
    axes[0].set_ylim(min(-9.5, values.min() - 2), max(8.0, values.max() + 2))
    axes[0].set_title("All three advancement conditions failed")

    per_object = result["per_object"]
    objects = np.asarray([int(row["object_id"]) for row in per_object])
    all_values = 100 * np.asarray(
        [float(row["relative_improvement"]) for row in per_object]
    )
    missing_values = 100 * np.asarray(
        [
            (
                float(row["baseline_natural_missing_loss"])
                - float(row["proposed_natural_missing_loss"])
            )
            / float(row["baseline_natural_missing_loss"])
            for row in per_object
        ]
    )
    positions = np.arange(len(objects))
    width = 0.37
    axes[1].bar(
        positions - width / 2,
        all_values,
        width,
        color=COLORS["blue"],
        label="All frames",
    )
    axes[1].bar(
        positions + width / 2,
        missing_values,
        width,
        color=COLORS["orange"],
        label="Natural missing",
    )
    axes[1].axhline(0.0, color=COLORS["black"], linewidth=0.8)
    axes[1].set_xticks(
        positions,
        [
            f"{object_id:02d}\n({row['track_count']}t)"
            for object_id, row in zip(objects, per_object)
        ],
    )
    axes[1].set_xlabel("Object ID (track count)")
    axes[1].set_ylabel("Relative loss improvement (pp)")
    axes[1].set_title("Benefits did not generalize uniformly")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")

    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontweight="bold")
    fig.suptitle(
        "M5-R3 sealed Photoneo outcome: FAIL — 23 tracks, 8 objects, 309 natural missing frames",
        y=1.025,
        fontsize=11,
    )
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, OUTPUT_DIR, "m5_r3_sealed_metrics")
    plt.close(fig)


if __name__ == "__main__":
    main()
