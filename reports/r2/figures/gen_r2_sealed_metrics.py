#!/usr/bin/env python3
"""Generate the quantitative PoseLoop R2 sealed-result figure."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from paper_plot_style import COLORS, save_figure
from r2_visual_common import OUTPUT_DIR, SEALED_ROOT, load_json, load_jsonl


def main() -> None:
    result = load_json(SEALED_ROOT / "sealed_result.json")
    outcomes = load_jsonl(SEALED_ROOT / "sealed_evaluated_outcomes.jsonl")
    m3 = result["stages"]["M3-R2"]
    m4 = result["stages"]["M4-R2"]
    m6 = result["stages"]["M6-R2"]

    y = np.asarray([not bool(row["joint_success"]) for row in outcomes], dtype=np.int64)
    learned_risk = np.asarray(
        [float(row["learned_failure_risk"]) for row in outcomes], dtype=np.float64
    )
    raw_risk = np.asarray(
        [float(row["raw_score_risk_baseline"]) for row in outcomes], dtype=np.float64
    )
    learned_fpr, learned_tpr, _ = roc_curve(y, learned_risk)
    raw_fpr, raw_tpr, _ = roc_curve(y, raw_risk)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.15))

    gains = np.asarray(
        [
            m3["gain_over_matched_random_pp"],
            m4["gain_over_fixed_slot_2_pp"],
            m4["gain_over_uniform_random_pp"],
        ],
        dtype=np.float64,
    )
    labels = ["M3\nrandom", "M4\nfixed", "M4\nrandom"]
    absolute = [
        f"{100*m3['active_macro_combined']:.2f} vs {100*m3['matched_random_macro_combined']:.2f}",
        f"{100*m4['active_macro_combined']:.2f} vs {100*m4['fixed_slot_2_macro_combined']:.2f}",
        f"{100*m4['active_macro_combined']:.2f} vs {100*m4['uniform_random_macro_combined']:.2f}",
    ]
    bars = axes[0].bar(
        np.arange(3), gains, color=[COLORS["blue"], COLORS["green"], COLORS["sky"]]
    )
    axes[0].axhline(0.0, color=COLORS["black"], linewidth=0.8)
    axes[0].set_ylabel("Macro combined utility gain (pp)")
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylim(0, max(gains) * 1.48)
    for bar, gain, detail in zip(bars, gains, absolute):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            gain + 0.05,
            f"+{gain:.2f}\n({detail})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axes[1].plot(
        learned_fpr,
        learned_tpr,
        color=COLORS["blue"],
        linewidth=2.0,
        label=f"R2 risk model ({m6['learned']['auroc']:.3f})",
    )
    axes[1].plot(
        raw_fpr,
        raw_tpr,
        color=COLORS["orange"],
        linewidth=1.8,
        label=f"Raw-score baseline ({m6['raw_score_baseline']['auroc']:.3f})",
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], linewidth=1)
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, loc="lower right")

    for key, color, label in (
        ("learned", COLORS["blue"], "R2 risk model"),
        ("raw_score_baseline", COLORS["orange"], "Raw-score baseline"),
    ):
        curve = m6[key]["risk_coverage_curve"]
        axes[2].plot(
            100 * np.asarray(curve["coverage"], dtype=np.float64),
            100 * np.asarray(curve["empirical_failure_risk"], dtype=np.float64),
            color=color,
            linewidth=2.0 if key == "learned" else 1.8,
            label=f"{label} (AURC {curve['aurc']:.3f})",
        )
    axes[2].set_xlabel("Accepted coverage (%)")
    axes[2].set_ylabel("Empirical failure risk (%)")
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(frameon=False, loc="upper left")

    for label, axis in zip(("(a)", "(b)", "(c)"), axes):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUTPUT_DIR, "r2_sealed_metrics")
    plt.close(fig)


if __name__ == "__main__":
    main()
