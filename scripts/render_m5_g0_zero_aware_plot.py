"""Render the zero-aware M5-G0 sealed dropout-recovery aggregate."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_m5_g0 as runner  # noqa: E402
from m1_common import load_jsonl, sha256_file, write_json_atomic  # noqa: E402


METHODS = runner.SEALED_METHODS
PLOT_NAME = "aggregate_dropout_recovery.png"


def dropout_recovery_values(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """Return the frozen median ten-frame-dropout recovery value by method."""

    return {
        method: runner._median(
            runner._finite_values(
                runner._filter_rows(
                    rows,
                    method,
                    symmetries=set(runner.SYMMETRIC_CLASSES),
                    stresses={"DROPOUT_10"},
                ),
                "metrics.dropout.frames_to_return_within_nominal.median",
            )
        )
        for method in METHODS
    }


def render(repo_root: Path) -> dict[str, Any]:
    artifacts = repo_root / "artifacts" / "m5_g0"
    sealed_path = artifacts / "sealed_results.jsonl"
    rows = load_jsonl(sealed_path)
    values = dropout_recovery_values(rows)
    reports = repo_root / "reports" / "m5_g0"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / PLOT_NAME
    temporary = path.with_name(f".{path.stem}.tmp.png")

    numeric = [values[method] for method in METHODS]
    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(
        METHODS,
        [math.nan if value is None else value for value in numeric],
        color="#4c78a8",
    )
    for index, (bar, value) in enumerate(zip(bars, numeric, strict=True)):
        if value is None:
            label = "N/A"
            marker_y = 0.0
            marker = "x"
        else:
            label = f"{value:.1f}"
            marker_y = float(value)
            marker = "o"
        axis.scatter(
            index,
            marker_y,
            marker=marker,
            facecolors="white" if marker == "o" else "none",
            edgecolors="#1f4e79",
            color="#1f4e79",
            linewidths=1.5,
            zorder=4,
        )
        axis.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2.0, marker_y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    finite = [float(value) for value in numeric if value is not None]
    all_zero = bool(finite) and all(abs(value) <= 1e-15 for value in finite)
    if all_zero:
        axis.set_ylim(-0.05, 1.0)
        axis.text(
            0.5,
            0.82,
            "All sealed median recovery values = 0 frames",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("median recovery frames")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    temporary.replace(path)

    receipt = {
        "schema_version": 1,
        "record_type": "m5_g0_zero_aware_plot_receipt",
        "plot": PLOT_NAME,
        "values": values,
        "all_finite_values_zero": all_zero,
        "source_hashes": {
            "sealed_results": sha256_file(sealed_path),
            "renderer": sha256_file(Path(__file__).resolve()),
            "plot": sha256_file(path),
        },
    }
    write_json_atomic(artifacts / "zero_aware_plot_receipt.json", receipt)
    return receipt


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    receipt = render(repo_root)
    print(
        "rendered M5-G0 zero-aware dropout plot: "
        f"all_zero={receipt['all_finite_values_zero']}"
    )


if __name__ == "__main__":
    main()
