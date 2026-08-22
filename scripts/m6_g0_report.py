#!/usr/bin/env python3
"""Render the tracked PoseLoop M6-G0 report and publication figures.

The renderer consumes only already-computed M2-development feature, evaluator,
and out-of-fold analysis structures.  It does not load protected holdout rows,
run inference, fit models, or create pose estimates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SCHEMA_VERSION = 1
FORMAL_METHODS: tuple[str, ...] = (
    "RAW_SCORE_RANK",
    "SCORE_ISOTONIC",
    "LOGISTIC_MULTIFEATURE",
    "SHALLOW_TREE_MULTIFEATURE",
    "NESTED_MULTIFEATURE",
)
PLOT_FILENAMES: tuple[str, ...] = (
    "failure_prevalence_by_object.png",
    "raw_score_by_outcome.png",
    "risk_coverage_curves.png",
    "calibration_reliability.png",
    "aurc_brier_bootstrap.png",
    "feature_ablation.png",
    "disagreement_vs_actual_error.png",
    "per_object_aurc_difference.png",
    "leave_one_object_out.png",
)

METHOD_LABELS = {
    "RAW_SCORE_RANK": "Raw score rank",
    "SCORE_ISOTONIC": "Score isotonic",
    "LOGISTIC_MULTIFEATURE": "Logistic multi",
    "SHALLOW_TREE_MULTIFEATURE": "Shallow tree multi",
    "NESTED_MULTIFEATURE": "Nested multi",
}
METHOD_COLORS = {
    "RAW_SCORE_RANK": "#4D4D4D",
    "SCORE_ISOTONIC": "#0072B2",
    "LOGISTIC_MULTIFEATURE": "#009E73",
    "SHALLOW_TREE_MULTIFEATURE": "#D55E00",
    "NESTED_MULTIFEATURE": "#CC79A7",
}
METHOD_LINESTYLES = {
    "RAW_SCORE_RANK": (0, (3, 2)),
    "SCORE_ISOTONIC": (0, (5, 2)),
    "LOGISTIC_MULTIFEATURE": (0, (1, 1)),
    "SHALLOW_TREE_MULTIFEATURE": (0, (5, 1, 1, 1)),
    "NESTED_MULTIFEATURE": "-",
}
ABLATION_LABELS = {
    "score_only": "Score only",
    "disagreement_only": "Disagreement only",
    "score_disagreement": "Score + disagreement",
    "all": "All permitted",
}


def _configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 9.0,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.7,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(
        _json_safe(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    _write_text_atomic(path, f"{payload}\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
        metadata={"Software": "PoseLoop M6-G0 deterministic renderer"},
    )
    plt.close(fig)


def _no_data(ax: Any, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#555555",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _records(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict(orient="records")
        except TypeError:
            value = value.to_dict("records")
    if isinstance(value, Mapping):
        return [value]
    return [row for row in value if isinstance(row, Mapping)]


def _object_id_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value)
    return (0, int(text), text) if text.lstrip("+-").isdigit() else (1, 0, text)


def _sorted_object_records(
    rows: Sequence[Mapping[str, Any]], field: str
) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: _object_id_sort_key(row.get(field)))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _target_id_from_feature(row: Mapping[str, Any]) -> str:
    metadata = _mapping(row.get("metadata"))
    value = (
        row.get("target_id")
        or metadata.get("target_id")
        or metadata.get("target_sample_id")
        or metadata.get("group_id")
    )
    return "" if value is None else str(value)


def _feature_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    features = row.get("features")
    return features if isinstance(features, Mapping) else row


def _evaluator_index(
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["target_id"]): row
        for row in evaluator_rows
        if row.get("target_id") is not None
    }


def _aggregate(evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(evaluation.get("aggregate_metrics"))


def _metric(aggregate: Mapping[str, Any], method: str, name: str) -> float | int | None:
    value = _mapping(aggregate.get(method)).get(name)
    if isinstance(value, bool):
        return int(value)
    number = _finite(value)
    if number is None:
        return None
    if isinstance(value, int):
        return int(value)
    return number


def _aurc_for_rows(rows: Sequence[Mapping[str, Any]]) -> float | None:
    usable: list[tuple[float, int]] = []
    for row in rows:
        risk = _finite(row.get("risk_score"))
        label = row.get("y_failure")
        if risk is None or label not in (0, 1, False, True):
            continue
        usable.append((risk, int(label)))
    if not usable:
        return None
    usable.sort(key=lambda item: item[0])

    # Match the formal selective-metric policy without importing the fitting
    # module into this renderer: every equal-risk block uses the expected
    # prefix failure count over uniform within-block permutations. Identifiers
    # and input order therefore cannot become undeclared secondary rankings.
    prefix_risks: list[float] = []
    failures_before = 0.0
    start = 0
    while start < len(usable):
        end = start + 1
        while end < len(usable) and usable[end][0] == usable[start][0]:
            end += 1
        block_size = end - start
        block_failures = float(sum(label for _, label in usable[start:end]))
        for offset in range(1, block_size + 1):
            accepted = start + offset
            expected_failures = failures_before + offset * block_failures / block_size
            prefix_risks.append(expected_failures / accepted)
        failures_before += block_failures
        start = end
    return float(np.mean(prefix_risks))


def _interval(
    bootstrap: Mapping[str, Any], method: str, metric: str
) -> Mapping[str, Any]:
    methods = _mapping(bootstrap.get("method_intervals"))
    return _mapping(_mapping(methods.get(method)).get(metric))


def _interval_error(
    point: float | None, interval: Mapping[str, Any]
) -> tuple[float, float] | None:
    lower = _finite(interval.get("lower"))
    upper = _finite(interval.get("upper"))
    if point is None or lower is None or upper is None:
        return None
    return max(point - lower, 0.0), max(upper - point, 0.0)


def _failure_prevalence_plot(
    path: Path,
    evaluator_rows: Sequence[Mapping[str, Any]],
    label_support: Mapping[str, Any],
) -> None:
    rows = _records(label_support.get("by_object"))
    if not rows:
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in evaluator_rows:
            key = str(row.get("object_id", "?"))
            label = int(row.get("y_failure", 0))
            counts[key][0] += 1
            counts[key][1] += label
        rows = [
            {
                "object_id": key,
                "target_count": total,
                "failure_count": failures,
                "failure_prevalence": failures / total if total else 0.0,
            }
            for key, (total, failures) in sorted(counts.items())
        ]
    rows = _sorted_object_records(rows, "object_id")
    labels = [str(row.get("object_id")) for row in rows]
    values = [_finite(row.get("failure_prevalence")) or 0.0 for row in rows]
    counts = [int(row.get("target_count", 0)) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    if values:
        x = np.arange(len(values))
        bars = ax.bar(
            x,
            values,
            color="#56B4E9",
            edgecolor="#333333",
            width=0.76,
        )
        overall = _finite(label_support.get("failure_prevalence"))
        if overall is not None:
            ax.axhline(
                overall,
                color="#D55E00",
                linestyle="--",
                linewidth=1.2,
                label=f"Overall prevalence ({overall:.1%})",
            )
            ax.legend(frameon=False, loc="upper right")
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                rotation=90,
            )
        ax.set_xticks(x, labels)
        ax.set_ylim(0.0, max(1.0, max(values) + 0.15))
        ax.set_xlabel("Object ID")
        ax.set_ylabel("Failure prevalence")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    else:
        _no_data(ax, "No per-object labels available")
    _save_figure(fig, path)


def _raw_score_plot(
    path: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> None:
    evaluator = _evaluator_index(evaluator_rows)
    by_outcome: dict[int, list[float]] = {0: [], 1: []}
    for row in feature_rows:
        target_id = _target_id_from_feature(row)
        label_row = evaluator.get(target_id)
        score = _finite(_feature_mapping(row).get("raw_selected_score"))
        if label_row is None or score is None:
            continue
        by_outcome[int(label_row.get("y_failure", 0))].append(score)

    fig, ax = plt.subplots(figsize=(5.2, 3.25), constrained_layout=True)
    all_values = [*by_outcome[0], *by_outcome[1]]
    if all_values:
        low, high = min(all_values), max(all_values)
        if math.isclose(low, high):
            low, high = low - 0.5, high + 0.5
        bins = np.linspace(low, high, 21)
        for label, text, color in (
            (0, "Correct", "#0072B2"),
            (1, "Failed", "#D55E00"),
        ):
            values = by_outcome[label]
            if values:
                ax.hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    color=color,
                    label=f"{text} (n={len(values)})",
                )
                ax.axvline(
                    float(np.median(values)),
                    color=color,
                    linestyle=(0, (2, 2)),
                    linewidth=1.0,
                )
        ax.set_xlabel("Frozen candidate FoundationPose score (higher is better)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    else:
        _no_data(ax, "No finite raw scores available")
    _save_figure(fig, path)


def _risk_coverage_plot(path: Path, evaluation: Mapping[str, Any]) -> None:
    aggregate = _aggregate(evaluation)
    fig, ax = plt.subplots(figsize=(5.4, 3.45), constrained_layout=True)
    plotted = False
    for method in FORMAL_METHODS:
        curve = _mapping(_mapping(aggregate.get(method)).get("risk_coverage_curve"))
        coverage = np.asarray(curve.get("coverage", []), dtype=float)
        risk = np.asarray(curve.get("empirical_failure_risk", []), dtype=float)
        if not len(coverage) or len(coverage) != len(risk):
            continue
        plotted = True
        ax.plot(
            coverage,
            risk,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.2 if method == "NESTED_MULTIFEATURE" else 1.35,
            zorder=4 if method == "NESTED_MULTIFEATURE" else 2,
        )
    prevalence = _metric(aggregate, "NESTED_MULTIFEATURE", "failure_prevalence")
    if prevalence is not None:
        ax.axhline(
            prevalence,
            color="#999999",
            linestyle=(0, (1, 2)),
            linewidth=0.9,
            label="Full-coverage prevalence",
        )
    if plotted:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Coverage (safest predictions accepted first)")
        ax.set_ylabel("Empirical failure risk")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(color="#E5E5E5", linewidth=0.5)
        ax.legend(frameon=False, ncol=2, loc="upper left")
    else:
        _no_data(ax, "No risk-coverage curves available")
    _save_figure(fig, path)


def _calibration_plot(path: Path, evaluation: Mapping[str, Any]) -> None:
    aggregate = _aggregate(evaluation)
    fig, ax = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#777777",
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        label="Ideal",
    )
    plotted = False
    marker_cycle = ("o", "s", "^", "D")
    calibrated = [method for method in FORMAL_METHODS if method != "RAW_SCORE_RANK"]
    for marker, method in zip(marker_cycle, calibrated, strict=True):
        bins = _records(_mapping(aggregate.get(method)).get("calibration_bins"))
        mean_probability = [_finite(row.get("mean_probability")) for row in bins]
        empirical = [_finite(row.get("empirical_failure_rate")) for row in bins]
        pairs = [
            (x_value, y_value)
            for x_value, y_value in zip(mean_probability, empirical, strict=True)
            if x_value is not None and y_value is not None
        ]
        if not pairs:
            continue
        plotted = True
        x_values, y_values = zip(*pairs, strict=True)
        ax.plot(
            x_values,
            y_values,
            marker=marker,
            markersize=3.8,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )
    if plotted:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Mean predicted failure probability")
        ax.set_ylabel("Tie-averaged empirical failure rate")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(color="#E5E5E5", linewidth=0.5)
        ax.legend(frameon=False, loc="upper left")
    else:
        _no_data(ax, "No calibrated probability bins available")
    _save_figure(fig, path)


def _aurc_brier_plot(path: Path, evaluation: Mapping[str, Any]) -> None:
    aggregate = _aggregate(evaluation)
    bootstrap = _mapping(evaluation.get("grouped_bootstrap"))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), constrained_layout=True)
    metrics = (("aurc", "AURC (lower is better)"), ("brier", "Brier score"))
    for ax, (metric_name, ylabel) in zip(axes, metrics, strict=True):
        methods: list[str] = []
        points: list[float] = []
        errors: list[tuple[float, float] | None] = []
        for method in FORMAL_METHODS:
            point = _finite(_mapping(aggregate.get(method)).get(metric_name))
            if point is None:
                continue
            methods.append(method)
            points.append(point)
            errors.append(
                _interval_error(point, _interval(bootstrap, method, metric_name))
            )
        if not points:
            _no_data(ax, f"No {metric_name.upper()} values available")
            continue
        x = np.arange(len(points))
        ax.scatter(
            x,
            points,
            s=32,
            c=[METHOD_COLORS[method] for method in methods],
            edgecolors="#222222",
            linewidths=0.45,
            zorder=3,
        )
        for index, (point, error) in enumerate(zip(points, errors, strict=True)):
            if error is None:
                continue
            ax.errorbar(
                index,
                point,
                yerr=np.asarray([[error[0]], [error[1]]]),
                color="#222222",
                capsize=2.5,
                linewidth=0.9,
                zorder=2,
            )
        ax.set_xticks(
            x,
            [METHOD_LABELS[method] for method in methods],
            rotation=24,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    _save_figure(fig, path)


def _ablation_metrics(value: Any) -> Mapping[str, Any]:
    record = _mapping(value)
    metrics = record.get("aggregate_metrics")
    if not isinstance(metrics, Mapping):
        return {}
    nested = metrics.get("NESTED_MULTIFEATURE")
    return _mapping(nested) if isinstance(nested, Mapping) else metrics


def _feature_ablation_plot(path: Path, evaluation: Mapping[str, Any]) -> None:
    ablations = _mapping(evaluation.get("feature_ablations"))
    names = [name for name in ABLATION_LABELS if name in ablations]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
    if not names:
        for ax in axes:
            _no_data(ax, "No ablation results available")
        _save_figure(fig, path)
        return
    x = np.arange(len(names))
    colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"][: len(names)]
    for ax, metric_name, ylabel in (
        (axes[0], "aurc", "AURC (lower is better)"),
        (axes[1], "auroc", "AUROC (higher is better)"),
    ):
        values = [
            _finite(_ablation_metrics(ablations[name]).get(metric_name))
            for name in names
        ]
        numeric = [value if value is not None else 0.0 for value in values]
        bars = ax.bar(x, numeric, color=colors, edgecolor="#333333", width=0.7)
        for bar, value in zip(bars, values, strict=True):
            label = "NA" if value is None else f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=7.0,
            )
        ax.set_xticks(
            x,
            [ABLATION_LABELS[name] for name in names],
            rotation=22,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    _save_figure(fig, path)


def _disagreement_error_plot(
    path: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> None:
    evaluator = _evaluator_index(evaluator_rows)
    points: list[tuple[float, float, int]] = []
    for row in feature_rows:
        label = evaluator.get(_target_id_from_feature(row))
        if label is None:
            continue
        features = _feature_mapping(row)
        disagreement = _finite(features.get("pair_normalized_mssd_p90"))
        if disagreement is None:
            disagreement = _finite(features.get("pair_normalized_mssd_median"))
        actual_error = _finite(label.get("normalized_mssd"))
        if disagreement is None or actual_error is None:
            continue
        points.append((disagreement, actual_error, int(label.get("y_failure", 0))))

    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    if points:
        array = np.asarray(points, dtype=float)
        for label, text, color, marker in (
            (0, "Correct", "#0072B2", "o"),
            (1, "Failed", "#D55E00", "x"),
        ):
            selected = array[:, 2] == label
            if np.any(selected):
                ax.scatter(
                    array[selected, 0],
                    array[selected, 1],
                    s=16,
                    alpha=0.48,
                    color=color,
                    marker=marker,
                    linewidths=0.7,
                    label=f"{text} (n={int(np.sum(selected))})",
                )
        order = np.argsort(array[:, 0], kind="stable")
        bins = np.array_split(order, min(6, len(order)))
        bin_x = [float(np.median(array[index, 0])) for index in bins if len(index)]
        bin_y = [float(np.mean(array[index, 1])) for index in bins if len(index)]
        ax.plot(
            bin_x,
            bin_y,
            color="#000000",
            marker="D",
            markersize=3.2,
            linewidth=1.2,
            label="Equal-count-bin mean",
            zorder=4,
        )
        ax.axhline(
            0.10,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            label="MSSD correctness threshold",
        )
        ax.set_xlabel("Pairwise normalized MSSD disagreement (p90)")
        ax.set_ylabel("Actual normalized MSSD error")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(color="#E5E5E5", linewidth=0.5)
        ax.legend(frameon=False, loc="upper left")
    else:
        _no_data(ax, "No finite disagreement/error pairs available")
    _save_figure(fig, path)


def _per_object_aurc_differences(
    evaluation: Mapping[str, Any],
) -> list[tuple[str, float]]:
    predictions = _records(evaluation.get("oof_predictions"))
    by_method_object: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        method = str(row.get("method", ""))
        object_id = str(row.get("object_id", "?"))
        by_method_object[(method, object_id)].append(row)
    objects = sorted(
        {
            object_id
            for method, object_id in by_method_object
            if method == "NESTED_MULTIFEATURE"
        },
        key=_object_id_sort_key,
    )
    result: list[tuple[str, float]] = []
    for object_id in objects:
        candidate = _aurc_for_rows(by_method_object[("NESTED_MULTIFEATURE", object_id)])
        isotonic = _aurc_for_rows(by_method_object[("SCORE_ISOTONIC", object_id)])
        if candidate is not None and isotonic is not None:
            result.append((object_id, isotonic - candidate))
    return result


def _per_object_difference_plot(path: Path, evaluation: Mapping[str, Any]) -> None:
    differences = _per_object_aurc_differences(evaluation)
    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    if differences:
        labels, values = zip(*differences, strict=True)
        x = np.arange(len(values))
        colors = ["#009E73" if value >= 0 else "#D55E00" for value in values]
        ax.bar(x, values, color=colors, edgecolor="#333333", width=0.74)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Object ID")
        ax.set_ylabel("Isotonic AURC − nested AURC")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    else:
        _no_data(ax, "No aligned per-object OOF predictions available")
    _save_figure(fig, path)


def _leave_one_object_plot(path: Path, evaluation: Mapping[str, Any]) -> None:
    robustness = _mapping(evaluation.get("object_jackknife"))
    rows = _sorted_object_records(
        _records(robustness.get("leave_one_object_out")), "removed_object_id"
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    if rows:
        labels = [str(row.get("removed_object_id", "?")) for row in rows]
        values = [
            _finite(row.get("candidate_minus_isotonic_aurc_improvement"))
            for row in rows
        ]
        numeric = [value if value is not None else 0.0 for value in values]
        x = np.arange(len(rows))
        colors = [
            "#009E73" if value is not None and value >= 0 else "#D55E00"
            for value in values
        ]
        ax.bar(x, numeric, color=colors, edgecolor="#333333", width=0.74)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Removed object ID")
        ax.set_ylabel("Remaining isotonic AURC − nested AURC")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    else:
        _no_data(ax, "No leave-one-object-out results available")
    _save_figure(fig, path)


def _render_plots(
    report_dir: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    label_support: Mapping[str, Any],
) -> list[Path]:
    plot_paths = [report_dir / filename for filename in PLOT_FILENAMES]
    _failure_prevalence_plot(plot_paths[0], evaluator_rows, label_support)
    _raw_score_plot(plot_paths[1], feature_rows, evaluator_rows)
    _risk_coverage_plot(plot_paths[2], evaluation)
    _calibration_plot(plot_paths[3], evaluation)
    _aurc_brier_plot(plot_paths[4], evaluation)
    _feature_ablation_plot(plot_paths[5], evaluation)
    _disagreement_error_plot(plot_paths[6], feature_rows, evaluator_rows)
    _per_object_difference_plot(plot_paths[7], evaluation)
    _leave_one_object_plot(plot_paths[8], evaluation)
    return plot_paths


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite(value)
    if number is None:
        return "NA"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    number = _finite(value)
    return "NA" if number is None else f"{number:.{digits}%}"


def _ci_text(interval: Any, *, percent: bool = False) -> str:
    record = _mapping(interval)
    lower = _finite(record.get("lower"))
    upper = _finite(record.get("upper"))
    if lower is None or upper is None:
        return "NA"
    if percent:
        return f"[{lower:.1%}, {upper:.1%}]"
    return f"[{lower:.4f}, {upper:.4f}]"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_cell(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _pooled_metric_tables(evaluation: Mapping[str, Any]) -> str:
    aggregate = _aggregate(evaluation)
    bootstrap = _mapping(evaluation.get("grouped_bootstrap"))
    discrimination_rows: list[list[str]] = []
    selective_rows: list[list[str]] = []
    bootstrap_rows: list[list[str]] = []
    for method in FORMAL_METHODS:
        metrics = _mapping(aggregate.get(method))
        discrimination_rows.append(
            [
                METHOD_LABELS[method],
                _fmt(metrics.get("auroc")),
                _fmt(metrics.get("auprc")),
                _pct(metrics.get("failure_prevalence")),
                _fmt(metrics.get("auprc_lift_over_prevalence")),
                _fmt(metrics.get("brier")),
                _fmt(metrics.get("log_loss")),
                _fmt(metrics.get("ece")),
                _fmt(metrics.get("mce")),
                _fmt(metrics.get("calibration_slope")),
                _fmt(metrics.get("calibration_intercept")),
            ]
        )
        selective_rows.append(
            [
                METHOD_LABELS[method],
                _fmt(metrics.get("aurc")),
                _pct(metrics.get("relative_aurc_improvement_vs_raw_score_rank")),
                _pct(metrics.get("relative_aurc_improvement_vs_score_isotonic")),
                _pct(metrics.get("risk_at_coverage_0_50")),
                _pct(metrics.get("risk_at_coverage_0_80")),
                _pct(metrics.get("risk_at_coverage_0_90")),
                _pct(metrics.get("coverage_at_empirical_risk_0_05")),
                _pct(metrics.get("coverage_at_empirical_risk_0_10")),
                _pct(metrics.get("coverage_at_empirical_risk_0_20")),
                _fmt(metrics.get("predicted_risk_le_0_05_count"), 0),
                _pct(metrics.get("predicted_risk_le_0_05_coverage")),
                _pct(metrics.get("predicted_risk_le_0_05_empirical_failure_rate")),
            ]
        )
        bootstrap_rows.append(
            [
                METHOD_LABELS[method],
                _ci_text(_interval(bootstrap, method, "auroc")),
                _ci_text(_interval(bootstrap, method, "auprc")),
                _ci_text(_interval(bootstrap, method, "brier")),
                _ci_text(_interval(bootstrap, method, "aurc")),
                _ci_text(
                    _interval(bootstrap, method, "risk_at_coverage_0_80"),
                    percent=True,
                ),
                _ci_text(
                    _interval(
                        bootstrap,
                        method,
                        "predicted_risk_le_0_05_coverage",
                    ),
                    percent=True,
                ),
            ]
        )

    paired = _mapping(bootstrap.get("paired_differences"))
    paired_rows = [
        [name.replace("_", " "), _ci_text(interval)]
        for name, interval in paired.items()
    ]
    parts = [
        "### Discrimination and calibration\n",
        _table(
            (
                "Formal method",
                "AUROC",
                "AUPRC",
                "Failure prevalence",
                "AUPRC lift",
                "Brier",
                "Log loss",
                "ECE",
                "MCE",
                "Calibration slope",
                "Calibration intercept",
            ),
            discrimination_rows,
        ),
        "\n\nRAW_SCORE_RANK has no fitted probability, so its calibration fields are "
        "intentionally `NA`.\n",
        "\n### Selective prediction\n",
        _table(
            (
                "Formal method",
                "AURC",
                "Rel. vs raw",
                "Rel. vs isotonic",
                "Risk @ 50% cov.",
                "Risk @ 80% cov.",
                "Risk @ 90% cov.",
                "Cov. @ ≤5% empirical",
                "Cov. @ ≤10% empirical",
                "Cov. @ ≤20% empirical",
                "Count p≤.05",
                "Cov. p≤.05",
                "Empirical risk p≤.05",
            ),
            selective_rows,
        ),
        "\n\nCoverage metrics use the complete safest-to-riskiest OOF ordering. "
        "Equal-risk blocks use their tie-invariant expected prefix risk, so "
        "audit identifiers never determine model ranking; lower AURC is better.\n",
        "\n### Grouped-bootstrap 95% confidence intervals\n",
        _table(
            (
                "Formal method",
                "AUROC CI",
                "AUPRC CI",
                "Brier CI",
                "AURC CI",
                "Risk @ 80% CI",
                "Coverage p≤.05 CI",
            ),
            bootstrap_rows,
        ),
    ]
    if paired_rows:
        parts.extend(
            [
                "\n\nPaired grouped-bootstrap differences use whole physical "
                "instances as the resampling unit. Positive AURC improvement "
                "means the nested candidate has lower AURC.\n\n",
                _table(("Paired quantity", "95% CI"), paired_rows),
            ]
        )
    return "".join(parts)


def _per_fold_table(evaluation: Mapping[str, Any]) -> str:
    rows = []
    for record in sorted(
        _records(evaluation.get("per_fold_metrics")),
        key=lambda row: (int(row.get("outer_fold", -1)), str(row.get("method"))),
    ):
        metrics = _mapping(record.get("metrics"))
        method = str(record.get("method", ""))
        rows.append(
            [
                record.get("outer_fold", "NA"),
                METHOD_LABELS.get(method, method),
                metrics.get("target_count", "NA"),
                metrics.get("failure_count", "NA"),
                _fmt(metrics.get("auroc")),
                _fmt(metrics.get("auprc")),
                _fmt(metrics.get("brier")),
                _fmt(metrics.get("ece")),
                _fmt(metrics.get("aurc")),
                _pct(metrics.get("risk_at_coverage_0_80")),
            ]
        )
    if not rows:
        return "No fold-level metrics were supplied."
    return _table(
        (
            "Outer fold",
            "Method",
            "Targets",
            "Failures",
            "AUROC",
            "AUPRC",
            "Brier",
            "ECE",
            "AURC",
            "Risk @ 80%",
        ),
        rows,
    )


def _model_selection_table(evaluation: Mapping[str, Any]) -> str:
    rows = []
    for record in _records(evaluation.get("model_selections")):
        parameters = record.get("nested_selected_parameters", {})
        rows.append(
            [
                record.get("outer_fold", "NA"),
                record.get("outer_train_target_count", "NA"),
                record.get("outer_test_target_count", "NA"),
                record.get("nested_selected_family", "NA"),
                f"`{json.dumps(_json_safe(parameters), sort_keys=True)}`",
                _fmt(record.get("best_logistic_inner_mean_aurc")),
                _fmt(record.get("best_tree_inner_mean_aurc")),
                _fmt(record.get("tree_inner_aurc_advantage")),
                str(bool(record.get("logistic_tie_rule_applied"))),
                str(bool(record.get("outer_test_labels_used_for_selection"))),
            ]
        )
    if not rows:
        return "No model-selection records were supplied."
    return _table(
        (
            "Outer fold",
            "Train n",
            "Test n",
            "Nested family",
            "Selected parameters",
            "Inner logistic AURC",
            "Inner tree AURC",
            "Tree advantage",
            "Logistic tie rule",
            "Outer labels used",
        ),
        rows,
    )


def _ablation_table(evaluation: Mapping[str, Any]) -> str:
    ablations = _mapping(evaluation.get("feature_ablations"))
    rows = []
    for name in ABLATION_LABELS:
        if name not in ablations:
            continue
        record = _mapping(ablations[name])
        metrics = _ablation_metrics(record)
        fold_metrics = _records(record.get("per_fold_metrics"))
        rows.append(
            [
                ABLATION_LABELS[name],
                len(record.get("feature_names", [])),
                _fmt(metrics.get("auroc")),
                _fmt(metrics.get("auprc")),
                _fmt(metrics.get("brier")),
                _fmt(metrics.get("ece")),
                _fmt(metrics.get("aurc")),
                _pct(metrics.get("relative_aurc_improvement_vs_raw_score_rank")),
                sum(
                    _finite(_mapping(row.get("metrics")).get("aurc")) is not None
                    for row in fold_metrics
                ),
            ]
        )
    if not rows:
        return "No feature-ablation results were supplied."
    return _table(
        (
            "Ablation",
            "Features",
            "AUROC",
            "AUPRC",
            "Brier",
            "ECE",
            "AURC",
            "Rel. AURC vs raw",
            "OOF folds",
        ),
        rows,
    )


def _contribution_tables(evaluation: Mapping[str, Any]) -> str:
    contributions = _mapping(evaluation.get("feature_contributions"))
    stability = sorted(
        _records(contributions.get("logistic_coefficient_stability")),
        key=lambda row: (
            -abs(_finite(row.get("coefficient_mean")) or 0.0),
            str(row.get("feature", "")),
        ),
    )
    coefficient_rows = [
        [
            row.get("feature", "NA"),
            _fmt(row.get("coefficient_mean")),
            _fmt(row.get("coefficient_std")),
            _fmt(row.get("coefficient_min")),
            _fmt(row.get("coefficient_max")),
            _pct(row.get("sign_consistency")),
        ]
        for row in stability[:15]
    ]

    importance_by_feature: dict[str, list[float]] = defaultdict(list)
    for row in _records(contributions.get("tree_outer_test_permutation_importance")):
        value = _finite(row.get("importance_mean"))
        if value is not None:
            importance_by_feature[str(row.get("feature", "NA"))].append(value)
    importance = sorted(
        (
            (feature, float(np.mean(values)), float(np.std(values)))
            for feature, values in importance_by_feature.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    importance_rows = [
        [feature, _fmt(mean), _fmt(std), len(importance_by_feature[feature])]
        for feature, mean, std in importance[:15]
    ]
    parts = [
        "Logistic values are standardized coefficients. The table shows the 15 "
        "largest absolute fold means; the complete fold vectors remain in the "
        "machine-readable artifact.\n\n",
        _table(
            (
                "Feature",
                "Mean coefficient",
                "Fold SD",
                "Minimum",
                "Maximum",
                "Sign consistency",
            ),
            coefficient_rows,
        )
        if coefficient_rows
        else "No logistic contribution rows were supplied.",
        "\n\nTree values are outer-test permutation AURC increases, never "
        "impurity-only importance. The table shows the 15 largest means.\n\n",
        _table(
            ("Feature", "Mean AURC increase", "Across-fold SD", "Fold rows"),
            importance_rows,
        )
        if importance_rows
        else "No tree permutation-importance rows were supplied.",
    ]
    return "".join(parts)


def _robustness_tables(evaluation: Mapping[str, Any]) -> str:
    robustness = _mapping(evaluation.get("object_jackknife"))
    per_object = _sorted_object_records(
        _records(robustness.get("per_object_metrics")), "object_id"
    )
    object_rows = [
        [
            row.get("object_id", "NA"),
            row.get("target_count", "NA"),
            row.get("failure_count", "NA"),
            _pct(row.get("failure_prevalence")),
            _fmt(row.get("auroc")),
            _fmt(row.get("aurc")),
            _pct(row.get("risk_at_coverage_0_80")),
        ]
        for row in per_object
    ]
    jackknife_rows = [
        [
            row.get("removed_object_id", "NA"),
            row.get("remaining_target_count", "NA"),
            _fmt(row.get("candidate_minus_isotonic_aurc_improvement")),
            _pct(row.get("relative_improvement")),
            str(bool(row.get("improvement_remains_non_negative"))),
            _fmt(row.get("aggregate_gain_change")),
            _fmt(row.get("additive_aurc_gain_contribution")),
            _pct(row.get("fraction_of_aggregate_gain")),
        ]
        for row in _sorted_object_records(
            _records(robustness.get("leave_one_object_out")), "removed_object_id"
        )
    ]
    deferred_rows = [
        [
            row.get("object_id", "NA"),
            _fmt(row.get("correctly_deferred_failure_count"), 2),
            _pct(row.get("fraction_of_correctly_deferred_failures")),
        ]
        for row in _sorted_object_records(
            _records(robustness.get("object_correctly_deferred_failure_contributions")),
            "object_id",
        )
    ]
    instance_rows = [
        [
            row.get("physical_instance_id", "NA"),
            row.get("target_count", "NA"),
            _fmt(row.get("leave_one_instance_out_gain")),
            _fmt(row.get("aggregate_gain_contribution")),
            _pct(row.get("fraction_of_aggregate_gain")),
        ]
        for row in _records(robustness.get("physical_instance_gain_contributions"))
    ]
    flags = _mapping(robustness.get("object_driven_flags"))
    parts = [
        f"Aggregate isotonic-minus-nested absolute AURC gain: "
        f"**{_fmt(robustness.get('aggregate_absolute_aurc_gain'))}**. "
        f"Object-driven flag: **{bool(robustness.get('object_driven', True))}**. "
        f"Frozen flags: `{json.dumps(_json_safe(flags), sort_keys=True)}`.\n\n",
        "### Per-object OOF diagnostics\n\n",
        _table(
            (
                "Object",
                "Targets",
                "Failures",
                "Prevalence",
                "Nested AUROC",
                "Nested AURC",
                "Risk @ 80%",
            ),
            object_rows,
        )
        if object_rows
        else "No per-object rows were supplied.",
        "\n\n### Leave-one-object-out jackknife\n\n",
        _table(
            (
                "Removed object",
                "Remaining n",
                "Absolute gain",
                "Relative gain",
                "Non-negative",
                "Aggregate-gain change",
                "Additive gain contribution",
                "Fraction of gain",
            ),
            jackknife_rows,
        )
        if jackknife_rows
        else "No object jackknife rows were supplied.",
        "\n\n### Object contribution to correctly deferred failures\n\n",
        _table(
            ("Object", "Expected correctly deferred failures", "Fraction"),
            deferred_rows,
        )
        if deferred_rows
        else "No deferred-failure contribution rows were supplied.",
        "\n\n<details><summary>Complete physical-instance AURC-gain "
        "contribution table</summary>\n\n",
        _table(
            (
                "Physical instance",
                "Targets",
                "Leave-one-out gain",
                "Gain contribution",
                "Fraction of gain",
            ),
            instance_rows,
        )
        if instance_rows
        else "No physical-instance contribution rows were supplied.",
        "\n\n</details>",
    ]
    return "".join(parts)


GATE_DESCRIPTIONS = {
    "auroc_at_least_0_75": "Nested grouped AUROC ≥ 0.75",
    "relative_aurc_improvement_vs_raw_at_least_0_10": (
        "Relative AURC improvement vs raw ≥ 10%"
    ),
    "relative_aurc_improvement_vs_isotonic_at_least_0_05": (
        "Relative AURC improvement vs isotonic ≥ 5%"
    ),
    "bootstrap_isotonic_aurc_improvement_ci_excludes_zero": (
        "Grouped-bootstrap isotonic AURC-gain CI lower bound > 0"
    ),
    "predicted_risk_0_05_coverage_at_least_0_40": "Coverage at p≤.05 ≥ 40%",
    "predicted_risk_0_05_empirical_failure_at_most_0_05": (
        "Empirical failure among p≤.05 ≤ 5%"
    ),
    "predicted_risk_0_05_accepts_at_least_30": "Accepted targets at p≤.05 ≥ 30",
    "brier_not_more_than_2_percent_worse_than_isotonic": (
        "Brier no more than 2% worse than isotonic"
    ),
    "ece_at_most_0_05": "Nested ECE ≤ 0.05",
    "every_object_jackknife_non_negative": "Every object jackknife gain ≥ 0",
    "not_object_or_instance_driven": "No object/instance concentration flag",
    "stable_disagreement_benefit": "Stable disagreement-feature benefit",
    "leakage_checks_passed": "Causal/leakage checks pass",
    "label_support_passed": "Label-support gate passes",
    "oof_complete": "Exactly one OOF prediction per target/method",
    "m3_access_boundary_passed": "Strict report-information boundary passes",
    "auroc_at_least_0_70": "Nested grouped AUROC ≥ 0.70",
    "relative_aurc_improvement_vs_raw_at_least_0_05": (
        "Relative AURC improvement vs raw ≥ 5%"
    ),
}


def _gate_observation(name: str, decision: Mapping[str, Any]) -> str:
    values = _mapping(decision.get("derived_values"))
    mapping = {
        "auroc_at_least_0_75": _fmt(values.get("candidate_auroc")),
        "auroc_at_least_0_70": _fmt(values.get("candidate_auroc")),
        "relative_aurc_improvement_vs_raw_at_least_0_10": _pct(
            values.get("relative_aurc_improvement_vs_raw")
        ),
        "relative_aurc_improvement_vs_raw_at_least_0_05": _pct(
            values.get("relative_aurc_improvement_vs_raw")
        ),
        "relative_aurc_improvement_vs_isotonic_at_least_0_05": _pct(
            values.get("relative_aurc_improvement_vs_isotonic")
        ),
        "bootstrap_isotonic_aurc_improvement_ci_excludes_zero": _ci_text(
            values.get("isotonic_aurc_improvement_bootstrap_ci")
        ),
        "predicted_risk_0_05_coverage_at_least_0_40": _pct(
            values.get("predicted_risk_0_05_coverage")
        ),
        "predicted_risk_0_05_empirical_failure_at_most_0_05": _pct(
            values.get("predicted_risk_0_05_empirical_failure_rate")
        ),
        "predicted_risk_0_05_accepts_at_least_30": _fmt(
            values.get("predicted_risk_0_05_accepted_count"), 0
        ),
        "brier_not_more_than_2_percent_worse_than_isotonic": _pct(
            values.get("candidate_brier_relative_change_vs_isotonic")
        ),
        "ece_at_most_0_05": _fmt(values.get("candidate_ece")),
        "not_object_or_instance_driven": str(values.get("object_driven")),
    }
    return mapping.get(name, "Boolean protocol check")


def _decision_tables(decision: Mapping[str, Any]) -> str:
    parts = []
    for heading, key in (
        ("SIGNAL GO gates", "go_conditions"),
        ("Weak-signal numerical gates", "weak_conditions"),
        ("Hard-validity gates", "hard_validity_conditions"),
    ):
        conditions = _mapping(decision.get(key))
        rows = [
            [
                GATE_DESCRIPTIONS.get(name, name.replace("_", " ")),
                _gate_observation(name, decision),
                "PASS" if bool(passed) else "FAIL",
            ]
            for name, passed in conditions.items()
        ]
        parts.extend(
            [
                f"### {heading}\n\n",
                _table(("Frozen condition", "Observed", "Result"), rows)
                if rows
                else "No conditions supplied.",
                "\n\n",
            ]
        )
    reasons = decision.get("reasons", [])
    if reasons:
        parts.append(
            "Formal failure reasons: "
            + ", ".join(f"`{reason}`" for reason in reasons)
            + "."
        )
    return "".join(parts)


def _qualitative_table(qualitative: Mapping[str, Any]) -> str:
    groups = (
        (
            "Lowest-risk actual failures",
            ("lowest_risk_failures", "lowest_predicted_risk_failures"),
        ),
        (
            "Highest-risk actual successes",
            ("highest_risk_successes", "highest_predicted_risk_successes"),
        ),
        (
            "Largest nested-versus-isotonic disagreements",
            (
                "largest_candidate_isotonic_disagreements",
                "largest_candidate_vs_isotonic_disagreements",
            ),
        ),
    )
    parts = [
        "Rows are selected by the frozen deterministic extrema rules over all OOF "
        "targets; there is no manual or favorable example selection. IDs appear "
        "only for audit and never enter a model matrix.\n\n"
    ]
    for heading, aliases in groups:
        rows: list[Mapping[str, Any]] = []
        for alias in aliases:
            rows = _records(qualitative.get(alias))
            if rows:
                break
        table_rows = []
        for row in rows:
            candidate = row.get("nested_failure_risk", row.get("candidate_risk"))
            isotonic = row.get("isotonic_failure_risk", row.get("isotonic_risk"))
            difference = row.get(
                "absolute_candidate_isotonic_difference",
                row.get("candidate_minus_isotonic_risk"),
            )
            table_rows.append(
                [
                    f"`{row.get('target_id', 'NA')}`",
                    row.get("object_id", "NA"),
                    f"`{row.get('physical_instance_id', 'NA')}`",
                    row.get("y_failure", "NA"),
                    _fmt(candidate),
                    _fmt(isotonic),
                    _fmt(difference),
                ]
            )
        parts.extend(
            [
                f"### {heading}\n\n",
                _table(
                    (
                        "Target",
                        "Object",
                        "Physical instance",
                        "Failure",
                        "Nested risk",
                        "Isotonic risk",
                        "Difference",
                    ),
                    table_rows,
                )
                if table_rows
                else "No qualifying rows.",
                "\n\n",
            ]
        )
    return "".join(parts)


def _ablation_gain(evaluation: Mapping[str, Any]) -> tuple[float | None, int, int]:
    ablations = _mapping(evaluation.get("feature_ablations"))
    score = _mapping(ablations.get("score_only"))
    score_disagreement = _mapping(ablations.get("score_disagreement"))
    score_aurc = _finite(_ablation_metrics(score).get("aurc"))
    disagreement_aurc = _finite(_ablation_metrics(score_disagreement).get("aurc"))
    score_folds = {
        int(row.get("outer_fold", -1)): _finite(
            _mapping(row.get("metrics")).get("aurc")
        )
        for row in _records(score.get("per_fold_metrics"))
    }
    disagreement_folds = {
        int(row.get("outer_fold", -1)): _finite(
            _mapping(row.get("metrics")).get("aurc")
        )
        for row in _records(score_disagreement.get("per_fold_metrics"))
    }
    common = sorted(set(score_folds).intersection(disagreement_folds))
    positive = sum(
        score_folds[fold] is not None
        and disagreement_folds[fold] is not None
        and float(disagreement_folds[fold]) < float(score_folds[fold])
        for fold in common
    )
    gain = (
        score_aurc - disagreement_aurc
        if score_aurc is not None and disagreement_aurc is not None
        else None
    )
    return gain, positive, len(common)


def _anti_cherry_answers(
    evaluation: Mapping[str, Any], decision: Mapping[str, Any]
) -> str:
    aggregate = _aggregate(evaluation)
    nested = _mapping(aggregate.get("NESTED_MULTIFEATURE"))
    raw = _mapping(aggregate.get("RAW_SCORE_RANK"))
    logistic = _mapping(aggregate.get("LOGISTIC_MULTIFEATURE"))
    tree = _mapping(aggregate.get("SHALLOW_TREE_MULTIFEATURE"))
    derived = _mapping(decision.get("derived_values"))
    robustness = _mapping(evaluation.get("object_jackknife"))
    stable = bool(
        _mapping(evaluation.get("decision_inputs")).get(
            "stable_disagreement_benefit",
            _mapping(decision.get("go_conditions")).get(
                "stable_disagreement_benefit", False
            ),
        )
    )
    ablation_gain, positive_folds, fold_count = _ablation_gain(evaluation)
    selections = _records(evaluation.get("model_selections"))
    families = Counter(
        str(row.get("nested_selected_family", "unknown")) for row in selections
    )
    iso_ci = derived.get("isotonic_aurc_improvement_bootstrap_ci")
    iso_ci_lower = _finite(_mapping(iso_ci).get("lower"))
    essentially_logistic = (
        _finite(logistic.get("aurc")) is not None
        and _finite(nested.get("aurc")) is not None
        and abs(float(logistic["aurc"]) - float(nested["aurc"])) <= 0.005
    )
    nested_auroc = _finite(nested.get("auroc"))
    grouped_signal_passed = nested_auroc is not None and nested_auroc >= 0.70
    low_risk_count = _finite(nested.get("predicted_risk_le_0_05_count"))
    low_risk_coverage = _finite(nested.get("predicted_risk_le_0_05_coverage"))
    low_risk_empirical = _finite(
        nested.get("predicted_risk_le_0_05_empirical_failure_rate")
    )
    useful_low_risk = (
        low_risk_count is not None
        and low_risk_count >= 30
        and low_risk_coverage is not None
        and low_risk_coverage >= 0.40
        and low_risk_empirical is not None
        and low_risk_empirical <= 0.05
    )
    object_driven = bool(robustness.get("object_driven", True))
    concentration_flags = _mapping(robustness.get("object_driven_flags"))
    one_object_concentrated = bool(
        concentration_flags.get("one_object_exceeds_40_percent_of_gain", False)
        or concentration_flags.get("removing_one_object_reverses_improvement", False)
    )
    one_instance_concentrated = bool(
        concentration_flags.get("one_instance_exceeds_20_percent_of_gain", False)
    )
    questions = [
        (
            "Does candidate disagreement add information beyond raw FoundationPose score?",
            f"{'Yes, stably' if stable else 'Not stably under the frozen rule'}. "
            f"Score plus disagreement changed AURC versus the score-only ablation by "
            f"{_fmt(ablation_gain)} and improved {positive_folds}/{fold_count} grouped "
            f"outer folds. Nested relative AURC improvement versus raw rank was "
            f"{_pct(derived.get('relative_aurc_improvement_vs_raw'))}.",
        ),
        (
            "Does it add information beyond score-only isotonic calibration?",
            f"{'Yes on grouped M2 OOF ranking' if iso_ci_lower is not None and iso_ci_lower > 0 else 'Not reliably'}. "
            f"Nested relative AURC improvement versus isotonic was "
            f"{_pct(derived.get('relative_aurc_improvement_vs_isotonic'))}; the paired "
            f"grouped-bootstrap absolute-gain CI was {_ci_text(iso_ci)}. "
            f"The CI {'excludes' if iso_ci_lower is not None and iso_ci_lower > 0 else 'does not exclude'} zero on the favorable side.",
        ),
        (
            "Does the signal survive grouping by physical instance?",
            f"{'Yes at the frozen weak-signal discrimination threshold' if grouped_signal_passed else 'Not at the frozen weak-signal discrimination threshold'}. "
            f"All reported predictions are physical-instance-grouped OOF predictions. "
            f"Nested AUROC was {_fmt(nested.get('auroc'))} and AURC was "
            f"{_fmt(nested.get('aurc'))}. This establishes only grouped-development "
            f"behavior, not independent holdout generalization.",
        ),
        (
            "Is the signal concentrated in one object?",
            f"{'Yes' if one_object_concentrated else 'No single object crossed the frozen threshold'}. "
            f"A physical instance {'did' if one_instance_concentrated else 'did not'} cross its threshold, "
            f"so the combined object/instance-driven flag was **{object_driven}**. "
            f"The frozen concentration flags were "
            f"`{json.dumps(_json_safe(robustness.get('object_driven_flags', {})), sort_keys=True)}`.",
        ),
        (
            "Does the model identify actual failures or merely low-score cases?",
            f"It captures outcome information beyond raw score, but the formal "
            f"discrimination gate still applies. The outcome-based OOF comparison "
            f"gives nested AUROC/AUPRC "
            f"{_fmt(nested.get('auroc'))}/{_fmt(nested.get('auprc'))} versus raw "
            f"{_fmt(raw.get('auroc'))}/{_fmt(raw.get('auprc'))}. The ablations and "
            f"qualitative error rows retain negative evidence; this diagnostic does "
            f"not prove a causal failure mechanism.",
        ),
        (
            "Does good AUROC translate into useful low-risk coverage?",
            f"{'Yes' if useful_low_risk else 'No'}. At predicted failure probability ≤5%, coverage was "
            f"{_pct(nested.get('predicted_risk_le_0_05_coverage'))}, empirical failure "
            f"was {_pct(nested.get('predicted_risk_le_0_05_empirical_failure_rate'))}, "
            f"and accepted count was {_fmt(nested.get('predicted_risk_le_0_05_count'), 0)}.",
        ),
        (
            "Is predicted 5% risk actually close to 5% empirical failure?",
            f"{'The support gate is sufficient' if useful_low_risk else 'The accepted set is too small to establish this'}. "
            f"The observed OOF rate among p≤.05 predictions was "
            f"{_pct(nested.get('predicted_risk_le_0_05_empirical_failure_rate'))}. "
            f"This is an empirical grouped-CV diagnostic without a conformal or "
            f"formal coverage guarantee.",
        ),
        (
            "Are tree-model gains stable enough to justify their complexity?",
            f"{'Yes for this audit' if decision.get('classification') == 'SIGNAL GO' else 'No for holdout progression under the frozen decision'}. "
            f"Nested inner selection chose families {dict(families)} across outer "
            f"folds. Fixed shallow-tree AURC was {_fmt(tree.get('aurc'))}, versus "
            f"fixed logistic AURC {_fmt(logistic.get('aurc'))}; the 0.005 tie rule "
            f"preferred logistic when differences were small.",
        ),
        (
            "Would logistic regression provide essentially the same result?",
            f"{'Yes under a 0.005 AURC comparison' if essentially_logistic else 'No under a 0.005 AURC comparison'}: fixed logistic AURC was "
            f"{_fmt(logistic.get('aurc'))}, nested AURC {_fmt(nested.get('aurc'))}.",
        ),
        (
            "Is there enough evidence to spend the untouched M3 holdout?",
            f"No. Formal classification is **{decision.get('classification', 'NO-GO')}**, "
            f"the strict report-information boundary failed, and holdout evaluation "
            f"remains unauthorized regardless of the numerical counterfactual.",
        ),
        (
            "What claims remain impossible without M3?",
            "Sealed-holdout generalization, final confidence calibration, a deployment "
            "risk bound, a conformal guarantee, and a final resume/paper claim all "
            "remain impossible.",
        ),
        (
            "Which features depend on multiple views and would not exist for single-view deployment?",
            "Pairwise translation/rotation/MSSD disagreement, candidate-to-output "
            "agreement, agreeing-view counts, strongest-view-versus-remaining "
            "disagreement, cross-view score dispersion, medoid support, and "
            "multi-view availability features require multiple candidates/views. "
            "Their single-view behavior was not established here.",
        ),
    ]
    return "\n\n".join(
        f"{index}. **{question}**\n\n   {answer}"
        for index, (question, answer) in enumerate(questions, start=1)
    )


def _compact_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _aggregate(evaluation)
    metric_names = (
        "target_count",
        "failure_count",
        "failure_prevalence",
        "auroc",
        "auprc",
        "auprc_lift_over_prevalence",
        "brier",
        "log_loss",
        "ece",
        "mce",
        "calibration_slope",
        "calibration_intercept",
        "aurc",
        "risk_at_coverage_0_50",
        "risk_at_coverage_0_80",
        "risk_at_coverage_0_90",
        "coverage_at_empirical_risk_0_05",
        "coverage_at_empirical_risk_0_10",
        "coverage_at_empirical_risk_0_20",
        "predicted_risk_le_0_05_count",
        "predicted_risk_le_0_05_coverage",
        "predicted_risk_le_0_05_empirical_failure_rate",
        "relative_aurc_improvement_vs_raw_score_rank",
        "relative_aurc_improvement_vs_score_isotonic",
    )
    return {
        method: {
            name: _mapping(aggregate.get(method)).get(name) for name in metric_names
        }
        for method in FORMAL_METHODS
    }


def _compact_ablations(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in _mapping(evaluation.get("feature_ablations")).items():
        record = _mapping(value)
        metrics = _ablation_metrics(record)
        result[name] = {
            "feature_count": len(record.get("feature_names", [])),
            "auroc": metrics.get("auroc"),
            "auprc": metrics.get("auprc"),
            "brier": metrics.get("brier"),
            "ece": metrics.get("ece"),
            "aurc": metrics.get("aurc"),
            "relative_aurc_improvement_vs_raw_score_rank": metrics.get(
                "relative_aurc_improvement_vs_raw_score_rank"
            ),
        }
    return result


def _source_hashes(
    repo_root: Path, report_path: Path, plot_paths: Sequence[Path]
) -> dict[str, str]:
    disclosure_path = (
        repo_root / "precomputed" / "m6_g0" / "access_boundary_disclosure.json"
    )
    if not disclosure_path.is_file():
        raise FileNotFoundError(
            f"M6-G0 access-boundary disclosure is missing: {disclosure_path}"
        )
    artifact_names = (
        "features.jsonl",
        "evaluator_rows.jsonl",
        "oof_predictions.jsonl",
        "ablation_oof_predictions.jsonl",
        "per_fold_metrics.jsonl",
        "aggregate_metrics.json",
        "grouped_bootstrap.json",
        "object_jackknife.json",
        "feature_ablations.json",
        "feature_contributions.json",
        "model_selections.json",
        "prediction_validation.json",
        "decision_inputs.json",
        "decision.json",
        "qualitative_error_table.json",
    )
    candidates = [
        *(repo_root / "artifacts" / "m6_g0" / name for name in artifact_names),
        *(
            repo_root / "scripts" / name
            for name in (
                "m6_g0_features.py",
                "m6_g0_labels.py",
                "m6_g0_analysis.py",
                "m6_g0_report.py",
                "run_m6_g0.py",
            )
        ),
        disclosure_path,
        report_path,
        *plot_paths,
    ]
    hashes: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        lowered = f"/{relative.lower()}/"
        if "/m3/" in lowered or "/m3_" in lowered or "/reports/m3" in lowered:
            continue
        hashes[relative] = _sha256_file(resolved)
    return dict(sorted(hashes.items()))


def _report_text(
    *,
    evaluation: Mapping[str, Any],
    label_support: Mapping[str, Any],
    decision: Mapping[str, Any],
    qualitative: Mapping[str, Any],
    chronology: Mapping[str, Any],
    protected_receipt: Mapping[str, Any],
) -> str:
    classification = str(decision.get("classification", "NO-GO"))
    counterfactual = decision.get(
        "counterfactual_classification_without_m3_boundary", "NA"
    )
    aggregate = _aggregate(evaluation)
    nested = _mapping(aggregate.get("NESTED_MULTIFEATURE"))
    fold_ids = sorted(
        {
            int(row.get("outer_fold", -1))
            for row in _records(evaluation.get("per_fold_metrics"))
            if row.get("outer_fold") is not None
        }
    )
    bootstrap = _mapping(evaluation.get("grouped_bootstrap"))
    receipt_sources = _mapping(protected_receipt.get("source_sha256"))
    source_count = len(receipt_sources)
    plot_blocks = "\n\n".join(
        f"### {index}. {caption}\n\n![{caption}]({filename})"
        for index, (filename, caption) in enumerate(
            zip(
                PLOT_FILENAMES,
                (
                    "Failure prevalence by object",
                    "Frozen raw score by outcome",
                    "Complete formal-method risk-coverage curves",
                    "Equal-frequency calibration reliability",
                    "AURC and Brier with grouped-bootstrap intervals",
                    "Frozen feature-ablation comparison",
                    "Candidate disagreement versus actual symmetry-aware error",
                    "Per-object nested-versus-isotonic AURC difference",
                    "Leave-one-object-out robustness",
                ),
                strict=True,
            ),
            start=1,
        )
    )
    return f"""# PoseLoop M6-G0 Existing-Data Confidence Signal Audit

> **Formal classification: {classification}. M3 holdout authorized: NO.**
>
> M5-G0 remained NO-GO. M6-G0 uses only M2 development data. No
> FoundationPose inference was run and no new pose estimates were generated.
> These are grouped cross-validation results, not sealed holdout evidence, and
> no conformal or formal risk guarantee is claimed.

## Executive result

The formal candidate was `NESTED_MULTIFEATURE`. Its pooled OOF AUROC was
**{_fmt(nested.get("auroc"))}**, AUPRC **{_fmt(nested.get("auprc"))}**, Brier
**{_fmt(nested.get("brier"))}**, ECE **{_fmt(nested.get("ece"))}**, and AURC
**{_fmt(nested.get("aurc"))}**. The frozen decision is **{classification}**.
The purely numerical counterfactual with the report-information gate removed
would be **{counterfactual}**; it is disclosed only as a diagnostic and does not
authorize holdout use.

## Non-negotiable evidence boundary

- M5-G0 remained NO-GO; this audit neither reopens nor upgrades it.
- The analysis used only the existing M2 development targets.
- M3 raw artifacts and protected hashes remained unchanged. No raw structured
  M3 target rows, labels, predictions, candidate values, object outcomes, or
  per-target metrics entered features, fitting, scoring, or selection.
- The strict report-information boundary **FAILED** because
  `reports/m3_active_budget.md` was accidentally displayed as documentation
  context before this audit. Displayed values were not used by M6-G0. This gate
  failure conservatively forces formal **NO-GO** and keeps the M3 holdout
  unauthorized.
- No FoundationPose inference was run; no new pose estimates were generated.
- Grouped cross-validation is development evidence, not a sealed holdout.
- No conformal guarantee, formal risk guarantee, deployment guarantee,
  FoundationPose improvement, or robot-control claim is made.

## Frozen output and correctness label

The calibrated output is the pre-existing five-view `symmetry_aware_medoid`,
implemented by `scripts/evaluate_m2.py:choose_medoid`. It receives the five
fixed-order M2 candidates transformed into the target-camera frame, excludes
failed/non-finite poses, minimizes mean bidirectional official symmetry-aware
MSSD normalized by official object diameter, and breaks ties by acquisition
rank then sample ID. It returns one existing candidate pose in
`T_target_camera_object` convention (metres), not a fused pose. It was frozen
before M6 labels and is the correct non-oracle deployable pose output to audit;
M6 did not compare final-pose methods.

The evaluator-only label is
`y_failure = int(not diagnostic_success.joint)` for the frozen medoid at view
budget 5. Correct means a finite pose with normalized symmetry-aware MSSD ≤
0.10 object diameter and symmetry-aware MSPD ≤ `10*r`, where
`r = target image width / 640`. Comparisons are inclusive. The label was not
tuned for class balance or predictability.

{
        _table(
            ("Support quantity", "Observed", "Frozen minimum", "Result"),
            (
                ("Targets", label_support.get("target_count", "NA"), "—", "INFO"),
                (
                    "Failures",
                    label_support.get("failure_count", "NA"),
                    "30",
                    "PASS"
                    if int(label_support.get("failure_count", 0)) >= 30
                    else "FAIL",
                ),
                (
                    "Successes",
                    label_support.get("success_count", "NA"),
                    "30",
                    "PASS"
                    if int(label_support.get("success_count", 0)) >= 30
                    else "FAIL",
                ),
                (
                    "Physical instances with failures",
                    label_support.get("physical_instances_with_failures", "NA"),
                    "10",
                    "PASS"
                    if int(label_support.get("physical_instances_with_failures", 0))
                    >= 10
                    else "FAIL",
                ),
                (
                    "Objects with failures",
                    label_support.get("objects_with_failures", "NA"),
                    "5",
                    "PASS"
                    if int(label_support.get("objects_with_failures", 0)) >= 5
                    else "FAIL",
                ),
                (
                    "Failure prevalence",
                    _pct(label_support.get("failure_prevalence")),
                    "—",
                    "INFO",
                ),
            ),
        )
    }

## Features, prohibitions, and causal separation

One inference-time row was constructed per M2 target. Frozen families were raw
score; score distribution; candidate/view availability; pairwise
symmetry-aware translation, rotation, and MSSD disagreement; candidate-to-output
agreement; and explicit view consistency. Model matrices excluded ground-truth
pose, target pose error, evaluator/BOP residuals, correctness/success/failure,
oracle-best-candidate information, object ID, physical-instance ID, scene ID,
target ID, split ID, M3 metadata, and all future/held-out outcomes. Identifiers
were used only for grouping, alignment, robustness, and qualitative audit.

## Formal methods and grouped nested cross-validation

{
        _table(
            ("Method", "Frozen role", "Fitting/selection boundary"),
            (
                (
                    "RAW_SCORE_RANK",
                    "Primary raw-confidence ranking baseline",
                    "No fitted model; frozen score direction",
                ),
                (
                    "SCORE_ISOTONIC",
                    "One-dimensional score calibration",
                    "Fit on outer training only",
                ),
                (
                    "LOGISTIC_MULTIFEATURE",
                    "Standardized L2 logistic regression",
                    "Training-fold median imputation; inner grouped C grid",
                ),
                (
                    "SHALLOW_TREE_MULTIFEATURE",
                    "Depth≤3 random forest with ≤100 trees",
                    "Inner grouped grid; training-only inner-OOF sigmoid calibration",
                ),
                (
                    "NESTED_MULTIFEATURE",
                    "Formal candidate",
                    "Per outer fold, inner grouped AURC family/parameter selection; logistic preferred within 0.005",
                ),
            ),
        )
    }

There were **{len(fold_ids)}** frozen outer folds (`{fold_ids}`), grouped by
physical instance and stratified by failure label plus object where feasible.
Every formal method used identical outer folds; preprocessing, imputation,
hyperparameter selection, and calibration remained inside outer training data.
Fold-manifest hash: `{
        evaluation.get(
            "fold_manifest_hash", chronology.get("fold_manifest_sha256", "NA")
        )
    }`.

## Full pooled out-of-fold results

{_pooled_metric_tables(evaluation)}

## Fold-level OOF results

{_per_fold_table(evaluation)}

## Training-only model selections

{_model_selection_table(evaluation)}

The fixed logistic grid was `C ∈ {{0.01, 0.1, 1, 10}}`. The fixed shallow
random-forest grid used depth 2 or 3, 50 or 100 trees, and minimum leaf size 10
or 20. Outer-test labels were never used for model or hyperparameter selection.

## Feature ablations

{_ablation_table(evaluation)}

## Feature contributions

{_contribution_tables(evaluation)}

## Object and physical-instance robustness

{_robustness_tables(evaluation)}

## Frozen decision-threshold audit

{_decision_tables(decision)}

## Deterministic qualitative error audit

{_qualitative_table(qualitative)}

## Anti-cherry-picking questions

{_anti_cherry_answers(evaluation, decision)}

## Figures

All plots use every applicable row or a declared deterministic aggregation; no
favorable examples are selected. Error bars are percentile 95% intervals from
**{bootstrap.get("n_resamples", "NA")}** grouped bootstrap resamples over whole
physical instances.

{plot_blocks}

## Reproduction and validation

From the repository root, with the documented XYZ-IBD and BOP Toolkit inputs:

```powershell
python -B scripts/build_m6_g0_manifests.py
python -B scripts/run_m6_g0.py
python -B scripts/validate_m6_g0.py
python -B scripts/validate_m6_g0.py --check
python -m pytest tests
python -m compileall scripts tests
python -m ruff check --ignore E402 scripts tests
git diff --check
```

`python -B scripts/run_m6_g0.py` writes the raw formal-method and ablation OOF
predictions, fold metrics, pooled metrics, grouped-bootstrap intervals,
ablations, robustness artifacts,
this report, the nine figures, and
[`precomputed/m6_g0/results.json`](../../precomputed/m6_g0/results.json).
The bare repository-root pytest command is intentionally not prescribed when it
would collect inaccessible vendored third-party suites; `pytest tests` is the
first-party validation boundary. The protected M1–M5 scripts intentionally add
their local import path before first-party imports, so the whole-tree Ruff check
uses the explicit pre-existing `E402` exception shown above.

## Receipts and limitations

The static protocol receipt records **{
        protected_receipt.get("static_frozen_file_count", "NA")
    }**
frozen files and **{source_count}** source hashes, tied to starting commit
`{
        protected_receipt.get(
            "starting_commit", "315ca6036df00aeb3a76bd639dae7dca0ac4bc86"
        )
    }`.
The audit remains limited by development-only grouped CV, historical M2 candidate
grouping, a single frozen output definition, empirical rather than guaranteed
low-risk coverage, finite object/instance support, and the failed strict
report-information boundary. The formal classification is **{classification}**;
M3 holdout use is unauthorized.
"""


def render_m6_g0_outputs(
    *,
    repo_root: Path | str,
    feature_rows: Any,
    evaluator_rows: Any,
    evaluation: Mapping[str, Any],
    label_support: Mapping[str, Any],
    decision: Mapping[str, Any],
    qualitative: Mapping[str, Any],
    chronology: Mapping[str, Any],
    protected_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Render the complete tracked M6-G0 publication bundle.

    All model fitting and metric computation must already be complete. The
    returned paths are absolute and stable so the runner can hash the tracked
    publication set without discovering unrelated files.
    """

    root = Path(repo_root).resolve()
    features = _records(feature_rows)
    evaluators = _records(evaluator_rows)
    evaluation_map = _mapping(evaluation)
    support_map = _mapping(label_support)
    decision_map = _mapping(decision)
    qualitative_map = _mapping(qualitative)
    chronology_map = _mapping(chronology)
    receipt_map = _mapping(protected_receipt)
    classification = str(decision_map.get("classification", ""))
    if classification not in {
        "SIGNAL GO",
        "WEAK SIGNAL / HOLDOUT NOT AUTHORIZED",
        "NO-GO",
    }:
        raise ValueError("decision must contain one frozen M6-G0 classification")
    if not features or not evaluators:
        raise ValueError("feature_rows and evaluator_rows must be non-empty")

    report_dir = root / "reports" / "m6_g0"
    report_path = report_dir / "existing_data_confidence_signal_audit.md"
    precomputed_path = root / "precomputed" / "m6_g0" / "results.json"
    disclosure_path = root / "precomputed" / "m6_g0" / "access_boundary_disclosure.json"
    if not disclosure_path.is_file():
        raise FileNotFoundError(
            f"M6-G0 access-boundary disclosure is missing: {disclosure_path}"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    precomputed_path.parent.mkdir(parents=True, exist_ok=True)

    _configure_style()
    plot_paths = _render_plots(
        report_dir,
        features,
        evaluators,
        evaluation_map,
        support_map,
    )
    report = _report_text(
        evaluation=evaluation_map,
        label_support=support_map,
        decision=decision_map,
        qualitative=qualitative_map,
        chronology=chronology_map,
        protected_receipt=receipt_map,
    )
    _write_text_atomic(report_path, report)
    source_hashes = _source_hashes(root, report_path, plot_paths)

    robustness = _mapping(evaluation_map.get("object_jackknife"))
    bootstrap = _mapping(evaluation_map.get("grouped_bootstrap"))
    compact_result = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_precomputed_result",
        "title": "Existing-Data Confidence Signal Audit",
        "classification": classification,
        "counterfactual_classification_without_m3_boundary": decision_map.get(
            "counterfactual_classification_without_m3_boundary"
        ),
        "formal_candidate": "NESTED_MULTIFEATURE",
        "m3_holdout_authorized": False,
        "scope": {
            "data": "M2 development only",
            "m5_g0_remained_no_go": True,
            "foundationpose_inference_run": False,
            "new_pose_estimates_generated": False,
            "grouped_cross_validation_not_holdout": True,
            "conformal_or_formal_risk_guarantee": False,
            "strict_report_information_boundary_passed": False,
            "raw_structured_m3_access": False,
        },
        "label_support": {
            key: support_map.get(key)
            for key in (
                "target_count",
                "failure_count",
                "success_count",
                "failure_prevalence",
                "physical_instance_count",
                "physical_instances_with_failures",
                "object_count",
                "objects_with_failures",
                "criteria",
                "passed",
            )
        },
        "formal_method_metrics": _compact_metrics(evaluation_map),
        "grouped_bootstrap": {
            "resampling_unit": bootstrap.get("resampling_unit"),
            "paired": bootstrap.get("paired"),
            "n_resamples": bootstrap.get("n_resamples"),
            "seed": bootstrap.get("seed"),
            "confidence_level": bootstrap.get("confidence_level"),
            "interval": bootstrap.get("interval"),
            "method_intervals": bootstrap.get("method_intervals"),
            "paired_differences": bootstrap.get("paired_differences"),
        },
        "feature_ablations": _compact_ablations(evaluation_map),
        "robustness": {
            "aggregate_candidate_aurc": robustness.get("aggregate_candidate_aurc"),
            "aggregate_isotonic_aurc": robustness.get("aggregate_isotonic_aurc"),
            "aggregate_absolute_aurc_gain": robustness.get(
                "aggregate_absolute_aurc_gain"
            ),
            "object_driven_flags": robustness.get("object_driven_flags"),
            "object_driven": robustness.get("object_driven"),
        },
        "decision": {
            key: decision_map.get(key)
            for key in (
                "classification",
                "reasons",
                "counterfactual_classification_without_m3_boundary",
                "m3_access_boundary_passed",
                "signal_go_forbidden_by_m3_boundary",
                "go_conditions",
                "weak_conditions",
                "hard_validity_conditions",
                "derived_values",
                "rules_version",
            )
        },
        "fold_manifest_hash": evaluation_map.get(
            "fold_manifest_hash", chronology_map.get("fold_manifest_sha256")
        ),
        "qualitative_audit": qualitative_map,
        "report": report_path.relative_to(root).as_posix(),
        "plots": [path.relative_to(root).as_posix() for path in plot_paths],
        "source_hashes": source_hashes,
    }
    _write_json_atomic(precomputed_path, compact_result)

    tracked_paths = [disclosure_path, report_path, precomputed_path, *plot_paths]
    return {
        "report": str(report_path.resolve()),
        "plots": [str(path.resolve()) for path in plot_paths],
        "tracked_paths": [str(path.resolve()) for path in tracked_paths],
    }


__all__ = ["render_m6_g0_outputs"]
