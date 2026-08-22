#!/usr/bin/env python3
"""Render the PoseLoop release summary from compact precomputed aggregates."""

from __future__ import annotations

import argparse
import difflib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
RELEASE = "v1.0.0"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=repo_root / "precomputed" / RELEASE / "results.json",
        help="Path to the compact aggregate result bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "reports" / "release_summary.md",
        help="Markdown report destination.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that --output is current without writing it.",
    )
    return parser.parse_args()


def _number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: expected {expected}, got {actual}")


def _contains_zero(interval: Sequence[Any]) -> bool:
    if len(interval) != 2:
        raise ValueError(f"Expected a two-element interval, got {interval!r}")
    lower = _number(interval[0], "interval lower")
    upper = _number(interval[1], "interval upper")
    if lower > upper:
        raise ValueError(f"Reversed interval: {interval!r}")
    return lower <= 0.0 <= upper


def _validate_source_path(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"{label} must be repository-relative: {value!r}")


def validate_bundle(bundle: Mapping[str, Any], raw_text: str) -> None:
    forbidden_tokens = (
        "C:" + "/Users/",
        "C:" + "\\Users\\",
        "/mnt/c/" + "Users/",
        "file:" + "//",
    )
    hits = [token for token in forbidden_tokens if token in raw_text]
    if hits:
        raise ValueError(f"Bundle contains local absolute path tokens: {hits}")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION}")
    if bundle.get("release") != RELEASE:
        raise ValueError(f"Expected release={RELEASE!r}")
    commit = bundle.get("result_source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("result_source_commit must be a full Git SHA")

    phases = bundle["phases"]
    m1 = phases["m1"]
    m2 = phases["m2"]
    m3 = phases["m3"]
    m4 = phases["m4"]
    for name, phase in phases.items():
        _validate_source_path(phase["source_report"], f"{name}.source_report")

    _close(
        _number(m1["combined"], "M1 combined"),
        (_number(m1["ar_mssd"], "M1 AR_MSSD") + _number(m1["ar_mspd"], "M1 AR_MSPD"))
        / 2.0,
        "M1 combined definition",
    )
    _close(
        _number(m2["baseline"]["combined"], "M2 baseline combined"),
        _number(m1["combined"], "M1 combined"),
        "M2 target-only reproduction of M1",
    )
    _close(
        _number(m2["selected"]["combined"], "M2 selected combined"),
        (
            _number(m2["selected"]["ar_mssd"], "M2 selected AR_MSSD")
            + _number(m2["selected"]["ar_mspd"], "M2 selected AR_MSPD")
        )
        / 2.0,
        "M2 combined definition",
    )
    m2_gain_pp = 100.0 * (
        _number(m2["selected"]["combined"], "M2 selected combined")
        - _number(m2["baseline"]["combined"], "M2 baseline combined")
    )
    _close(
        m2_gain_pp,
        _number(m2["reported_gain_pp"], "M2 reported gain"),
        "M2 +8.40 pp result",
    )

    fast = m3["active_fast"]
    _close(
        _number(fast["combined"], "M3 fast combined")
        - _number(fast["matched_random_combined"], "M3 fast random combined"),
        _number(fast["gain_vs_matched_random"], "M3 fast random gain"),
        "M3 fast comparison",
    )
    if not _contains_zero(fast["paired_95_ci_vs_matched_random"]):
        raise ValueError("M3 fast paired interval must include zero")
    conservative = m3["active_conservative"]
    if (
        _number(
            conservative["paired_95_ci_vs_matched_random"][1],
            "M3 conservative CI upper",
        )
        >= 0.0
    ):
        raise ValueError("M3 conservative interval must remain below zero")

    scores = m4["scores"]
    gains = m4["learned_gains"]
    learned = _number(scores["learned_voi"], "M4 learned score")
    for baseline_name, gain_name in (
        ("target_only", "vs_target_only"),
        ("fixed_first", "vs_fixed_first"),
        ("random_candidate", "vs_random_candidate"),
    ):
        _close(
            learned - _number(scores[baseline_name], f"M4 {baseline_name} score"),
            _number(gains[gain_name], f"M4 {gain_name}"),
            f"M4 learned {gain_name}",
        )
        if not _contains_zero(m4["paired_95_intervals"][gain_name]):
            raise ValueError(f"M4 {gain_name} paired interval must include zero")
    if int(m4["low_visibility_holdout_target_count"]) != 0:
        raise ValueError("M4 low-visibility holdout count must remain zero")


def percent(value: Any) -> str:
    return f"{100.0 * _number(value, 'percentage'):.2f}%"


def points(value: Any) -> str:
    return f"{100.0 * _number(value, 'percentage-point difference'):+.2f} pp"


def interval_points(interval: Sequence[Any]) -> str:
    return f"[{points(interval[0]).removesuffix(' pp')}, {points(interval[1])}]"


def report_link(phase: Mapping[str, Any]) -> str:
    return PurePosixPath(str(phase["source_report"])).name


def render_report(bundle: Mapping[str, Any]) -> str:
    phases = bundle["phases"]
    m0 = phases["m0"]
    m1 = phases["m1"]
    m2 = phases["m2"]
    m3 = phases["m3"]
    m4 = phases["m4"]
    fast = m3["active_fast"]
    conservative = m3["active_conservative"]
    scores = m4["scores"]
    gains = m4["learned_gains"]
    intervals = m4["paired_95_intervals"]

    return f"""<!-- Generated by scripts/render_precomputed_report.py. -->

# PoseLoop {bundle["release"]} result summary

This report is rendered from
[`precomputed/{bundle["release"]}/results.json`](../precomputed/{bundle["release"]}/results.json).
It contains aggregate results only and requires no dataset, weights, GPU, or
third-party runtime.

## Results at a glance

| Phase | Scope | Result | Interpretation |
|---|---|---|---|
| M0 | {m0["sample_count"]} smoke sample | Runtime/checkpoint/evaluator evidence | No accuracy claim |
| M1 | {m1["target_count"]} fixed targets | Combined `{percent(m1["combined"])}` | Single-view diagnostic baseline |
| **M2** | Same {m2["target_count"]} targets, `k={m2["view_budget"]}` | **`{m2["selector"]}` `{percent(m2["selected"]["combined"])}`, {points(m2["selected"]["combined"] - m2["baseline"]["combined"])}** | **Main positive diagnostic** |
| M3 | {m3["holdout_target_count"]}-instance M2-disjoint holdout | Fast `{percent(fast["combined"])}` at {fast["mean_views"]:.2f} views, but {points(fast["gain_vs_matched_random"])} vs matched random | Budget saving; allocation hypothesis not supported |
| M4 | Same {m4["holdout_target_count"]}-instance holdout, one extra view | Learned `{percent(scores["learned_voi"])}`: {points(gains["vs_target_only"])} vs target-only | No end-to-end gain; ranking advantage unresolved |

## M2: main positive diagnostic

On the same fixed {m2["target_count"]}-target oracle-mask subset as M1,
`{m2["selector"]}` at `k={m2["view_budget"]}` changes:

| Metric | Target-only | M2 selected | Change |
|---|---:|---:|---:|
| AR_MSSD | {percent(m2["baseline"]["ar_mssd"])} | {percent(m2["selected"]["ar_mssd"])} | {points(m2["selected"]["ar_mssd"] - m2["baseline"]["ar_mssd"])} |
| AR_MSPD | {percent(m2["baseline"]["ar_mspd"])} | {percent(m2["selected"]["ar_mspd"])} | {points(m2["selected"]["ar_mspd"] - m2["baseline"]["ar_mspd"])} |
| **Combined** | **{percent(m2["baseline"]["combined"])}** | **{percent(m2["selected"]["combined"])}** | **{points(m2["selected"]["combined"] - m2["baseline"]["combined"])}** |

The selector uses {m2["selector_input"]}. The acquisition order is
{m2["acquisition_order"]}. This is a positive multi-view diagnostic, not an
independent holdout result or deployable selection policy. See the
[full M2 report]({report_link(m2)}).

## M3: budget hypothesis not supported

The fast policy reaches {percent(fast["combined"])} at
{fast["mean_views"]:.2f} mean views and reduces recorded registration latency by
{percent(fast["registration_latency_reduction_vs_fixed_k5"])} relative to fixed
`k=5`. It is nevertheless {points(fast["gain_vs_matched_random"])} below its
matched-random control; the paired 95% interval is
`{interval_points(fast["paired_95_ci_vs_matched_random"])}`.

The conservative policy is {points(conservative["gain_vs_matched_random"])}
below matched random, with paired 95% interval
`{interval_points(conservative["paired_95_ci_vs_matched_random"])}`. Thus M3
supports a budget/latency trade-off, but not a learned allocation advantage.
See the [full M3 report]({report_link(m3)}).

## M4: ranking hypothesis unresolved and no end-to-end gain

Learned VOI scores {percent(scores["learned_voi"])} macro combined:

| Comparison | Point estimate | Paired 95% interval |
|---|---:|---:|
| vs target-only | {points(gains["vs_target_only"])} | `{interval_points(intervals["vs_target_only"])}` |
| vs fixed-first | {points(gains["vs_fixed_first"])} | `{interval_points(intervals["vs_fixed_first"])}` |
| vs random candidate | {points(gains["vs_random_candidate"])} | `{interval_points(intervals["vs_random_candidate"])}` |

Every interval includes zero, and learned VOI remains below target-only. The
holdout contains {m4["low_visibility_holdout_target_count"]} low-visibility
targets, so no low-visibility claim can be tested. See the
[full M4 report]({report_link(m4)}).

## Scope and provenance

The reported combined metric is `{bundle["metric"]["name"]}`:
{bundle["metric"]["definition"]}

- Known object IDs and ground-truth visible masks are used.
- These are diagnostic subset results, not official BOP AP or full-dataset
  performance.
- M1/M2 share the same targets. M3/M4 share a
  physical-instance-disjoint-but-not-scene-disjoint holdout.
- M3/M4 use macro-object aggregation because holdout object counts are unequal.

Data attribution and third-party terms are documented in
[`LICENSES.md`](../LICENSES.md).

Frozen experimental source commit:
`{bundle["result_source_commit"]}`.
"""


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    bundle_path = args.bundle.resolve()
    output_path = args.output.resolve()
    raw_text = bundle_path.read_text(encoding="utf-8")
    bundle = json.loads(raw_text)
    validate_bundle(bundle, raw_text)
    rendered = render_report(bundle)

    repo_root = Path(__file__).resolve().parents[1]
    display_output = (
        output_path.relative_to(repo_root)
        if output_path.is_relative_to(repo_root)
        else output_path
    )
    if args.check:
        if not output_path.is_file():
            raise FileNotFoundError(f"Missing rendered report: {display_output}")
        existing = output_path.read_text(encoding="utf-8")
        if existing != rendered:
            diff = "".join(
                difflib.unified_diff(
                    existing.splitlines(keepends=True),
                    rendered.splitlines(keepends=True),
                    fromfile=str(display_output),
                    tofile=f"{display_output} (expected)",
                )
            )
            print(diff, end="")
            return 1
        print(f"PASS: {display_output} matches {bundle_path.name}")
        return 0

    write_atomic(output_path, rendered)
    print(f"Rendered: {display_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
