#!/usr/bin/env python3
"""Freeze the immutable protocol inputs for PoseLoop M6-G0."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
EXPECTED_STARTING_BRANCH = "master"
EXPECTED_STARTING_COMMIT = "315ca6036df00aeb3a76bd639dae7dca0ac4bc86"
EXPECTED_PROTECTED_ENTRY_COUNT = 182
EXPECTED_PROTECTED_TOTAL_BYTES = 554_872_235
EXPECTED_PROTECTED_ROOT_SHA256 = (
    "85ae92db17682612f942400a4aab35205ca15a429ee97cbafb2bbafda51d49e8"
)
EXPECTED_PROTECTED_RECEIPT_SHA256 = (
    "8592728f7732f64f3e5660f2e0ac3bdd7eea275c2212a98b45d9455be9871411"
)
EXPECTED_PROTECTED_RAW_ROOTS = (
    "artifacts/m2",
    "artifacts/m3",
    "artifacts/m4",
    "artifacts/m5_g0*",
)
AUTHORIZED_M6_MODIFIED_PATHS = ("README.md",)
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pypng",
)

ARTIFACT_DIR = Path("artifacts/m6_g0")
FROZEN_DIR = ARTIFACT_DIR / "frozen"
PROTECTED_RECEIPT = Path("precomputed/m6_g0/protected_m2_m5_hashes.json")

STATIC_FROZEN_FILENAMES = (
    "protected_hash_reference.json",
    "source_audit.json",
    "target_output_definition.json",
    "label_definition.json",
    "feature_specification.json",
    "feature_schema.json",
    "evaluator_schema.json",
    "fold_protocol.json",
    "model_grid.json",
    "decision_rules.json",
    "input_allowlist.json",
)
COPIED_FROZEN_FILENAMES = ("protected_m2_m5_hashes.json",)
RUNNER_OWNED_FROZEN_FILENAMES = ("fold_manifest.json",)
FROZEN_FILENAMES = (
    *COPIED_FROZEN_FILENAMES,
    *STATIC_FROZEN_FILENAMES,
    "manifest_hashes.json",
)

FORMAL_METHODS = (
    "RAW_SCORE_RANK",
    "SCORE_ISOTONIC",
    "LOGISTIC_MULTIFEATURE",
    "SHALLOW_TREE_MULTIFEATURE",
    "NESTED_MULTIFEATURE",
)
PROHIBITED_MODEL_FIELDS = (
    "ground_truth_pose",
    "gt_model_to_camera_pose_m",
    "target_pose_error",
    "translation_error_mm",
    "raw_rotation_error_degrees",
    "normalized_mssd",
    "mspd_px",
    "sample_ar_mssd",
    "sample_ar_mspd",
    "diagnostic_success",
    "correct",
    "success",
    "failure",
    "y_failure",
    "oracle_best_candidate",
    "object_id",
    "physical_instance_id",
    "scene_id",
    "target_id",
    "target_sample_id",
    "group_id",
    "split_id",
    "dataset_split",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing freeze without writing anything.",
    )
    return parser.parse_args(argv)


def _git(repo_root: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_binary(path: Path) -> str:
    """Hash raw bytes; this is the only operation permitted on M3 artifacts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return text.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def freeze_bytes(path: Path, payload: bytes) -> None:
    """Atomically write once, accepting an existing byte-identical freeze."""

    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to overwrite non-identical frozen file: {path}"
            )
        return
    _write_bytes_atomic(path, payload)


def freeze_json(path: Path, payload: Any) -> None:
    """Atomically freeze JSON, or prove an existing file is byte-identical."""

    freeze_bytes(path, _json_bytes(payload))


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} in {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(
                            f"Non-finite JSON constant {value!r} at "
                            f"{path}:{line_number}"
                        )
                    ),
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _raw_protected_paths(repo_root: Path) -> set[str]:
    roots = [
        repo_root / "artifacts" / "m2",
        repo_root / "artifacts" / "m3",
        repo_root / "artifacts" / "m4",
        *sorted((repo_root / "artifacts").glob("m5_g0*")),
    ]
    paths: set[str] = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Protected raw root is missing: {root}")
        if root.is_file():
            paths.add(root.relative_to(repo_root).as_posix())
            continue
        for path in root.rglob("*"):
            if path.is_file():
                paths.add(path.relative_to(repo_root).as_posix())
    return paths


def validate_protected_baseline(repo_root: Path) -> dict[str, Any]:
    """Validate the immutable tracked receipt and every protected byte stream."""

    receipt_path = repo_root / PROTECTED_RECEIPT
    if _sha256_binary(receipt_path) != EXPECTED_PROTECTED_RECEIPT_SHA256:
        raise ValueError("Tracked M2-M5 protected receipt bytes changed")
    receipt = _strict_json(receipt_path)
    exact_fields = {
        "record_type": "m6_g0_protected_m2_m5_hashes",
        "schema_version": SCHEMA_VERSION,
        "starting_branch": EXPECTED_STARTING_BRANCH,
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "file_count": EXPECTED_PROTECTED_ENTRY_COUNT,
        "total_bytes": EXPECTED_PROTECTED_TOTAL_BYTES,
        "root_sha256": EXPECTED_PROTECTED_ROOT_SHA256,
        "protected_raw_roots": list(EXPECTED_PROTECTED_RAW_ROOTS),
    }
    for key, expected in exact_fields.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"Protected receipt {key} changed: {receipt.get(key)!r} != {expected!r}"
            )
    if receipt.get("m3_handling") != (
        "SHA-256 over binary bytes only; no M3 structured content was parsed "
        "to create this receipt."
    ):
        raise ValueError("Protected receipt no longer preserves the M3 byte-only rule")

    entries = receipt.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_PROTECTED_ENTRY_COUNT:
        raise ValueError("Protected receipt entry count is malformed")
    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Protected receipt contains a non-object entry")
        relative = str(entry.get("path", "")).replace("\\", "/")
        if not relative or relative in indexed or relative.startswith("/"):
            raise ValueError(f"Invalid protected path: {relative!r}")
        indexed[relative] = entry

    tracked_at_start = set(
        _git(
            repo_root,
            "ls-tree",
            "-r",
            "--name-only",
            EXPECTED_STARTING_COMMIT,
        ).splitlines()
    )
    expected_paths = tracked_at_start | _raw_protected_paths(repo_root)
    if set(indexed) != expected_paths:
        missing = sorted(expected_paths - set(indexed))
        unexpected = sorted(set(indexed) - expected_paths)
        raise ValueError(
            "Protected receipt path set changed: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    baseline_total = 0
    m3_entry_count = 0
    for relative in sorted(indexed):
        entry = indexed[relative]
        path = repo_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Protected file is missing: {path}")
        baseline_total += int(entry["bytes"])
        if relative == "README.md":
            baseline_bytes = subprocess.run(
                ["git", "show", f"{EXPECTED_STARTING_COMMIT}:README.md"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            ).stdout
            if len(baseline_bytes) != entry.get("bytes") or hashlib.sha256(
                baseline_bytes
            ).hexdigest() != entry.get("sha256"):
                raise ValueError("Starting-commit README does not match the receipt")
            current_bytes = path.read_bytes()
            if current_bytes != baseline_bytes:
                if not current_bytes.startswith(baseline_bytes):
                    raise ValueError(
                        "README modification is not an append-only M6 reproduction section"
                    )
                appended = current_bytes[len(baseline_bytes) :].decode("utf-8")
                required = (
                    "M6-G0",
                    "python -B scripts/build_m6_g0_manifests.py",
                    "python -B scripts/run_m6_g0.py",
                    "python -B scripts/validate_m6_g0.py",
                    "python -B scripts/validate_m6_g0.py --check",
                )
                missing_markers = [
                    marker for marker in required if marker not in appended
                ]
                if missing_markers:
                    raise ValueError(
                        "Appended README M6 section omits required markers: "
                        f"{missing_markers}"
                    )
            continue
        byte_count = path.stat().st_size
        digest = _sha256_binary(path)
        if byte_count != entry.get("bytes") or digest != entry.get("sha256"):
            raise ValueError(f"Protected file changed: {relative}")
        m3_entry_count += relative.startswith("artifacts/m3/")
    if baseline_total != EXPECTED_PROTECTED_TOTAL_BYTES:
        raise ValueError("Protected receipt baseline total byte count changed")

    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            EXPECTED_STARTING_COMMIT,
            "HEAD",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("M6-G0 starting commit is not an ancestor of HEAD")
    return {
        "passed": True,
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "entry_count": len(indexed),
        "total_bytes": baseline_total,
        "root_sha256": EXPECTED_PROTECTED_ROOT_SHA256,
        "receipt_sha256": EXPECTED_PROTECTED_RECEIPT_SHA256,
        "m3_entry_count": m3_entry_count,
        "m3_hashes_unchanged": True,
        "m3_access_mode": "binary_sha256_only",
        "authorized_m6_modified_paths": list(AUTHORIZED_M6_MODIFIED_PATHS),
        "readme_authorization": "append-only M6 reproduction section",
        "all_m1_m5_evidence_unchanged": True,
    }


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _m2_source_audit(repo_root: Path) -> dict[str, Any]:
    groups_path = repo_root / "artifacts" / "m2" / "groups.jsonl"
    metrics_path = repo_root / "artifacts" / "m2" / "metrics.jsonl"
    groups = _strict_jsonl(groups_path)
    metrics = _strict_jsonl(metrics_path)
    if len(groups) != 300:
        raise ValueError(f"Expected 300 M2 groups, found {len(groups)}")

    by_target: dict[str, dict[str, Any]] = {}
    view_counts: Counter[int] = Counter()
    object_ids: set[int] = set()
    physical_instances: set[str] = set()
    for group in groups:
        target_id = str(group.get("target_sample_id"))
        if target_id in by_target:
            raise ValueError(f"Duplicate M2 target group: {target_id}")
        views = group.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError(f"M2 target has no candidate views: {target_id}")
        sample_ids = [str(view.get("sample_id")) for view in views]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"M2 target has duplicate candidate rows: {target_id}")
        scene_object_instance = {
            (
                int(view.get("scene_id")),
                int(view.get("object_id")),
                int(view.get("gt_instance_index")),
            )
            for view in views
        }
        if len(scene_object_instance) != 1:
            raise ValueError(
                f"M2 candidate rows do not share one physical instance: {target_id}"
            )
        association = group.get("oracle_association", {})
        physical_id = str(association.get("track_id", ""))
        if not physical_id:
            raise ValueError(f"M2 group lacks a physical-instance ID: {target_id}")
        by_target[target_id] = group
        view_counts[len(views)] += 1
        object_ids.add(int(group.get("object_id")))
        physical_instances.add(physical_id)

    formal_rows = [
        row
        for row in metrics
        if row.get("method") == "symmetry_aware_medoid"
        and row.get("requested_view_budget") == 5
    ]
    if len(formal_rows) != 300:
        raise ValueError(
            "Expected one k=5 symmetry-aware-medoid result per M2 target, "
            f"found {len(formal_rows)}"
        )
    formal_by_target = {str(row.get("target_sample_id")): row for row in formal_rows}
    if len(formal_by_target) != 300 or set(formal_by_target) != set(by_target):
        raise ValueError("M2 formal target rows do not join one-to-one with groups")

    label_counts: Counter[int] = Counter()
    by_object: dict[int, Counter[int]] = defaultdict(Counter)
    by_instance: dict[str, Counter[int]] = defaultdict(Counter)
    for target_id, row in formal_by_target.items():
        flags = row.get("diagnostic_success")
        if not isinstance(flags, dict) or not isinstance(flags.get("joint"), bool):
            raise ValueError(f"M2 row lacks frozen joint correctness: {target_id}")
        failure = int(not flags["joint"])
        group = by_target[target_id]
        object_id = int(group["object_id"])
        physical_id = str(group["oracle_association"]["track_id"])
        label_counts[failure] += 1
        by_object[object_id][failure] += 1
        by_instance[physical_id][failure] += 1

    failures = label_counts[1]
    successes = label_counts[0]
    failure_instances = sum(counts[1] > 0 for counts in by_instance.values())
    failure_objects = sum(counts[1] > 0 for counts in by_object.values())
    support_passed = (
        failures >= 30
        and successes >= 30
        and failure_instances >= 10
        and failure_objects >= 5
    )
    if not support_passed:
        raise ValueError("Frozen M2 label does not meet the M6-G0 support gate")
    return {
        "target_count": len(by_target),
        "candidate_rows_per_target": _counter_dict(view_counts),
        "physical_instance_count": len(physical_instances),
        "object_count": len(object_ids),
        "object_ids": sorted(object_ids),
        "candidate_target_join_one_to_one": True,
        "candidate_rows_share_physical_instance": True,
        "physical_instance_grouping_complete": True,
        "formal_output_row_count": len(formal_rows),
        "label_counts": {"failure": failures, "success": successes},
        "failure_prevalence": failures / len(formal_rows),
        "physical_instances_containing_failures": failure_instances,
        "objects_containing_failures": failure_objects,
        "label_support_passed": support_passed,
        "counts_by_object": {
            str(object_id): {
                "failure": counts[1],
                "success": counts[0],
            }
            for object_id, counts in sorted(by_object.items())
        },
        "source_sha256": {
            "artifacts/m2/groups.jsonl": _sha256_binary(groups_path),
            "artifacts/m2/metrics.jsonl": _sha256_binary(metrics_path),
            "artifacts/m2/view_predictions.jsonl": _sha256_binary(
                repo_root / "artifacts" / "m2" / "view_predictions.jsonl"
            ),
        },
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Feature contract contains unsupported value: {type(value)}")


def _feature_contract() -> tuple[dict[str, Any], list[str], dict[str, str]]:
    try:
        module = importlib.import_module("m6_g0_features")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scripts/m6_g0_features.py must exist before freezing M6-G0"
        ) from exc
    raw_spec = getattr(module, "FEATURE_SPEC", None)
    if not isinstance(raw_spec, Mapping) or not raw_spec:
        raise ValueError("m6_g0_features.FEATURE_SPEC must be a non-empty mapping")
    specification = _plain(raw_spec)
    if "features" in specification and isinstance(specification["features"], Mapping):
        feature_names = list(specification["features"])
    else:
        feature_names = list(specification)
    raw_families = getattr(
        module,
        "FEATURE_FAMILY",
        getattr(module, "FEATURE_FAMILIES", {}),
    )
    family_by_feature: dict[str, str] = {}
    if isinstance(raw_families, Mapping):
        if set(map(str, raw_families)) >= set(feature_names):
            family_by_feature = {
                name: str(raw_families[name]) for name in feature_names
            }
        else:
            for family, names in raw_families.items():
                if isinstance(names, (list, tuple, set)):
                    for name in names:
                        family_by_feature[str(name)] = str(family)
    for name in feature_names:
        family_by_feature.setdefault(name, "declared_in_feature_spec")
    return specification, feature_names, family_by_feature


def _target_output_definition() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_target_output_definition",
        "definition_status": "PASS",
        "method_name": "symmetry_aware_medoid",
        "source_symbol": "scripts/evaluate_m2.py:choose_medoid",
        "requested_view_budget": 5,
        "input_candidates": (
            "The five fixed-order M2 candidate poses for one target, transformed "
            "into that target camera frame; failed/non-finite poses are excluded."
        ),
        "configuration": {
            "pairwise_metric": (
                "mean of forward and reverse official symmetry-aware MSSD"
            ),
            "normalization": "official object diameter",
            "candidate_score": "mean pairwise normalized symmetric MSSD",
            "selection": "minimum candidate score",
            "tie_break": ["acquisition_rank", "sample_id"],
            "ground_truth_used_by_selection": False,
            "new_tuning": False,
        },
        "output_pose_convention": (
            "T_target_camera_object, a 4x4 homogeneous transform mapping "
            "object/model-frame column vectors into the M2 target camera; "
            "translation is in metres."
        ),
        "return_type": "one existing finite candidate pose, not a fused aggregate",
        "selection_rationale": (
            "It is the single fixed non-oracle symmetry-aware five-view method "
            "already evaluated by M2 and frozen before M6-G0. The M2 report "
            "shows positive multi-view evidence for this method; M6-G0 does not "
            "compare output methods or select one from M6 labels."
        ),
        "deployment_boundary": (
            "The medoid selection itself does not use ground-truth pose error. "
            "The historical M2 candidate grouping used oracle physical-instance "
            "association, so this existing-data audit does not establish an "
            "end-to-end deployable association pipeline."
        ),
    }


def _label_definition(m2_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_label_definition",
        "definition_status": "PASS",
        "formal_label": "y_failure",
        "positive_class": "frozen final pose is incorrect",
        "formula": "y_failure = int(not diagnostic_success.joint)",
        "source": {
            "path": "artifacts/m2/metrics.jsonl",
            "row_filter": {
                "method": "symmetry_aware_medoid",
                "requested_view_budget": 5,
            },
            "field": "diagnostic_success.joint",
            "source_symbol": "scripts/evaluate_m2.py:diagnostic_flags",
        },
        "correctness_rule": {
            "correct": (
                "finite pose AND normalized symmetry-aware MSSD <= 0.10 "
                "object diameter AND symmetry-aware MSPD <= 10*r"
            ),
            "normalized_mssd_threshold": 0.10,
            "normalized_mssd_units": "fraction of official object diameter",
            "mspd_threshold_multiplier": 10,
            "mspd_units": "pixels",
            "r_definition": "target image width / 640",
            "comparison": "inclusive less than or equal to",
            "new_threshold_tuning": False,
        },
        "target_count": m2_audit["target_count"],
        "failure_count": m2_audit["label_counts"]["failure"],
        "success_count": m2_audit["label_counts"]["success"],
        "failure_prevalence": m2_audit["failure_prevalence"],
        "counts_by_object": m2_audit["counts_by_object"],
        "physical_instances_containing_failures": m2_audit[
            "physical_instances_containing_failures"
        ],
        "objects_containing_failures": m2_audit["objects_containing_failures"],
        "minimum_support": {
            "failure_count_min": 30,
            "success_count_min": 30,
            "physical_instances_containing_failures_min": 10,
            "objects_containing_failures_min": 5,
        },
        "label_support_passed": m2_audit["label_support_passed"],
        "evaluator_only": True,
    }


def _feature_specification(
    specification: Mapping[str, Any],
    feature_names: Sequence[str],
    family_by_feature: Mapping[str, str],
) -> dict[str, Any]:
    score_only = [name for name in feature_names if family_by_feature[name] == "score"]
    disagreement_only = [
        name
        for name in feature_names
        if family_by_feature[name] in {"disagreement", "agreement", "view"}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_feature_specification",
        "frozen_before_model_evaluation": True,
        "source_symbol": "scripts/m6_g0_features.py:FEATURE_SPEC",
        "row_granularity": "exactly one inference-time feature row per M2 target",
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "family_by_feature": dict(sorted(family_by_feature.items())),
        "definitions": specification,
        "ablations": {
            "score_only": score_only,
            "disagreement_only": disagreement_only,
            "score_disagreement": [*score_only, *disagreement_only],
            "all": list(feature_names),
            "availability_features_enter_all_only": True,
        },
        "raw_score_orientation": {
            "higher_score_means": "more confident",
            "failure_risk_orientation": "negative raw FoundationPose top score",
            "frozen_source_semantics": (
                "The upstream registration returns its selected pose first; "
                "PoseLoop stores scores[0] as top score and scores[0]-scores[1] "
                "as the top-score margin. Orientation is not selected from labels."
            ),
        },
        "availability": (
            "Features use only saved candidate outputs, saved camera calibration, "
            "and declared symmetry metadata available at inference time."
        ),
        "model_input_is_features_mapping_only": True,
        "prohibited_model_fields": list(PROHIBITED_MODEL_FIELDS),
        "multiple_view_dependency_must_be_declared_per_feature": True,
    }


def _feature_schema(feature_names: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_inference_feature_schema",
        "artifact": "artifacts/m6_g0/features.jsonl",
        "row_record_type": "m6_g0_feature_row",
        "required_top_level_fields": ["features", "metadata"],
        "additional_top_level_fields_allowed": False,
        "features": {
            "required": list(feature_names),
            "additional_fields_allowed": False,
            "value_type": "finite number or null",
            "null_policy": (
                "Nulls are retained; imputation and missing indicators are fit "
                "inside training folds only."
            ),
        },
        "metadata": {
            "required": [
                "target_sample_id",
                "group_id",
                "object_id",
                "physical_instance_id",
            ],
            "identifier_fields_are_never_model_inputs": True,
        },
        "model_api_contract": "pass row.features only",
        "prohibited_feature_fields": list(PROHIBITED_MODEL_FIELDS),
    }


def _evaluator_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_evaluator_schema",
        "artifact": "artifacts/m6_g0/evaluator_rows.jsonl",
        "row_record_type": "m6_g0_evaluator_row",
        "required_fields": [
            "record_type",
            "schema_version",
            "target_id",
            "object_id",
            "physical_instance_id",
            "y_failure",
        ],
        "optional_diagnostic_fields": [
            "normalized_mssd",
            "mspd_px",
            "diagnostic_success",
        ],
        "one_row_per_target": True,
        "ground_truth_allowed": True,
        "may_enter_model_api": False,
        "join_contract": (
            "Join to feature metadata by target_sample_id only after the model "
            "matrix has been constructed from row.features."
        ),
    }


def _fold_protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_fold_protocol",
        "manifest_artifact": "artifacts/m6_g0/frozen/fold_manifest.json",
        "manifest_write_rule": "atomic write once; existing file must be byte-identical",
        "manifest_written_before_model_fitting": True,
        "source_symbol": "scripts/m6_g0_analysis.py:make_fold_manifest",
        "group_field": "physical_instance_id",
        "outer": {
            "preferred_fold_count": 5,
            "fallback_fold_count": 4,
            "fallback_allowed_once_only_for_class_support": True,
            "splitter": "StratifiedGroupKFold",
            "shuffle": True,
            "random_seed": 6001,
            "stratification": (
                "preserve failure prevalence and object distribution as well as "
                "possible without splitting a physical instance"
            ),
            "each_target_test_count": 1,
        },
        "inner": {
            "fold_count": 4,
            "splitter": "StratifiedGroupKFold",
            "shuffle": True,
            "outer_fold_index_base": 0,
            "random_seed_formula": "6100 + outer_fold_index",
            "outer_test_rows_forbidden": True,
        },
        "all_formal_methods_share_outer_folds": True,
        "fold_replacement_after_results_forbidden": True,
    }


def _model_grid() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_model_grid",
        "formal_methods": list(FORMAL_METHODS),
        "selection_metric": "mean inner-fold AURC; lower is better",
        "nested_tie_rule": (
            "When best logistic and tree inner AURC differ by at most 0.005 "
            "absolute, select logistic regression."
        ),
        "RAW_SCORE_RANK": {
            "fitted": False,
            "confidence_score": "raw_selected_score",
            "orientation": "higher is more confident",
        },
        "SCORE_ISOTONIC": {
            "input_features": ["raw_selected_score"],
            "model": "IsotonicRegression",
            "out_of_bounds": "clip",
            "increasing_failure_risk": False,
            "fit_scope": "outer training rows only",
        },
        "LOGISTIC_MULTIFEATURE": {
            "model": "LogisticRegression",
            "penalty": "l2",
            "solver": "lbfgs",
            "C": [0.01, 0.1, 1.0, 10.0],
            "max_iter": 5000,
            "random_state": 6001,
            "class_weight": None,
            "preprocessing": [
                "training-fold median imputation",
                "missing indicators",
                "training-fold standardization",
            ],
        },
        "SHALLOW_TREE_MULTIFEATURE": {
            "model": "RandomForestClassifier",
            "n_estimators": [50, 100],
            "max_depth": [2, 3],
            "min_samples_leaf": [10, 20],
            "max_features": "sqrt",
            "criterion": "log_loss",
            "bootstrap": True,
            "class_weight": None,
            "random_state": {
                "inner_grid_fit": ("6100 + outer_fold + 100*grid_index + inner_fold"),
                "selected_tree_inner_oof_calibration_fit": (
                    "6100 + outer_fold + inner_fold"
                ),
                "outer_refit": "6001 + outer_fold",
            },
            "n_jobs": 1,
            "probability_calibration": {
                "method": "sigmoid logistic regression on clipped logits",
                "fit_scope": "outer-training inner out-of-fold predictions only",
                "selection_boundary": (
                    "inner AURC selection uses raw inner-OOF tree probabilities; "
                    "the calibrator never scores the labels used to fit it"
                ),
                "C": 1000000.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 5000,
                "random_state": 6001,
            },
        },
        "NESTED_MULTIFEATURE": {
            "candidates": [
                "LOGISTIC_MULTIFEATURE",
                "SHALLOW_TREE_MULTIFEATURE",
            ],
            "hyperparameters_selected_in_inner_grouped_cv": True,
            "refit_on_complete_outer_training_fold": True,
            "outer_test_prediction_count": 1,
        },
        "forbidden_models": [
            "neural_network",
            "svm",
            "knn",
            "large_ensemble",
            "conformal_prediction",
            "test_time_adaptation",
        ],
    }


def _decision_rules() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_decision_rules",
        "formal_method": "NESTED_MULTIFEATURE",
        "classification_values": [
            "SIGNAL GO",
            "WEAK SIGNAL / HOLDOUT NOT AUTHORIZED",
            "NO-GO",
        ],
        "access_disclosure_override": {
            "m3_access_boundary_passed_required_for_signal_go": True,
            "current_m3_access_boundary_passed": False,
            "current_raw_structured_m3_access": False,
            "forced_classification": "NO-GO",
            "m3_holdout_authorized": False,
            "reason": (
                "Documentation audits accidentally displayed aggregate and "
                "per-object context from reports/m3_active_budget.md. No raw "
                "M3 artifact rows, target IDs, predictions, labels, candidates, or "
                "per-target metrics were accessed and no displayed value is used, "
                "but pristine no-M3-inspection cannot be certified."
            ),
        },
        "signal_go": {
            "auroc_min": 0.75,
            "aurc_relative_improvement_vs_raw_min": 0.10,
            "aurc_relative_improvement_vs_isotonic_min": 0.05,
            "bootstrap_95_ci_aurc_improvement_vs_isotonic_excludes_zero": True,
            "risk_threshold": 0.05,
            "coverage_at_risk_threshold_min": 0.40,
            "empirical_failure_rate_at_risk_threshold_max": 0.05,
            "accepted_count_at_risk_threshold_min": 30,
            "brier_relative_degradation_vs_isotonic_max": 0.02,
            "ece_max": 0.05,
            "leave_one_object_out_aurc_improvement_nonnegative": True,
            "single_object_gain_fraction_max": 0.40,
            "single_physical_instance_gain_fraction_max": 0.20,
            "all_leakage_and_support_checks_pass": True,
            "each_target_exactly_one_oof_prediction": True,
            "m3_access_boundary_passed": True,
        },
        "weak_signal": {
            "auroc_min": 0.70,
            "aurc_relative_improvement_vs_raw_min": 0.05,
            "stable_disagreement_benefit_required": True,
            "holdout_authorized": False,
        },
        "no_go_material_conditions": [
            "invalid frozen target or label definition",
            "insufficient class or group support",
            "grouped AUROC below 0.70",
            "multifeature AURC improvement versus raw below 5%",
            "isotonic captures essentially all benefit",
            "benefit disappears under physical-instance grouping",
            "object- or instance-driven benefit",
            "negligible 5% risk coverage",
            "materially poor calibration",
            "leakage-free feature generation not established",
            "M3 data required for a positive result",
            "post-hoc thresholds, objects, features, or folds required",
        ],
        "relative_improvement_formula": "(comparator_aurc - candidate_aurc) / comparator_aurc",
        "zero_comparator_rule": "A zero comparator AURC cannot establish a positive relative improvement.",
        "selective_prediction_ties": {
            "rule": (
                "Every equal-risk block uses the expected prefix failure count "
                "under a uniform ordering within that block"
            ),
            "identifier_tie_break_for_metrics": False,
            "purpose": (
                "Make risk-coverage, AURC, bootstrap, and robustness metrics "
                "invariant to target, object, and instance identifiers"
            ),
        },
        "equal_frequency_calibration_ties": {
            "rule": (
                "When an equal-probability block crosses equal-frequency bin "
                "boundaries, distribute its failures in expectation in "
                "proportion to the block positions assigned to each bin"
            ),
            "identifier_tie_break_for_bins": False,
        },
        "aurc_gain_attribution": {
            "rule": (
                "Decompose each method AURC into additive failed-target rank "
                "contributions, using the equal-risk block expectation, then "
                "subtract candidate from isotonic per target"
            ),
            "aggregation": ["object_id", "physical_instance_id"],
            "sum_must_equal_aggregate_aurc_gain": True,
            "deletion_influence_is_not_a_gain_share": True,
        },
        "coverage_failure_attribution": {
            "coverage": 0.80,
            "rule": (
                "If an equal-risk block crosses the coverage boundary, allocate "
                "accepted/deferred failures in expectation under the same uniform "
                "within-block policy before aggregating by object"
            ),
            "identifier_tie_break_for_attribution": False,
        },
        "bootstrap": {
            "resampling_unit": "physical_instance_id",
            "minimum_resamples": 2000,
            "interval": "percentile 95%",
        },
    }


def _input_allowlist() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_input_allowlist",
        "milestone_scope": "M2 development targets only",
        "feature_generation": {
            "allowed_paths": {
                "artifacts/m1/predictions.jsonl": [
                    "sample_id",
                    "status",
                    "predicted_model_to_camera_pose_m",
                    "foundationpose_top_score",
                    "foundationpose_top_score_margin",
                    "pose_hypothesis_count",
                ],
                "artifacts/m2/view_predictions.jsonl": [
                    "sample_id",
                    "status",
                    "predicted_model_to_camera_pose_m",
                    "foundationpose_top_score",
                    "foundationpose_top_score_margin",
                    "pose_hypothesis_count",
                ],
                "artifacts/m2/groups.jsonl": [
                    "group_id",
                    "target_sample_id",
                    "object_id",
                    "views.sample_id",
                    "views.acquisition_rank",
                    "views.camera_world_to_camera_pose_m",
                    "oracle_association.track_id (metadata/grouping only)",
                ],
            },
            "allowed_declared_symmetry_metadata": {
                "source": "XYZ-IBD models_eval through scripts/evaluate_m1.py",
                "fields": [
                    "official model points in millimetres",
                    "official object diameter in millimetres",
                    "declared discrete and continuous symmetry transforms",
                ],
                "ground_truth_target_pose_allowed": False,
                "identity_fields_are_model_features": False,
            },
            "field_projection_required_before_feature_computation": True,
            "ground_truth_fields_forbidden": True,
        },
        "evaluator_only": {
            "allowed_paths": ["artifacts/m2/metrics.jsonl"],
            "allowed_row_filter": {
                "method": "symmetry_aware_medoid",
                "requested_view_budget": 5,
            },
            "ground_truth_fields_may_be_used_only_to_reconstruct_label": True,
        },
        "prohibited_input_roots": [
            "artifacts/m3",
            "any M3-derived target-level table",
        ],
        "m3_target_ids_forbidden": True,
        "m3_raw_structured_access_forbidden": True,
        "m3_protected_artifacts_may_only_be_read_as_binary_bytes_for_sha256": True,
        "model_api_receives": "features mapping only",
        "model_api_never_receives": ["metadata", "evaluator_rows"],
    }


def expected_static_payloads(repo_root: Path) -> dict[str, dict[str, Any]]:
    protected = validate_protected_baseline(repo_root)
    m2_audit = _m2_source_audit(repo_root)
    specification, feature_names, family_by_feature = _feature_contract()
    source_audit = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_source_audit",
        "verified_starting_state": {
            "branch": EXPECTED_STARTING_BRANCH,
            "commit": EXPECTED_STARTING_COMMIT,
            "baseline_worktree": (
                "clean before creation of the tracked protection receipt"
            ),
            "continuation_worktree": (
                "pre-existing uncommitted M6-G0 files and an append-only README "
                "section were present when this audit resumed; protected M1-M5 "
                "paths were unchanged"
            ),
            "remote_v": [],
            "log_3_oneline": [
                "315ca60 m5: validate symmetry-quotient temporal pose estimation",
                "3b97284 docs: prepare PoseLoop v1.0.0 release",
                "11b6d6a feat: add M4 one-step view utility ranking",
            ],
        },
        "runtime_environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "distributions": {
                name: distribution_version(name) for name in RUNTIME_DISTRIBUTIONS
            },
        },
        "protected_m2_m5": protected,
        "m2_development_audit": m2_audit,
        "acceptance": {
            "status": "PASS",
            "protected_path_set_and_hashes": True,
            "m2_target_candidate_group_and_label_contract": True,
            "existing_validation_commands": [
                "python -B scripts/render_precomputed_report.py --check",
                "python -B scripts/validate_m5_g0.py --check",
            ],
        },
        "m3_access_disclosure": {
            "m3_access_boundary_passed": False,
            "raw_structured_m3_access": False,
            "m3_artifact_rows_accessed": False,
            "m3_target_ids_accessed": False,
            "m3_predictions_or_labels_accessed": False,
            "m3_hashes_unchanged": True,
            "accidental_documentation_display": "reports/m3_active_budget.md",
            "displayed_scope": "aggregate and per-object context only",
            "displayed_values_used_by_m6_g0": False,
            "consequence": "SIGNAL GO barred; M3 holdout remains unauthorized",
        },
        "scope": {
            "m2_development_only": True,
            "foundationpose_inference_run": False,
            "new_pose_estimates_generated": False,
            "m5_g0_remained_no_go": True,
            "conformal_or_formal_risk_guarantee_claimed": False,
        },
    }
    return {
        "protected_hash_reference.json": {
            "schema_version": SCHEMA_VERSION,
            "record_type": "m6_g0_protected_hash_reference",
            "tracked_receipt_path": PROTECTED_RECEIPT.as_posix(),
            "frozen_copy_path": (FROZEN_DIR / "protected_m2_m5_hashes.json").as_posix(),
            "tracked_receipt_sha256": EXPECTED_PROTECTED_RECEIPT_SHA256,
            "protected_root_sha256": EXPECTED_PROTECTED_ROOT_SHA256,
            "protected_entry_count": EXPECTED_PROTECTED_ENTRY_COUNT,
            "starting_commit": EXPECTED_STARTING_COMMIT,
            "m3_handling": "binary SHA-256 only",
        },
        "source_audit.json": source_audit,
        "target_output_definition.json": _target_output_definition(),
        "label_definition.json": _label_definition(m2_audit),
        "feature_specification.json": _feature_specification(
            specification,
            feature_names,
            family_by_feature,
        ),
        "feature_schema.json": _feature_schema(feature_names),
        "evaluator_schema.json": _evaluator_schema(),
        "fold_protocol.json": _fold_protocol(),
        "model_grid.json": _model_grid(),
        "decision_rules.json": _decision_rules(),
        "input_allowlist.json": _input_allowlist(),
    }


def _required_source_paths(repo_root: Path) -> dict[str, Path]:
    names = (
        "build_m6_g0_manifests.py",
        "m6_g0_features.py",
        "m6_g0_labels.py",
        "m6_g0_analysis.py",
        "m6_g0_report.py",
        "run_m6_g0.py",
        "validate_m6_g0.py",
    )
    paths = {name: repo_root / "scripts" / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"M6-G0 source files must exist before freezing: {missing}"
        )
    return paths


def _manifest_receipt(
    repo_root: Path,
    frozen_dir: Path,
    payloads: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_hashes = {
        name: _sha256_binary(frozen_dir / name)
        for name in sorted((*COPIED_FROZEN_FILENAMES, *payloads))
    }
    source_hashes = {
        f"scripts/{name}": _sha256_binary(path)
        for name, path in sorted(_required_source_paths(repo_root).items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_manifest_hashes",
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "protected_root_sha256": EXPECTED_PROTECTED_ROOT_SHA256,
        "static_frozen_file_count": len(frozen_hashes),
        "frozen_sha256": frozen_hashes,
        "source_sha256": source_hashes,
        "runner_owned_write_once_files": list(RUNNER_OWNED_FROZEN_FILENAMES),
        "fold_manifest_hash_contract": (
            "fold_manifest.json stores the canonical SHA-256 of its assignment "
            "payload and is validated independently before any fitting"
        ),
    }


def freeze(repo_root: Path, *, check_only: bool = False) -> dict[str, Any]:
    """Create or byte-validate the complete static M6-G0 protocol freeze."""

    repo_root = repo_root.resolve()
    frozen_dir = repo_root / FROZEN_DIR
    payloads = expected_static_payloads(repo_root)
    receipt_bytes = (repo_root / PROTECTED_RECEIPT).read_bytes()

    if check_only:
        if not frozen_dir.is_dir():
            raise FileNotFoundError(f"Frozen directory does not exist: {frozen_dir}")
        expected_bytes: dict[str, bytes] = {
            "protected_m2_m5_hashes.json": receipt_bytes,
            **{name: _json_bytes(value) for name, value in payloads.items()},
        }
        for name, expected in expected_bytes.items():
            path = frozen_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Required frozen file is missing: {path}")
            if path.read_bytes() != expected:
                raise ValueError(f"Frozen file differs from protocol: {path}")
        expected_receipt = _manifest_receipt(
            repo_root,
            frozen_dir,
            payloads,
        )
        receipt_path = frozen_dir / "manifest_hashes.json"
        if receipt_path.read_bytes() != _json_bytes(expected_receipt):
            raise ValueError("M6-G0 manifest hash receipt differs from current sources")
        return expected_receipt

    frozen_dir.mkdir(parents=True, exist_ok=True)
    freeze_bytes(frozen_dir / "protected_m2_m5_hashes.json", receipt_bytes)
    for name, payload in payloads.items():
        freeze_json(frozen_dir / name, payload)
    receipt = _manifest_receipt(repo_root, frozen_dir, payloads)
    freeze_json(frozen_dir / "manifest_hashes.json", receipt)
    return receipt


def target_ids_from_m2(repo_root: Path) -> set[str]:
    """Return the allowed M2 target-ID universe without consulting M3."""

    return {
        str(row["target_sample_id"])
        for row in _strict_jsonl(repo_root / "artifacts" / "m2" / "groups.jsonl")
    }


def canonical_sha256(value: Any) -> str:
    """Public canonical hash helper shared with the runner and validator."""

    return _canonical_sha256(value)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    receipt = freeze(repo_root, check_only=args.check)
    action = "validated" if args.check else "frozen"
    fold_path = repo_root / FROZEN_DIR / "fold_manifest.json"
    fold_state = "present" if fold_path.is_file() else "awaiting runner"
    print(
        f"{action} M6-G0 protocol: protected="
        f"{EXPECTED_PROTECTED_ENTRY_COUNT}, static_frozen="
        f"{receipt['static_frozen_file_count']}, fold_manifest={fold_state}"
    )


if __name__ == "__main__":
    main()
