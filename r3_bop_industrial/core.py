"""Leakage-resistant contracts and official-evaluator orchestration for R3.

This module intentionally does not import BOP Toolkit or parse BOP ground-truth
files.  It validates prediction-side artifacts, freezes their hashes, and then
invokes the pinned official evaluator only behind an explicit authorization
receipt.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import PACKAGE_VERSION, PROTOCOL_ID


class ContractError(ValueError):
    """Raised when an R3 boundary or frozen-input contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError("R3 protocol must be a JSON object")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError(
            f"Expected protocol_id {PROTOCOL_ID!r}, got {value.get('protocol_id')!r}"
        )
    if value.get("state") != "frozen_before_r3_predictions":
        raise ContractError("R3 protocol is not frozen before predictions")
    if value.get("prediction_bundle", {}).get("hash_algorithm") != "sha256":
        raise ContractError("R3 protocol must use SHA-256")
    hash_contracts = {
        "toolkit source archive": value.get("toolkit", {}).get("source_archive_sha256"),
        "toolkit source tree": value.get("toolkit", {})
        .get("source_tree_contract", {})
        .get("sha256"),
    }
    for archive_name, archive in value.get("dataset", {}).get("source", {}).get(
        "archives", {}
    ).items():
        hash_contracts[f"dataset archive {archive_name}"] = archive.get("sha256")
    for label, digest in hash_contracts.items():
        if not _valid_sha256(digest):
            raise ContractError(f"R3 protocol {label} SHA-256 is invalid")
    if value.get("anti_leakage", {}).get("result_tuning_after_score") is not False:
        raise ContractError("R3 protocol must forbid tuning after scoring")
    return value


def _run_git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_worktree_git(
    root: Path, arguments: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    completed = _run_git(root, arguments)
    if completed.returncode == 0:
        return completed
    marker = root / ".git"
    if os.name != "posix" or not marker.is_file():
        return completed
    text = marker.read_text(encoding="utf-8").strip()
    if not text.lower().startswith("gitdir:"):
        return completed
    raw_git_dir = text.split(":", 1)[1].strip().replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", raw_git_dir)
    if match:
        raw_git_dir = f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    return subprocess.run(
        [
            "git",
            f"--git-dir={raw_git_dir}",
            f"--work-tree={root}",
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_r3_output_scope(path: Path, repo_root: Path, protocol: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    repository = repo_root.resolve()
    if not _is_relative_to(resolved, repository):
        return
    allowed = [
        (repository / relative).resolve()
        for relative in protocol["r3_output_namespaces"]
    ]
    if not any(_is_relative_to(resolved, root) for root in allowed):
        raise ContractError(
            f"R3 output inside the repository must stay in an R3 namespace: {resolved}"
        )


def repo_state(repo_root: Path, baseline_commit: str) -> dict[str, Any]:
    head = _run_worktree_git(repo_root, ["rev-parse", "HEAD"])
    ancestor = _run_worktree_git(
        repo_root, ["merge-base", "--is-ancestor", baseline_commit, "HEAD"]
    )
    branch = _run_worktree_git(repo_root, ["branch", "--show-current"])
    status = _run_worktree_git(repo_root, ["status", "--short"])
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "baseline_commit": baseline_commit,
        "baseline_is_ancestor": (
            True if ancestor.returncode == 0 else False if ancestor.returncode == 1 else None
        ),
        "git_query_errors": [
            item.stderr.strip()
            for item in (head, ancestor, branch, status)
            if item.returncode not in (0, 1) and item.stderr.strip()
        ],
        "status_entries": [line for line in status.stdout.splitlines() if line],
    }


def dataset_preflight(dataset_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    root = dataset_root.resolve()
    required = list(protocol["dataset"]["required_public_inputs"])
    entries: dict[str, Any] = {}
    missing: list[str] = []
    accessed: list[str] = []
    for relative in required:
        candidate = root / relative
        exists = candidate.is_file() or candidate.is_dir()
        item: dict[str, Any] = {
            "path": str(candidate),
            "exists": exists,
            "kind": "directory" if candidate.is_dir() else "file",
        }
        if candidate.is_file():
            item["sha256"] = sha256_file(candidate)
            accessed.append(str(candidate))
        elif candidate.is_dir():
            # Deliberately do not enumerate the split directory; it contains labels.
            accessed.append(str(candidate))
        else:
            missing.append(relative)
        entries[relative] = item
    name_ok = root.name == protocol["dataset"]["name"]
    errors = [] if name_ok else [f"dataset root basename must be {protocol['dataset']['name']}"]
    if missing:
        errors.append(f"missing public dataset inputs: {', '.join(missing)}")
    return {
        "root": str(root),
        "dataset_name_matches": name_ok,
        "required_public_inputs": entries,
        "accessed_paths": accessed,
        "evaluator_only_label_paths_accessed": [],
        "errors": errors,
        "ready": not errors,
    }


def toolkit_preflight(toolkit_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    root = toolkit_root.resolve()
    expected_commit = str(protocol["toolkit"]["commit"])
    evaluator_relpaths = list(protocol["toolkit"]["official_evaluators"].values())
    result: dict[str, Any] = {
        "root": str(root),
        "expected_commit": expected_commit,
        "actual_commit": None,
        "source_kind": None,
        "source_tree_sha256": None,
        "source_tree_file_count": 0,
        "expected_source_tree_sha256": protocol["toolkit"]["source_tree_contract"][
            "sha256"
        ],
        "official_evaluators": {},
        "tracked_sources_clean": False,
        "errors": [],
    }
    if not root.is_dir():
        result["errors"].append("BOP Toolkit root is missing")
        result["ready"] = False
        return result

    tree = toolkit_source_tree_audit(root, protocol)
    result["source_tree_sha256"] = tree["sha256"]
    result["source_tree_file_count"] = tree["file_count"]

    rev = _run_git(root, ["rev-parse", "HEAD"])
    top_level = _run_git(root, ["rev-parse", "--show-toplevel"])
    is_checkout_root = (
        (root / ".git").exists()
        and rev.returncode == 0
        and top_level.returncode == 0
        and Path(top_level.stdout.strip()).resolve() == root
    )
    if is_checkout_root:
        result["source_kind"] = "git_checkout"
        result["actual_commit"] = rev.stdout.strip()
        if result["actual_commit"] != expected_commit:
            result["errors"].append(
                f"BOP Toolkit commit mismatch: {result['actual_commit']} != {expected_commit}"
            )

    for relative in evaluator_relpaths:
        script = root / relative
        item = {"path": str(script), "exists": script.is_file()}
        if script.is_file():
            item["sha256"] = sha256_file(script)
        else:
            result["errors"].append(f"missing official evaluator: {relative}")
        result["official_evaluators"][relative] = item

    if is_checkout_root:
        tracked_diff = _run_git(
            root,
            [
                "diff",
                "--quiet",
                "--ignore-cr-at-eol",
                "HEAD",
                "--",
                *protocol["toolkit"]["source_tree_contract"]["included_roots"],
            ],
        )
        if tracked_diff.returncode not in (0, 1):
            result["errors"].append("could not audit BOP Toolkit tracked sources")
        elif tracked_diff.returncode == 1:
            result["errors"].append("BOP Toolkit evaluator/library sources have tracked changes")
        else:
            result["tracked_sources_clean"] = True
    else:
        result["source_kind"] = "pinned_archive_tree"
        tree_contract = protocol["toolkit"]["source_tree_contract"]
        if (
            tree["sha256"] != tree_contract["sha256"]
            or tree["file_count"] != tree_contract["file_count"]
        ):
            result["errors"].append(
                "BOP Toolkit is not a Git checkout and its source-tree hash does not "
                "match the pinned official archive"
            )
        else:
            result["tracked_sources_clean"] = True

    result["ready"] = not result["errors"]
    return result


def toolkit_source_tree_audit(
    toolkit_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Hash evaluator code from either a Git checkout or a verified archive tree."""

    root = toolkit_root.resolve()
    contract = protocol["toolkit"]["source_tree_contract"]
    text_extensions = {
        str(value).lower() for value in contract["text_eol_normalized_extensions"]
    }
    relative_paths: set[Path] = set()
    for relative_root in contract["included_roots"]:
        candidate = root / relative_root
        if candidate.is_file():
            relative_paths.add(candidate.relative_to(root))
        elif candidate.is_dir():
            for path in candidate.rglob("*"):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix.lower() != ".pyc"
                ):
                    relative_paths.add(path.relative_to(root))

    digest = hashlib.sha256()
    for relative in sorted(relative_paths, key=lambda value: value.as_posix()):
        data = (root / relative).read_bytes()
        if relative.suffix.lower() in text_extensions:
            data = data.replace(b"\r\n", b"\n")
        file_sha256 = hashlib.sha256(data).hexdigest()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "file_count": len(relative_paths)}


def expected_prediction_paths(
    input_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Path]:
    root = input_root.resolve()
    return {
        role: root / filename
        for role, filename in protocol["prediction_bundle"]["files"].items()
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def validate_coco_predictions(
    path: Path, required_fields: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ContractError("COCO prediction input must be a non-empty JSON list")
    required = set(required_fields)
    allowed = required
    timing_by_image: dict[tuple[int, int], float] = {}
    keys: set[tuple[int, int, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"COCO prediction row {index} is not an object")
        if set(row) != allowed:
            raise ContractError(
                f"COCO prediction row {index} fields differ from frozen schema: "
                f"missing={sorted(required - set(row))}, extra={sorted(set(row) - allowed)}"
            )
        for field in ("scene_id", "image_id", "category_id"):
            if isinstance(row[field], bool) or not isinstance(row[field], int):
                raise ContractError(f"COCO row {index} {field} must be an integer")
        _finite_number(row["score"], f"COCO row {index} score")
        time_value = _finite_number(row["time"], f"COCO row {index} time")
        if time_value < 0.0:
            raise ContractError("R3 requires non-negative measured inference time")
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ContractError(f"COCO row {index} bbox must contain four numbers")
        bbox_values = [
            _finite_number(value, f"COCO row {index} bbox") for value in bbox
        ]
        if any(value < 0.0 for value in bbox_values[2:]):
            raise ContractError(f"COCO row {index} bbox width/height must be non-negative")
        segmentation = row["segmentation"]
        if not isinstance(segmentation, dict) or set(segmentation) != {"counts", "size"}:
            raise ContractError(f"COCO row {index} segmentation must be exact COCO RLE")
        size = segmentation["size"]
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size)
        ):
            raise ContractError(f"COCO row {index} RLE size is invalid")
        if not isinstance(segmentation["counts"], (str, list)):
            raise ContractError(f"COCO row {index} RLE counts are invalid")
        image_key = (row["scene_id"], row["image_id"])
        previous_time = timing_by_image.setdefault(image_key, time_value)
        if abs(previous_time - time_value) > 0.001:
            raise ContractError(f"COCO timings differ within image {image_key}")
        keys.add((row["scene_id"], row["image_id"], row["category_id"]))
    return rows, {
        "row_count": len(rows),
        "image_count": len(timing_by_image),
        "scene_image_category_count": len(keys),
        "sha256": sha256_file(path),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"Association line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ContractError("Predicted association JSONL must not be empty")
    return rows


def validate_predicted_association(
    path: Path,
    coco_rows: Sequence[Mapping[str, Any]],
    required_fields: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_jsonl(path)
    required = set(required_fields)
    used_detection_indices: set[int] = set()
    tracks: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        if set(row) != required:
            raise ContractError(
                f"Association row {row_number} fields differ from frozen schema: "
                f"missing={sorted(required - set(row))}, extra={sorted(set(row) - required)}"
            )
        for field in ("scene_id", "image_id", "category_id", "detection_index", "view_rank"):
            if isinstance(row[field], bool) or not isinstance(row[field], int):
                raise ContractError(f"Association row {row_number} {field} must be an integer")
        detection_index = row["detection_index"]
        if detection_index < 0 or detection_index >= len(coco_rows):
            raise ContractError(f"Association row {row_number} detection_index is out of range")
        if detection_index in used_detection_indices:
            raise ContractError(f"COCO detection {detection_index} is associated more than once")
        used_detection_indices.add(detection_index)
        source = coco_rows[detection_index]
        for association_field, coco_field in (
            ("scene_id", "scene_id"),
            ("image_id", "image_id"),
            ("category_id", "category_id"),
        ):
            if row[association_field] != source[coco_field]:
                raise ContractError(
                    f"Association row {row_number} does not match COCO detection {detection_index}"
                )
        track_id = row["predicted_track_id"]
        if not isinstance(track_id, str) or not track_id.strip():
            raise ContractError(f"Association row {row_number} has no predicted track ID")
        if row["view_rank"] < 0:
            raise ContractError(f"Association row {row_number} has a negative view rank")
        score = _finite_number(row["association_score"], "association score")
        if score < 0.0 or score > 1.0:
            raise ContractError("Association score must be in [0, 1]")
        if not isinstance(row["is_target_view"], bool):
            raise ContractError("is_target_view must be boolean")
        tracks[(row["scene_id"], row["category_id"], track_id)].append(row)

    for track_key, track_rows in tracks.items():
        ranks = [row["view_rank"] for row in track_rows]
        if len(ranks) != len(set(ranks)):
            raise ContractError(f"Predicted track {track_key} repeats a view rank")
        target_count = sum(bool(row["is_target_view"]) for row in track_rows)
        if target_count != 1:
            raise ContractError(
                f"Predicted track {track_key} must contain exactly one target view"
            )
    return rows, {
        "row_count": len(rows),
        "track_count": len(tracks),
        "target_view_count": len(tracks),
        "sha256": sha256_file(path),
    }


def _rotation_determinant(values: Sequence[float]) -> float:
    a, b, c, d, e, f, g, h, i = values
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_rotation(values: Sequence[float], row_number: int) -> None:
    for first in range(3):
        for second in range(3):
            dot = sum(values[k * 3 + first] * values[k * 3 + second] for k in range(3))
            expected = 1.0 if first == second else 0.0
            if abs(dot - expected) > 5e-3:
                raise ContractError(f"Pose row {row_number} rotation is not orthonormal")
    if abs(_rotation_determinant(values) - 1.0) > 5e-3:
        raise ContractError(f"Pose row {row_number} rotation determinant is not +1")


def validate_pose_csv(
    path: Path, expected_header: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timings: dict[tuple[int, int], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ContractError(f"Pose result is empty: {path}") from exc
        if ",".join(header) != expected_header:
            raise ContractError(f"Pose result header differs from frozen BOP header: {path}")
        for row_number, values in enumerate(reader, start=2):
            if len(values) != 7:
                raise ContractError(f"Pose row {row_number} must contain seven CSV fields")
            try:
                scene_id, image_id, object_id = (int(values[index]) for index in range(3))
                score = float(values[3])
                rotation = [float(value) for value in values[4].split()]
                translation = [float(value) for value in values[5].split()]
                time_value = float(values[6])
            except ValueError as exc:
                raise ContractError(f"Pose row {row_number} contains an invalid number") from exc
            if len(rotation) != 9 or len(translation) != 3:
                raise ContractError(f"Pose row {row_number} has invalid R/t dimensions")
            if not all(math.isfinite(value) for value in [score, *rotation, *translation, time_value]):
                raise ContractError(f"Pose row {row_number} contains non-finite values")
            if time_value < 0.0:
                raise ContractError("R3 requires non-negative measured pose inference time")
            _validate_rotation(rotation, row_number)
            image_key = (scene_id, image_id)
            previous_time = timings.setdefault(image_key, time_value)
            if abs(previous_time - time_value) > 0.001:
                raise ContractError(f"Pose timings differ within image {image_key}")
            rows.append(
                {
                    "scene_id": scene_id,
                    "im_id": image_id,
                    "obj_id": object_id,
                    "score": score,
                    "R": rotation,
                    "t": translation,
                    "time": time_value,
                }
            )
    if not rows:
        raise ContractError(f"Pose result contains no estimates: {path}")
    keys = {(row["scene_id"], row["im_id"], row["obj_id"]) for row in rows}
    return rows, {
        "row_count": len(rows),
        "image_count": len(timings),
        "scene_image_object_count": len(keys),
        "sha256": sha256_file(path),
    }


def validate_provenance(
    path: Path,
    prediction_paths: Mapping[str, Path],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError("Prediction provenance must be a JSON object")
    required = {
        "schema_version",
        "protocol_id",
        "input_origin",
        "producer",
        "producer_version",
        "source_uri_or_run_id",
        "created_utc",
        "label_blind",
        "uses_gt_visible_masks",
        "uses_oracle_association",
        "result_selection_uses_evaluator_metrics",
        "source_artifacts",
        "files",
        "lineage",
    }
    if set(value) != required:
        raise ContractError(
            "Prediction provenance fields differ from the frozen schema: "
            f"missing={sorted(required - set(value))}, extra={sorted(set(value) - required)}"
        )
    if value["schema_version"] != "poseloop.r3.prediction-provenance.v1":
        raise ContractError("Prediction provenance schema version is invalid")
    if value["protocol_id"] != protocol["protocol_id"]:
        raise ContractError("Prediction provenance protocol ID mismatch")
    if value["input_origin"] not in protocol["prediction_bundle"]["allowed_origins"]:
        raise ContractError("Prediction provenance origin is not allowed")
    for field in ("producer", "producer_version", "source_uri_or_run_id", "created_utc"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ContractError(f"Prediction provenance {field} must be non-empty")
    try:
        datetime.fromisoformat(value["created_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("Prediction provenance created_utc is invalid") from exc
    if value["label_blind"] is not True:
        raise ContractError("Prediction production must be declared label-blind")
    for field in (
        "uses_gt_visible_masks",
        "uses_oracle_association",
        "result_selection_uses_evaluator_metrics",
    ):
        if value[field] is not False:
            raise ContractError(f"Prediction provenance must declare {field}=false")

    source_artifacts = value["source_artifacts"]
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise ContractError("Prediction provenance source_artifacts must be a non-empty list")
    artifact_required = set(
        protocol["prediction_bundle"]["source_artifact_required_fields"]
    )
    artifact_roles: set[str] = set()
    for index, artifact in enumerate(source_artifacts):
        if not isinstance(artifact, dict) or set(artifact) != artifact_required:
            raise ContractError(f"Prediction source artifact {index} fields are invalid")
        role = artifact["role"]
        if not isinstance(role, str) or not role.strip() or role in artifact_roles:
            raise ContractError(f"Prediction source artifact {index} role is invalid or repeated")
        artifact_roles.add(role)
        if not isinstance(artifact["uri"], str) or not artifact["uri"].strip():
            raise ContractError(f"Prediction source artifact {index} URI is empty")
        if not _valid_sha256(artifact["sha256"]):
            raise ContractError(f"Prediction source artifact {index} SHA-256 is invalid")
    origin_requirement_key = (
        "audited_prediction_required_artifact_roles"
        if value["input_origin"] == "audited_prediction"
        else "bop_default_required_artifact_roles"
    )
    required_artifact_roles = set(protocol["prediction_bundle"][origin_requirement_key])
    missing_artifact_roles = required_artifact_roles - artifact_roles
    if missing_artifact_roles:
        raise ContractError(
            "Prediction provenance is missing source artifact roles: "
            + ", ".join(sorted(missing_artifact_roles))
        )

    file_roles = {"coco_predictions", "predicted_association", "single_view_pose", "multi_view_pose"}
    if not isinstance(value["files"], dict) or set(value["files"]) != file_roles:
        raise ContractError("Prediction provenance file roles differ from the frozen schema")
    for role in sorted(file_roles):
        record = value["files"][role]
        if not isinstance(record, dict) or set(record) != {"filename", "sha256"}:
            raise ContractError(f"Prediction provenance file record is invalid: {role}")
        expected_path = prediction_paths[role]
        if record["filename"] != expected_path.name:
            raise ContractError(f"Prediction provenance filename mismatch: {role}")
        actual_sha256 = sha256_file(expected_path)
        if record["sha256"] != actual_sha256:
            raise ContractError(f"Prediction provenance SHA-256 mismatch: {role}")

    if value["lineage"] != protocol["lineage"]:
        raise ContractError("Prediction provenance lineage differs from the frozen protocol")
    return {
        "input_origin": value["input_origin"],
        "producer": value["producer"],
        "producer_version": value["producer_version"],
        "source_uri_or_run_id": value["source_uri_or_run_id"],
        "created_utc": value["created_utc"],
        "source_artifacts": source_artifacts,
        "sha256": sha256_file(path),
        "hash_linkage_verified": True,
        "label_blind_contract_verified": True,
    }


def validate_prediction_bundle(
    input_root: Path,
    dataset_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    paths = expected_prediction_paths(input_root, protocol)
    missing = [role for role, path in paths.items() if not path.is_file()]
    summary: dict[str, Any] = {
        "root": str(input_root.resolve()),
        "expected_files": {role: str(path) for role, path in paths.items()},
        "missing_roles": missing,
        "errors": [],
        "ready": False,
    }
    dataset = dataset_root.resolve()
    for role, path in paths.items():
        if path.exists() and _is_relative_to(path.resolve(), dataset):
            summary["errors"].append(
                f"prediction role {role} is inside the dataset root and may expose GT"
            )
    if missing or summary["errors"]:
        if missing:
            summary["errors"].append(f"missing prediction roles: {', '.join(missing)}")
        return summary

    try:
        coco_rows, coco_audit = validate_coco_predictions(
            paths["coco_predictions"],
            protocol["prediction_bundle"]["coco_required"],
        )
        association_rows, association_audit = validate_predicted_association(
            paths["predicted_association"],
            coco_rows,
            protocol["prediction_bundle"]["association_required"],
        )
        single_rows, single_audit = validate_pose_csv(
            paths["single_view_pose"],
            protocol["prediction_bundle"]["pose_csv_header"],
        )
        multi_rows, multi_audit = validate_pose_csv(
            paths["multi_view_pose"],
            protocol["prediction_bundle"]["pose_csv_header"],
        )
        coco_keys = {
            (row["scene_id"], row["image_id"], row["category_id"])
            for row in coco_rows
        }
        single_keys = {(row["scene_id"], row["im_id"], row["obj_id"]) for row in single_rows}
        multi_keys = {(row["scene_id"], row["im_id"], row["obj_id"]) for row in multi_rows}
        target_association_keys = {
            (row["scene_id"], row["image_id"], row["category_id"])
            for row in association_rows
            if row["is_target_view"]
        }
        if not single_keys.issubset(coco_keys):
            raise ContractError("Single-view pose outputs are not linked to COCO predictions")
        if not multi_keys.issubset(target_association_keys):
            raise ContractError("Multi-view pose outputs are not linked to predicted target views")
        provenance_audit = validate_provenance(paths["provenance"], paths, protocol)
        summary.update(
            {
                "coco_predictions": coco_audit,
                "predicted_association": association_audit,
                "single_view_pose": single_audit,
                "multi_view_pose": multi_audit,
                "provenance": provenance_audit,
                "comparison_population": {
                    "single_target_key_count": len(single_keys),
                    "multi_target_key_count": len(multi_keys),
                    "shared_target_key_count": len(single_keys & multi_keys),
                    "official_target_denominator_source": "dataset target files, not result intersection",
                },
                "ready": True,
            }
        )
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        summary["errors"].append(str(exc))
    return summary


def build_official_commands(
    input_root: Path,
    toolkit_root: Path,
    eval_root: Path,
    protocol: Mapping[str, Any],
    python_executable: str = sys.executable,
) -> list[dict[str, Any]]:
    files = protocol["prediction_bundle"]["files"]
    evaluators = protocol["toolkit"]["official_evaluators"]
    workers = int(protocol["toolkit"]["workers"])
    commands: list[dict[str, Any]] = []
    for variant, role in (("single_view", "single_view_pose"), ("multi_view", "multi_view_pose")):
        filename = files[role]
        result_name = Path(filename).stem
        commands.append(
            {
                "metric": "official_bop19_localization_ar",
                "variant": variant,
                "argv": [
                    python_executable,
                    str((toolkit_root / evaluators["localization_ar"]).resolve()),
                    "--renderer_type=vispy",
                    f"--result_filenames={filename}",
                    f"--results_path={input_root.resolve()}",
                    f"--eval_path={eval_root.resolve()}",
                    f"--targets_filename={protocol['toolkit']['localization_targets']}",
                    f"--num_workers={workers}",
                ],
                "expected_score": str(
                    (eval_root / result_name / "scores_bop19.json").resolve()
                ),
            }
        )
        commands.append(
            {
                "metric": "official_bop24_pose_detection_ap",
                "variant": variant,
                "argv": [
                    python_executable,
                    str((toolkit_root / evaluators["pose_detection_ap"]).resolve()),
                    f"--result_filenames={filename}",
                    f"--results_path={input_root.resolve()}",
                    f"--eval_path={eval_root.resolve()}",
                    f"--targets_filename={protocol['toolkit']['detection_targets']}",
                    f"--num_workers={workers}",
                ],
                "expected_score": str(
                    (eval_root / result_name / "scores_bop24.json").resolve()
                ),
            }
        )

    coco_filename = files["coco_predictions"]
    coco_result_name = Path(coco_filename).stem
    for annotation_type in ("bbox", "segm"):
        score_filename = f"scores_bop22_coco_{annotation_type}.json"
        commands.append(
            {
                "metric": f"official_bop22_coco_{annotation_type}_ap",
                "variant": "shared_predicted_input",
                "argv": [
                    python_executable,
                    str((toolkit_root / evaluators["bbox_segmentation_ap"]).resolve()),
                    f"--result_filenames={coco_filename}",
                    f"--results_path={input_root.resolve()}",
                    f"--eval_path={eval_root.resolve()}",
                    f"--targets_filename={protocol['toolkit']['localization_targets']}",
                    f"--ann_type={annotation_type}",
                    "--bbox_type=amodal",
                ],
                "expected_score": str(
                    (eval_root / coco_result_name / score_filename).resolve()
                ),
            }
        )
    return commands


def _fingerprint(receipt: Mapping[str, Any]) -> dict[str, Any]:
    dataset_files = receipt["dataset"]["required_public_inputs"]
    toolkit_scripts = receipt["toolkit"]["official_evaluators"]
    predictions = receipt["prediction_bundle"]
    prediction_hashes = {
        role: predictions[role]["sha256"]
        for role in (
            "coco_predictions",
            "predicted_association",
            "single_view_pose",
            "multi_view_pose",
            "provenance",
        )
        if role in predictions
    }
    return {
        "protocol_sha256": receipt["protocol"]["sha256"],
        "dataset_public_input_sha256": {
            relative: item.get("sha256")
            for relative, item in dataset_files.items()
            if "sha256" in item
        },
        "toolkit_commit": receipt["toolkit"]["actual_commit"],
        "toolkit_verified_revision": (
            receipt["toolkit"]["actual_commit"]
            or receipt["toolkit"]["expected_commit"]
        ),
        "toolkit_pinned_source_tree_sha256": receipt["toolkit"][
            "expected_source_tree_sha256"
        ],
        "toolkit_evaluator_sha256": {
            relative: item.get("sha256")
            for relative, item in toolkit_scripts.items()
        },
        "prediction_sha256": prediction_hashes,
    }


def create_preflight_receipt(
    *,
    protocol_path: Path,
    dataset_root: Path,
    toolkit_root: Path,
    input_root: Path,
    eval_root: Path,
    repo_root: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    dataset = dataset_preflight(dataset_root, protocol)
    toolkit = toolkit_preflight(toolkit_root, protocol)
    predictions = validate_prediction_bundle(input_root, dataset_root, protocol)
    repository = repo_state(repo_root, str(protocol["baseline_commit"]))
    commands = build_official_commands(
        input_root,
        toolkit_root,
        eval_root,
        protocol,
        python_executable,
    )
    errors = [
        *dataset["errors"],
        *toolkit["errors"],
        *predictions["errors"],
    ]
    if repository["baseline_is_ancestor"] is False:
        errors.append("declared R3 baseline is not an ancestor of the current checkout")
    if repository["baseline_is_ancestor"] is None:
        errors.append("could not verify declared R3 baseline ancestry")
    receipt: dict[str, Any] = {
        "schema_version": "poseloop.r3.preflight.v1",
        "protocol_id": protocol["protocol_id"],
        "package_version": PACKAGE_VERSION,
        "created_utc": utc_now(),
        "status": "ready" if not errors else "blocked",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256_file(protocol_path),
            "state": protocol["state"],
        },
        "repository": repository,
        "dataset": dataset,
        "toolkit": toolkit,
        "prediction_bundle": predictions,
        "official_commands": commands,
        "official_evaluation_executed": False,
        "evaluator_only_label_paths_accessed": [],
        "errors": errors,
    }
    fingerprint = _fingerprint(receipt)
    receipt["fingerprint"] = fingerprint
    receipt["fingerprint_sha256"] = canonical_sha256(fingerprint)
    return receipt


def freeze_input_lock(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if preflight.get("status") != "ready":
        raise ContractError("Cannot freeze an input lock from a blocked preflight")
    lock = dict(preflight)
    lock["schema_version"] = "poseloop.r3.input-lock.v1"
    lock["frozen_utc"] = utc_now()
    lock["state"] = "frozen_before_official_evaluation"
    return lock


def validate_input_lock(lock_path: Path, current: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(lock_path)
    if not isinstance(lock, dict) or lock.get("schema_version") != "poseloop.r3.input-lock.v1":
        raise ContractError("R3 input lock schema is invalid")
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R3 input lock protocol ID mismatch")
    if current.get("status") != "ready":
        raise ContractError("Current R3 inputs are not ready")
    if lock.get("fingerprint_sha256") != current.get("fingerprint_sha256"):
        raise ContractError("Current R3 inputs differ from the frozen input lock")
    return lock


def validate_label_access_receipt(
    authorization_path: Path,
    input_lock_path: Path,
) -> dict[str, Any]:
    value = read_json(authorization_path)
    if not isinstance(value, dict):
        raise ContractError("Label-access authorization must be a JSON object")
    required = {
        "schema_version",
        "protocol_id",
        "approved",
        "scope",
        "approved_by",
        "approved_at_utc",
        "input_lock_sha256",
    }
    if set(value) != required:
        raise ContractError("Label-access authorization fields differ from the frozen schema")
    if value["schema_version"] != "poseloop.r3.label-access.v1":
        raise ContractError("Label-access authorization schema is invalid")
    if value["protocol_id"] != PROTOCOL_ID or value["approved"] is not True:
        raise ContractError("Official evaluation is not explicitly approved for this protocol")
    if value["scope"] != "official_bop_toolkit_evaluator_only":
        raise ContractError("Label access must be restricted to the official BOP evaluator")
    if value["input_lock_sha256"] != sha256_file(input_lock_path):
        raise ContractError("Label-access authorization does not match the frozen input lock")
    for field in ("approved_by", "approved_at_utc"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ContractError(f"Label-access authorization {field} is empty")
    return value


def _load_official_scores(commands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for command in commands:
        path = Path(str(command["expected_score"]))
        if not path.is_file():
            raise ContractError(f"Official evaluator did not produce expected score: {path}")
        value = read_json(path)
        if not isinstance(value, dict):
            raise ContractError(f"Official score file is not a JSON object: {path}")
        key = f"{command['variant']}:{command['metric']}"
        scores[key] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "scores": value,
        }
    return scores


def _score_delta(
    scores: Mapping[str, Any],
    metric: str,
    score_name: str,
) -> float | None:
    single_key = f"single_view:{metric}"
    multi_key = f"multi_view:{metric}"
    if single_key not in scores or multi_key not in scores:
        return None
    single = scores[single_key]["scores"].get(score_name)
    multi = scores[multi_key]["scores"].get(score_name)
    if not isinstance(single, (int, float)) or not isinstance(multi, (int, float)):
        return None
    return float(multi) - float(single)


def execute_official_evaluation(
    *,
    current_preflight: Mapping[str, Any],
    input_lock_path: Path,
    authorization_path: Path,
    dataset_root: Path,
    output_root: Path,
    repo_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    validate_input_lock(input_lock_path, current_preflight)
    authorization = validate_label_access_receipt(authorization_path, input_lock_path)
    assert_r3_output_scope(output_root, repo_root, protocol)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BOP_PATH"] = str(dataset_root.resolve().parent)
    env["BOP_RESULTS_PATH"] = str(
        Path(current_preflight["prediction_bundle"]["root"]).resolve()
    )
    env["BOP_EVAL_PATH"] = str(output_root.resolve())

    python_command = str(current_preflight["official_commands"][0]["argv"][0])
    python_path = shutil.which(python_command)
    if python_path is None and Path(python_command).is_file():
        python_path = str(Path(python_command).resolve())
    if python_path is not None:
        env["PATH"] = str(Path(python_path).parent) + os.pathsep + env.get("PATH", "")

    executions: list[dict[str, Any]] = []
    commands = list(current_preflight["official_commands"])
    for index, command in enumerate(commands, start=1):
        completed = subprocess.run(
            list(command["argv"]),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        log_path = output_root / "logs" / f"{index:02d}_{command['metric']}_{command['variant']}.log"
        write_text_atomic(
            log_path,
            f"argv={json.dumps(command['argv'])}\nexit_code={completed.returncode}\n"
            f"\n[stdout]\n{completed.stdout}\n[stderr]\n{completed.stderr}",
        )
        executions.append(
            {
                "metric": command["metric"],
                "variant": command["variant"],
                "exit_code": completed.returncode,
                "log": str(log_path.resolve()),
                "log_sha256": sha256_file(log_path),
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Official evaluator failed for {command['metric']} / {command['variant']}; "
                f"see {log_path}"
            )

    scores = _load_official_scores(commands)
    result = {
        "schema_version": "poseloop.r3.official-scores.v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "input_lock": {
            "path": str(input_lock_path.resolve()),
            "sha256": sha256_file(input_lock_path),
        },
        "authorization": {
            "path": str(authorization_path.resolve()),
            "sha256": sha256_file(authorization_path),
            "approved_by": authorization["approved_by"],
            "scope": authorization["scope"],
        },
        "official_evaluation_executed": True,
        "executions": executions,
        "scores": scores,
        "single_vs_multiview_delta": {
            "bop19_average_recall": _score_delta(
                scores,
                "official_bop19_localization_ar",
                "bop19_average_recall",
            ),
            "bop24_mAP": _score_delta(
                scores,
                "official_bop24_pose_detection_ap",
                "bop24_mAP",
            ),
        },
        "claim_boundary": protocol["claim_scope"],
    }
    result_path = output_root / "official_scores.json"
    write_json_atomic(result_path, result)
    return result
