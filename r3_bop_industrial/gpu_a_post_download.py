"""Verify and stage the fixed XYZ-IBD archives on GPU-A without reading labels.

This module is intentionally separate from the frozen R3 protocol/scaffold.  It
only verifies the three protocol-pinned archives, normalizes their documented
top-level layout, and validates public target/model metadata.  It never walks
the extracted ``val`` directory and it never invokes an evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


class StageError(RuntimeError):
    """A fail-closed dataset staging error."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_progress(path: Path | None, fields: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered + "\n")


def verify_archive(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise StageError(f"missing archive: {path.name}")
    actual_bytes = path.stat().st_size
    expected_bytes = int(contract["bytes"])
    if actual_bytes != expected_bytes:
        raise StageError(
            f"archive size mismatch for {path.name}: {actual_bytes} != {expected_bytes}"
        )
    actual_sha256 = sha256_file(path)
    expected_sha256 = str(contract["sha256"])
    if actual_sha256 != expected_sha256:
        raise StageError(f"archive SHA-256 mismatch for {path.name}")
    return {
        "path": str(path.resolve()),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "verified": True,
    }


def _validate_int_fields(
    rows: Any, *, name: str, exact_fields: set[str], positive_fields: set[str]
) -> int:
    if not isinstance(rows, list) or not rows:
        raise StageError(f"{name} must be a non-empty JSON list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != exact_fields:
            raise StageError(f"{name} row {index} fields differ from the public contract")
        for field in exact_fields:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise StageError(f"{name} row {index} field {field} must be an integer")
            if field in positive_fields and value <= 0:
                raise StageError(f"{name} row {index} field {field} must be positive")
            if field not in positive_fields and value < 0:
                raise StageError(f"{name} row {index} field {field} must be non-negative")
    return len(rows)


def validate_public_dataset(dataset_root: Path) -> dict[str, Any]:
    """Validate only public files; the val directory is never enumerated."""

    if dataset_root.name != "xyzibd":
        raise StageError("dataset root basename must be xyzibd")
    bop19_path = dataset_root / "test_targets_bop19.json"
    bop24_path = dataset_root / "test_targets_bop24.json"
    models_info_path = dataset_root / "models_eval" / "models_info.json"
    val_path = dataset_root / "val"
    for path in (bop19_path, bop24_path, models_info_path):
        if not path.is_file():
            raise StageError(f"missing public input: {path.relative_to(dataset_root)}")
    if not val_path.is_dir():
        raise StageError("missing public split directory marker: val")

    bop19 = read_json(bop19_path)
    bop24 = read_json(bop24_path)
    models_info = read_json(models_info_path)
    bop19_count = _validate_int_fields(
        bop19,
        name="test_targets_bop19.json",
        exact_fields={"scene_id", "im_id", "obj_id", "inst_count"},
        positive_fields={"inst_count"},
    )
    bop24_count = _validate_int_fields(
        bop24,
        name="test_targets_bop24.json",
        exact_fields={"scene_id", "im_id"},
        positive_fields=set(),
    )
    if not isinstance(models_info, dict) or not models_info:
        raise StageError("models_eval/models_info.json must be a non-empty object")
    return {
        "root": str(dataset_root.resolve()),
        "test_targets_bop19.json": {
            "bytes": bop19_path.stat().st_size,
            "sha256": sha256_file(bop19_path),
            "row_count": bop19_count,
        },
        "test_targets_bop24.json": {
            "bytes": bop24_path.stat().st_size,
            "sha256": sha256_file(bop24_path),
            "row_count": bop24_count,
        },
        "models_eval/models_info.json": {
            "bytes": models_info_path.stat().st_size,
            "sha256": sha256_file(models_info_path),
            "object_count": len(models_info),
        },
        "val": {"exists": True, "enumerated": False},
        "accessed_paths": [
            str(bop19_path.resolve()),
            str(bop24_path.resolve()),
            str(models_info_path.resolve()),
            str(val_path.resolve()),
        ],
        "evaluator_only_label_paths_accessed": [],
    }


def _exact_top_level(root: Path, expected: set[str]) -> None:
    if not root.is_dir():
        raise StageError(f"missing extraction directory: {root.name}")
    actual = {path.name for path in root.iterdir()}
    if actual != expected:
        raise StageError(f"unexpected top-level archive layout under {root.name}")


def _run_unzip(archive: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        ["unzip", "-q", "-o", str(archive), "-d", str(destination)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        # Do not preserve unzip's per-file diagnostics because they may contain
        # evaluator-only path names from the split archive.
        raise StageError(f"unzip failed for {archive.name} with exit {completed.returncode}")
    return {"archive": archive.name, "exit_code": completed.returncode}


def extract_normalized_dataset(
    archives: Mapping[str, Path], data_parent: Path
) -> tuple[Path, dict[str, Any]]:
    if shutil.which("unzip") is None:
        raise StageError("required command is unavailable: unzip")
    final_container = data_parent / "r3_xyzibd_official"
    dataset_root = final_container / "xyzibd"
    if final_container.exists():
        public = validate_public_dataset(dataset_root)
        return dataset_root, {"reused": True, "commands": [], "public": public}

    data_parent.mkdir(parents=True, exist_ok=True)
    staging = data_parent / f".r3_xyzibd_official.extract.{os.getpid()}"
    staging.mkdir(parents=False, exist_ok=False)
    raw = staging / "raw"
    raw.mkdir()
    commands = [
        _run_unzip(archives["xyzibd_base.zip"], raw / "base"),
        _run_unzip(archives["xyzibd_models.zip"], raw / "models"),
        _run_unzip(archives["xyzibd_val.zip"], raw / "val"),
    ]

    _exact_top_level(raw / "base", {"xyzibd"})
    _exact_top_level(raw / "models", {"models", "models_eval"})
    _exact_top_level(raw / "val", {"xyzibd_val"})
    _exact_top_level(raw / "val" / "xyzibd_val", {"val"})

    normalized_container = staging / "normalized_container"
    normalized_container.mkdir()
    normalized = normalized_container / "xyzibd"
    os.replace(raw / "base" / "xyzibd", normalized)
    os.replace(raw / "models" / "models", normalized / "models")
    os.replace(raw / "models" / "models_eval", normalized / "models_eval")
    os.replace(raw / "val" / "xyzibd_val" / "val", normalized / "val")
    public = validate_public_dataset(normalized)
    if final_container.exists():
        raise StageError("final dataset container appeared during extraction")
    os.replace(normalized_container, final_container)
    public = validate_public_dataset(dataset_root)
    return dataset_root, {
        "reused": False,
        "staging_path": str(staging.resolve()),
        "commands": commands,
        "public": public,
    }


def prepare(
    *,
    protocol_path: Path,
    cache_root: Path,
    data_parent: Path,
    json_receipt: Path,
    progress_receipt: Path | None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": "poseloop.r3.gpu-a-dataset-stage.v1",
        "created_utc": utc_now(),
        "status": "blocked",
        "protocol_path": str(protocol_path.resolve()),
        "archives": {},
        "dataset": {},
        "evaluator_only_label_paths_accessed": [],
        "official_evaluation_executed": False,
        "errors": [],
    }
    try:
        protocol = read_json(protocol_path)
        if protocol.get("protocol_id") != "poseloop.r3.bop-industrial.e2e.v1":
            raise StageError("unexpected R3 protocol ID")
        contracts = protocol["dataset"]["source"]["archives"]

        val_final = cache_root / "xyzibd_val.zip"
        val_part = cache_root / "xyzibd_val.zip.part"
        promoted_val_audit: dict[str, Any] | None = None
        if not val_final.exists():
            promoted_val_audit = verify_archive(
                val_part, contracts["xyzibd_val.zip"]
            )
            os.replace(val_part, val_final)
            promoted_val_audit["path"] = str(val_final.resolve())
            promoted_val_audit["promoted_from_part"] = True
        archive_paths = {name: cache_root / name for name in contracts}
        receipt["archives"] = {}
        for name, path in archive_paths.items():
            if name == "xyzibd_val.zip" and promoted_val_audit is not None:
                receipt["archives"][name] = promoted_val_audit
            else:
                receipt["archives"][name] = verify_archive(path, contracts[name])
        dataset_root, extraction = extract_normalized_dataset(archive_paths, data_parent)
        receipt["dataset"] = extraction["public"]
        receipt["extraction"] = {
            key: value for key, value in extraction.items() if key != "public"
        }
        receipt["dataset_root"] = str(dataset_root.resolve())
        receipt["status"] = "ready"
    except (KeyError, OSError, json.JSONDecodeError, StageError) as exc:
        receipt["errors"].append(str(exc))
    receipt["completed_utc"] = utc_now()
    write_json_atomic(json_receipt, receipt)
    append_progress(
        progress_receipt,
        {
            "utc": receipt["completed_utc"],
            "task": "poseloop_r3_xyzibd",
            "post_download_status": receipt["status"],
            "json_receipt": str(json_receipt.resolve()),
        },
    )
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_r3_bop_industrial_protocol.json",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--json-receipt", type=Path, required=True)
    parser.add_argument("--progress-receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = prepare(
        protocol_path=args.protocol.resolve(),
        cache_root=args.cache_root.resolve(),
        data_parent=args.data_parent.resolve(),
        json_receipt=args.json_receipt.resolve(),
        progress_receipt=(
            args.progress_receipt.resolve() if args.progress_receipt is not None else None
        ),
    )
    print(f"R3 GPU-A dataset stage: {receipt['status']}")
    print(f"receipt: {args.json_receipt.resolve()}")
    for error in receipt["errors"]:
        print(f"blocker: {error}")
    return 0 if receipt["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
