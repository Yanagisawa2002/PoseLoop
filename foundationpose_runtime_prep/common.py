"""Shared strict-JSON and atomic-I/O helpers for the PREP namespace."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


PREP_PROTOCOL_ID = "poseloop-r4c-foundationpose-runtime-prep-v1"
MANIFEST_SCHEMA = "poseloop.r4c.prep.label-free-manifest.v1"
RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2 = (
    "poseloop.r4c.prep.runtime-isolated-manifest.v2"
)
A_PRODUCER_OUTPUT_SCHEMA = "poseloop.pose-accuracy-recovery.producer-output.v1"
CHECKPOINT_SCHEMA = "poseloop.r4c.prep.item-checkpoint.v1"
RUN_LOCK_SCHEMA = "poseloop.r4c.prep.run-lock.v1"
RESULT_SCHEMA = "poseloop.r4c.prep.producer-result.v1"
RUN_RECEIPT_SCHEMA = "poseloop.r4c.prep.run-receipt.v1"
PROFILE_SAMPLE_SCHEMA = "poseloop.r4c.prep.profile-sample.v1"
PROFILE_SUMMARY_SCHEMA = "poseloop.r4c.prep.profile-summary.v1"
EQUIVALENCE_SCHEMA = "poseloop.r4c.prep.exact-equivalence.v1"
VISUALIZATION_PLAN_SCHEMA = "poseloop.r4c.prep.visualization-plan.v1"


class PrepError(RuntimeError):
    """Raised when an immutable PREP contract is violated."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PrepError(f"Cannot hash file: {path}") from exc
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepError(f"Cannot read strict JSON: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PrepError(f"Cannot read strict JSONL: {path}") from exc
    if not lines:
        raise PrepError(f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise PrepError(f"Blank JSONL line: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PrepError(f"Malformed JSONL: {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise PrepError(f"Non-object JSONL row: {path}:{line_number}")
        rows.append(value)
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(str(destination.parent), os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
    )


def write_bytes_atomic(path: Path, value: bytes) -> None:
    _atomic_write(path, value)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(path, b"".join(canonical_bytes(dict(row)) + b"\n" for row in rows))


def write_text_atomic(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))
