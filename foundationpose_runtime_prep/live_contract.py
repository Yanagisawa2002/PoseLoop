"""Hash-locked live Python runtime preflight and transport acknowledgement."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .common import (
    PREP_PROTOCOL_ID,
    PrepError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
)
from .preflight import validate_python_runtime_contract
from .producer import FIXED_INFERENCE, load_protocol


LIVE_RUNTIME_LOCK_SCHEMA = "poseloop.r4c.prep.python-live-runtime-lock.v1"
LIVE_BACKEND_ACK_SCHEMA = "poseloop.r4c.prep.backend-transport-ack.v1"
LIVE_BACKEND_ID = "official-foundationpose-python-live-v1"

# These are the already-audited production implementations that the live
# adapter imports.  Hashing the exact files prevents a similarly named local
# replacement from silently changing candidate attention or batching.
REUSED_RUNTIME_SOURCES = {
    "scripts/run_r1_sealed_inference.py": (
        "1acf3bad5e64950831d1b137d8fa89ca1182c482952cfb42d293581077a7f1d4"
    ),
    "scripts/run_r3_foundationpose_shard.py": (
        "998e52724e3cedf5789d38dd4a160d922bd2ef38222eb94480934afeb9f6e439"
    ),
    "scripts/run_xyzibd_batch.py": (
        "979622519b98f6249cb69af932180495716cd047303533943c10ceedb37c5b4a"
    ),
}

# The deployed implementation identity is content-addressed independently of
# Git metadata.  The commit remains a separate required provenance field.
IMPLEMENTATION_FILES = (
    "foundationpose_runtime_prep/__init__.py",
    "foundationpose_runtime_prep/__main__.py",
    "foundationpose_runtime_prep/a_output.py",
    "foundationpose_runtime_prep/cli.py",
    "foundationpose_runtime_prep/common.py",
    "foundationpose_runtime_prep/equivalence.py",
    "foundationpose_runtime_prep/live_backend.py",
    "foundationpose_runtime_prep/live_contract.py",
    "foundationpose_runtime_prep/live_producer.py",
    "foundationpose_runtime_prep/manifest.py",
    "foundationpose_runtime_prep/preflight.py",
    "foundationpose_runtime_prep/producer.py",
    "foundationpose_runtime_prep/results.py",
    "foundationpose_runtime_prep/visualization.py",
    "foundationpose_runtime_prep/launch_official_python.sh",
    "foundationpose_runtime_prep/contracts/backend_transport_v1.json",
    "foundationpose_runtime_prep/contracts/producer_protocol_v1.json",
    "foundationpose_runtime_prep/contracts/python_runtime_contract_v1.json",
)

RUNTIME_LOCK_KEYS = {
    "schema_version",
    "protocol_id",
    "backend_id",
    "created_at_utc",
    "protocol_sha256",
    "runtime_contract_sha256",
    "backend_transport_contract_sha256",
    "implementation_root",
    "implementation_commit",
    "implementation_files_sha256",
    "implementation_sha256",
    "poseloop_runtime_root",
    "reused_runtime_sources_sha256",
    "foundationpose_root",
    "foundationpose_source",
    "runtime_sources",
    "environment",
    "gpu",
    "checkpoints",
    "checkpoint_configs",
    "producer_runtime_lock",
    "frozen_inference",
    "label_access_count",
    "gt_path_open_count",
    "evaluator_path_open_count",
    "scorer_path_open_count",
    "official_scorer_run",
}

ACK_KEYS = {
    "schema_version",
    "protocol_id",
    "backend_id",
    "created_at_utc",
    "runtime_lock_sha256",
    "backend_transport_contract_sha256",
    "implementation_commit",
    "implementation_sha256",
    "acknowledged",
    "frozen_inference",
    "commitments",
    "label_access_count",
    "official_scorer_run",
}

ACK_COMMITMENTS = {
    "candidate_limit": 252,
    "pose_hypothesis_count": 252,
    "candidate_pruning": False,
    "score_data_chunk": 8,
    "score_feature_chunk": 32,
    "refiner_feature_chunk": 32,
    "iterations": 5,
    "top_k_count": 5,
    "refiner_trace_state_count": 6,
    "raw_gt_or_evaluator_paths": False,
    "official_scorer": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PrepError(f"Cannot inspect Git source {root}: {' '.join(arguments)}")
    return completed.stdout.strip()


def _validate_commit(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrepError(f"Invalid live runtime commit: {field}")
    return value


def _hash_implementation(root: Path) -> dict[str, str]:
    resolved = root.resolve()
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = resolved / relative
        if not path.is_file() or path.is_symlink():
            raise PrepError(f"Live implementation file is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def _hash_reused_sources(root: Path) -> dict[str, str]:
    resolved = root.resolve()
    observed: dict[str, str] = {}
    for relative, expected in REUSED_RUNTIME_SOURCES.items():
        path = resolved / relative
        if not path.is_file() or path.is_symlink():
            raise PrepError(f"Audited production source is missing: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise PrepError(f"Audited production source SHA differs: {relative}")
        observed[relative] = digest
    return observed


def _verify_foundationpose_source(
    root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = root.resolve()
    commit = _run_git(resolved, "rev-parse", "HEAD")
    if commit != expected.get("commit"):
        raise PrepError("FoundationPose source commit differs from runtime contract")
    origin = _run_git(resolved, "remote", "get-url", "origin").rstrip("/")
    if origin != str(expected.get("repository", "")).rstrip("/"):
        raise PrepError("FoundationPose source origin differs from runtime contract")
    if _run_git(resolved, "status", "--short", "--untracked-files=no"):
        raise PrepError("FoundationPose tracked source tree is not clean")
    return {"repository": origin, "commit": commit, "tracked_tree_clean": True}


def _local_source_commit_from_direct_url(distribution_name: str) -> str:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise PrepError(
            f"Missing live runtime distribution: {distribution_name}"
        ) from exc
    text = distribution.read_text("direct_url.json")
    if not text:
        raise PrepError(
            f"Runtime distribution has no source receipt: {distribution_name}"
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PrepError(
            f"Runtime source receipt is malformed: {distribution_name}"
        ) from exc
    commit = value.get("vcs_info", {}).get("commit_id")
    if commit:
        return _validate_commit(commit, field=distribution_name)
    url = value.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        raise PrepError(
            f"Runtime source receipt has no immutable commit: {distribution_name}"
        )
    # Paths in pip direct_url receipts are URL encoded.  urllib is deliberately
    # local-only; no network request occurs.
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    source = Path(unquote(parsed.path))
    if os.name == "nt" and source.as_posix().startswith("/"):
        source = Path(source.as_posix()[1:])
    commit = _run_git(source, "rev-parse", "HEAD")
    if _run_git(source, "status", "--short", "--untracked-files=no"):
        raise PrepError(f"Runtime source tree is not clean: {distribution_name}")
    return _validate_commit(commit, field=distribution_name)


def _checkpoint_records(
    foundationpose_root: Path, runtime: Mapping[str, Any], field: str
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, expected in runtime[field].items():
        declared = foundationpose_root / str(expected["relative_path"])
        resolved = declared.resolve()
        if not resolved.is_file():
            raise PrepError(f"Live {field} file is missing: {name}")
        if resolved.stat().st_size != int(expected["bytes"]):
            raise PrepError(f"Live {field} byte count differs: {name}")
        digest = sha256_file(resolved)
        if digest != expected["sha256"]:
            raise PrepError(f"Live {field} SHA-256 differs: {name}")
        records[name] = {
            "declared_relative_path": expected["relative_path"],
            "resolved_path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": digest,
        }
    return records


def _gpu_identity(torch: Any, gpu_index: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise PrepError("CUDA is unavailable for the live Python backend")
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise PrepError("Requested live GPU index does not exist")
    torch.cuda.set_device(gpu_index)
    props = torch.cuda.get_device_properties(gpu_index)
    uuid = None
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        uuid = completed.stdout.strip().splitlines()[0]
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        raise PrepError("Cannot freeze the live GPU UUID")
    return {
        "index": gpu_index,
        "name": props.name,
        "uuid": uuid,
        "total_memory_bytes": int(props.total_memory),
        "cuda_runtime": str(torch.version.cuda),
    }


def inspect_live_runtime(
    *,
    protocol_path: Path,
    runtime_contract_path: Path,
    implementation_root: Path,
    implementation_commit: str,
    poseloop_runtime_root: Path,
    foundationpose_root: Path,
    gpu_index: int,
) -> dict[str, Any]:
    """Inspect the live runtime without inference and return immutable evidence."""

    protocol = load_protocol(protocol_path)
    runtime = validate_python_runtime_contract(runtime_contract_path)
    implementation_commit = _validate_commit(
        implementation_commit, field="implementation_commit"
    )
    implementation_files = _hash_implementation(implementation_root)
    implementation_sha = canonical_sha256(implementation_files)
    reused_sources = _hash_reused_sources(poseloop_runtime_root)
    foundationpose = _verify_foundationpose_source(
        foundationpose_root, runtime["foundationpose_source"]
    )

    original_cwd = Path.cwd()
    fp_root = foundationpose_root.resolve()
    if str(fp_root) not in sys.path:
        sys.path.insert(0, str(fp_root))
    os.chdir(fp_root)
    try:
        imported: dict[str, str | None] = {}
        for module_name in runtime["environment"]["required_imports"]:
            module = importlib.import_module(module_name)
            imported[module_name] = getattr(module, "__version__", None)
        import torch
    except Exception as exc:
        raise PrepError(
            f"Live Python runtime import failed: {type(exc).__name__}"
        ) from exc
    finally:
        os.chdir(original_cwd)
    if platform.python_version() != runtime["environment"]["python"]:
        raise PrepError("Live Python version differs from runtime contract")
    if str(torch.__version__) != runtime["environment"]["torch"]:
        raise PrepError("Live Torch version differs from runtime contract")

    runtime_sources = {
        "pytorch3d": _local_source_commit_from_direct_url("pytorch3d"),
        "nvdiffrast": _local_source_commit_from_direct_url("nvdiffrast"),
    }
    for name, field in (
        ("pytorch3d", "pytorch3d_commit"),
        ("nvdiffrast", "nvdiffrast_commit"),
    ):
        if runtime_sources[name] != runtime["runtime_sources"][field]:
            raise PrepError(f"Live runtime source commit differs: {name}")

    checkpoints = _checkpoint_records(foundationpose_root, runtime, "checkpoints")
    configs = _checkpoint_records(foundationpose_root, runtime, "checkpoint_configs")
    checkpoint_hashes = {
        name: checkpoints[name]["sha256"] for name in ("refiner", "scorer")
    }
    model_sha = canonical_sha256(
        {
            "foundationpose_commit": foundationpose["commit"],
            "checkpoint_sha256": checkpoint_hashes,
            "checkpoint_config_sha256": {
                name: configs[name]["sha256"] for name in ("refiner", "scorer")
            },
        }
    )
    producer_runtime_lock = {
        "implementation_commit": implementation_commit,
        "implementation_sha256": implementation_sha,
        "model_sha256": model_sha,
        "checkpoint_sha256": checkpoint_hashes,
    }
    transport_sha = runtime["backend_transport_contract"]["sha256"]
    return {
        "schema_version": LIVE_RUNTIME_LOCK_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": LIVE_BACKEND_ID,
        "created_at_utc": utc_now(),
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "runtime_contract_sha256": sha256_file(runtime_contract_path.resolve()),
        "backend_transport_contract_sha256": transport_sha,
        "implementation_root": str(implementation_root.resolve()),
        "implementation_commit": implementation_commit,
        "implementation_files_sha256": implementation_files,
        "implementation_sha256": implementation_sha,
        "poseloop_runtime_root": str(poseloop_runtime_root.resolve()),
        "reused_runtime_sources_sha256": reused_sources,
        "foundationpose_root": str(fp_root),
        "foundationpose_source": foundationpose,
        "runtime_sources": runtime_sources,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "required_imports": imported,
        },
        "gpu": _gpu_identity(torch, gpu_index),
        "checkpoints": checkpoints,
        "checkpoint_configs": configs,
        "producer_runtime_lock": producer_runtime_lock,
        "frozen_inference": protocol["inference"],
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "scorer_path_open_count": 0,
        "official_scorer_run": False,
    }


def write_live_preflight(
    *,
    protocol_path: Path,
    runtime_contract_path: Path,
    implementation_root: Path,
    implementation_commit: str,
    poseloop_runtime_root: Path,
    foundationpose_root: Path,
    gpu_index: int,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = output_root.resolve()
    lock_path = root / "live-runtime-lock.json"
    ack_path = root / "backend-transport-ack.json"
    if lock_path.exists() or ack_path.exists():
        raise PrepError("Live preflight output namespace is already frozen")
    lock = inspect_live_runtime(
        protocol_path=protocol_path,
        runtime_contract_path=runtime_contract_path,
        implementation_root=implementation_root,
        implementation_commit=implementation_commit,
        poseloop_runtime_root=poseloop_runtime_root,
        foundationpose_root=foundationpose_root,
        gpu_index=gpu_index,
    )
    write_json_atomic(lock_path, lock)
    ack = {
        "schema_version": LIVE_BACKEND_ACK_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "backend_id": LIVE_BACKEND_ID,
        "created_at_utc": utc_now(),
        "runtime_lock_sha256": sha256_file(lock_path),
        "backend_transport_contract_sha256": lock["backend_transport_contract_sha256"],
        "implementation_commit": lock["implementation_commit"],
        "implementation_sha256": lock["implementation_sha256"],
        "acknowledged": True,
        "frozen_inference": FIXED_INFERENCE,
        "commitments": ACK_COMMITMENTS,
        "label_access_count": 0,
        "official_scorer_run": False,
    }
    write_json_atomic(ack_path, ack)
    return lock, ack


def validate_live_lock_and_ack(
    *,
    protocol_path: Path,
    runtime_contract_path: Path,
    runtime_lock_path: Path,
    backend_ack_path: Path,
    revalidate_environment: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = read_json(runtime_lock_path.resolve())
    ack = read_json(backend_ack_path.resolve())
    if not isinstance(lock, dict) or set(lock) != RUNTIME_LOCK_KEYS:
        raise PrepError("Live runtime lock fields differ")
    if lock.get("schema_version") != LIVE_RUNTIME_LOCK_SCHEMA:
        raise PrepError("Live runtime lock schema differs")
    if (
        lock.get("protocol_id") != PREP_PROTOCOL_ID
        or lock.get("backend_id") != LIVE_BACKEND_ID
    ):
        raise PrepError("Live runtime lock identity differs")
    if lock.get("protocol_sha256") != sha256_file(protocol_path.resolve()):
        raise PrepError("Live runtime lock protocol SHA differs")
    if lock.get("runtime_contract_sha256") != sha256_file(
        runtime_contract_path.resolve()
    ):
        raise PrepError("Live runtime contract SHA differs")
    if lock.get("frozen_inference") != FIXED_INFERENCE:
        raise PrepError("Live runtime lock changed the frozen inference contract")
    for counter in (
        "label_access_count",
        "gt_path_open_count",
        "evaluator_path_open_count",
        "scorer_path_open_count",
    ):
        if lock.get(counter) != 0:
            raise PrepError(f"Live runtime lock access counter is nonzero: {counter}")
    if lock.get("official_scorer_run") is not False:
        raise PrepError("Live runtime lock declares official scorer use")
    runtime = validate_python_runtime_contract(runtime_contract_path)
    if (
        lock.get("backend_transport_contract_sha256")
        != runtime["backend_transport_contract"]["sha256"]
    ):
        raise PrepError("Live runtime transport contract SHA differs")

    if not isinstance(ack, dict) or set(ack) != ACK_KEYS:
        raise PrepError("Live backend acknowledgement fields differ")
    if (
        ack.get("schema_version") != LIVE_BACKEND_ACK_SCHEMA
        or ack.get("protocol_id") != PREP_PROTOCOL_ID
        or ack.get("backend_id") != LIVE_BACKEND_ID
        or ack.get("acknowledged") is not True
    ):
        raise PrepError("Live backend acknowledgement identity differs")
    if ack.get("runtime_lock_sha256") != sha256_file(runtime_lock_path.resolve()):
        raise PrepError("Live backend acknowledgement does not bind the runtime lock")
    if (
        ack.get("backend_transport_contract_sha256")
        != lock["backend_transport_contract_sha256"]
    ):
        raise PrepError("Live backend acknowledgement transport SHA differs")
    if (
        ack.get("implementation_commit") != lock["implementation_commit"]
        or ack.get("implementation_sha256") != lock["implementation_sha256"]
    ):
        raise PrepError("Live backend acknowledgement implementation differs")
    if (
        ack.get("frozen_inference") != FIXED_INFERENCE
        or ack.get("commitments") != ACK_COMMITMENTS
    ):
        raise PrepError("Live backend acknowledgement changed frozen semantics")
    if (
        ack.get("label_access_count") != 0
        or ack.get("official_scorer_run") is not False
    ):
        raise PrepError(
            "Live backend acknowledgement crosses the label/scorer boundary"
        )

    if revalidate_environment:
        observed = inspect_live_runtime(
            protocol_path=protocol_path,
            runtime_contract_path=runtime_contract_path,
            implementation_root=Path(lock["implementation_root"]),
            implementation_commit=lock["implementation_commit"],
            poseloop_runtime_root=Path(lock["poseloop_runtime_root"]),
            foundationpose_root=Path(lock["foundationpose_root"]),
            gpu_index=int(lock["gpu"]["index"]),
        )
        # Timestamps are receipt metadata, not runtime identity.
        observed.pop("created_at_utc")
        frozen = dict(lock)
        frozen.pop("created_at_utc")
        if observed != frozen:
            raise PrepError("Live environment differs from the frozen runtime lock")
    return lock, ack
