#!/usr/bin/env python3
"""Hash the frozen legacy M3-M6 evidence without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_RELATIVE_PATH = Path("protocols/poseloop_r1_protocol.json")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "freeze_receipt.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def protected_files(repo_root: Path, protected: list[str]) -> list[Path]:
    files: set[Path] = set()
    for relative in protected:
        path = (repo_root / relative).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Frozen path is missing: {relative}")
        if path.is_file():
            files.add(path)
        else:
            files.update(candidate for candidate in path.rglob("*") if candidate.is_file())
    return sorted(files, key=lambda path: path.relative_to(repo_root).as_posix())


def build_receipt(repo_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    freeze = protocol["legacy_freeze"]
    expected_commit = str(freeze["commit"])
    tag = str(freeze["tag"])
    actual_commit = git_output(repo_root, "rev-parse", f"{tag}^{{}}")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"Freeze tag mismatch: expected {expected_commit}, got {actual_commit}"
        )

    entries = []
    for path in protected_files(repo_root, list(freeze["protected_namespaces"])):
        entries.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "freeze_tag": tag,
        "freeze_commit": actual_commit,
        "current_branch": git_output(repo_root, "branch", "--show-current"),
        "protected_file_count": len(entries),
        "protected_total_bytes": sum(entry["size_bytes"] for entry in entries),
        "protected_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "protected_files": entries,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    protocol_path = repo_root / PROTOCOL_RELATIVE_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    r1_root = (repo_root / "artifacts" / "r1").resolve()
    if not output.is_relative_to(r1_root):
        raise ValueError(f"Freeze receipt must stay under {r1_root}")
    receipt = build_receipt(repo_root, protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        f"Frozen {receipt['protected_file_count']} files / "
        f"{receipt['protected_total_bytes']} bytes; "
        f"manifest={receipt['protected_manifest_sha256']}"
    )
    print(output)


if __name__ == "__main__":
    main()
