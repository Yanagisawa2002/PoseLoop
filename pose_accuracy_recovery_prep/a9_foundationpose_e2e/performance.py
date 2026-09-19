"""Explicit create-only refiner=64 experiment; historical A9 CLI stays unchanged."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from . import runtime


def variant_identity(protocol: Path, manifest: Path, commit: str, *, safety: bool) -> dict[str, Any]:
    """Bind the one allowed override and optional five-scene safety population."""
    selected: list[str] = []
    if safety:
        seen: set[int] = set()
        for item in runtime._read_json(manifest)["items"]:
            scene = int(item["scene_id"])
            if scene not in seen:
                selected.append(str(item["item_id"]))
                seen.add(scene)
    return {
        "schema_version": "poseloop.performance.refiner64.v1",
        "is_frozen_release_run": False,
        "base_protocol_sha256": runtime._sha256_file(protocol),
        "base_manifest_sha256": runtime._sha256_file(manifest),
        "variant_git_commit": commit,
        "variant_source_sha256": runtime._sha256_file(Path(__file__)),
        "overrides": {"refine_batch_size": 64},
        "safety_item_ids": selected,
    }


def validate_variant(variant: dict[str, Any], protocol: Path, manifest: Path, commit: str) -> None:
    expected = variant_identity(protocol, manifest, commit, safety=bool(variant.get("safety_item_ids")))
    if variant != expected:
        raise runtime.ContractError("Performance variant identity or single-variable override changed")
    relative = Path(__file__).resolve().relative_to(runtime._repo_root()).as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(runtime._repo_root()), "ls-files", "--error-unmatch", relative],
        capture_output=True, text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(runtime._repo_root()), "status", "--porcelain", "--", relative],
        check=True, capture_output=True, text=True,
    ).stdout
    if tracked.returncode or dirty.strip():
        raise runtime.ContractError("Performance entry point is not clean committed source")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=runtime._repo_root() / "protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--safety-check", action="store_true", help="Only first frozen item per scene; cannot be evaluated as a full run")
    args = parser.parse_args(argv)
    variant = variant_identity(args.protocol, args.manifest, args.implementation_commit, safety=args.safety_check)
    result = runtime.run_primary(
        protocol_path=args.protocol, manifest_path=args.manifest,
        foundationpose_root=args.foundationpose_root, output_root=args.output_root,
        implementation_commit=args.implementation_commit, resume=False,
        performance_variant=variant,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
