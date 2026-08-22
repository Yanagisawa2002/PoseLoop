#!/usr/bin/env python3
"""Fetch and verify the four FoundationPose model-based checkpoint files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


PRIMARY_REPO = "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
PRIMARY_REVISION = "39a143ab8ff830a593f92b7ddf5d66c99e59bb44"


@dataclass(frozen=True)
class Asset:
    run: str
    name: str
    size: int | None = None
    sha256: str | None = None

    @property
    def relative_weight_path(self) -> Path:
        return Path(self.run) / self.name


ASSETS = (
    Asset("2023-10-28-18-33-37", "config.yml"),
    Asset(
        "2023-10-28-18-33-37",
        "model_best.pth",
        68_220_109,
        "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
    ),
    Asset("2024-01-11-20-02-45", "config.yml"),
    Asset(
        "2024-01-11-20-02-45",
        "model_best.pth",
        190_229_389,
        "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_asset(path: Path, asset: Asset) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Empty file: {path}")
    digest = sha256_file(path)
    if asset.size is not None and size != asset.size:
        raise RuntimeError(
            f"Size mismatch for {path}: expected {asset.size}, measured {size}"
        )
    if asset.sha256 is not None and digest != asset.sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {asset.sha256}, measured {digest}"
        )
    return {"size": size, "sha256": digest}


def source_path(asset: Asset) -> str:
    relative = asset.relative_weight_path.as_posix()
    return f"checkpoint/FoundationPose/weights/{relative}"


def download_url(repo_id: str, revision: str, filename: str) -> str:
    encoded = quote(filename, safe="/")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{encoded}"


def download_asset(
    asset: Asset,
    destination: Path,
) -> dict[str, object]:
    if destination.exists():
        return inspect_asset(destination, asset)

    part = destination.with_name(f"{destination.name}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if part.exists():
        if not part.is_file():
            raise RuntimeError(f"Partial path is not a regular file: {part}")
        if asset.size is None:
            part.unlink()
        elif part.stat().st_size == asset.size:
            metadata = inspect_asset(part, asset)
            os.replace(part, destination)
            return metadata
        elif part.stat().st_size > asset.size:
            raise RuntimeError(
                f"Partial file is larger than expected ({part.stat().st_size} > "
                f"{asset.size}): {part}"
            )

    repo_id, revision = PRIMARY_REPO, PRIMARY_REVISION
    filename = source_path(asset)
    url = download_url(repo_id, revision, filename)
    command = [
        "curl",
        "--fail",
        "--location",
        "--show-error",
        "--retry",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
    ]
    if part.exists():
        command.extend(["--continue-at", "-"])
    command.extend(["--output", str(part), url])
    print(f"Downloading {repo_id}@{revision}:{filename}", flush=True)
    subprocess.run(command, check=True)

    metadata = inspect_asset(part, asset)
    os.replace(part, destination)
    return metadata


def materialize_expected_path(source: Path, destination: Path, asset: Asset) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        destination.unlink()
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(
                f"Existing FoundationPose path is not a regular file: {destination}"
            )
        inspect_asset(destination, asset)
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"Existing FoundationPose file differs: {destination}")
        return
    temporary = destination.with_name(f"{destination.name}.part")
    if temporary.exists():
        if not temporary.is_file() or temporary.is_symlink():
            raise RuntimeError(f"Invalid materialization partial: {temporary}")
        temporary.unlink()
    shutil.copyfile(source, temporary)
    inspect_asset(temporary, asset)
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_asset_root = Path(
        os.environ.get(
            "POSELOOP_ASSET_ROOT",
            Path.home() / ".cache" / "poseloop" / "foundationpose-assets",
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=repo_root / "third_party" / "FoundationPose",
    )
    parser.add_argument("--asset-root", type=Path, default=default_asset_root)
    parser.add_argument(
        "--source",
        choices=("primary",),
        default="primary",
        help="Use the pinned official NVIDIA source.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    foundationpose_root = args.foundationpose_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    if not (foundationpose_root / "estimater.py").is_file():
        raise RuntimeError(f"Not a FoundationPose checkout: {foundationpose_root}")
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for resumable downloads")

    repo_id, revision = PRIMARY_REPO, PRIMARY_REVISION
    weight_cache = asset_root / "weights"
    records: list[dict[str, object]] = []
    for asset in ASSETS:
        cached = weight_cache / asset.relative_weight_path
        metadata = download_asset(asset, cached)
        expected = foundationpose_root / "weights" / asset.relative_weight_path
        materialize_expected_path(cached, expected, asset)
        records.append(
            {
                "path": asset.relative_weight_path.as_posix(),
                "source_path": source_path(asset),
                **metadata,
            }
        )
        print(
            f"Verified {asset.relative_weight_path}: "
            f"{metadata['size']} bytes, sha256={metadata['sha256']}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "files": records,
    }
    manifest_path = weight_cache / "poseloop_checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expected_manifest = (
        foundationpose_root / "weights" / "poseloop_checkpoint_manifest.json"
    )
    if expected_manifest.is_symlink():
        if expected_manifest.resolve() != manifest_path.resolve():
            raise RuntimeError(
                f"Existing manifest symlink differs: {expected_manifest}"
            )
        expected_manifest.unlink()
    if expected_manifest.exists():
        if (
            not expected_manifest.is_file()
            or expected_manifest.read_bytes() != manifest_path.read_bytes()
        ):
            raise RuntimeError(
                f"Existing FoundationPose manifest differs: {expected_manifest}"
            )
    else:
        temporary_manifest = expected_manifest.with_name(
            f"{expected_manifest.name}.part"
        )
        if temporary_manifest.exists():
            if not temporary_manifest.is_file() or temporary_manifest.is_symlink():
                raise RuntimeError(
                    f"Invalid manifest materialization partial: {temporary_manifest}"
                )
            temporary_manifest.unlink()
        shutil.copyfile(manifest_path, temporary_manifest)
        os.replace(temporary_manifest, expected_manifest)
    print(f"Checkpoint manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
