"""Fetch pinned base/models/val archives, extract only A-R9 RealSense assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import zipfile

REVISION = "4fe4671783172622313ac0c7182012cee618f217"
ARCHIVES = {
    "xyzibd_base.zip": (3121, "0eac3085c378001cd942515e85ee38ba6c0fbd9c4c5ccfd067f52f50810a96aa"),
    "xyzibd_models.zip": (4077458, "5ca56d98177c1ec3d083de661449dd0f532554a7af61fb18524730d2a6da7fc9"),
    "xyzibd_val.zip": (7710607370, "09c5639c6e55b8c9a0708a344037e98918c5935ddbd91b5db5a9b06f5484c659"),
}
TRAIN = {0, 5, 15, 20, 35, 45, 50, 55, 60, 70}
EVAL = {10, 25, 30, 40, 65}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_path(name):
    parts = PurePosixPath(name).parts
    if ".." in parts or name.startswith("/"):
        raise ValueError("Unsafe ZIP member")
    if "val" in parts:
        parts = parts[parts.index("val"):]
        if len(parts) < 3 or not parts[1].isdigit():
            return None
        scene = int(parts[1])
        if scene not in TRAIN | EVAL:
            return None
        if len(parts) == 3:
            return Path(*parts) if parts[2] in {"scene_camera_realsense.json", "scene_gt_realsense.json"} else None
        sensor = parts[2]
        if sensor not in {"rgb_realsense", "depth_realsense", "mask_visib_realsense"}:
            return None
        image_id = int(Path(parts[-1]).stem.split("_")[0])
        ids = range(50) if scene in TRAIN else (0, 10, 20, 30, 40)
        return Path(*parts) if image_id in ids else None
    for directory in ("models", "models_eval"):
        if directory in parts:
            return Path(*parts[parts.index(directory):])
    if parts[-1].startswith("camera") and parts[-1].endswith(".json"):
        return Path(parts[-1])
    return None


def fetch(url, path, expected_size, expected_sha):
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        command = ["curl", "--fail", "--location", "--retry", "3", "--retry-all-errors",
                   "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), url]
        # Every download explicitly loads the user's acceleration setup first.
        subprocess.run(["bash", "-c", "source /etc/network_turbo >/dev/null 2>&1 && exec " + shlex.join(command)], check=True)
        if partial.stat().st_size != expected_size or sha(partial) != expected_sha:
            raise RuntimeError(f"Download identity mismatch: {path.name}")
        partial.rename(path)
    if path.stat().st_size != expected_size or sha(path) != expected_sha:
        raise RuntimeError(f"Existing asset identity mismatch: {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    archives = root / "downloads"
    data = root / "xyzibd"
    archives.mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)
    records = []
    for filename, (size, digest) in ARCHIVES.items():
        if shutil.disk_usage(root).free < size + 15 * 1024**3:
            raise RuntimeError("Insufficient disk reserve before download")
        path = archives / filename
        fetch(f"https://huggingface.co/datasets/bop-benchmark/xyzibd/resolve/{REVISION}/{filename}", path, size, digest)
        selected = []
        with zipfile.ZipFile(path) as archive:
            members = [(info, selected_path(info.filename)) for info in archive.infolist() if not info.is_dir()]
            required = sum(info.file_size for info, dest in members if dest is not None)
            if shutil.disk_usage(root).free < required + 15 * 1024**3:
                raise RuntimeError("Insufficient disk reserve before extraction")
            for info, relative in members:
                if relative is None:
                    continue
                target = data / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                # zipfile validates each member's CRC on complete read.
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                selected.append(relative.as_posix())
        records.append({"archive": filename, "archive_sha256": digest, "selected_members": len(selected),
                        "selected_paths_sha256": hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest()})
        print(json.dumps(records[-1]), flush=True)
    weights = root / "weights"
    weights.mkdir(exist_ok=True)
    fetch("https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
          weights / "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth", 185828065,
          "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e")
    receipt = root / "receipts" / "data-download.json"
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(json.dumps({"revision": REVISION, "archives": records, "scene9_extracted": False,
        "remaining_disk_bytes": shutil.disk_usage(root).free}, indent=2) + "\n")


if __name__ == "__main__":
    main()
