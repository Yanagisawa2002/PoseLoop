from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "reports" / "r2" / "figures"
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

import gen_m5_r6a_content_replay as visual  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_m5_r6a_visual_selection_recomputes_from_frozen_results() -> None:
    root = REPO_ROOT / "artifacts" / "r2" / "m5_r6a"
    selection = _load_json(FIGURE_DIR / "m5_r6a_content_selection.json")
    result = _load_json(root / "development_result.json")
    tracks = visual.measurement_first_tracks(
        visual.load_jsonl(root / "development_track_results.jsonl")
    )
    global_improvement = result["development_gate"]["observed"][
        "cross_fitted_macro_object_relative_improvement"
    ]
    selected = visual.choose_range_tracks(tracks, global_improvement)
    selected_frames = [visual.choose_rescue_frame(track) for _, track in selected]

    assert [role for role, _ in selected] == ["weakest", "typical", "strongest"]
    assert [track["object_id"] for _, track in selected] == [10, 8, 9]
    assert [row["object_id"] for row in selection["range_examples"]] == [10, 8, 9]
    assert [frame["replay_frame_index"] for frame in selected_frames] == [85, 38, 66]
    assert [row["replay_frame_index"] for row in selection["range_examples"]] == [
        85,
        38,
        66,
    ]
    assert selection["used_for_method_configuration_threshold_metric_or_gate"] is False
    assert selection["selection_timing"] == "after_frozen_development_evaluation"


def test_m5_r6a_missing_storyboard_is_exact_fallback_with_calibrated_decay() -> None:
    root = REPO_ROOT / "artifacts" / "r2" / "m5_r6a"
    tracks = visual.measurement_first_tracks(
        visual.load_jsonl(root / "development_track_results.jsonl")
    )
    track, frames = visual.choose_missing_storyboard(tracks)
    selection = _load_json(FIGURE_DIR / "m5_r6a_content_selection.json")
    recorded = selection["missing_confidence_storyboard"]

    assert track["object_id"] == recorded["object_id"] == 13
    assert [frame["proposed_gap_frames"] for frame in frames] == [0, 1, 6, 18, 36, 0]
    assert [row["gap_frames"] for row in recorded["frames"]] == [0, 1, 6, 18, 36, 0]
    missing = [frame for frame in frames if frame["natural_input_missing"]]
    for frame in missing:
        np.testing.assert_allclose(
            frame["baseline_output_pose_m"],
            frame["proposed_output_pose_m"],
            rtol=0,
            atol=1e-12,
        )
        assert frame["baseline_normalized_pose_error"] == (
            frame["proposed_normalized_pose_error"]
        )
    confidences = [frame["proposed_confidence"] for frame in missing]
    uncertainties = [frame["proposed_uncertainty"] for frame in missing]
    assert confidences == sorted(confidences, reverse=True)
    assert uncertainties == sorted(uncertainties)
    assert recorded["post_gap_confidence"] > recorded["terminal_missing_confidence"]
    assert recorded["post_gap_uncertainty"] < recorded["terminal_missing_uncertainty"]


def test_m5_r6a_visual_outputs_match_receipt_and_video_is_readable() -> None:
    selection = _load_json(FIGURE_DIR / "m5_r6a_content_selection.json")
    input_root = REPO_ROOT / "artifacts" / "r2" / "m5_r6a"
    for receipt in selection["inputs"].values():
        path = input_root / Path(receipt["path"]).name
        assert path.is_file()
        assert _sha256(path) == receipt["sha256"]
    for name, receipt in selection["outputs"].items():
        path = FIGURE_DIR / name
        assert path.is_file()
        assert path.stat().st_size == receipt["size_bytes"]
        assert _sha256(path) == receipt["sha256"]

    for name in ("m5_r6a_content_examples.png", "m5_r6a_missing_confidence.png"):
        image = cv2.imread(str(FIGURE_DIR / name), cv2.IMREAD_UNCHANGED)
        assert image is not None
        assert image.shape[0] >= 1500
        assert image.shape[1] >= 2500

    video = cv2.VideoCapture(str(FIGURE_DIR / "m5_r6a_typical_replay.mp4"))
    assert video.isOpened()
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 128
    readable, frame = video.read()
    video.release()
    assert readable
    assert frame.shape[:2] == (458, 1080)
