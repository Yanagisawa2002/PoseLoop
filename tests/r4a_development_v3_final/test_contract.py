from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from r4a_development_repair.core import ContractError, sha256_file
from r4a_development_v3_final import (
    PROTOCOL_ID,
    PROTOCOL_ID_V2,
    PROTOCOL_ID_V3,
    PROTOCOL_ID_V4,
)
from r4a_development_v3_final.cli import parse_args
from r4a_development_v3_final.core import (
    _c_result_rows,
    _decode_rle,
    _environment_python,
    load_protocol,
    multi_view_predictions,
    validate_results,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_v3_development_final_evaluation_v1.json"
PROTOCOL_V2 = ROOT / "protocols" / "poseloop_r4a_v3_development_final_evaluation_v2.json"
PROTOCOL_V3 = ROOT / "protocols" / "poseloop_r4a_v3_development_final_evaluation_v3.json"
PROTOCOL_V4 = ROOT / "protocols" / "poseloop_r4a_v3_development_final_evaluation_v4.json"


def _workload(index: int, *, scene: int = 4, obj: int = 2) -> dict[str, object]:
    return {
        "item_id": f"item-{index}",
        "scene_id": scene,
        "image_id": index,
        "object_id": obj,
        "detection_index": index,
    }


def _result(index: int, *, scene: int = 4, obj: int = 2, tx: float = 0.0) -> dict[str, object]:
    pose = np.eye(4)
    pose[0, 3] = tx
    return {
        **_workload(index, scene=scene, obj=obj),
        "status": "success",
        "candidate_limit": 252,
        "pose_hypothesis_count": 252,
        "evaluator_label_read": False,
        "uses_gt_visible_mask": False,
        "uses_oracle_association": False,
        "result_selection_uses_evaluator_metrics": False,
        "predicted_model_to_camera_pose_m": pose.tolist(),
        "camera_world_to_camera_pose_m": np.eye(4).tolist(),
        "registration_seconds": 1.0,
    }


def test_frozen_protocol_identity_and_boundaries() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["role"] == "DEVELOPMENT_ONLY"
    assert protocol["bop24_development"]["variant_order"] == ["single_view", "multi_view"]
    assert protocol["bop24_development"]["maximum_calls_per_variant"] == 1
    assert protocol["immutable_boundaries"]["xyzibd_val_access_permitted"] is False
    assert protocol["immutable_boundaries"]["official_evaluator_rerun_after_success_or_failure_permitted"] is False


def test_protocol_binds_exact_upstream_and_toolkit_hashes() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert sha256_file(PROTOCOL) == "741b7b5f436ba1212cb6445bc94523b798e546bd97df3b75af83d27bd0ab0528"
    for section, names in {
        "upstream_r4a": ["protocol_sha256", "exact_selection_sha256", "no_gt_archive_sha256", "evaluator_only_archive_sha256"],
        "upstream_r4c": [
            "execution_protocol_sha256",
            "adapter_protocol_sha256",
            "adapter_lock_sha256",
            "flat_camera_intrinsics_amendment_protocol_sha256",
            "flat_camera_intrinsics_amendment_lock_sha256",
        ],
    }.items():
        for name in names:
            assert len(protocol[section][name]) == 64
    assert protocol["toolkit"]["commit"] == "cea62d651c7e395b2e1962b9749e4e89693c6ac4"


def test_v2_binds_exact_return_and_known_metadata_adapter() -> None:
    protocol = load_protocol(PROTOCOL_V2)
    assert sha256_file(PROTOCOL_V2) == "d6acf8f0d5f00a7d88e6bcfe69b98de3d1f798b083f0a258f8771311fd6bd96b"
    assert protocol["protocol_id"] == PROTOCOL_ID_V2
    returned = protocol["upstream_r4c"]["return_bundle"]
    assert returned["archive_sha256"] == "87d32fcdbb4be77fa63f284d11ddb1c59285411385c457cd25abd77094641136"
    assert returned["raw_results_sha256"] == "8bb6f6723d473376feb2f649444dc3d4bee4e0bcde86d7a9176b2e7918b66dde"
    assert protocol["gpu_c_result_adapter"]["metadata_rows"] == 1
    assert protocol["supersedes_without_scoring"]["official_evaluator_invocation_count"] == 0


def test_v2_adapter_skips_only_one_known_metadata_row(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL_V2)
    rows = [
        {"record_type": "poseloop_gpu_merged_metadata", "item_count": 10},
        *[{**_result(index), "record_type": "poseloop_gpu_result"} for index in range(10)],
    ]
    path = tmp_path / "merged.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = _c_result_rows(protocol, path)
    assert len(output) == 10
    assert output[0]["item_id"] == "item-0"
    rows[0]["record_type"] = "unknown"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ContractError, match="metadata row is missing"):
        _c_result_rows(protocol, path)


def test_v3_inherits_v2_and_changes_only_interpreter_fix_and_result_names() -> None:
    protocol = load_protocol(PROTOCOL_V3)
    assert sha256_file(PROTOCOL_V3) == "9dd72269331f5ae315d3395741b5d9ee6b0b396de8ec687867ed0932207248ee"
    assert protocol["protocol_id"] == PROTOCOL_ID_V3
    assert protocol["upstream_r4c"]["return_bundle"]["archive_sha256"] == "87d32fcdbb4be77fa63f284d11ddb1c59285411385c457cd25abd77094641136"
    assert protocol["supersedes_without_scoring"]["fixed_command_exit_codes"] == [1, 1]
    assert protocol["supersedes_without_scoring"]["rerun_v2_permitted"] is False
    assert protocol["implementation_fix"]["prediction_input_candidate_association_or_multi_view_change"] is False
    assert protocol["bop24_development"]["result_filenames"]["single_view"].startswith("poseloopr4av3singlev3_")


def test_environment_python_preserves_given_absolute_path(tmp_path: Path) -> None:
    interpreter = tmp_path / "environment" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fixture")
    assert _environment_python(interpreter) == interpreter.absolute()


def test_v4_corrects_v3_static_receipt_hash_before_pre_score() -> None:
    protocol = load_protocol(PROTOCOL_V4)
    assert sha256_file(PROTOCOL_V4) == "b15359855924dcd7ed24bd5db42ee44d9a5e4b0d94b88ca06527997067fbea3a"
    assert protocol["protocol_id"] == PROTOCOL_ID_V4
    assert protocol["v3_static_rejection"]["pre_score_created"] is False
    assert protocol["v3_static_rejection"]["official_evaluator_invocation_count"] == 0
    assert len(protocol["supersedes_without_scoring"]["invocation_started_sha256"]) == 64
    for name in (
        "protocol_sha256",
        "pre_score_sha256",
        "invocation_started_sha256",
        "official_once_sha256",
        "official_log_sha256",
    ):
        assert len(protocol["supersedes_without_scoring"][name]) == 64
    assert protocol["bop24_development"]["result_filenames"]["multi_view"].startswith("poseloopr4av3multiv4_")


def test_result_gate_requires_exact_coverage_and_label_blind_c252() -> None:
    workload = [_workload(0), _workload(1)]
    results = [_result(1), _result(0)]
    validated = validate_results(workload, results)
    assert [row["item_id"] for row in validated] == ["item-0", "item-1"]
    invalid = [_result(0), _result(1)]
    invalid[1]["uses_oracle_association"] = True
    with pytest.raises(ContractError, match="label-blind"):
        validate_results(workload, invalid)


def test_fixed_world_medoid_is_deterministic_and_emits_every_item() -> None:
    rows = [_result(0, tx=0.0), _result(1, tx=0.1), _result(2, tx=1.0)]
    output = multi_view_predictions(rows)
    assert len(output) == 3
    assert [row["item_id"] for row in output] == ["item-0", "item-1", "item-2"]
    assert all(row["multi_view_group_size"] == 3 for row in output)
    assert all(np.isclose(row["predicted_model_to_camera_pose_m"][0][3], 0.1) for row in output)


def test_uncompressed_rle_decoding_matches_column_major_contract() -> None:
    mask = _decode_rle({"size": [2, 3], "counts": [1, 2, 1, 2]})
    np.testing.assert_array_equal(mask, np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.uint8))
    with pytest.raises(ContractError, match="does not cover"):
        _decode_rle({"size": [2, 3], "counts": [1, 2]})


def test_cli_exposes_separate_prepare_once_summary_and_visualization() -> None:
    parsed = parse_args(
        [
            "visualize",
            "--protocol-path",
            "protocol.json",
            "--pre-score-path",
            "pre.json",
            "--output-root",
            "visuals",
            "--output-path",
            "visuals.json",
        ]
    )
    assert parsed.command == "visualize"
    assert parsed.protocol_path == Path("protocol.json")
