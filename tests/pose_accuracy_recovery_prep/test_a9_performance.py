import json
from pathlib import Path

import pytest

from pose_accuracy_recovery_prep.a9_foundationpose_e2e import performance, runtime


def test_safety_selection_keeps_one_original_item_per_scene(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [
        {"scene_id": 10, "item_id": "a"}, {"scene_id": 10, "item_id": "b"},
        {"scene_id": 25, "item_id": "c"},
    ]}))
    safety = performance.variant_identity(protocol, manifest, "a" * 40, safety=True)
    full = performance.variant_identity(protocol, manifest, "a" * 40, safety=False)
    assert safety["safety_item_ids"] == ["a", "c"]
    assert full["safety_item_ids"] == []
    assert full["overrides"] == {"refine_batch_size": 64}
    assert full["is_frozen_release_run"] is False
    full["overrides"]["refine_batch_size"] = 128
    with pytest.raises(runtime.ContractError, match="single-variable"):
        performance.validate_variant(full, protocol, manifest, "a" * 40)


def test_historical_cli_does_not_expose_performance_override() -> None:
    args = runtime._parser().parse_args([
        "run-primary", "--manifest", "m", "--foundationpose-root", "f",
        "--output-root", "o", "--implementation-commit", "a" * 40,
    ])
    assert not hasattr(args, "performance_variant")
    protocol = runtime.load_protocol(args.protocol)
    assert protocol["foundationpose"]["resource_batches"] == {
        "warp": 32, "refine": 32, "score_data": 8, "score_feature": 32,
    }


def test_historical_resume_rejects_performance_output_before_completion(tmp_path: Path) -> None:
    lock = {"performance_variant": {"overrides": {"refine_batch_size": 64}}}
    lock["run_lock_sha256"] = runtime._canonical_sha256(lock)
    (tmp_path / "run-lock.json").write_text(json.dumps(lock))
    with pytest.raises(runtime.ContractError, match="run lock changed"):
        runtime.preflight_primary_output(tmp_path, resume=True, expected={"performance_variant": None})


def test_performance_root_is_create_only(tmp_path: Path) -> None:
    with pytest.raises(runtime.ContractError, match="exists"):
        runtime.preflight_primary_output(tmp_path, resume=False, expected={})
