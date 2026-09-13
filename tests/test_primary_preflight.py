from pathlib import Path

import pytest

from pose_accuracy_recovery_prep.a9_foundationpose_e2e import runtime


def test_fresh_output_and_absent_resume(tmp_path):
    output = tmp_path / 'primary'
    assert runtime.preflight_primary_output(output, resume=False, expected={}) is None
    with pytest.raises(runtime.ContractError, match='absent'):
        runtime.preflight_primary_output(output, resume=True, expected={})


def test_same_lock_resume_completed_and_changed_lock(tmp_path):
    output = tmp_path / 'primary'
    output.mkdir()
    lock = {'implementation_commit': 'a' * 40}
    lock['run_lock_sha256'] = runtime._canonical_sha256(lock)
    runtime._write_json_atomic(output / 'run-lock.json', lock)
    expected = {'implementation_commit': 'a' * 40}
    assert runtime.preflight_primary_output(output, resume=True, expected=expected) is None
    with pytest.raises(runtime.ContractError, match='exists'):
        runtime.preflight_primary_output(output, resume=False, expected=expected)
    with pytest.raises(runtime.ContractError, match='changed'):
        runtime.preflight_primary_output(output, resume=True, expected={'implementation_commit': 'b' * 40})
    predictions = output / 'predictions.jsonl'
    predictions.write_text('{}\n', encoding='utf-8')
    receipt = {'run_lock_sha256': lock['run_lock_sha256'], 'predictions_sha256': runtime._sha256_file(predictions)}
    runtime._write_json_atomic(output / 'completion-receipt.json', receipt)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert runtime.preflight_primary_output(output, resume=True, expected=expected) == receipt
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    predictions.write_text('changed', encoding='utf-8')
    with pytest.raises(runtime.ContractError, match='changed'):
        runtime.preflight_primary_output(output, resume=True, expected=expected)
