import copy
import json
import re
from pathlib import Path

import pytest
from pose_accuracy_recovery_prep.a9_foundationpose_e2e import runtime

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json'


@pytest.mark.parametrize('mutation', ['extra_item', 'duplicate_frame', 'unknown_frame'])
def test_rehashed_manifest_cannot_hide_inventory_changes(tmp_path, mutation):
    manifest = {
        'schema_version': runtime.INPUT_SCHEMA, 'protocol_id': runtime.PROTOCOL_ID,
        'protocol_sha256': runtime._sha256_file(PROTOCOL),
        'dataset_role': 'ALREADY_CONSUMED_REAL_DEVELOPMENT', 'frame_count': 25, 'item_count': 820,
        'runtime_boundary': {k: 0 for k in ['label_access_count', 'gt_path_open_count', 'evaluator_path_open_count', 'official_scorer_run_count', 'scene9_read_count']},
        'frames': [{'frame_id': str(i)} for i in range(25)],
        'items': [{'item_id': str(i), 'frame_id': str(i % 25), 'scene_id': 10} for i in range(820)],
    }
    def save(value):
        value['manifest_lock_sha256'] = runtime._canonical_sha256(runtime._without_lock(value, 'manifest_lock_sha256'))
        path = tmp_path / 'input.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        return path
    runtime.validate_input_manifest(save(manifest), PROTOCOL, verify_assets=False)
    bad = copy.deepcopy(manifest)
    if mutation == 'extra_item': bad['items'].append(bad['items'][0])
    elif mutation == 'duplicate_frame': bad['frames'][-1] = bad['frames'][0]
    else: bad['items'][0]['frame_id'] = 'absent'
    with pytest.raises(runtime.ContractError):
        runtime.validate_input_manifest(save(bad), PROTOCOL, verify_assets=False)


def test_provenance_erratum_is_explicit_and_does_not_rewrite_history():
    erratum = json.loads((ROOT / 'docs/corrected-provenance.json').read_text())
    frozen = json.loads((ROOT / erratum['frozen_file']).read_text())
    assert frozen['provenance'][erratum['field']] == erratum['original']
    assert len(erratum['original']) == 65
    assert re.fullmatch('[0-9a-f]{64}', erratum['corrected'])
    assert re.fullmatch('[0-9a-f]{64}', erratum['source_archive_sha256'])
