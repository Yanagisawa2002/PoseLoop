#!/usr/bin/env python3
"""CPU-only evidence reconciliation; optionally hash an externally supplied model.

This does not rerun training/inference or establish availability of external data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(root: Path = ROOT, checkpoint: Path | None = None) -> dict:
    def read(path):
        return json.loads((root / path).read_text(encoding='utf-8'))

    prefix = 'reports/reproduction_5090/'
    summary = read(prefix + 'summary.json')
    detector = read(prefix + 'detector-replay.json')
    a9 = read(prefix + 'a9-replay.json')
    tax = read(prefix + 'taxonomy-summary.json')
    release = read('release/v1.1.0/results.json')
    protocol = read('protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json')
    provenance = release['provenance']
    require(summary['training']['final_checkpoint']['sha256'] == detector['checkpoint_sha256']
            == provenance['detector_checkpoint_sha256'], 'Checkpoint identity mismatch')
    require(detector['prediction_manifest_exact'] is True and detector['prediction_manifest_sha256']
            == provenance['detector_prediction_manifest_sha256'], 'Detector manifest identity mismatch')
    aliases = {'instance_precision_iou50': 'precision_iou50', 'instance_recall_iou50': 'recall_iou50',
               'instance_f1_iou50': 'f1_iou50'}
    for key, row in detector['historical_metrics_comparison'].items():
        require(row['actual'] == row['historical'] and row['exact_numeric_match'] is True,
                f'Detector metric mismatch: {key}')
        release_key = aliases.get(key, key)
        if release_key in release['detector']:
            require(row['actual'] == release['detector'][release_key], f'Detector release mismatch: {key}')
    comparison = a9['recorded_metric_comparison']
    aliases = {'runtime_success_count': 'runtime_completed', 'mask_iou50_match_count': 'mask_iou50_matches',
               'joint_pose_success_count': 'joint_pose_successes', 'joint_pose_precision': 'joint_precision',
               'joint_pose_recall': 'joint_recall', 'joint_pose_f1': 'joint_f1', 'joint_pose_ap': 'joint_ap'}
    for key, row in comparison['aggregate'].items():
        require(row['actual'] == row['historical'] == release['pose_pipeline'][aliases.get(key, key)]
                and row['exact'] is True and row['delta'] == 0, f'A9 metric mismatch: {key}')
    require(comparison['per_instance_exact_claim'] is False and comparison['artifact_bit_exact_claim'] is False,
            'Unsupported historical raw-pose identity claim')
    require(set(tax['per_scene']) == {str(s) for s in release['dataset']['scene_ids']}, 'Scene coverage mismatch')
    for scene, counts in tax['per_scene'].items():
        row = comparison['per_scene'][scene]
        require(row['exact'] is True and counts['gt'] == row['historical_gt'] == row['actual']['gt']
                and counts['joint_successes'] == row['historical_joint'] == row['actual']['joint']
                and counts['failures'] == counts['gt'] - counts['joint_successes'], f'Scene mismatch: {scene}')
    buckets = tax['bucket_counts']
    require(sum(buckets.values()) == tax['ground_truth_instance_count'] == release['dataset']['ground_truth_instance_count'],
            'Taxonomy GT total mismatch')
    upstream = sum(buckets[k] for k in ('DETECTOR_MISS', 'MASK_BOUNDARY_IOU_FAILURE', 'OVER_SEGMENTATION', 'UNDER_SEGMENTATION_OR_MERGE'))
    pose = sum(buckets[k] for k in ('POSE_MSPD_ONLY_FAILURE', 'POSE_MSSD_ONLY_FAILURE', 'POSE_MSSD_AND_MSPD_FAILURE'))
    require(upstream == 193 and pose == 95 and buckets['SUCCESS'] == tax['joint_pose_success_count'] == 482
            and pose + buckets['SUCCESS'] == tax['mask_iou50_match_count'] == 577, 'Taxonomy waterfall mismatch')
    require(sum(v['gt'] for v in tax['per_scene'].values()) == 770
            and sum(v['joint_successes'] for v in tax['per_scene'].values()) == 482, 'Scene totals mismatch')
    lock = a9['run_lock']
    primary = a9['primary_receipt']
    require(tax['primary_predictions_sha256'] == primary['predictions_sha256'] == primary['completion']['predictions_sha256'],
            'Primary prediction binding mismatch')
    require(tax['manifest_sha256'] == lock['manifest_sha256'] and tax['protocol_sha256'] == lock['protocol_sha256']
            == provenance['pose_protocol_sha256'], 'A9 protocol/manifest binding mismatch')
    require(tax['primary_completion_receipt_sha256'] == primary['completion_receipt_sha256'], 'Receipt binding mismatch')
    require(lock['frozen_inference']['resource_batches'] == protocol['foundationpose']['resource_batches'], 'Frozen batching changed')
    for key, value in lock['runtime_boundary'].items():
        require(value == primary['completion'][key] == 0, f'Primary boundary changed: {key}')
    for relative, expected in lock['implementation']['files_sha256'].items():
        require(sha256(root / relative) == expected, f'Frozen source checksum mismatch: {relative}')
    performance = summary['performance']
    ab = performance['ab']
    require(performance['decision'] == 'REJECTED' and performance['evaluation_comparison']['pass_gate'] is False
            and ab['successful_optimization'] is False and ab['equivalence_preserving'] is False,
            'Rejected variant misrepresented as accepted')
    a, b = ab['runs']['A1'], ab['runs']['B1']
    require(abs(ab['observed_single_pair']['wall_reduction_percent'] - (a-b)/a*100) < 1e-10, 'Timing arithmetic mismatch')
    require(a == primary['completion']['wall_time_seconds'] and ab['runs']['A2'] is None
            and ab['runs']['B2'] is None and performance['optimized_profile'] is None, 'Timing/profile provenance mismatch')
    require(summary['project_status'] == 'FROZEN', 'Project experimental freeze changed')
    if checkpoint is not None:
        require(sha256(checkpoint) == provenance['detector_checkpoint_sha256'], 'External checkpoint SHA-256 mismatch')
    return {'status': 'PASS_TRACKED_REPRODUCTION_INTEGRITY', 'gpu_rerun': False,
            'external_checkpoint_verified': checkpoint is not None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Optional local checkpoint; never downloaded by this tool')
    args = parser.parse_args()
    print(json.dumps(validate(checkpoint=args.checkpoint), indent=2))
