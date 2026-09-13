#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: run_release_pipeline.sh --dataset-root PATH --detector-checkpoint PATH' \
    '  --foundationpose-root PATH --toolkit-root PATH --output-root PATH'
}

dataset_root=''
detector_checkpoint=''
foundationpose_root=''
toolkit_root=''
output_root=''

while (($#)); do
  case "$1" in
    --dataset-root) dataset_root="$2"; shift 2 ;;
    --detector-checkpoint) detector_checkpoint="$2"; shift 2 ;;
    --foundationpose-root) foundationpose_root="$2"; shift 2 ;;
    --toolkit-root) toolkit_root="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in dataset_root detector_checkpoint foundationpose_root toolkit_root output_root; do
  if [[ -z "${!value_name}" ]]; then
    printf 'Missing required argument: %s\n' "$value_name" >&2
    usage >&2
    exit 2
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
protocol_detector="$repo_root/protocols/poseloop_pose_accuracy_recovery_real_instance_detector_v1.json"
protocol_pose="$repo_root/protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json"

for path in "$dataset_root" "$detector_checkpoint" "$foundationpose_root" "$toolkit_root"; do
  if [[ ! -e "$path" ]]; then
    printf 'Required path does not exist: %s\n' "$path" >&2
    exit 3
  fi
done

if [[ -e "$output_root" ]]; then
  printf 'Output root is create-only and already exists: %s\n' "$output_root" >&2
  exit 4
fi

expected_checkpoint_sha='a90d4134cb36cb242e98481440cfdc782c15b65f8d1068c2964243d7152fec2b'
actual_checkpoint_sha="$(python - "$detector_checkpoint" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"
if [[ "$actual_checkpoint_sha" != "$expected_checkpoint_sha" ]]; then
  printf 'Detector checkpoint SHA-256 mismatch: %s\n' "$actual_checkpoint_sha" >&2
  exit 5
fi

mkdir -p "$output_root"
dataset_dir="$output_root/dataset"
prediction_dir="$output_root/detector-predictions"
freeze_dir="$output_root/freeze"
primary_dir="$output_root/foundationpose-primary"
evaluation_dir="$output_root/evaluation"
archive_path="$output_root/poseloop-v1.1.0-evidence.tar.gz"

python -B -m pose_accuracy_recovery_prep.real_instance_detector_v1 \
  protocol-check --protocol "$protocol_detector"
python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e \
  contract-check --protocol "$protocol_pose"
python -B -m pose_accuracy_recovery_prep.real_instance_detector_v1 \
  build-dataset --protocol "$protocol_detector" --dataset-root "$dataset_root" \
  --output-root "$dataset_dir"
python -B -m pose_accuracy_recovery_prep.real_instance_detector_v1 \
  predict --protocol "$protocol_detector" --dataset-root "$dataset_root" \
  --dataset-manifest "$dataset_dir/dataset-manifest.json" \
  --checkpoint "$detector_checkpoint" --output-root "$prediction_dir"
python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e \
  freeze-inputs --protocol "$protocol_pose" --dataset-root "$dataset_root" \
  --dataset-manifest "$dataset_dir/dataset-manifest.json" \
  --predictions-root "$prediction_dir" --output-root "$freeze_dir"
python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e \
  validate-inputs --protocol "$protocol_pose" \
  --manifest "$freeze_dir/input-manifest.json" --verify-assets

implementation_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]]; then
  printf 'Tracked release worktree must be clean before primary inference.\n' >&2
  exit 6
fi

python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e \
  run-primary --protocol "$protocol_pose" \
  --manifest "$freeze_dir/input-manifest.json" \
  --foundationpose-root "$foundationpose_root" --output-root "$primary_dir" \
  --implementation-commit "$implementation_commit" --resume
python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e \
  evaluate --protocol "$protocol_pose" \
  --manifest "$freeze_dir/input-manifest.json" --primary-root "$primary_dir" \
  --dataset-root "$dataset_root" --toolkit-root "$toolkit_root" \
  --output-root "$evaluation_dir"
python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e \
  package --protocol "$protocol_pose" --freeze-root "$freeze_dir" \
  --primary-root "$primary_dir" --evaluation-root "$evaluation_dir" \
  --archive "$archive_path"

printf 'PoseLoop release pipeline complete: %s\n' "$archive_path"
