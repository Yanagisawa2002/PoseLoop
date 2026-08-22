#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Download and extract the three XYZ-IBD archives required by PoseLoop M0.

Usage:
  scripts/prepare_xyzibd.sh [--archive-dir PATH] [--data-root PATH]

Options:
  --archive-dir PATH  Archive cache (default: <asset-base>/archives/xyzibd)
  --data-root PATH    Extraction destination (default: <asset-base>/xyzibd)
  -h, --help          Show this help

The asset base is POSELOOP_DATA_ROOT when set, otherwise $HOME/datasets.
Large downloads use POSELOOP_DOWNLOAD_JOBS parallel HTTP ranges (default: 8).
Interrupted downloads remain as *.part files and are resumed on the next run.
The data root must be empty or previously completed by this script.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

asset_base="${POSELOOP_DATA_ROOT:-${HOME}/datasets}"
archive_dir="${asset_base}/archives/xyzibd"
data_root="${asset_base}/xyzibd"

while (($# > 0)); do
  case "$1" in
    --archive-dir)
      (($# >= 2)) || die "--archive-dir requires a path"
      archive_dir="$2"
      shift 2
      ;;
    --archive-dir=*)
      archive_dir="${1#*=}"
      shift
      ;;
    --data-root)
      (($# >= 2)) || die "--data-root requires a path"
      data_root="$2"
      shift 2
      ;;
    --data-root=*)
      data_root="${1#*=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$archive_dir" ]] || die "--archive-dir must not be empty"
[[ -n "$data_root" ]] || die "--data-root must not be empty"

require_command curl
require_command unzip
require_command mktemp
require_command sha256sum
require_command stat
require_command cat

mkdir -p -- "$archive_dir"
archive_dir="$(cd -- "$archive_dir" && pwd -P)"

data_name="$(basename -- "$data_root")"
[[ "$data_name" != "." && "$data_name" != ".." && -n "$data_name" ]] \
  || die "--data-root must name a dedicated directory"
data_parent="$(dirname -- "$data_root")"
mkdir -p -- "$data_parent"
data_parent="$(cd -- "$data_parent" && pwd -P)"
data_root="${data_parent}/${data_name}"
[[ "$archive_dir" != "$data_root" ]] \
  || die "Archive directory and data root must be different paths"

validate_data_root() {
  local root="$1"
  local camera_files=("${root}"/camera*.json)
  local model_files=("${root}"/models/obj_*.ply)

  [[ -e "${camera_files[0]}" ]] || die "Data root has no camera*.json: $root"
  [[ -f "${root}/models/models_info.json" ]] \
    || die "Data root is missing models/models_info.json: $root"
  [[ -e "${model_files[0]}" ]] || die "Data root has no models/obj_*.ply: $root"
  [[ -d "${root}/models_eval" ]] \
    || die "Data root is missing models_eval/: $root"
  [[ -d "${root}/val" ]] || die "Data root is missing val/: $root"
  [[ -n "$(find "${root}/val" -mindepth 1 -print -quit)" ]] \
    || die "Data root has an empty val/: $root"
}

completion_marker="${data_root}/.poseloop_xyzibd_complete"
if [[ -f "$completion_marker" ]]; then
  validate_data_root "$data_root"
  printf 'XYZ-IBD is already prepared at %s\n' "$data_root"
  exit 0
fi

data_root_was_empty=0
if [[ -e "$data_root" ]]; then
  [[ -d "$data_root" ]] || die "Data root exists but is not a directory: $data_root"
  if [[ -n "$(find "$data_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "Data root is non-empty but has no completion marker: $data_root"
  fi
  data_root_was_empty=1
fi

readonly xyzibd_revision="4fe4671783172622313ac0c7182012cee618f217"
readonly xyzibd_base_url="https://huggingface.co/datasets/bop-benchmark/xyzibd/resolve/${xyzibd_revision}"
readonly urls=(
  "${xyzibd_base_url}/xyzibd_base.zip"
  "${xyzibd_base_url}/xyzibd_models.zip"
  "${xyzibd_base_url}/xyzibd_val.zip"
)
readonly archive_names=(
  "xyzibd_base.zip"
  "xyzibd_models.zip"
  "xyzibd_val.zip"
)
readonly archive_sha256s=(
  "0eac3085c378001cd942515e85ee38ba6c0fbd9c4c5ccfd067f52f50810a96aa"
  "5ca56d98177c1ec3d083de661449dd0f532554a7af61fb18524730d2a6da7fc9"
  "09c5639c6e55b8c9a0708a344037e98918c5935ddbd91b5db5a9b06f5484c659"
)
readonly archive_sizes=(
  "3121"
  "4077458"
  "7710607370"
)

verify_zip() {
  local zip_path="$1"
  printf 'Verifying %s\n' "$zip_path"
  unzip -tq "$zip_path"
}

verify_sha256() {
  local archive_path="$1"
  local expected_sha256="$2"
  local actual_sha256

  read -r actual_sha256 _ < <(sha256sum "$archive_path")
  [[ "$actual_sha256" == "$expected_sha256" ]] \
    || die "SHA-256 mismatch for $archive_path (expected $expected_sha256, got $actual_sha256)"
}

verify_archive() {
  local archive_path="$1"
  local expected_sha256="$2"

  verify_zip "$archive_path" || die "Archive failed ZIP integrity check: $archive_path"
  verify_sha256 "$archive_path" "$expected_sha256"
}

download_archive() {
  local url="$1"
  local archive_name="$2"
  local expected_sha256="$3"
  local expected_size="$4"
  local final_path="${archive_dir}/${archive_name}"
  local part_path="${final_path}.part"
  local segment_dir="${part_path}.segments"

  if [[ -e "$final_path" && ! -f "$final_path" ]]; then
    die "Archive path exists but is not a regular file: $final_path"
  fi
  if [[ -f "$final_path" ]]; then
    verify_archive "$final_path" "$expected_sha256"
    printf 'Using verified archive %s\n' "$final_path"
    return
  fi

  if [[ -e "$part_path" && ! -f "$part_path" ]]; then
    die "Partial archive path exists but is not a regular file: $part_path"
  fi
  if [[ -f "$part_path" ]] && unzip -tq "$part_path"; then
    verify_sha256 "$part_path" "$expected_sha256"
    printf 'Promoting already-complete partial archive %s\n' "$part_path"
    mv -- "$part_path" "$final_path"
    return
  fi

  if ((expected_size >= 1073741824)); then
    download_archive_parallel "$url" "$part_path" "$expected_size"
  else
    printf 'Downloading %s\n' "$url"
    curl \
      --fail \
      --location \
      --show-error \
      --retry 5 \
      --retry-all-errors \
      --connect-timeout 30 \
      --continue-at - \
      --output "$part_path" \
      "$url"
  fi

  verify_archive "$part_path" "$expected_sha256"
  mv -- "$part_path" "$final_path"
  if [[ -d "$segment_dir" ]]; then
    [[ "$segment_dir" == "${archive_dir}/"*".part.segments" ]] \
      || die "Refusing to remove unexpected segment directory: $segment_dir"
    rm -rf -- "$segment_dir"
  fi
  printf 'Stored verified archive %s\n' "$final_path"
}

download_archive_parallel() {
  local url="$1"
  local part_path="$2"
  local expected_size="$3"
  local jobs="${POSELOOP_DOWNLOAD_JOBS:-8}"
  local segment_bytes=268435456
  local segment_count=$(((expected_size + segment_bytes - 1) / segment_bytes))
  local segment_dir="${part_path}.segments"
  local assembled_path="${part_path}.assembled"
  local existing_size=0
  local index
  local failed=0
  local pids=()

  [[ "$jobs" =~ ^[0-9]+$ ]] \
    || die "POSELOOP_DOWNLOAD_JOBS must be an integer from 1 to 32"
  jobs=$((10#$jobs))
  ((jobs >= 1 && jobs <= 32)) \
    || die "POSELOOP_DOWNLOAD_JOBS must be an integer from 1 to 32"
  [[ ! -L "$part_path" && ! -L "$segment_dir" && ! -L "$assembled_path" ]] \
    || die "Refusing to write through a symlink in the parallel download cache"
  mkdir -p -- "$segment_dir"

  if [[ -f "$part_path" ]]; then
    existing_size="$(stat -c %s "$part_path")"
    if ((existing_size == expected_size)); then
      printf 'Using already-assembled partial archive %s\n' "$part_path"
      return
    fi
    ((existing_size <= segment_bytes)) \
      || die "Existing partial is too large to seed parallel ranges: $part_path"
    if [[ ! -e "${segment_dir}/000000.part" ]]; then
      mv -- "$part_path" "${segment_dir}/000000.part"
      printf 'Seeded parallel ranges with %s existing bytes\n' "$existing_size"
    fi
  fi

  download_segment() {
    local segment_index="$1"
    local start=$((segment_index * segment_bytes))
    local end=$((start + segment_bytes - 1))
    local expected_segment_size
    local segment_path
    local current_size=0
    local request_start
    local tail_path
    local tail_size

    if ((end >= expected_size)); then
      end=$((expected_size - 1))
    fi
    expected_segment_size=$((end - start + 1))
    segment_path="$(printf '%s/%06d.part' "$segment_dir" "$segment_index")"
    tail_path="${segment_path}.tail"
    [[ ! -L "$segment_path" && ! -L "$tail_path" ]] \
      || die "Refusing to write through a symlink in range segment $segment_index"
    if [[ -f "$segment_path" ]]; then
      current_size="$(stat -c %s "$segment_path")"
    fi
    ((current_size <= expected_segment_size)) \
      || die "Oversized range segment: $segment_path"
    if [[ -f "$tail_path" ]]; then
      tail_size="$(stat -c %s "$tail_path")"
      ((current_size + tail_size <= expected_segment_size)) \
        || die "Oversized partial range response: $tail_path"
      cat -- "$tail_path" >>"$segment_path"
      rm -f -- "$tail_path"
      current_size=$((current_size + tail_size))
    fi
    if ((current_size == expected_segment_size)); then
      return
    fi

    request_start=$((start + current_size))
    printf 'Downloading byte range %d-%d\n' "$request_start" "$end"
    curl \
      --fail \
      --location \
      --silent \
      --show-error \
      --retry 5 \
      --retry-all-errors \
      --connect-timeout 30 \
      --range "${request_start}-${end}" \
      --output "$tail_path" \
      "$url"
    tail_size="$(stat -c %s "$tail_path")"
    ((tail_size == expected_segment_size - current_size)) \
      || die "Wrong response size for byte range ${request_start}-${end}"
    cat -- "$tail_path" >>"$segment_path"
    rm -f -- "$tail_path"
  }

  printf 'Downloading %s in %d resumable ranges with %d jobs\n' \
    "$url" "$segment_count" "$jobs"
  for ((index = 0; index < segment_count; index++)); do
    download_segment "$index" &
    pids+=("$!")
    if ((${#pids[@]} >= jobs)); then
      wait "${pids[0]}" || failed=1
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  ((failed == 0)) || die "One or more parallel byte ranges failed"

  : >"$assembled_path"
  for ((index = 0; index < segment_count; index++)); do
    cat -- "$(printf '%s/%06d.part' "$segment_dir" "$index")" >>"$assembled_path"
  done
  [[ "$(stat -c %s "$assembled_path")" == "$expected_size" ]] \
    || die "Assembled archive has the wrong size: $assembled_path"
  mv -- "$assembled_path" "$part_path"
}

for index in "${!urls[@]}"; do
  download_archive \
    "${urls[$index]}" \
    "${archive_names[$index]}" \
    "${archive_sha256s[$index]}" \
    "${archive_sizes[$index]}"
done

staging_prefix="${data_parent}/.${data_name}.extract."
staging_dir="$(mktemp -d "${staging_prefix}XXXXXX")"

cleanup() {
  if [[ -n "${staging_dir:-}" && -d "$staging_dir" && "$staging_dir" == "$staging_prefix"* ]]; then
    rm -rf -- "$staging_dir"
  fi
}
trap cleanup EXIT

raw_root="${staging_dir}/raw"
mkdir -p -- "${raw_root}/base" "${raw_root}/models" "${raw_root}/val"

printf 'Extracting %s\n' "${archive_dir}/xyzibd_base.zip"
unzip -q -o "${archive_dir}/xyzibd_base.zip" -d "${raw_root}/base"
printf 'Extracting %s\n' "${archive_dir}/xyzibd_models.zip"
unzip -q -o "${archive_dir}/xyzibd_models.zip" -d "${raw_root}/models"
printf 'Extracting %s\n' "${archive_dir}/xyzibd_val.zip"
unzip -q -o "${archive_dir}/xyzibd_val.zip" -d "${raw_root}/val"

assert_single_root() {
  local extraction_dir="$1"
  local expected_name="$2"
  local entries=()

  mapfile -d '' entries < <(find "$extraction_dir" -mindepth 1 -maxdepth 1 -print0)
  ((${#entries[@]} == 1)) \
    || die "Unexpected archive layout under $extraction_dir (expected only $expected_name/)"
  [[ -d "${extraction_dir}/${expected_name}" && "${entries[0]}" == "${extraction_dir}/${expected_name}" ]] \
    || die "Unexpected archive root under $extraction_dir (expected $expected_name/)"
}

assert_exact_roots() {
  local extraction_dir="$1"
  shift
  local expected_names=("$@")
  local entries=()
  local expected_name

  mapfile -d '' entries < <(find "$extraction_dir" -mindepth 1 -maxdepth 1 -print0)
  ((${#entries[@]} == ${#expected_names[@]})) \
    || die "Unexpected archive layout under $extraction_dir"
  for expected_name in "${expected_names[@]}"; do
    [[ -d "${extraction_dir}/${expected_name}" ]] \
      || die "Missing expected archive root: ${expected_name}/"
  done
}

assert_single_root "${raw_root}/base" "xyzibd"
assert_exact_roots "${raw_root}/models" "models" "models_eval"
assert_single_root "${raw_root}/val" "xyzibd_val"
assert_single_root "${raw_root}/val/xyzibd_val" "val"

normalized_root="${staging_dir}/normalized"
mv -- "${raw_root}/base/xyzibd" "$normalized_root"
[[ ! -e "${normalized_root}/models" ]] \
  || die "Base archive unexpectedly contains models/"
[[ ! -e "${normalized_root}/val" ]] \
  || die "Base archive unexpectedly contains val/"
mv -- "${raw_root}/models/models" "${normalized_root}/models"
mv -- "${raw_root}/models/models_eval" "${normalized_root}/models_eval"
mv -- "${raw_root}/val/xyzibd_val/val" "${normalized_root}/val"

validate_data_root "$normalized_root"

{
  printf 'format=poseloop-xyzibd-v1\n'
  for index in "${!urls[@]}"; do
    printf 'archive=%s\n' "${archive_names[$index]}"
    printf 'sha256=%s\n' "${archive_sha256s[$index]}"
    printf 'source=%s\n' "${urls[$index]}"
  done
} >"${normalized_root}/.poseloop_xyzibd_complete"

if ((data_root_was_empty)); then
  rmdir -- "$data_root" || die "Data root changed during extraction: $data_root"
fi
[[ ! -e "$data_root" ]] || die "Data root appeared during extraction: $data_root"

mv -- "$normalized_root" "$data_root"
rm -rf -- "$staging_dir"
staging_dir=""
trap - EXIT

printf 'XYZ-IBD preparation complete.\n'
printf 'Archives: %s\n' "$archive_dir"
printf 'Data root: %s\n' "$data_root"
