#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUNTIME_ROOT BASE_PYTHON" >&2
  exit 64
fi

RUNTIME_ROOT=$(realpath -m -- "$1")
BASE_PYTHON=$(realpath -- "$2")
FOUNDATIONPOSE_COMMIT=a1b694b83e633c2cb6115b9063d940a687759392
NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
PYTORCH3D_COMMIT=3143b3baf8ef8b1023ed76f225af59e2e8a71e06

mkdir -p -- "$RUNTIME_ROOT/sources" "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/receipts"
source /etc/network_turbo
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.version
export GIT_CONFIG_VALUE_0=HTTP/1.1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=16

MYCPP_APT_PACKAGES=(
  libboost-system-dev
  libboost-program-options-dev
  libeigen3-dev
  pybind11-dev
)
missing_apt_packages=()
for package in "${MYCPP_APT_PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | \
    grep -qx 'install ok installed'; then
    missing_apt_packages+=("$package")
  fi
done
if [[ ${#missing_apt_packages[@]} -gt 0 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends "${missing_apt_packages[@]}"
fi

unexpected_source_status() {
  git -C "$1" status --porcelain | \
    grep -Ev '^\?\? (build/|nvdiffrast\.egg-info/|pytorch3d\.egg-info/)$' || true
}

clone_exact() {
  local repository=$1
  local commit=$2
  local destination=$3
  shift 3
  local slug=${repository#https://github.com/}
  slug=${slug%.git}
  local marker="$destination/.git/poseloop-materialized-commit"
  local archive="$RUNTIME_ROOT/downloads/${slug//\//-}-$commit.tar.gz"
  if [[ ! -d "$destination/.git" ]]; then
    [[ ! -e "$destination" ]] || {
      echo "non-Git source destination already exists: $destination" >&2
      return 65
    }
    mkdir -p -- "$destination"
    git -C "$destination" init
    git -C "$destination" remote add origin "$repository"
  fi
  [[ ! -e "$destination/.git/index.lock" ]] || {
    echo "stale or active Git index lock requires an explicit audit: $destination" >&2
    return 66
  }
  if [[ $# -eq 0 ]] \
    && [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" == "$commit" ]] \
    && [[ -z "$(unexpected_source_status "$destination")" ]]; then
    printf '%s\n' "$commit" >"$marker"
    return 0
  fi
  if ! git -C "$destination" cat-file -e "$commit^{commit}" 2>/dev/null; then
    local attempt
    for attempt in 1 2 3; do
      if git -C "$destination" -c http.version=HTTP/1.1 fetch \
        --depth=1 --filter=blob:none --no-tags origin "$commit"; then
        break
      fi
      echo "bounded fetch retry $attempt/3 for $repository" >&2
    done
    git -C "$destination" cat-file -e "$commit^{commit}"
  fi
  if [[ $# -gt 0 ]]; then
    git -C "$destination" update-ref --no-deref HEAD "$commit"
    git -C "$destination" sparse-checkout init --no-cone
    git -C "$destination" sparse-checkout set "$@"
    git -C "$destination" checkout --detach "$commit"
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]]
    [[ "$(git -C "$destination" remote get-url origin)" == "$repository" ]]
    [[ -z "$(unexpected_source_status "$destination")" ]]
    printf '%s\n' "$commit" >"$marker"
    return 0
  fi
  mkdir -p -- "$RUNTIME_ROOT/downloads"
  if [[ ! -f "$archive" ]]; then
    curl --fail --location --show-error --retry 5 --retry-all-errors \
      --connect-timeout 30 --continue-at - \
      --output "$archive.part" \
      "https://codeload.github.com/$slug/tar.gz/$commit"
    tar -tzf "$archive.part" >/dev/null
    mv -- "$archive.part" "$archive"
  fi
  tar -tzf "$archive" >/dev/null
  git -C "$destination" read-tree --reset "$commit"
  git -C "$destination" update-ref --no-deref HEAD "$commit"
  tar -xzf "$archive" --strip-components=1 -C "$destination"
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]]
  [[ "$(git -C "$destination" remote get-url origin)" == "$repository" ]]
  [[ -z "$(unexpected_source_status "$destination")" ]]
  printf '%s\n' "$commit" >"$marker"
}

if [[ ! -x "$RUNTIME_ROOT/venv/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$RUNTIME_ROOT/venv"
fi
PYTHON="$RUNTIME_ROOT/venv/bin/python"
export PATH="$RUNTIME_ROOT/venv/bin:$PATH"

"$PYTHON" -m pip install --prefer-binary \
  ruamel.yaml==0.18.10 fvcore iopath trimesh==5.0.0 imageio scipy ninja \
  pybind11==2.13.6

clone_exact \
  https://github.com/NVlabs/FoundationPose.git \
  "$FOUNDATIONPOSE_COMMIT" \
  "$RUNTIME_ROOT/sources/FoundationPose"
clone_exact \
  https://github.com/NVlabs/nvdiffrast.git \
  "$NVDIFFRAST_COMMIT" \
  "$RUNTIME_ROOT/sources/nvdiffrast"
clone_exact \
  https://github.com/facebookresearch/pytorch3d.git \
  "$PYTORCH3D_COMMIT" \
  "$RUNTIME_ROOT/sources/pytorch3d" \
  setup.py setup.cfg pyproject.toml MANIFEST.in requirements.txt pytorch3d \
  projects/implicitron_trainer

"$PYTHON" -m pip uninstall -y opencv-python-headless
"$PYTHON" -m pip install --prefer-binary \
  -r "$RUNTIME_ROOT/sources/FoundationPose/requirements.txt"

if ! "$PYTHON" -c 'import nvdiffrast.torch' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --no-deps --no-build-isolation \
    "$RUNTIME_ROOT/sources/nvdiffrast"
fi
if ! "$PYTHON" -c 'import pytorch3d' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --no-deps --no-build-isolation \
    "$RUNTIME_ROOT/sources/pytorch3d"
fi

export CONDA_PREFIX="$RUNTIME_ROOT/venv"
export CMAKE_PREFIX_PATH="$("$PYTHON" -m pybind11 --cmakedir):${CMAKE_PREFIX_PATH:-}"
(
  cd "$RUNTIME_ROOT/sources/FoundationPose"
  bash build_all_conda.sh
)

"$PYTHON" - <<'PY'
import importlib
import json
import torch

versions = {}
for name in (
    "cv2",
    "numpy",
    "trimesh",
    "nvdiffrast.torch",
    "pytorch3d",
):
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", None)
versions["torch"] = torch.__version__
versions["cuda"] = torch.version.cuda
versions["cuda_available"] = torch.cuda.is_available()
if not versions["cuda_available"]:
    raise RuntimeError("CUDA is unavailable after FoundationPose provisioning")
print(json.dumps(versions, sort_keys=True))
PY
