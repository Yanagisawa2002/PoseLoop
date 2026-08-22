#!/usr/bin/env bash
set -euo pipefail

if [[ "${AUTO_DEPLOY:-false}" != "true" ]]; then
  echo "BLOCKED: AUTO_DEPLOY must be explicitly set to true by a future authorized GPU-C task." >&2
  exit 64
fi

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <static-preflight args...>" >&2
  exit 64
fi

python -m foundationpose_runtime_prep static-preflight "$@"
echo "BLOCKED: Isaac ROS/TensorRT image, release, CUDA/TensorRT, engine, and adapter hashes are unresolved." >&2
exit 78
