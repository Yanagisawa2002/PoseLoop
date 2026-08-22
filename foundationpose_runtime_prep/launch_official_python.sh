#!/usr/bin/env bash
set -euo pipefail

if [[ "${AUTO_DEPLOY:-false}" != "true" ]]; then
  echo "BLOCKED: AUTO_DEPLOY must be explicitly set to true by a future authorized GPU-C task." >&2
  exit 64
fi

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <live-preflight|live-backend-smoke|live-produce> <args...>" >&2
  exit 64
fi

case "$1" in
  live-preflight|live-backend-smoke|live-produce)
    command="$1"
    shift
    exec python -m foundationpose_runtime_prep "$command" "$@"
    ;;
  *)
    echo "BLOCKED: launcher permits only explicit live-preflight, live-backend-smoke, or live-produce." >&2
    exit 64
    ;;
esac
