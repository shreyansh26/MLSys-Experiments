#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: $(basename "$0") <model> [extra vllm serve args...]" >&2
  exit 2
fi
shift || true

vllm serve "${MODEL}" \
        --port 9256 \
        --max_num_seqs 16 \
        "$@"