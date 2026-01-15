#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: $(basename "$0") <model> [extra vllm bench serve args...]" >&2
  exit 2
fi
shift || true

vllm bench serve --model "${MODEL}" \
                --dataset-name random \
                --host 127.0.0.1 \
                --port 9256 \
                --random-input-len 8192 \
                --random-output-len 0 \
                --request-rate inf \
                --num-prompts 512 \
                "$@"