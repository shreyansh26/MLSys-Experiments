#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: $(basename "$0") <model> [extra vllm bench serve args...]" >&2
  exit 2
fi
shift || true

SAFETENSORS_FAST_GPU=1 vllm bench latency \
    --model "${MODEL}" --trust-remote-code \
    --enable_expert_parallel --tensor-parallel-size 8 \
    --input-len 8192     \
  	--output-len 128    \
  	--batch-size 1    \
  	--num-iters-warmup 3    \
  	--num-iters 3
