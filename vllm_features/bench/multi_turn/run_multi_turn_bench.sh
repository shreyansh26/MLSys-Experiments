# Files here - https://github.com/vllm-project/vllm/tree/a5aa4d5c0f31bba0491a2d9328785dd39dac33c0/benchmarks/multi_turn
#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: $(basename "$0") <model> [extra vllm bench serve args...]" >&2
  exit 2
fi
shift || true

uv run python benchmark_serving_multi_turn.py --model "${MODEL}" \
--input-file generate_multi_turn.json --num-clients 2 --max-active-conversations 6
