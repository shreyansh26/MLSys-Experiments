# lm_eval[api] and lm_eval[vllm]

#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: $(basename "$0") <model> [extra vllm bench serve args...]" >&2
  exit 2
fi
shift || true

lm_eval --model local-completions --model_args model="${MODEL}",base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,num_concurrent=100,trust_remote_code=True --tasks gsm8k
