#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  DATASET_NAME=$1
  shift
else
  DATASET_NAME="infinity_instruct"
fi

CONDA_ENV="shreyansh-env-py11-torch26-v083"
export CUDA_VISIBLE_DEVICES=4

python train.py \
  --dataset-name "${DATASET_NAME}" \
  --num-epochs 2 \
  --use-bf16 \
  "$@"

# ./run_training.sh ifeval_like_data --batch-size 8 --gradient-accumulation-steps 2 --learning-rate 5e-5
