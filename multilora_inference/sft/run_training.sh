#!/usr/bin/env bash
set -xeuo pipefail

if [[ $# -gt 0 ]]; then
  DATASET_NAME=$1
  shift
else
  DATASET_NAME="infinity_instruct"
fi

python train.py \
  --dataset-name "${DATASET_NAME}" \
  --num-epochs 2 \
  --use-bf16 \
  "$@"

# cuda4 ./run_training.sh infinity_instruct --batch-size 4 --gradient-accumulation-steps 2 --learning-rate 5e-5
