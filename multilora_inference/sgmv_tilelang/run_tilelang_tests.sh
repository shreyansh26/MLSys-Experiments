#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_HOME=/usr
export TILELANG_LIBRARY_PATH=/mnt/ssd1/shreyansh/home_dir/misc_experiments/multilora_inference/sgmv_triton/tilelang/build
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export NV_BASE=$(python - <<'PY'
import sysconfig, os
for key in ("purelib", "platlib"):
    base = sysconfig.get_paths()[key]
    cand = os.path.join(base, "nvidia")
    if os.path.isdir(cand):
        print(cand)
        break
PY
)

export CUDAHOSTCXX="$(command -v g++)"
export LD_LIBRARY_PATH=$NV_BASE/nvjitlink/lib:$NV_BASE/cusparse/lib:$NV_BASE/cublas/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cudnn/lib:$TILELANG_LIBRARY_PATH:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH}

python -m test_sgmv_tilelang "$@"
