# Decompose-K Experiments

Standalone Triton and PyTorch Inductor experiments for Decompose-K matmul,
custom-op autotuning, and ReLU epilogue fusion.

## Environment

Use the project virtual environment:

```bash
.venv/bin/python --version
```

Most scripts require CUDA.

## ReLU Epilogue And Figure 5 Benchmarks

`bench_decompose_k.py` runs the Figure 5 shape grid:

- `M=N`: `16,32,48,64`
- `K`: `8192,12288,16384,20480,24576,28672,32768`

It supports three benchmark suites:

- `epilogue-bf16`: BF16 `torch.mm + relu`, compiled `torch.mm + relu`,
  Decompose-K + separate ReLU, and Decompose-K fused ReLU.
- `matmul-bf16`: BF16 plain `torch.mm`, compiled `torch.mm`, and Decompose-K.
- `matmul-fp32`: FP32 plain `torch.mm`, compiled `torch.mm`, and Decompose-K.

Run all suites:

```bash
.venv/bin/python -u bench_decompose_k.py \
  --suites all \
  --out-dir fig5_epilogue_results \
  2>&1 | tee fig5_epilogue_results.log
```

The default `--compile-mode` is `max-autotune-no-cudagraphs`. A quick GPU3
comparison on representative Figure 5 shapes picked it over `max-autotune` for
all tested BF16 epilogue, BF16 matmul, and FP32 matmul cases.

For the FP32 matmul suite, the benchmark uses true FP32 behavior: PyTorch matmul
precision is set to `highest`, and the standalone Triton Decompose-K kernel uses
IEEE input precision for FP32 `tl.dot`. This avoids mixing TF32 references with
split FP32 accumulation.

Important naming distinction:

- `compiled_ms` is the `torch.compile` / Inductor path for `torch.mm` or
  `torch.mm + relu`.
- `decompose_k_unfused_ms` and `decompose_k_fused_ms` are standalone handwritten
  Triton Decompose-K paths from `decompose_k_kernel.py`.
- `decompose_k_unfused_ms` is not expected to exactly match compiled
  `torch.mm + relu`, even when Inductor also chooses a Decompose-K lowering.
  Inductor's generated path can use `extern_kernels.bmm_dtype` plus generated
  Triton reduction and pointwise ReLU kernels, while the benchmark's Decompose-K
  curves use our standalone Triton partial-matmul and reduction kernels.

Run one suite:

```bash
.venv/bin/python -u bench_decompose_k.py \
  --suites epilogue-bf16 \
  --out-dir fig5_epilogue_results

.venv/bin/python -u bench_decompose_k.py \
  --suites matmul-bf16 \
  --out-dir fig5_epilogue_results

.venv/bin/python -u bench_decompose_k.py \
  --suites matmul-fp32 \
  --out-dir fig5_epilogue_results
```

Quick smoke test:

```bash
.venv/bin/python -u bench_decompose_k.py \
  --suites all \
  --mns 16 \
  --ks 256 \
  --warmup 1 \
  --rep 3 \
  --out-dir fig5_epilogue_smoke
```

Quick GPU3 compile-mode comparison:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python - <<'PY'
import torch
import triton

shapes = [(16, 8192), (32, 16384), (64, 32768)]
modes = ["max-autotune", "max-autotune-no-cudagraphs"]
cases = [
    ("epilogue-bf16", torch.bfloat16, lambda a, b: torch.relu(torch.mm(a, b))),
    ("matmul-bf16", torch.bfloat16, lambda a, b: torch.mm(a, b)),
    ("matmul-fp32", torch.float32, lambda a, b: torch.mm(a, b)),
]

torch.manual_seed(0)
print(f"torch={torch.__version__} triton={triton.__version__}")
print(f"device={torch.cuda.get_device_name()}")

for case_name, dtype, fn in cases:
    torch.set_float32_matmul_precision("highest" if dtype == torch.float32 else "high")
    compiled = {mode: torch.compile(fn, mode=mode) for mode in modes}
    for mn, k in shapes:
        a = torch.randn((mn, k), device="cuda", dtype=dtype)
        b = torch.randn((k, mn), device="cuda", dtype=dtype)
        eager = fn(a, b)
        for mode in modes:
            cf = compiled[mode]
            torch.testing.assert_close(cf(a, b), eager, rtol=5e-2, atol=5e-2)
            ms = triton.testing.do_bench(lambda: cf(a, b), warmup=10, rep=50)
            print(f"{case_name} M=N={mn} K={k} {mode}: {ms:.5f} ms")
PY
```

Rerun only the real-FP32 matmul suite on GPU3:

```bash
CUDA_VISIBLE_DEVICES=3 DECOMPOSE_K_FORCE_EXIT=1 \
  .venv/bin/python -u bench_decompose_k.py \
  --suites matmul-fp32 \
  --out-dir fp32_matmul_rerun
```

Expected FP32 rerun outputs:

- `fp32_matmul_rerun/plain_matmul_fp32.csv`
- `fp32_matmul_rerun/plain_matmul_fp32_overall_grid.png`
- `fp32_matmul_rerun/plain_matmul_fp32_mn16.png`
- `fp32_matmul_rerun/plain_matmul_fp32_mn32.png`
- `fp32_matmul_rerun/plain_matmul_fp32_mn48.png`
- `fp32_matmul_rerun/plain_matmul_fp32_mn64.png`

Expected outputs:

- `fig5_epilogue_results/epilogue_relu_bf16.csv`
- `fig5_epilogue_results/plain_matmul_bf16.csv`
- `fig5_epilogue_results/plain_matmul_fp32.csv`
- `fig5_epilogue_results/*_overall_grid.png`
- `fig5_epilogue_results/*_mn16.png`, `*_mn32.png`, `*_mn48.png`, `*_mn64.png`

## Standalone Decompose-K ReLU Kernel

`decompose_k_kernel.py` contains the handwritten Triton Decompose-K
partial matmul and reduction/epilogue kernels. It also has a small usage test.

```bash
.venv/bin/python decompose_k_kernel.py --warmup 5 --rep 20
```

Custom shape:

```bash
.venv/bin/python decompose_k_kernel.py \
  --m 16 \
  --n 16 \
  --k 8192 \
  --dtype bf16 \
  --warmup 5 \
  --rep 20
```

## Inductor Custom-Op Autotuning Exploration

`custom_op_autotune_relu_dispatch.py` registers custom autotuning configs on
`torch.ops.aten.mm.default`. It does not handwrite a Triton Decompose-K kernel;
instead, it supplies PyTorch-level decompositions and lets Inductor lower and
autotune them.

## Inductor `torch.mm + relu` Epilogue POC

`inductor_mm_relu_epilogue_poc.py` checks what Inductor emits for a plain
`torch.mm` followed by `relu` at small and large K.

Small K:

```bash
TORCH_LOGS=output_code uv run python inductor_mm_relu_epilogue_poc.py \
  --m 16 \
  --n 16 \
  --k 256 \
  --dtype bf16 \
  --mode max-autotune \
  --no-dynamic \
  2>&1 | tee compile_logs/torch_mm_epilogue_small_k.py
```

Expected small-K result:

- Inductor emits one fused template kernel for both ops.
- In the saved log, look for:
  - `Topologically Sorted Source Nodes: [mm, relu]`
  - `Original ATen: [aten.mm, aten.relu]`
  - `triton_tem_fused_mm_relu_0`
- This means the ReLU epilogue is fused into the generated matmul kernel.

Large K:

```bash
TORCH_LOGS=output_code uv run python inductor_mm_relu_epilogue_poc.py \
  --m 16 \
  --n 16 \
  --k 32768 \
  --dtype bf16 \
  --mode max-autotune \
  --no-dynamic \
  2>&1 | tee compile_logs/torch_mm_epilogue_large_k.py
```

Expected large-K result:

- Inductor selects the Decompose-K matmul path.
- In the saved log, look for:
  - `decompose_k_mm_64_split_5`
  - `extern_kernels.bmm_dtype`
  - `triton_per_fused_mm_0`
  - `triton_poi_fused_relu_1`
- The matmul is decomposed into batched partial matmuls plus a reduction, then
  ReLU is emitted as a separate pointwise kernel. In this path, ReLU is not
  fused into the Decompose-K reduction/store epilogue.

Existing saved logs:

- `compile_logs/torch_mm_epilogue_small_k.py`
- `compile_logs/torch_mm_epilogue_large_k.py`

Run the dynamic-shape exploration:

```bash
.venv/bin/python -u custom_op_autotune_relu_dispatch.py \
  --t-values 1,16,64,256,768 \
  --k 7168 \
  --n 256 \
  --mode max-autotune \
  --dynamic
```

Dump generated Inductor code:

```bash
TORCH_LOGS=output_code DECOMPOSE_K_FORCE_EXIT=1 \
  .venv/bin/python -u custom_op_autotune_relu_dispatch.py \
  --t-values 16 \
  --k 7168 \
  --n 256 \
  --mode max-autotune \
  --dynamic \
  2>&1 | tee custom_op_autotune_relu_dispatch_k7168_output_code.log
```

Useful things to look for in the generated code:

```bash
grep -n "torch.cond\\|bmm_dtype\\|triton_poi_fused_relu\\|CustomOp decompose_k_impl" \
  custom_op_autotune_relu_dispatch_k7168_output_code.log
```
