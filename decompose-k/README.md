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

It supports four benchmark suites:

- `epilogue-bf16`: BF16 `torch.mm + relu`, compiled `torch.mm + relu`,
  custom-op autotuned `mm+relu`, Decompose-K + separate ReLU, and Decompose-K
  fused ReLU.
- `matmul-bf16`: BF16 plain `torch.mm`, compiled `torch.mm`, custom-op
  autotuned `mm`, and standalone Decompose-K.
- `matmul-fp16`: FP16 plain `torch.mm`, compiled `torch.mm`, custom-op
  autotuned `mm`, and standalone Decompose-K.
- `matmul-fp32`: FP32 plain `torch.mm`, compiled `torch.mm`, custom-op
  autotuned `mm`, and standalone Decompose-K.

Run all suites:

```bash
.venv/bin/python -u bench_decompose_k.py \
  --suites all \
  --out-dir bench_results \
  2>&1 | tee bench_results.log
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
- `custom_op_mm_ms` and `custom_op_mm_relu_ms` are also `torch.compile` /
  Inductor paths, but through the registered custom-op autotuning API. Inductor
  internally chooses among the candidates returned by `generate_custom_mm_configs`
  or `generate_mm_relu_configs`, then the benchmark measures the compiled result.
- `decompose_k_unfused_ms` and `decompose_k_fused_ms` are standalone handwritten
  Triton Decompose-K paths from `kernels/decompose_k_triton_kernel.py`.
- `decompose_k_fused_vs_unfused_speedup` is the standalone ReLU epilogue fusion
  benefit: `decompose_k_unfused_ms / decompose_k_fused_ms`. It compares the same
  standalone Decompose-K config with ReLU applied as a separate in-place op
  versus ReLU fused into the reduction/store epilogue.
- `decompose_k_unfused_ms` is not expected to exactly match compiled
  `torch.mm + relu`, even when Inductor also chooses a Decompose-K lowering.
  Inductor's generated path can use `extern_kernels.bmm_dtype` plus generated
  Triton reduction and pointwise ReLU kernels, while the benchmark's Decompose-K
  curves use our standalone Triton partial-matmul and reduction kernels.

CSV columns are grouped as:

1. Shape and dtype metadata.
2. Timing columns, such as `eager_ms`, `compiled_ms`, `custom_op_mm_ms` or
   `custom_op_mm_relu_ms`, and the standalone Decompose-K timing columns.
3. Speedup columns. Ratios are always `baseline_ms / implementation_ms`, so
   values above `1.0` mean the implementation is faster than the named baseline.
4. Config columns. `custom_op_*` columns describe the Inductor custom-op
   autotune winner captured during compile, while `standalone_*` columns
   describe the handwritten Triton config chosen by the explicit
   `candidate_configs(...)` search in `bench_decompose_k.py`.

### Custom-Op Autotune Flow

The custom-op benchmark has two layers of timing.

First, Inductor performs its own internal autotuning while compiling the custom
op. For plain matmul, `register_mm_static_autotune()` registers
`decompose_k::mm` with `generate_custom_mm_configs(...)`. For ReLU epilogue,
`register_mm_relu_static_autotune()` registers `decompose_k::mm_relu` with
`generate_mm_relu_configs(...)`. These config generators inspect the fake tensor
metadata for the current compile shape, enumerate valid Decompose-K split counts
where `K % k_splits == 0`, and return candidates such as:

```python
CustomOpConfig(mm_impl)
CustomOpConfig(decompose_k_impl, k_splits=...)
CustomOpConfig(decompose_k_relu_impl, k_splits=...)
```

Inductor benchmarks those candidates during `torch.compile(...)`, picks the
fastest one for that exact shape, and lowers the selected decomposition to
generated code. That internal autotune step decides whether the custom-op path
uses plain `mm_impl` or a Decompose-K decomposition.

`bench_decompose_k.py` captures that Inductor decision while compiling the
custom-op path and writes it to the CSV:

- `custom_op_autotune_name`: the Inductor custom-op autotune group name.
- `custom_op_best_impl`: selected implementation, such as `mm_impl`,
  `mm_relu_impl`, `decompose_k_impl`, `decompose_k_relu_impl`, or `fallback`.
- `custom_op_k_splits`: selected Decompose-K split count when the winner is a
  Decompose-K candidate; empty for plain/fallback winners.
- `custom_op_choice_name`: lower-level Inductor choice name for debugging and
  matching against autotune logs.

Second, `bench_decompose_k.py` measures the already-compiled callable with
`triton.testing.do_bench`. It does not choose the custom-op candidate directly;
it only measures the result of Inductor's choice:

```text
generate_*_configs
  -> Inductor internal autotune chooses a custom-op decomposition
  -> torch.compile emits the lowered implementation
  -> bench_decompose_k.py measures the compiled callable
```

This is separate from the standalone `decompose_k_ms`,
`decompose_k_unfused_ms`, and `decompose_k_fused_ms` columns. Those standalone
columns are chosen by explicit Python loops over `candidate_configs(...)` in
`bench_decompose_k.py`, not by Inductor's custom-op autotuner.

The standalone config columns are:

- `standalone_split_k`
- `standalone_block_m`
- `standalone_block_n`
- `standalone_block_k`

Those are handwritten Triton kernel parameters, not the Inductor custom-op
winner.

### Dynamo Recompile Limit

The Figure 5 grid is easy to mis-benchmark because it asks `torch.compile` to
specialize the same Python function over many exact shapes. TorchDynamo's
default `config.recompile_limit` is `8` per code object. If the benchmark keeps
recompiling the same function for new `K` values without resetting Dynamo or
using a valid dynamic-shape strategy, Dynamo eventually logs:

```text
torch._dynamo hit config.recompile_limit (8)
```

After that point, later shapes can stop getting fresh optimized graphs and fall
back to slower execution. In the ReLU epilogue grid this can make
`custom_op_mm_relu_ms` look excellent through `(M=N=32, K=8192)` and then jump
up toward the eager/compiled `torch.mm + relu` band for subsequent shapes. That
is a benchmark cache/recompile artifact, not evidence that the custom-op
Decompose-K candidate itself became slower.

When changing `bench_decompose_k.py`, keep the exact-shape compiled baselines
isolated from this limit. Compile time is not part of the latency measurement,
so it is valid for the harness to reset Dynamo between exact-shape grid points
before compiling the next measured callable.

Run one suite:

```bash
.venv/bin/python -u bench_decompose_k.py \
  --suites epilogue-bf16 \
  --out-dir bench_results

.venv/bin/python -u bench_decompose_k.py \
  --suites matmul-bf16 \
  --out-dir bench_results

.venv/bin/python -u bench_decompose_k.py \
  --suites matmul-fp32 \
  --out-dir bench_results
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

- `bench_results/epilogue_relu_bf16.csv`
- `bench_results/plain_matmul_bf16.csv`
- `bench_results/plain_matmul_fp16.csv`
- `bench_results/plain_matmul_fp32.csv`
- `bench_results/*_overall_grid.png`
- `bench_results/*_mn16.png`, `*_mn32.png`, `*_mn48.png`, `*_mn64.png`

## Standalone Decompose-K ReLU Kernel

`kernels/decompose_k_triton_kernel.py` contains the handwritten Triton Decompose-K
partial matmul and reduction/epilogue kernels. It also has a small usage test.

```bash
.venv/bin/python -m kernels.decompose_k_triton_kernel --warmup 5 --rep 20
```

Custom shape:

```bash
.venv/bin/python -m kernels.decompose_k_triton_kernel \
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

The registration call is:

```python
register_custom_op_autotuning(
    custom_op=torch.ops.aten.mm.default,
    config_generator=generate_configs,
    name="router_mm_relu_autotune",
    input_gen_fns={
        "self": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
        "mat2": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
    },
    dispatch_on={"tensor_name": "self", "dim": 0, "range_upper_bound": 1024},
    split_points=SPLIT_POINTS,
    benchmark_with_cudagraphs=True,
)
```

Argument meanings:

- `custom_op`: the operation whose Inductor lowering is being customized. Despite
  the API name, this can be either a real `@torch.library.custom_op` or an
  existing ATen `OpOverload`. This experiment targets
  `torch.ops.aten.mm.default`, so the registration applies when compiled code
  contains `torch.mm`.
- `configs`: an optional static list of candidate configs. Each entry can be a
  `CustomOpConfig` or a callable decomposition. Use this when the candidate list
  is independent of input tensor metadata. This script does not pass `configs`.
- `config_generator`: an optional function that receives fake tensors keyed by
  operator argument name and returns the candidate list for the current compile
  shape. This script uses `generate_configs` so it can inspect `K` and include
  only Decompose-K split counts where `K % k_splits == 0`. `configs` and
  `config_generator` are mutually exclusive.
- `name`: a readable prefix used in autotune logs and generated candidate names.
  For example, this script produces names beginning with
  `router_mm_relu_autotune`.
- `input_gen_fns`: benchmark input generators. During autotuning, Inductor has
  fake tensors with shape/dtype/stride metadata; these functions create real CUDA
  tensors with matching metadata so each candidate can be timed. The dictionary
  keys must match the target op's schema argument names.
- `dispatch_on`: enables range-based dispatch. Here,
  `{"tensor_name": "self", "dim": 0, "range_upper_bound": 1024}` means benchmark
  and dispatch based on `self.shape[0]`, i.e. the `M` dimension of
  `M x K @ K x N`. `range_upper_bound` is the representative benchmark size for
  the final open-ended range; it is not an input validity limit.
- `split_points`: endpoints used with `dispatch_on` to form benchmark ranges. For
  `SPLIT_POINTS = [1, 8, 32, 128, 512]`, the ranges are approximately `[1, 1]`,
  `[2, 8]`, `[9, 32]`, `[33, 128]`, `[129, 512]`, and `[513, inf]`. Inductor
  picks a winner per range and emits a runtime dispatch tree when different
  ranges need different winners.
- `min_speedup_threshold`: optional threshold for selecting a non-fallback
  candidate. The default is `1.0`, meaning any measured speedup over fallback is
  enough. A value like `1.1` would require a 10 percent speedup. This script uses
  the default.
- `benchmark_with_cudagraphs`: whether to use CUDA graph replay for timing the
  fallback/default implementation during autotuning. This can make comparisons
  fairer against compiled/generated candidates. It changes autotune measurement,
  not the mathematical result.

The `input_gen_fns` keys must match the ATen operator schema argument names for
the op being registered. For `torch.ops.aten.mm.default`, the schema names are
`self` for the left matrix and `mat2` for the right matrix:

```bash
.venv/bin/python - <<'PY'
import torch

op = torch.ops.aten.mm.default
print(op._schema)
print([(arg.name, arg.type) for arg in op._schema.arguments])
PY
```

Expected output:

```text
aten::mm(Tensor self, Tensor mat2) -> Tensor
[('self', Tensor), ('mat2', Tensor)]
```

That is why the custom autotune registration uses:

```python
input_gen_fns={
    "self": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
    "mat2": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
}
```

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
