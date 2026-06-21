# Decompose-K Experiments

Standalone Triton and PyTorch Inductor experiments for Decompose-K matmul,
custom-op autotuning, and ReLU epilogue fusion.

## Environment

<details>
<summary>Fresh clone setup with uv</summary>

This folder is a self-contained [uv](https://docs.astral.sh/uv/) project. A
fresh checkout should be set up from the project files in this directory:

- `pyproject.toml`: declares Python and package requirements.
- `uv.lock`: pins the exact package versions.
- `.python-version`: selects Python 3.12.

Do not copy an existing `.venv/` from another machine. Recreate it with `uv`.

### Prerequisites

- Linux with an NVIDIA GPU and a driver that can run CUDA 12.8 wheels (`cu128`).
  `nvidia-smi` should report CUDA 12.8 or newer.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.
- Network access to PyPI and the PyTorch nightly CUDA 12.8 wheel index:
  `https://download.pytorch.org/whl/nightly/cu128`.

### Fresh Clone Setup

```bash
git clone <repo-url>
cd <repo>/misc_experiments/decompose-k

# Optional sanity check: these files should all exist in this folder.
ls pyproject.toml uv.lock .python-version
```

Install exactly what the lockfile specifies:

```bash
uv sync --frozen
```

What this does:

- Creates `.venv/` with Python 3.12 if it does not exist.
- Installs the exact versions from `uv.lock`.
- Pulls `torch` and `triton` from the PyTorch nightly CUDA 12.8 index configured
  in `pyproject.toml` (`https://download.pytorch.org/whl/nightly/cu128`).
- Allows prerelease/nightly wheels via `[tool.uv] prerelease = "allow"`.

If `uv` cannot find Python 3.12 locally, install it through `uv` and retry:

```bash
uv python install 3.12
uv sync --frozen
```

At the time the lockfile was generated, that resolved to roughly:

- `torch==2.12.0.dev20260408+cu128`
- `triton==3.7.0+git282c8251`
- `numpy>=2.2.6`

Use `uv sync --frozen` for a reproducible clone. Use plain `uv sync` only if you
intend to let `uv` update the lockfile metadata.

To refresh nightly PyTorch/Triton pins later:

```bash
uv lock --upgrade-package torch --upgrade-package triton
uv sync
```

### Verify the setup

```bash
uv run python --version
uv run python -c "import torch, triton; print(torch.__version__); print(triton.__version__); print('cuda', torch.cuda.is_available())"
```

You should see Python 3.12.x, CUDA available as `True`, and `+cu128` in the
PyTorch version string.

</details>

## What Is Decompose-K?

Decompose-K is a matmul strategy for shapes where the reduction dimension is
large but the output tile is small. For a normal GEMM:

```text
C[M, N] = A[M, K] @ B[K, N]
```

Decompose-K splits the K dimension into `S` independent chunks, computes `S`
partial GEMMs, then reduces those partial outputs:

```text
A[M, K] @ B[K, N]
  -> partials[S, M, N]
  -> sum(partials, dim=0)
```

This creates extra parallelism for skinny, K-dominant matmuls where a normal
`M x N` output grid may not expose enough work to fill the GPU.

The minimal PyTorch version in `gemm.py` is:

```python
def decomposeK(a, b, k_splits):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    assert k == b.shape[0], "Incompatible dimensions"
    assert k % k_splits == 0, "k must be divisible by k_splits"

    k_parts = k // k_splits

    # [m, k_splits, k_parts] -> [k_splits, m, k_parts]
    a_reshaped = a.reshape(m, k_splits, k_parts).permute(1, 0, 2)
    b_reshaped = b.reshape(k_splits, k_parts, n)

    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    reduced_result = result.sum(dim=0)
    return reduced_result.to(a.dtype)
```

`custom_op_autotune_relu_dispatch.py` uses the same reshape-to-batched-GEMM idea
and also defines a `torch.mm + relu` variant where ReLU is applied after the
split reduction. Those implementations are registered as custom-op autotune
candidates. Inductor sees the current fake tensor shape, enumerates valid
`k_splits`, benchmarks plain `mm` or `mm+relu` against the Decompose-K
candidates, and lowers the fastest candidate:

```python
K_SPLITS = (2, 4, 8, 16, 32, 64, 128, 256)


def generate_mm_relu_configs(
    fake_tensors: dict[str, torch.Tensor],
) -> list[CustomOpConfig]:
    k = int(fake_tensors["a"].shape[1])
    splits = [k_splits for k_splits in K_SPLITS if k % k_splits == 0]

    configs = [CustomOpConfig(mm_relu_impl)]
    configs.extend(
        CustomOpConfig(decompose_k_relu_impl, k_splits=k_splits)
        for k_splits in splits
    )
    return configs
```

This is most useful when:

- `K` is very large and `M`/`N` are small, for example `M=N=16..64` with
  `K=8192..32768`.
- The workload is latency-sensitive and single-shape performance matters more
  than amortizing a generic GEMM implementation.
- The leading dimension is dynamic and different ranges want different kernels.
  One example is a DeepSeek V3 MoE router GEMM: `[T, 7168] @ [7168, 256]`,
  where decode has small dynamic token counts (`T=1..256`) and prefill has
  larger sizes.
- A fused epilogue such as ReLU can be applied after the split reduction,
  avoiding a separate pointwise pass in the standalone kernel.

It is usually less attractive when `M` and `N` are already large enough to give
the GPU plenty of output-tile parallelism, when `K` is small, when `K` has poor
divisibility for the candidate split counts, or when the extra
`partials[S, M, N]` buffer and reduction cost dominate.


## ReLU Epilogue And Large-K Benchmarks

`bench_decompose_k.py` runs a small-M/N, large-K shape grid:

- `M=N`: `16,32,48,64`
- `K`: `8192,12288,16384,20480,24576,28672,32768`

It supports four benchmark suites:

- `epilogue-bf16`: BF16 `torch.mm + relu`, compiled `torch.mm + relu`,
  custom-op autotuned `mm+relu`, standalone Triton Decompose-K + separate ReLU,
  and standalone Triton Decompose-K fused ReLU.
- `matmul-bf16`: BF16 plain `torch.mm`, compiled `torch.mm`, custom-op
  autotuned `mm`, and standalone Triton Decompose-K.
- `matmul-fp16`: FP16 plain `torch.mm`, compiled `torch.mm`, custom-op
  autotuned `mm`, and standalone Triton Decompose-K.
- `matmul-fp32`: FP32 plain `torch.mm`, compiled `torch.mm`, custom-op
  autotuned `mm`, and standalone Triton Decompose-K.

The standalone Triton path has two implementations in this folder: the baseline
`kernels/decompose_k_triton_kernel.py` and the optimized
`kernels/decompose_k_triton_kernel_optimized.py`. The current benchmark harness
imports the optimized kernel.

Saved result directories use the same CSV schema but different standalone Triton
implementations:

- `bench_results`: original standalone Triton kernel run.
- `bench_results_v2`: optimized standalone Triton kernel run.

In both directories, the `standalone_*` CSV columns describe the Triton config
that produced the `decompose_k_*` timings; they do not refer to Inductor's
custom-op autotune winner.

Run all suites:

```bash
uv run python -u bench_decompose_k.py \
  --suites all \
  --out-dir bench_results_v2 \
  2>&1 | tee bench_results_v2.log
```

The default `--compile-mode` is `max-autotune-no-cudagraphs`. A quick
comparison on representative benchmark shapes picked it over `max-autotune` for
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
- `decompose_k_ms` is the standalone handwritten Triton Decompose-K timing for
  the plain matmul suites.
- `decompose_k_unfused_ms` and `decompose_k_fused_ms` are the standalone
  handwritten Triton Decompose-K timings for the ReLU epilogue suite.
- The CSV schema does not have separate `baseline_*` or `optimized_*` timing
  columns. `bench_results` uses the original standalone Triton kernel, and
  `bench_results_v2` uses the optimized standalone Triton kernel, but both
  directories keep the same `decompose_k_*` and `standalone_*` column names.
- `decompose_k_fused_vs_unfused_speedup` is the standalone ReLU epilogue fusion
  benefit: `decompose_k_unfused_ms / decompose_k_fused_ms`. It compares the same
  standalone Decompose-K config with ReLU applied as a separate in-place op
  versus ReLU fused into the reduction/store epilogue.
- The standalone `decompose_k_*` timings are not expected to exactly match
  compiled `torch.mm` or compiled
  `torch.mm + relu`, even when Inductor also chooses a Decompose-K lowering.
  Inductor's generated path can use `extern_kernels.bmm_dtype` plus generated
  Triton reduction and pointwise ReLU kernels, while the benchmark's Decompose-K
  curves use our standalone Triton partial-matmul and reduction kernels.

CSV columns are grouped as:

1. Shape and dtype metadata.
2. Timing columns, such as `eager_ms`, `compiled_ms`, `custom_op_mm_ms` or
   `custom_op_mm_relu_ms`, and the standalone Triton Decompose-K timing columns.
3. Speedup columns. Ratios are always `baseline_ms / implementation_ms`, so
   values above `1.0` mean the implementation is faster than the named baseline.
4. Config columns. `custom_op_*` columns describe the Inductor custom-op
   autotune winner captured during compile, while `standalone_*` columns
   describe the handwritten Triton config chosen by the explicit
   `candidate_configs(...)` search. `standalone_*` means "standalone Triton
   config" in both `bench_results` and `bench_results_v2`; it does not mean the
   original/baseline Triton module specifically.

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
`decompose_k_unfused_ms`, and `decompose_k_fused_ms` columns. Those
`decompose_k_*` columns are chosen by explicit Python loops over
`candidate_configs(...)` in `bench_decompose_k.py`, not by Inductor's custom-op
autotuner. The column names are the same in `bench_results` and
`bench_results_v2`: `bench_results` is the original Triton run, while
`bench_results_v2` is the optimized Triton run.

The standalone config columns are:

- `standalone_split_k`
- `standalone_block_m`
- `standalone_block_n`
- `standalone_block_k`

Those are handwritten Triton kernel parameters, not the Inductor custom-op
winner. They use the `standalone_*` prefix in both result directories because
they describe standalone Triton kernel configs.

### Dynamo Recompile Limit

This shape grid is easy to mis-benchmark because it asks `torch.compile` to
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
uv run python -u bench_decompose_k.py \
  --suites epilogue-bf16 \
  --out-dir bench_results

uv run python -u bench_decompose_k.py \
  --suites matmul-bf16 \
  --out-dir bench_results

uv run python -u bench_decompose_k.py \
  --suites matmul-fp32 \
  --out-dir bench_results
```

Quick smoke test:

```bash
uv run python -u bench_decompose_k.py \
  --suites all \
  --mns 16 \
  --ks 256 \
  --warmup 1 \
  --rep 3 \
  --out-dir fig5_epilogue_smoke
```

Quick compile-mode comparison:

```bash
uv run python - <<'PY'
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

Rerun only the real-FP32 matmul suite:

```bash
DECOMPOSE_K_FORCE_EXIT=1 \
  uv run python -u bench_decompose_k.py \
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
uv run python -m kernels.decompose_k_triton_kernel --warmup 5 --rep 20
```

Custom shape:

```bash
uv run python -m kernels.decompose_k_triton_kernel \
  --m 16 \
  --n 16 \
  --k 8192 \
  --dtype bf16 \
  --warmup 5 \
  --rep 20
```

## Optimized Triton Decompose-K Kernel

`kernels/decompose_k_triton_kernel_optimized.py` keeps the same high-level
Decompose-K structure as `kernels/decompose_k_triton_kernel.py`:

1. Compute fp32 partial matmul outputs shaped like `[split_k, M, N]`.
2. Reduce those partials into the final output, optionally applying ReLU during
   the final store.

The optimized version does not use split-K atomics. That matters for epilogues:
an atomic-add design would still need all partial updates to finish before the
epilogue result is meaningful. The optimized kernel instead stays in the
Decompose-K setup and makes the explicit reduction cheaper and more parallel.

### Reducer Shape

The baseline reducer is output-tile shaped. One Triton program owns a
`BLOCK_M x BLOCK_N` tile, carries an accumulator with that same 2D shape, and
loops over all `SPLIT_K` partials:

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
for split_id in range(0, SPLIT_K):
    acc += tl.load(partials + split_id * stride_ps + tile_offsets)
```

This works, but it ties reduction parallelism to the matmul output tile. For a
tiny output such as `M=N=16`, a `16x16` reducer tile can mean only one reducer
program for the whole output, and that program serially walks the split
dimension.

The optimized reducer flattens the output matrix into a 1D element index:

```text
x = m * N + n
```

It then treats the Decompose-K split as the reduction axis:

```text
vals: [XBLOCK, RBLOCK]
acc:  [XBLOCK]
```

Conceptually, each reducer program does:

```python
x_base = program_id * XBLOCK + arange(0, XBLOCK)
r = arange(0, RBLOCK)
vals = partials[r, x_base]
acc = sum(vals, axis=1)
store c[x_base] = acc
```

Each `x` is one final output element. There is no later combine step across
programs: every reducer program owns a disjoint slice of flattened output
elements, reduces all split partials for those elements, applies ReLU if
requested, and writes the final values directly.

Example for `M=N=16`, where the output has 256 elements:

```text
XBLOCK=32
program 0 writes C_flat[0:32]
program 1 writes C_flat[32:64]
...
program 7 writes C_flat[224:256]
```

This helps because the reduction is now expressed as an actual vector reduction
over the split axis via `tl.sum(vals, 1)`, rather than as a serial loop over
`SPLIT_K` inside a tile-shaped accumulator. It also decouples reducer tiling
from matmul tiling: the partial matmul can use `16x16`, `64x32`, or `64x64`
tiles, while the reducer can independently use a small `XBLOCK` chosen for
split-reduction efficiency.

### Flat Contiguous Fast Path

The optimized reducer has a fast path for the benchmark's normal allocation
pattern:

```python
if partials.is_contiguous() and c.is_contiguous():
    ...
```

For contiguous row-major tensors, the memory layout is:

```text
partials[s, m, n] address = partials_base + s * (M * N) + (m * N + n)
c[m, n] address           = c_base + (m * N + n)
```

Since `x = m * N + n`, the hot reducer address calculation becomes:

```python
tl.load(partials + r * stride_ps + x)
tl.store(c + x_base, acc)
```

This avoids converting `x` back into `(m, n)` with division/modulo and avoids
general strided address math in the common path. The generic strided reducer is
still present as a fallback for non-contiguous views or `empty_strided` outputs,
for example `c = base[:, ::2]` or `partials = partials_base[:, :, ::2]`.

### Warp Count And Tile Size

The optimized config set adds small low-warp partial matmul tiles as well as
larger BMM-like tiles. This is important because the benchmark shapes have tiny
outputs and large K. A `16x16` output tile has only 256 fp32 accumulators. Even
with a K slice such as 64 or 128, many of these partial matmul programs are too
small to justify 4 warps.

Using 4 warps means 128 CUDA threads are assigned to one Triton program. For
small output tiles this can add overhead without enough independent work to keep
all warps busy:

- more warp scheduling and synchronization overhead per program
- more register pressure per resident program
- fewer resident programs per SM
- less ability to cover latency by running many independent split programs

The optimized search therefore includes 1-warp and 2-warp configs such as
`16x16x64` and `16x16x128`. These can be faster for small `M/N` because the GPU
can keep more independent programs resident and each program has lower overhead.

Larger output tiles are different. A `64x32` or `64x64` output tile has many
more accumulators, and with a nontrivial K slice it has substantially more work
per program:

```text
16x16x128 = 32,768 multiply-add positions
64x32x64  = 131,072 multiply-add positions
64x64x64  = 262,144 multiply-add positions
64x64x128 = 524,288 multiply-add positions
```

So 4 warps can win for the larger tiles. The optimized config set includes both
families and lets the benchmark choose per shape.

### Split Selection

The optimized kernel also expands the split candidates. The baseline split
heuristic prefers a limited ordered set of divisors where the K partition is
large enough. The optimized version explicitly tries power-of-two-style split
counts:

```python
K_SPLITS = (2, 4, 8, 16, 32, 64, 128, 256)
```

and then appends the baseline split candidates. This better matches the custom-op
Decompose-K setup and exposes more useful parallelism for small output, large-K
shapes. The reducer uses a power-of-two `RBLOCK` for the split axis, so these
split counts also map naturally to the vectorized reduction.

### Final BF16 Results

The strict validation runs benchmark the optimized Triton candidate against
saved custom-op timings without rerunning custom-op autotune. For the optimized
standalone Triton result directory, see `bench_results_v2`:

```bash
uv run python -u bench_candidate_decompose_k.py \
  --module kernels.decompose_k_triton_kernel_optimized \
  --baseline-dir bench_results_v2 \
  --suites epilogue-bf16 \
  --warmup 10 \
  --rep 50 \
  --out-csv bench_results_v2/optimized_epilogue_bf16_rep50_full.csv

uv run python -u bench_candidate_decompose_k.py \
  --module kernels.decompose_k_triton_kernel_optimized \
  --baseline-dir bench_results_v2 \
  --suites matmul-bf16 \
  --warmup 10 \
  --rep 50 \
  --out-csv bench_results_v2/optimized_matmul_bf16_rep50_full.csv
```

The saved BF16 CSVs compare the custom-op autotuned timing against the
standalone Triton timing: `custom_op_mm_relu_ms / decompose_k_fused_ms` for the
epilogue suite, and `custom_op_mm_ms / decompose_k_ms` for the plain matmul
suite. Values above `1.0x` mean the standalone Triton path is faster than the
custom-op path.

From `bench_results` (original standalone Triton kernel):

- `epilogue-bf16`: `0/28` wins versus custom-op timings, with
  min/median/max of `0.874x / 0.917x / 0.982x`.
- `matmul-bf16`: `0/28` wins versus custom-op timings, with
  min/median/max of `0.886x / 0.920x / 0.956x`.

From `bench_results_v2` (optimized standalone Triton kernel):

- `epilogue-bf16`: `26/28` wins and `1` tie versus custom-op timings, with
  min/median/max of `0.990x / 1.026x / 1.080x`.
- `matmul-bf16`: `24/28` wins and `2` ties versus custom-op timings, with
  min/median/max of `0.997x / 1.022x / 1.052x`.


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
uv run python - <<'PY'
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
uv run python -u custom_op_autotune_relu_dispatch.py \
  --t-values 1,16,64,256,768 \
  --k 7168 \
  --n 256 \
  --mode max-autotune \
  --dynamic
```

Dump generated Inductor code:

```bash
TORCH_LOGS=output_code DECOMPOSE_K_FORCE_EXIT=1 \
  uv run python -u custom_op_autotune_relu_dispatch.py \
  --t-values 16 \
  --k 7168 \
  --n 256 \
  --mode max-autotune \
  --dynamic \
  2>&1 | tee custom_op_autotune_relu_dispatch_k7168_output_code.log
```