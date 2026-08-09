# Matmul interview lab: Triton TMA, TLX, and Gluon

This project implements `C[M, N] = A[M, K] @ B[K, N]` five ways: a PyTorch/cuBLAS reference and four optimized study kernels. Every custom kernel owns the complete K reduction for its output tile. There is no split-K grid, partial workspace, atomic accumulation, or reduction kernel.

## Kernels

| File | Scheduling | Data movement / compute | Warp specialization |
|---|---|---|---|
| `triton_tma_matmul.py` | One program per output tile; grouped ordering; autotuned | Host TMA descriptors, FP32 `tl.dot` accumulation | Compiler-managed `tl.range` path on SM100+ |
| `triton_tma_persistent_matmul.py` | SM-stride persistent tiles; grouped ordering; subtiled epilogue | Host TMA descriptors, FP32 `tl.dot` accumulation | Compiler-managed persistent-loop path on SM100+ |
| `tlx_matmul.py` | SM-stride persistent tiles | One TMA producer plus two replicated four-warp WGMMA consumers | Explicit `tlx.async_tasks` on Hopper |
| `gluon_matmul.py` | Grouped SM-stride persistent tiles | Three-stage TMA pipeline, four-warp WGMMA, overlapped previous-tile TMA store | Native `gl.warp_specialize` loader/compute partitions on Hopper |
| `reference.py` | PyTorch | `torch.matmul` / cuBLAS | N/A |

The two Triton files really do contain warp-specialized paths, but Triton's compiler-managed `warp_specialize=True` currently requires Blackwell (SM100+). On the H100 used here, their wrappers select the same TMA kernels with aggressive software pipelining. Passing `warp_specialize=True` explicitly on H100 raises instead of silently pretending it is active. TLX and Gluon use explicit Hopper warp specialization on SM90.

## Reproducible UV environment

Prerequisites are `uv`, Git, a C++ toolchain, a CUDA driver, and an NVIDIA H100 for all four custom implementations.

```bash
cd /mnt/ssd1/shreyansh/home_dir/misc_experiments/triton/matmul_triton_tlx
./bootstrap.sh
```

The bootstrap script:

1. installs managed CPython 3.12;
2. checks out the exact Triton commit required by `triton-utlx==3.7.1` under `.deps/triton`;
3. applies `patches/triton-tlx-warp-specialization.patch`;
4. creates `.venv`, resolves the locked PyTorch/TLX/plotting dependencies, and builds extension-enabled Triton;
5. smoke-tests the TLX compiler plugin.

The small source patch exports existing Hopper TMA/warp-specialization IR operations and bridges frontend API drift at the pinned Triton/TLX boundary. `tlx_plugin.py` contains the corresponding narrow Python compatibility shims, including coexistence with Gluon's native compiler pipeline.

## Run correctness, benchmarks, and plots

Correctness across arbitrary boundary shapes in both FP16 and BF16, followed by a short FP16 sweep:

```bash
uv run python benchmark.py --suite quick --dtype fp16 --output results/quick
```

Full tutorial-style sweeps:

```bash
uv run python benchmark.py --suite full --dtype fp16 --skip-check --output results/fp16
uv run python benchmark.py --suite full --dtype bf16 --skip-check --output results/bf16
```

Each sweep records median/20th/80th-percentile GPU timing, TFLOP/s, CSV data, Triton's tutorial plot, and a readable grouped horizontal-bar plot. Providers run sequentially so they do not contend for the GPU.

## Measured H100 results

Selected median throughput from the checked-in full sweeps (TFLOP/s):

| dtype / M×N×K | cuBLAS | Triton TMA | Triton persistent | TLX WS | Gluon WS |
|---|---:|---:|---:|---:|---:|
| FP16 / 4096×4096×4096 | 692.8 | 660.6 | 662.5 | 637.2 | 575.2 |
| FP16 / 1024×4096×4096 | 681.7 | 673.6 | 679.2 | 678.7 | 606.6 |
| FP16 / 4096×11008×4096 | 640.6 | 620.0 | 635.8 | 609.8 | 576.6 |
| BF16 / 4096×4096×4096 | 747.7 | 677.4 | 687.5 | 680.8 | 607.6 |
| BF16 / 4096×4096×11008 | 711.9 | 703.4 | 706.0 | 669.8 | 591.8 |

Full data: `results/fp16/*.csv` and `results/bf16/*.csv`.

![FP16 throughput](results/fp16/matmul-float16-tflops-comparison.png)

![BF16 throughput](results/bf16/matmul-bfloat16-tflops-comparison.png)

## Scope and reading order

- Inputs are contiguous row-major FP16 or BF16; accumulation is FP32 and output uses the input dtype.
- Positive arbitrary `M/N/K` are supported. The wrappers transparently pad TMA's contiguous K/N dimensions to 16 elements and crop the output; aligned benchmark shapes do not pay this cost.
- Start with `reference.py`, then read the tiled Triton kernel, persistent Triton kernel, TLX producer/consumer kernel, and Gluon partition functions. Finish with `benchmark.py`.
- Useful follow-ups: fuse bias/ReLU, accept transposed B, add batched GEMM, inspect TTGIR/PTX/SASS, profile barrier stalls, and explain why persistent scheduling or grouped tile order can help or hurt L2 reuse.

The implementations follow the official [Triton persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html), [Gluon WGMMA tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html), and [Gluon warp-specialization tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html).
