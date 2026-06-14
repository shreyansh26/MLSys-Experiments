"""Benchmark ReLU epilogue fusion for the Figure 5 Decompose-K shape grid.

The fused path applies ReLU in the final partial-sum reduction kernel, avoiding
the extra read/write pass used by `torch.mm(...).relu()` or Decompose-K plus a
separate `relu_()`.
"""

import argparse
import csv
from pathlib import Path

import torch
import triton

from decompose_k_relu_kernel import (
    KernelConfig,
    candidate_configs,
    decompose_k_relu_out,
    decompose_k_unfused_relu_out,
    inductor_like_splits,
)


def parse_csv_ints(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def parse_dtype(name: str) -> torch.dtype:
    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def mm_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.mm(a, b))


def bench_ms(fn, warmup: int, rep: int) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def best_decompose_k_config(
    a: torch.Tensor,
    b: torch.Tensor,
    ref: torch.Tensor,
    split_limit: int,
    warmup: int,
    rep: int,
    rtol: float,
    atol: float,
) -> tuple[float, float, KernelConfig]:
    m, k = a.shape
    n = b.shape[1]
    splits = inductor_like_splits(m, n, k, split_limit)
    if not splits:
        raise ValueError(f"no valid split_k values for M={m}, N={n}, K={k}")

    best_fused = (float("inf"), float("inf"), None)

    for config in candidate_configs(splits):
        c = torch.empty_like(ref)
        partials = torch.empty((config.split_k, m, n), device=a.device, dtype=torch.float32)
        try:
            decompose_k_relu_out(a, b, c, partials, config, fuse_relu=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)
            fused_ms = bench_ms(
                lambda: decompose_k_relu_out(a, b, c, partials, config, fuse_relu=True),
                warmup,
                rep,
            )
            unfused_ms = bench_ms(
                lambda: decompose_k_unfused_relu_out(a, b, c, partials, config),
                warmup,
                rep,
            )
        except Exception:
            continue

        if fused_ms < best_fused[0]:
            best_fused = (fused_ms, unfused_ms, config)

    if best_fused[2] is None:
        raise RuntimeError(f"no Decompose-K configs completed for M={m}, N={n}, K={k}")
    return best_fused[0], best_fused[1], best_fused[2]


def plot_results(rows: list[dict[str, object]], out_dir: Path, dtype_name: str) -> None:
    import matplotlib.pyplot as plt

    rows_by_shape = sorted(rows, key=lambda row: (int(row["m"]), int(row["k"])))
    labels = [f"({row['m']}, {row['k']})" for row in rows_by_shape]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(18, 6))
    for key, label in [
        ("eager_ms", "torch.mm + relu"),
        ("compiled_ms", "compiled torch.mm + relu"),
        ("decompose_k_unfused_ms", "decomposeK + relu"),
        ("decompose_k_fused_ms", "decomposeK fused relu"),
    ]:
        ax.plot(x, [float(row[key]) for row in rows_by_shape], marker="o", label=label)

    group_start = 0
    for mn in sorted({int(row["m"]) for row in rows_by_shape}):
        count = sum(1 for row in rows_by_shape if int(row["m"]) == mn)
        if group_start:
            ax.axvline(group_start - 0.5, color="0.75", linewidth=1)
        ax.text(
            group_start + (count - 1) / 2,
            0.985,
            f"M=N={mn}",
            ha="center",
            va="top",
            color="0.25",
            fontsize=10,
            transform=ax.get_xaxis_transform(),
        )
        group_start += count

    ax.set_title(f"ReLU Epilogue Fusion, {dtype_name.upper()}, Figure 5 Shape Grid")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("(M/N, K)")
    ax.set_xticks(x, labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"relu_epilogue_fig5_grid_{dtype_name}.png", dpi=180)
    plt.close(fig)

    for mn in sorted({int(row["m"]) for row in rows}):
        subset = [row for row in rows if int(row["m"]) == mn]
        subset.sort(key=lambda row: int(row["k"]))
        labels = [f"({row['m']}, {row['k']})" for row in subset]
        x = list(range(len(labels)))

        fig, ax = plt.subplots(figsize=(12, 5))
        for key, label in [
            ("eager_ms", "torch.mm + relu"),
            ("compiled_ms", "compiled torch.mm + relu"),
            ("decompose_k_unfused_ms", "decomposeK + relu"),
            ("decompose_k_fused_ms", "decomposeK fused relu"),
        ]:
            ax.plot(x, [float(row[key]) for row in subset], marker="o", label=label)

        ax.set_title(f"ReLU Epilogue Fusion, {dtype_name.upper()}, M=N={mn}")
        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel("(M/N, K)")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"relu_epilogue_mn{mn}_{dtype_name}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mns", type=parse_csv_ints, default=[16, 32, 48, 64])
    parser.add_argument(
        "--ks",
        type=parse_csv_ints,
        default=[8192, 12288, 16384, 20480, 24576, 28672, 32768],
    )
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    parser.add_argument("--split-limit", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--out-dir", type=Path, default=Path("epilogue_relu_results"))
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    compiled_mm_relu = torch.compile(mm_relu, mode=args.compile_mode)
    dtype_names = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }
    dtype_name = dtype_names[args.dtype]

    print(f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    print(f"device={torch.cuda.get_device_name()}", flush=True)
    print(f"dtype={args.dtype} compile_mode={args.compile_mode}", flush=True)

    rows: list[dict[str, object]] = []
    for mn in args.mns:
        for k in args.ks:
            a = torch.randn((mn, k), device="cuda", dtype=args.dtype)
            b = torch.randn((k, mn), device="cuda", dtype=args.dtype)
            ref = mm_relu(a, b)
            torch.cuda.synchronize()

            compiled_out = compiled_mm_relu(a, b)
            torch.cuda.synchronize()
            torch.testing.assert_close(compiled_out, ref, rtol=args.rtol, atol=args.atol)

            eager_ms = bench_ms(lambda: mm_relu(a, b), args.warmup, args.rep)
            compiled_ms = bench_ms(lambda: compiled_mm_relu(a, b), args.warmup, args.rep)
            fused_ms, unfused_ms, config = best_decompose_k_config(
                a,
                b,
                ref,
                args.split_limit,
                args.warmup,
                args.rep,
                args.rtol,
                args.atol,
            )

            row = {
                "m": mn,
                "n": mn,
                "k": k,
                "dtype": dtype_name,
                "eager_ms": eager_ms,
                "compiled_ms": compiled_ms,
                "decompose_k_unfused_ms": unfused_ms,
                "decompose_k_fused_ms": fused_ms,
                "fused_speedup_vs_eager": eager_ms / fused_ms,
                "fused_speedup_vs_compiled": compiled_ms / fused_ms,
                "epilogue_fusion_speedup": unfused_ms / fused_ms,
                "split_k": config.split_k,
                "block_m": config.block_m,
                "block_n": config.block_n,
                "block_k": config.block_k,
            }
            rows.append(row)
            print(
                f"(M=N={mn}, K={k}) eager={eager_ms:.4f}ms "
                f"compiled={compiled_ms:.4f}ms unfused={unfused_ms:.4f}ms "
                f"fused={fused_ms:.4f}ms speedup_vs_compiled={compiled_ms / fused_ms:.2f}x "
                f"config=split{config.split_k}/bm{config.block_m}/bn{config.block_n}/bk{config.block_k}",
                flush=True,
            )

    csv_path = args.out_dir / f"relu_epilogue_{dtype_name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_results(rows, args.out_dir, dtype_name)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote plots under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
