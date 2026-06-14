import argparse

import torch
import triton

from decompose_k_kernel import (
    KernelConfig,
    candidate_configs,
    decompose_k_matmul_out,
    inductor_like_splits,
)


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


def parse_splits(text: str) -> list[int] | str:
    if text == "auto":
        return text
    splits = [int(part) for part in text.split(",") if part]
    if not splits or any(split <= 0 for split in splits):
        raise argparse.ArgumentTypeError(
            "splits must be 'auto' or a comma-separated list of positive ints"
        )
    return splits


def tflops(ms: float, m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    parser.add_argument("--splits", type=parse_splits, default="auto")
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest" if args.dtype == torch.float32 else "high")

    device = "cuda"
    a = torch.randn((args.m, args.k), device=device, dtype=args.dtype)
    b = torch.randn((args.k, args.n), device=device, dtype=args.dtype)
    ref = torch.matmul(a, b)
    torch.cuda.synchronize()

    print(f"torch={torch.__version__} triton={triton.__version__}")
    print(f"device={torch.cuda.get_device_name()}")
    print(f"shape=({args.m}, {args.k}) x ({args.k}, {args.n}) dtype={args.dtype}")

    torch_out = torch.empty_like(ref)
    torch_ms = triton.testing.do_bench(
        lambda: torch.mm(a, b, out=torch_out),
        warmup=args.warmup,
        rep=args.rep,
        return_mode="median",
    )
    print(f"torch.mm: {torch_ms:.4f} ms, {tflops(torch_ms, args.m, args.n, args.k):.2f} TFLOP/s")

    if args.splits == "auto":
        valid_splits = inductor_like_splits(args.m, args.n, args.k, args.num_splits)
        print(f"inductor_like_splits={valid_splits}")
    else:
        valid_splits = [split for split in args.splits if args.k % split == 0]
    if not valid_splits:
        raise ValueError(f"none of {args.splits} divide K={args.k}")

    results: list[tuple[float, KernelConfig]] = []
    for config in candidate_configs(valid_splits):
        c = torch.empty_like(ref)
        partials = torch.empty(
            (config.split_k, args.m, args.n), device=device, dtype=torch.float32
        )
        try:
            decompose_k_matmul_out(a, b, c, partials, config)
            torch.cuda.synchronize()
            torch.testing.assert_close(c, ref, rtol=args.rtol, atol=args.atol)

            ms = triton.testing.do_bench(
                lambda: decompose_k_matmul_out(a, b, c, partials, config),
                warmup=args.warmup,
                rep=args.rep,
                return_mode="median",
            )
        except Exception as exc:
            print(f"skip {config}: {exc}")
            continue
        results.append((ms, config))
        print(
            "decompose_k "
            f"split={config.split_k:<2} bm={config.block_m:<2} bn={config.block_n:<3} "
            f"bk={config.block_k:<3}: {ms:.4f} ms, "
            f"{tflops(ms, args.m, args.n, args.k):.2f} TFLOP/s"
        )

    if not results:
        raise RuntimeError("no valid Decompose-K configs completed")

    best_ms, best_config = min(results, key=lambda item: item[0])
    print(
        "best: "
        f"split={best_config.split_k} bm={best_config.block_m} "
        f"bn={best_config.block_n} bk={best_config.block_k} "
        f"group_m={best_config.group_m} warps={best_config.num_warps} "
        f"stages={best_config.num_stages} -> {best_ms:.4f} ms, "
        f"{tflops(best_ms, args.m, args.n, args.k):.2f} TFLOP/s"
    )


if __name__ == "__main__":
    main()
