import argparse
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class KernelConfig:
    split_k: int
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int


@triton.jit
def _decompose_k_partial_mm(
    a,
    b,
    partials,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ps: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k_per_split = K // SPLIT_K
    split_k_start = split_id * k_per_split
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, k_per_split, BLOCK_K):
        k_offsets = k0 + offs_k
        a_ptrs = a + offs_m[:, None] * stride_am + (
            split_k_start + k_offsets[None, :]
        ) * stride_ak
        b_ptrs = b + (split_k_start + k_offsets[:, None]) * stride_bk + offs_n[
            None, :
        ] * stride_bn
        k_mask = k_offsets < k_per_split
        a_vals = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b_vals = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a_vals, b_vals, out_dtype=tl.float32)

    partial_ptrs = partials + split_id * stride_ps + offs_m[:, None] * stride_pm + offs_n[
        None, :
    ] * stride_pn
    tl.store(partial_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _decompose_k_reduce(
    partials,
    c,
    stride_ps: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs = partials + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for split_id in range(0, SPLIT_K):
        acc += tl.load(
            ptrs + split_id * stride_ps,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )

    c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _check_inputs(a: torch.Tensor, b: torch.Tensor, split_k: int) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected 2D tensors")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("expected CUDA tensors")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype=} {b.dtype=}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible matmul shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if a.shape[1] % split_k != 0:
        raise ValueError(f"K={a.shape[1]} must be divisible by split_k={split_k}")


def decompose_k_matmul_out(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    partials: torch.Tensor,
    config: KernelConfig,
) -> torch.Tensor:
    _check_inputs(a, b, config.split_k)
    m, k = a.shape
    n = b.shape[1]

    if c.shape != (m, n) or c.dtype != a.dtype or c.device != a.device:
        raise ValueError("c must have shape (M, N), input dtype, and input device")
    if partials.shape != (config.split_k, m, n):
        raise ValueError("partials must have shape (split_k, M, N)")
    if partials.dtype != torch.float32 or partials.device != a.device:
        raise ValueError("partials must be a float32 CUDA tensor")

    partial_grid = (
        triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),
        config.split_k,
    )
    _decompose_k_partial_mm[partial_grid](
        a,
        b,
        partials,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        M=m,
        N=n,
        K=k,
        SPLIT_K=config.split_k,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_M=config.group_m,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    reduce_grid = (triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),)
    _decompose_k_reduce[reduce_grid](
        partials,
        c,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        c.stride(0),
        c.stride(1),
        M=m,
        N=n,
        SPLIT_K=config.split_k,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        GROUP_M=config.group_m,
        num_warps=max(1, min(config.num_warps, 4)),
        num_stages=3,
    )
    return c


def decompose_k_matmul(a: torch.Tensor, b: torch.Tensor, config: KernelConfig) -> torch.Tensor:
    _check_inputs(a, b, config.split_k)
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    partials = torch.empty(
        (config.split_k, a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32
    )
    return decompose_k_matmul_out(a, b, c, partials, config)


def candidate_configs(split_values: list[int]) -> list[KernelConfig]:
    configs = []
    for split_k in split_values:
        configs.extend(
            [
                KernelConfig(split_k, 16, 16, 128, 8, 4, 4),
                KernelConfig(split_k, 16, 32, 128, 8, 4, 4),
                KernelConfig(split_k, 16, 32, 256, 8, 4, 4),
                KernelConfig(split_k, 16, 64, 64, 8, 4, 4),
                KernelConfig(split_k, 32, 16, 128, 8, 4, 4),
                KernelConfig(split_k, 32, 16, 256, 8, 4, 4),
                KernelConfig(split_k, 32, 32, 128, 8, 4, 4),
                KernelConfig(split_k, 16, 128, 64, 8, 4, 4),
                KernelConfig(split_k, 32, 64, 64, 8, 4, 4),
                KernelConfig(split_k, 32, 128, 64, 8, 4, 4),
                KernelConfig(split_k, 32, 128, 128, 8, 4, 4),
                KernelConfig(split_k, 64, 64, 64, 4, 4, 4),
            ]
        )
    return configs


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


def divisors(value: int) -> list[int]:
    out = []
    for candidate in range(1, math.isqrt(value) + 1):
        if value % candidate == 0:
            out.append(candidate)
            if candidate * candidate != value:
                out.append(value // candidate)
    return sorted(out)


def inductor_like_splits(m: int, n: int, k: int, limit: int) -> list[int]:
    max_split = min(k // m, k // n)
    pow2_k_parts = []
    multiple_32_k_parts = []
    rest = []

    for split in divisors(k):
        if split < 2 or split > max_split:
            continue
        k_part = k // split
        if k_part < 128:
            continue
        if k_part & (k_part - 1) == 0:
            pow2_k_parts.append(split)
        elif k_part % 32 == 0:
            multiple_32_k_parts.append(split)
        else:
            rest.append(split)

    return (pow2_k_parts + multiple_32_k_parts + rest)[:limit]


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
