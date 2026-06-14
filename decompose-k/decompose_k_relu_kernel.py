"""Standalone Triton Decompose-K matmul with an optional ReLU epilogue."""

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
def _partial_mm(
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
def _reduce_epilogue(
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
    FUSE_RELU: tl.constexpr,
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

    if FUSE_RELU:
        acc = tl.maximum(acc, 0.0)

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


def decompose_k_relu_out(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    partials: torch.Tensor,
    config: KernelConfig,
    *,
    fuse_relu: bool,
) -> torch.Tensor:
    _check_inputs(a, b, config.split_k)
    m, k = a.shape
    n = b.shape[1]

    partial_grid = (
        triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),
        config.split_k,
    )
    _partial_mm[partial_grid](
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
    _reduce_epilogue[reduce_grid](
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
        FUSE_RELU=fuse_relu,
        num_warps=max(1, min(config.num_warps, 4)),
        num_stages=3,
    )
    return c


def decompose_k_unfused_relu_out(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    partials: torch.Tensor,
    config: KernelConfig,
) -> torch.Tensor:
    decompose_k_relu_out(a, b, c, partials, config, fuse_relu=False)
    return c.relu_()


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


def _parse_dtype(name: str) -> torch.dtype:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--split-k", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    split_k = args.split_k
    if split_k is None:
        splits = inductor_like_splits(args.m, args.n, args.k, limit=1)
        if not splits:
            raise ValueError(
                f"no valid split_k values for M={args.m}, N={args.n}, K={args.k}"
            )
        split_k = splits[0]

    config = candidate_configs([split_k])[0]
    a = torch.randn((args.m, args.k), device="cuda", dtype=args.dtype)
    b = torch.randn((args.k, args.n), device="cuda", dtype=args.dtype)
    expected = torch.relu(torch.mm(a, b))

    fused = torch.empty((args.m, args.n), device="cuda", dtype=args.dtype)
    unfused = torch.empty_like(fused)
    partials = torch.empty((config.split_k, args.m, args.n), device="cuda", dtype=torch.float32)

    decompose_k_relu_out(a, b, fused, partials, config, fuse_relu=True)
    decompose_k_unfused_relu_out(a, b, unfused, partials, config)
    torch.cuda.synchronize()
    torch.testing.assert_close(fused, expected, rtol=args.rtol, atol=args.atol)
    torch.testing.assert_close(unfused, expected, rtol=args.rtol, atol=args.atol)

    fused_ms = triton.testing.do_bench(
        lambda: decompose_k_relu_out(a, b, fused, partials, config, fuse_relu=True),
        warmup=args.warmup,
        rep=args.rep,
        return_mode="median",
    )
    unfused_ms = triton.testing.do_bench(
        lambda: decompose_k_unfused_relu_out(a, b, unfused, partials, config),
        warmup=args.warmup,
        rep=args.rep,
        return_mode="median",
    )

    print(f"torch={torch.__version__} triton={triton.__version__}")
    print(f"shape=({args.m}, {args.k}) x ({args.k}, {args.n}) dtype={args.dtype}")
    print(
        f"config=split{config.split_k}/bm{config.block_m}/bn{config.block_n}/bk{config.block_k}"
    )
    print("correctness=passed")
    print(f"fused_relu={fused_ms:.4f} ms")
    print(f"unfused_relu={unfused_ms:.4f} ms")
    print(f"epilogue_fusion_speedup={unfused_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    main()
