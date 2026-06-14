"""Explore Inductor's custom-op autotuning API for matmul plus ReLU.

`register_custom_op_autotuning` can also target an existing OpOverload such as
`torch.ops.aten.mm.default`.  The registered configs are alternate
decompositions that Inductor autotunes per shape range, then lowers to either a
single winner or a `torch.cond` runtime dispatch tree.

The default path in this script uses a true `mm_relu` custom op so the autotuned
unit includes the ReLU epilogue.  Use `--boundary aten-mm` to reproduce the
older experiment where only `aten.mm` is custom-autotuned and ReLU stays outside
the custom-op boundary.
"""

import argparse
import os
import sys

import torch
import triton

from torch._inductor.kernel.custom_op import (
    CustomOpConfig,
    register_custom_op_autotuning,
)


K_SPLITS = (2, 4, 8, 16, 32, 64, 128, 256)
SPLIT_POINTS = [1, 8, 32, 128, 512]


def mm_impl(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return torch.mm(self, mat2)


def mm_relu_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.mm(a, b))


def decompose_k_impl(
    self: torch.Tensor,
    mat2: torch.Tensor,
    k_splits: int,
) -> torch.Tensor:
    m = self.shape[0]
    n = mat2.shape[1]
    k = self.shape[1]

    assert k == mat2.shape[0], "incompatible matmul dimensions"
    assert k % k_splits == 0, "K must be divisible by k_splits"

    k_part = k // k_splits
    a_reshaped = self.reshape(m, k_splits, k_part).permute(1, 0, 2)
    b_reshaped = mat2.reshape(k_splits, k_part, n)
    partials = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    return partials.sum(dim=0).to(self.dtype)


def decompose_k_relu_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    k_splits: int,
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    assert k == b.shape[0], "incompatible matmul dimensions"
    assert k % k_splits == 0, "K must be divisible by k_splits"

    k_part = k // k_splits
    a_reshaped = a.reshape(m, k_splits, k_part).permute(1, 0, 2)
    b_reshaped = b.reshape(k_splits, k_part, n)
    partials = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    return torch.relu(partials.sum(dim=0).to(a.dtype))


@torch.library.custom_op("decompose_k::mm_relu", mutates_args=())
def mm_relu_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return mm_relu_impl(a, b)


@mm_relu_op.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)


def generate_mm_configs(fake_tensors: dict[str, torch.Tensor]) -> list[CustomOpConfig]:
    k = int(fake_tensors["self"].shape[1])
    splits = [k_splits for k_splits in K_SPLITS if k % k_splits == 0]

    configs = [CustomOpConfig(mm_impl)]
    configs.extend(
        CustomOpConfig(decompose_k_impl, k_splits=k_splits) for k_splits in splits
    )
    return configs


def generate_mm_relu_configs(fake_tensors: dict[str, torch.Tensor]) -> list[CustomOpConfig]:
    k = int(fake_tensors["a"].shape[1])
    splits = [k_splits for k_splits in K_SPLITS if k % k_splits == 0]

    configs = [CustomOpConfig(mm_relu_impl)]
    configs.extend(
        CustomOpConfig(decompose_k_relu_impl, k_splits=k_splits)
        for k_splits in splits
    )
    return configs


def register_mm_autotune() -> None:
    register_custom_op_autotuning(
        # Intercept Inductor lowering for torch.mm / aten.mm.default. This API
        # also accepts true @torch.library.custom_op ops, but here we target an
        # existing ATen op.
        custom_op=torch.ops.aten.mm.default,
        # Build the candidate list from fake tensor metadata. For each compile
        # shape, this returns plain mm plus all valid Decompose-K split counts.
        config_generator=generate_mm_configs,
        # Prefix used in autotune logs and generated candidate names.
        name="router_mm_relu_autotune",
        # Create real CUDA benchmark inputs matching each fake tensor's
        # shape/dtype/stride. Keys must match the ATen schema argument names:
        # `torch.ops.aten.mm.default._schema` is
        # `aten::mm(Tensor self, Tensor mat2) -> Tensor`.
        input_gen_fns={
            "self": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
            "mat2": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
        },
        # Benchmark and dispatch by self.shape[0], i.e. M/T for an MxK @ KxN
        # matmul. range_upper_bound is the representative size for the final
        # open-ended bucket; it is not an input validity limit.
        dispatch_on={"tensor_name": "self", "dim": 0, "range_upper_bound": 1024},
        # Convert split points into ranges such as [1, 1], [2, 8], [9, 32],
        # etc. Inductor picks the fastest candidate per range and emits a
        # runtime torch.cond dispatch tree if adjacent ranges need different
        # implementations.
        split_points=SPLIT_POINTS,
        # Use CUDA graph replay during autotune measurements where supported.
        benchmark_with_cudagraphs=True,
    )


def register_mm_relu_autotune() -> None:
    register_custom_op_autotuning(
        # Intercept Inductor lowering for the true custom op. Unlike the
        # `aten.mm` registration above, this boundary includes the ReLU epilogue,
        # so every autotuned candidate returns relu(mm(a, b)).
        custom_op=mm_relu_op,
        # Build fused candidates from fake tensor metadata: plain mm+relu plus
        # all valid Decompose-K splits that reduce and apply ReLU before storing.
        config_generator=generate_mm_relu_configs,
        # Prefix used in autotune logs and generated candidate names.
        name="router_mm_relu_fused_autotune",
        # Create real CUDA benchmark inputs matching the custom op schema:
        # `decompose_k::mm_relu(Tensor a, Tensor b) -> Tensor`.
        input_gen_fns={
            "a": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
            "b": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
        },
        # Benchmark and dispatch by a.shape[0], i.e. M/T for an MxK @ KxN
        # matmul. This mirrors the `aten.mm` experiment while moving the
        # autotune boundary one op later to include ReLU.
        dispatch_on={"tensor_name": "a", "dim": 0, "range_upper_bound": 1024},
        # Use the same runtime ranges as the `aten.mm` path so the two boundary
        # choices can be compared directly.
        split_points=SPLIT_POINTS,
        # Use CUDA graph replay during autotune measurements where supported.
        benchmark_with_cudagraphs=True,
    )


def register_mm_relu_k_autotune(
    split_points: list[int],
    range_upper_bound: int,
) -> None:
    register_custom_op_autotuning(
        # Experimental range-dispatch variant for K sweeps. This exposes a
        # current API limitation: Inductor changes only this dispatch tensor's
        # representative dimension, but matmul K is shared by both inputs.
        custom_op=mm_relu_op,
        config_generator=generate_mm_relu_configs,
        name="router_mm_relu_fused_k_autotune",
        input_gen_fns={
            "a": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
            "b": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
        },
        dispatch_on={
            "tensor_name": "a",
            "dim": 1,
            "range_upper_bound": range_upper_bound,
        },
        split_points=split_points,
        benchmark_with_cudagraphs=True,
    )


def register_mm_relu_static_autotune() -> None:
    register_custom_op_autotuning(
        # Exact-shape autotuning for benchmark harnesses that compile one fixed
        # shape at a time. This avoids range representative inputs and compares
        # the fused custom-op candidates at the actual benchmark shape.
        custom_op=mm_relu_op,
        config_generator=generate_mm_relu_configs,
        name="static_mm_relu_fused_autotune",
        input_gen_fns={
            "a": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
            "b": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
        },
        benchmark_with_cudagraphs=True,
    )


def relu_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.mm(a, b))


def custom_relu_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return mm_relu_op(a, b)


def bench(label: str, fn, flops: int) -> float:
    ms = triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{label}: {ms:.4f} ms, {tflops:.2f} TFLOP/s", flush=True)
    return ms


def parse_t_values(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated T values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-values", type=parse_t_values, default=[1, 16, 64, 256, 768])
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--mode", default="max-autotune")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--boundary",
        choices=["mm-relu", "aten-mm"],
        default="mm-relu",
        help="Autotune the fused custom mm_relu op or only aten.mm.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.boundary == "mm-relu":
        register_mm_relu_autotune()
        compiled_relu_mm = torch.compile(
            custom_relu_mm,
            mode=args.mode,
            dynamic=args.dynamic,
        )
    else:
        register_mm_autotune()
        compiled_relu_mm = torch.compile(relu_mm, mode=args.mode, dynamic=args.dynamic)

    print(f"torch={torch.__version__} triton={triton.__version__}")
    print(f"device={torch.cuda.get_device_name()}")
    print(f"K={args.k} N={args.n} dtype={dtype} dynamic={args.dynamic}")
    print(f"autotune_boundary={args.boundary}")
    print(f"registered split_points={SPLIT_POINTS} k_splits={K_SPLITS}")
    print(
        "Set TORCH_LOGS=output_code when running this script to dump the generated "
        "torch.cond dispatch and fused kernels.",
        flush=True,
    )

    b = torch.randn((args.k, args.n), device="cuda", dtype=dtype)
    for t in args.t_values:
        a = torch.randn((t, args.k), device="cuda", dtype=dtype)
        expected = relu_mm(a, b)
        actual = compiled_relu_mm(a, b)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-2)

        flops = 2 * t * args.k * args.n
        eager_ms = bench(f"eager_relu_mm_T{t}", lambda: relu_mm(a, b), flops)
        compiled_ms = bench(
            f"compiled_custom_autotune_relu_mm_T{t}",
            lambda: compiled_relu_mm(a, b),
            flops,
        )
        print(f"T={t}: compiled/eager={compiled_ms / eager_ms:.2f}x latency ratio", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("DECOMPOSE_K_FORCE_EXIT", "1") == "1":
        os._exit(0)


if __name__ == "__main__":
    main()
