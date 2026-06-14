import torch
import triton

from torch._inductor.kernel.custom_op import (
    CustomOpConfig,
    register_custom_op_autotuning,
)


K_SPLITS = (1, 2, 4, 8, 16, 32)
SPLIT_POINTS = [1, 8, 32, 128, 512]


def matmul_impl(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(a, b)


def decompose_k_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    k_splits: int = 4,
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    assert k == b.shape[0], "Incompatible dimensions"
    assert k % k_splits == 0, "k must be divisible by k_splits"

    k_parts = k // k_splits
    a_reshaped = a.reshape(m, k_splits, k_parts).permute(1, 0, 2)
    b_reshaped = b.reshape(k_splits, k_parts, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    return result.sum(dim=0).to(a.dtype)


@torch.library.custom_op("decompose_k::matmul", mutates_args=())
def decompose_k_op(
    a: torch.Tensor,
    b: torch.Tensor,
    k_splits: int = 4,
) -> torch.Tensor:
    return decompose_k_impl(a, b, k_splits=k_splits)


@decompose_k_op.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    k_splits: int = 4,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)


def generate_configs(fake_tensors: dict[str, torch.Tensor]) -> list[CustomOpConfig]:
    k = int(fake_tensors["a"].shape[1])
    valid_splits = [k_splits for k_splits in K_SPLITS if k % k_splits == 0]

    configs = [CustomOpConfig(matmul_impl)]
    configs.extend(
        CustomOpConfig(decompose_k_impl, k_splits=k_splits)
        for k_splits in valid_splits
    )
    return configs


register_custom_op_autotuning(
    custom_op=decompose_k_op,
    config_generator=generate_configs,
    name="decompose_k_autotuned",
    input_gen_fns={
        "a": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
        "b": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
    },
    dispatch_on={"tensor_name": "a", "dim": 0, "range_upper_bound": 1024},
    split_points=SPLIT_POINTS,
)


def custom_decompose_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return decompose_k_op(a, b)


def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return matmul_impl(a, b)


def bench(label: str, fn, flops: int) -> None:
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{label}: {ms:.3f} ms, {tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = "cuda"
    a = torch.randn(64, 7168, dtype=dtype, device=device)
    b = torch.randn(7168, 256, dtype=dtype, device=device)

    compiled_custom_decompose_k = torch.compile(
        custom_decompose_k,
        mode="max-autotune-no-cudagraphs",
    )
    # compiled_custom_decompose_k = torch.compile(
    #     custom_decompose_k,
    #     mode="max-autotune",
    # )

    expected = torch_matmul(a, b)
    actual = compiled_custom_decompose_k(a, b)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-2)

    flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
    bench("torch.matmul", lambda: torch_matmul(a, b), flops)
    bench("custom_op_decompose_k", lambda: compiled_custom_decompose_k(a, b), flops)
