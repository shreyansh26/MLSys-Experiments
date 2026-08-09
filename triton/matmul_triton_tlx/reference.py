"""Shared validation, alignment handling, and the PyTorch matmul reference."""

from __future__ import annotations

import torch
import triton


def validate_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"expected A[M, K] and B[K, N], got {a.shape} and {b.shape}")
    if a.device.type != "cuda" or a.device != b.device:
        raise ValueError("A and B must be on the same CUDA device")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("A and B must have the same fp16 or bf16 dtype")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("the study kernels require row-major contiguous inputs")
    return a.shape[0], b.shape[1], a.shape[1]


def aligned_inputs(
    a: torch.Tensor, b: torch.Tensor, alignment: int = 16
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    """Pad TMA's contiguous dimensions while preserving the public shape."""
    m, n, k = validate_inputs(a, b)
    original_n = n
    aligned_k = triton.cdiv(k, alignment) * alignment
    if aligned_k != k:
        a = torch.nn.functional.pad(a, (0, aligned_k - k))
        b = torch.nn.functional.pad(b, (0, 0, 0, aligned_k - k))
        k = aligned_k
    aligned_n = triton.cdiv(n, alignment) * alignment
    if aligned_n != n:
        b = torch.nn.functional.pad(b, (0, aligned_n - n))
        n = aligned_n
    return a, b, m, n, k, original_n


def restore_output(c: torch.Tensor, original_n: int) -> torch.Tensor:
    if c.shape[1] == original_n:
        return c
    return c[:, :original_n].contiguous()


def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    validate_inputs(a, b)
    return torch.matmul(a, b)


def supports_tma(device: torch.device | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    device = device or torch.device("cuda")
    return torch.cuda.get_device_capability(device)[0] >= 9


def supports_triton_warp_specialization(
    device: torch.device | None = None,
) -> bool:
    """Compiler-managed tl.range warp specialization is currently SM100+."""
    if not supports_tma(device):
        return False
    device = device or torch.device("cuda")
    return torch.cuda.get_device_capability(device)[0] >= 10


def resolve_triton_warp_specialization(
    device: torch.device, requested: bool | None
) -> bool:
    supported = supports_triton_warp_specialization(device)
    if requested and not supported:
        capability = torch.cuda.get_device_capability(device)
        raise RuntimeError(
            "Triton compiler-managed warp specialization requires SM100+; "
            f"this device is SM{capability[0]}{capability[1]}. Pass "
            "warp_specialize=False to exercise the same TMA kernel with "
            "software pipelining."
        )
    return supported if requested is None else requested
