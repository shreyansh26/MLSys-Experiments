import torch

try:
    from . import _bgmv_cuda as _C
except Exception as e:  # noqa: BLE001
    _C = None
    _load_error = e
else:
    _load_error = None


def lora_bgmv_cuda(y: torch.Tensor,
              x: torch.Tensor,
              A: torch.Tensor,
              B: torch.Tensor,
              I: torch.Tensor,
              num_layers: int = 1,
              layer_idx: int = 0,
              num_lora_adapters: int = 1000,
              scale: float = 1.0) -> torch.Tensor:
    """
    Compute Y = B @ (A @ X) scaled, using grouped BGMV shrink kernel.
    A is A^T, B is B^T.
    """
    assert y.is_cuda and x.is_cuda and A.is_cuda and B.is_cuda and I.is_cuda, "All tensors must be on CUDA"
    assert y.dim() == 3, "y must be [B, n, out_dim]"
    assert x.dim() == 3, "x must be [B, n, in_dim]"
    assert A.dim() == 3 and B.dim() == 3, "A [L*num_layers, r, in_dim], B [L*num_layers, out_dim, r]"
    assert I.dim() == 1 and I.size(0) == y.size(0), "I shape must match batch size"

    Bsz, n, in_dim  = x.shape
    By,  ny, out_dim = y.shape

    x = x.contiguous().view(Bsz * n, in_dim)
    y = y.contiguous().view(By * ny, out_dim)

    seq_len = n

    assert Bsz == By and n == ny, "x/y batch or sequence length mismatch"

    rank = A.size(1)
    L = A.size(0)

    y_intermediate = torch.zeros(Bsz * n, rank, dtype=x.dtype, device=x.device)

    _C.bgmv_forward(y_intermediate, x, A, I, int(seq_len), int(num_layers), int(layer_idx), int(num_lora_adapters), 1.0)
    _C.bgmv_forward(y, y_intermediate, B, I, int(seq_len), int(num_layers), int(layer_idx), int(num_lora_adapters), float(scale)) # Apply scale only in final matmul

    return y