import torch
from .bgmv_triton import bgmv_triton

def lora_bgmv_triton(y: torch.Tensor,
                     x: torch.Tensor,
                     A: torch.Tensor,
                     B: torch.Tensor,
                     I: torch.Tensor,
                     num_layers: int = 1,
                     layer_idx: int = 0,
                     num_lora_adapters: int = 1000,
                     scale: float = 1.0) -> torch.Tensor:
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

    bgmv_triton(y_intermediate, x, A, I, seq_len=int(seq_len), num_layers=int(num_layers), layer_idx=int(layer_idx), num_lora_adapters=int(num_lora_adapters), scale=1.0, accumulate=True)    # accumulate can be both True or False as Y_intermediate is zeroed
    bgmv_triton(y, y_intermediate, B, I, seq_len=int(seq_len), num_layers=int(num_layers), layer_idx=int(layer_idx), num_lora_adapters=int(num_lora_adapters), scale=float(scale), accumulate=True)

    return y