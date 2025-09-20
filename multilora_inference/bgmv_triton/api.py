import torch
from .bgmv_triton import bgmv_triton

def lora_bgmv_triton(y: torch.Tensor,
                     x: torch.Tensor,
                     A: torch.Tensor,
                     B: torch.Tensor,
                     I: torch.Tensor,
                     num_layers: int = 1,
                     layer_idx: int = 0,
                     scale: float = 1.0) -> torch.Tensor:
    assert y.is_cuda and x.is_cuda and A.is_cuda and B.is_cuda and I.is_cuda, "All tensors must be on CUDA"
    assert y.dim() == 3 and y.size(1) == 1, "y must be [B, 1, out_dim]"
    assert x.dim() == 3 and x.size(1) == 1, "x must be [B, 1, in_dim]"
    assert A.dim() == 3 and B.dim() == 3, "A [L*num_layers, r, in_dim], B [L*num_layers, out_dim, r]"
    assert I.dim() == 1 and I.size(0) == y.size(0), "I shape must match batch size"

    batch_size = x.size(0)
    in_dim = x.size(2)
    out_dim = y.size(2)
    rank = A.size(1)
    L = A.size(0)

    original_y_shape = y.shape
    x = x.view(batch_size, in_dim)
    y = y.view(batch_size, out_dim)

    y_intermediate = torch.zeros(batch_size, rank, dtype=x.dtype, device=y.device)

    bgmv_triton(y_intermediate, x, A, I, num_layers=int(num_layers), layer_idx=int(layer_idx), scale=float(scale), accumulate=False)
    bgmv_triton(y, y_intermediate, B, I, num_layers=int(num_layers), layer_idx=int(layer_idx), scale=float(scale), accumulate=True)

    # Return reshaped view
    return y.view(original_y_shape)