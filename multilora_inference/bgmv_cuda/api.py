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
              scale: float = 1.0) -> torch.Tensor:
    """
    Compute Y = B @ (A @ X) scaled, using grouped BGMV shrink kernel.
    A is A^T, B is B^T.
    """
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

    _C.bgmv_forward(y_intermediate, x, A, I, int(num_layers), int(layer_idx), float(scale))
    _C.bgmv_forward(y, y_intermediate, B, I, int(num_layers), int(layer_idx), float(scale))

    # Return reshaped view; storage already updated in-place via the 2D view
    return y.view(original_y_shape)