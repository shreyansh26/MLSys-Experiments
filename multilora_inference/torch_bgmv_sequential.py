import torch

def vectorized_size(dtype: torch.dtype) -> int:
    """
    Thread tile size - 16 byte vectorization
    """
    elem_bytes = dtype.itemsize
    assert 16 % elem_bytes == 0, f"Dtype {dtype} is not a multiple of 16"
    return 16 // elem_bytes

def bgmv_shrink_torch_sequential(Y: torch.Tensor, 
                                X: torch.Tensor, 
                                W: torch.Tensor,
                                indices: torch.Tensor,    
                                num_layers: int,
                                layer_idx: int,
                                scale: float):
    """
    Sequential implementation of bgmv shrink operation

    Shapes:
        X: [B, F_in]
        W: [L * num_layers, F_out, F_in]
        Y: [B, out] (accumulated into with +=)
        indices: [B] (int32/int64) selecting adapter id in [0, L)
        scale: alpha / r, where r is the rank of the adapter

    Logic:
        For each b, j: Y[b, j] += scale * dot(W[idx(b), j, :], X[b, :]).
    """
    B, F_in = X.shape
    _, F_out, F_in_W = W.shape
    assert F_in_W == F_in
    dtype = X.dtype
    vec = vectorized_size(dtype)

    tx = 32   # Same as warp size
    ty = 4    # Number of warps
    tile_size = tx * ty * vec

    Y_fp32 = Y.to(torch.float32)

    for b in range(B):
        w_idx_adapter = indices[b] * num_layers + layer_idx

        # One block per output j
        for j in range(F_out):
            y_acc = 0.0
            num_tiles = (F_in + tile_size - 1) // tile_size

            for t in range(num_tiles):
                # First reduction over warps and then across warps
                # Per-warp partial
                for warp_ind in range(ty):
                    s_warp = 0.0
                    for lane in range(tx):
                        # Reduce over per thread vectorized chunk
                        base_idx = t * tile_size + (warp_ind * tx + lane) * vec

                        s_local = 0.0
                        for v in range(vec):
                            i = base_idx + v
                            if i < F_in:
                                s_local += float(W[w_idx_adapter, j, i]) * float(X[b, i]) * scale

                        s_warp += s_local

                    y_acc += s_warp

            Y_fp32[b, j] += y_acc

    Y.copy_(Y_fp32.to(Y.dtype))

def bgmv_torch_sequential(Y: torch.Tensor, 
                          X: torch.Tensor, 
                          W: torch.Tensor,
                          indices: torch.Tensor,
                          num_layers: int,
                          layer_idx: int,
                          scale: float):
    """
    Sequential implementation of bgmv operation
    """
    B, F_in = X.shape
    _, F_out, F_in_W = W.shape

    # Shrink case
    if F_in > F_out:
        bgmv_shrink_torch_sequential(Y, X, W, indices, num_layers, layer_idx, scale)
    # Expand case
    else:
        raise NotImplementedError("Expand case not implemented")

if __name__ == "__main__":
    torch.manual_seed(1023)

    B = 3
    F_in = 1204
    F_out = 16
    num_layers = 2
    L = 10
    layeridx = 1
    scale = 0.25

    dtype = torch.bfloat16

    X = torch.randn(B, F_in, dtype=dtype)
    W = torch.randn(L * num_layers, F_out, F_in, dtype=dtype)
    indices = torch.randint(0, L, (B,), dtype=torch.int32)

    Y = torch.zeros(B, F_out, dtype=dtype)

    Y_torch_sequential = Y.clone()
    bgmv_torch_sequential(Y_torch_sequential, X, W, indices, num_layers, layeridx, scale)

    print(Y_torch_sequential)

    Y_mm_lora = Y.clone()
    for b in range(B):
        idx_b = indices[b] * num_layers + layeridx
        Y_mm_lora[b] += (W[idx_b].to(torch.float32) @ X[b].to(torch.float32)) * scale
    Y_mm_lora = Y_mm_lora.to(Y.dtype)

    print(Y_mm_lora)

    print(torch.allclose(Y_torch_sequential, Y_mm_lora))
