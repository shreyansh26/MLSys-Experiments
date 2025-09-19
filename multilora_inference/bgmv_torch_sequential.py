import torch

def vectorized_size(dtype: torch.dtype) -> int:
    """
    Thread tile size - 16 byte vectorization
    """
    elem_bytes = dtype.itemsize
    assert 16 % elem_bytes == 0, f"dtype {dtype} is not a multiple of 16"
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

        Each block computes one output row j of a single batch b. Grid spans over (b, j)
        tx -> Lanes per warp (full warp 32)
        ty -> Number of warps per block

        tile_size -> Inputs per tile -> block iterates over F_in
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

                        # Intra-warp: sum over x lane contributions s_local into s_warp (emulates warp shuffle reduction over 32 lanes)
                        s_warp += s_local
                    # Cross-warp: sum s_warp across ty warps into y_acc
                    y_acc += s_warp
            # Across tiles: y_acc accumulates over all tiles covering F_in
            Y_fp32[b, j] += y_acc

    Y.copy_(Y_fp32.to(Y.dtype))

def bgmv_expand_torch_sequential(Y: torch.Tensor, 
                                X: torch.Tensor, 
                                W: torch.Tensor,
                                indices: torch.Tensor,
                                num_layers: int,
                                layer_idx: int,
                                scale: float):
    """
    Sequential implementation of bgmv expand operation

    Shapes:
        X: [B, F_in]
        W: [L * num_layers, F_out, F_in]
        Y: [B, F_out] (accumulated into with +=)
        indices: [B] (int32/int64) selecting adapter id in [0, L)
        scale: alpha / r, where r is the rank of the adapter

    Logic:
        For each b, j: Y[b, j] += scale * dot(W[idx(b), j, :], X[b, :]).
            
        Each block computes many rows of output rows at once before the reduction per row is small
        tx -> Lanes participating in the reduction for one row; tx ≤ 32; tx * vec == F_in -> No tiling for F_in (tx lanes cover the entire input dimension once)
        ty -> How many independent row-reductions fit in one warp
        tz -> Number of warps per block; we replicate the "warp that computes ty rows" tz times inside the block

        rows_per_block -> ty * tz
    """
    B, F_in = X.shape
    _, F_out, F_in_W = W.shape
    assert F_in_W == F_in
    dtype = X.dtype
    vec = vectorized_size(dtype)
    assert F_in % vec == 0, "expand requires F_in % vec_size == 0"
    
    # Needed so that tx is an integer -> F_in covered by tx lanes once
    tx = F_in // vec
    # Needed so that warp can be partitioned into groups of size tx
    assert 32 % tx == 0, "expand requires 32 % tx == 0"
    # A single warp is split into ty disjoint subwarps, each of width tx. Each subwarp computes one row independently at the same time
    ty = 32 // tx
    tz = 4
    rows_per_block = ty * tz

    Y_fp32 = Y.to(torch.float32)

    for b in range(B):
        w_idx_adapter = indices[b] * num_layers + layer_idx
        num_tiles = (F_out + rows_per_block - 1) // rows_per_block

        # One block for a small tile of output rows
        for t in range(num_tiles):
            j0 = t * rows_per_block
            for warp_ind in range(tz):
                # subwarp in a warp
                for y_subwarp in range(ty):
                    # For tile starting at j0, row index j = j0 + warp_ind * ty + y_subwarp
                    # Different (warp_ind, y_subwarp) pairs select different rows; no cross-row sharing
                    j = j0 + warp_ind * ty + y_subwarp
                    if j >= F_out:
                        continue
                    # Reduce over each row
                    # Reduction is across the tx lanes (a subgroup of the warp)
                    # Intra-subwarp: sum across tx lanes to get the dot for one row (emulates cooperative_groups tiled_partition<tx>)
                    s_row = 0.0
                    for xlane in range(tx):
                        base = xlane * vec
                        s_local = 0.0
                        for v in range(vec):
                            i = base + v
                            if i < F_in:
                                s_local += float(W[w_idx_adapter, j, i]) * float(X[b, i]) * scale
                        s_row += s_local
                    Y_fp32[b, j] += s_row
    
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
        bgmv_expand_torch_sequential(Y, X, W, indices, num_layers, layer_idx, scale)

if __name__ == "__main__":
    torch.manual_seed(1023)

    MODE = "expand"
    B = 3
    num_layers = 2
    L = 10
    layeridx = 1
    scale = 0.25

    if MODE == "shrink":
        F_in = 1024
        F_out = 16
    elif MODE == "expand":
        F_in = 16
        F_out = 1024
    else:
        raise ValueError(f"Invalid mode: {MODE}")

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
