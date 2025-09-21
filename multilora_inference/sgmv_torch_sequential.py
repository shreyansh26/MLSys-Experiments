import math
from typing import List, Sequence, Optional
import torch

def ceil_div(a: int, b: int) -> int:
    return -(-a // b)

@torch.no_grad()
def sgmv_shrink_torch(
    y: torch.Tensor,           # [total_rows, d_out], dtype T (fp16/bf16), in-place updated (Y += X W^T)
    x: torch.Tensor,           # [total_rows, d_in], dtype T (fp16/bf16)
    w_list: Sequence[torch.Tensor],  # per-problem weights; each item is either [d_out, d_in] or [num_layers, d_out, d_in]
    s: torch.Tensor,           # [num_problems+1] (int), segment offsets
    *,
    layer_idx: int = 0,
    chunk_size: int = 128,     # K-chunk size per "bx" (gridDim.x). Kernel uses 128 per K-iteration; chunk_size is user-chosen.
    num_warps: int = 4,        # row tile is 16 * num_warps, matching kernel step height
    cooperative: bool = True   # if False, require num_chunks == 1 (like kernel)
) -> torch.Tensor:
    """
    PyTorch-sequential analogue of flashinfer::sgmv::sgmv_shrink.
    Computes, per problem p:
        Y_p[rows, :] += X_p[rows, :] @ W_p[layer]^T
    with the same segmentation/tiling/chunking scheme as the CUDA kernel.

    Shapes:
      - x, y are concatenations of all problems along axis 0.
      - s[p]:s[p+1] gives the row range for problem p.
      - w_list[p] is either [d_out, d_in] or [num_layers, d_out, d_in].
      - d_out must be a multiple of 16 (like the kernel template parameter).

    Dtypes:
      - x, y, w should be in fp16 or bf16 (T). Accumulation is in fp32.
    """

    assert x.dim() == 2 and y.dim() == 2, "x, y must be 2D"
    total_rows, d_in = x.shape
    total_rows_y, d_out = y.shape
    assert total_rows == total_rows_y, "x and y must have same number of rows"
    assert d_out % 16 == 0, "d_out must be a multiple of 16 to mirror the kernel"
    assert s.dim() == 1 and s.numel() >= 2, "s must be 1D with at least 2 elements"

    num_problems = s.numel() - 1
    assert len(w_list) == num_problems, "w_list length must match num_problems"

    device = y.device
    T_dtype = y.dtype
    # assert T_dtype in (torch.float16, torch.bfloat16), "T must be fp16 or bf16 (to match kernel intent)"

    # Number of K-chunks (gridDim.x analogue).
    num_chunks = ceil_div(d_in, chunk_size)
    if not cooperative:
        # Kernel's non-cooperative path assumes gridDim.x == 1.
        assert num_chunks == 1, "Non-cooperative path requires num_chunks == 1"

    num_blocks_n = d_out // 16              # N-tiling
    row_step = 16 * num_warps               # M-tiling per kernel step
    num_k_frags = 8                         # 8 fragments per K-iteration
    k_frag = 16                             # each fragment is 16 along K
    k_iter_size = num_k_frags * k_frag      # = 128

    # Ensure contiguity for predictable slicing cost (not required, but tidy).
    x = x.contiguous()
    y = y.contiguous()

    # Loop over problems (blockIdx.y analogue)
    for p in range(num_problems):
        s_start = int(s[p].item())
        s_end   = int(s[p+1].item())
        if s_end <= s_start:
            continue

        # Select weight for this problem and layer
        Wp = w_list[p]
        if Wp.dim() == 3:
            # [num_layers, d_out, d_in]
            assert 0 <= layer_idx < Wp.shape[0], "layer_idx out of range"
            W = Wp[layer_idx]
        else:
            # [d_out, d_in]
            W = Wp
        assert W.shape == (d_out, d_in), f"W[{p}] must be [d_out, d_in]"

        # Optional: ensure same dtype/device as x/y
        W = W.to(device=device, dtype=T_dtype).contiguous()

        # Number of row steps for this segment
        M_p = s_end - s_start
        num_steps = ceil_div(M_p, row_step)

        # Iterate over M-tiles (step i), 16 rows per warp
        for i in range(num_steps):
            row0 = s_start + i * row_step
            row1 = min(s_end, row0 + row_step)
            M_i = row1 - row0
            if M_i <= 0:
                continue

            # We will build per-chunk contributions (like tmp + grid.sync() reduce).
            # Each element is an fp32 [M_i, d_out] tensor.
            chunk_contribs: List[torch.Tensor] = []

            # K chunk loop (blockIdx.x analogue)
            for bx in range(num_chunks):
                k0 = bx * chunk_size
                k1 = min(d_in, (bx + 1) * chunk_size)
                if k1 <= k0:
                    # empty chunk
                    # still push a zero contrib to keep alignment with reduction
                    chunk_contribs.append(torch.zeros(M_i, d_out, dtype=torch.float32, device=device))
                    continue

                # Initialize Y_frag: chunk 0 starts from existing Y (like kernel load);
                # other chunks start from zeros. Accumulate in fp32.
                if bx == 0:
                    y_init = y[row0:row1, :].to(torch.float32)
                else:
                    y_init = torch.zeros(M_i, d_out, dtype=torch.float32, device=device)
                Y_frag = y_init  # fp32 accumulator for this chunk

                # Number of K-iterations in this chunk (each iteration covers up to 128 K)
                num_iterations = ceil_div(k1 - k0, k_iter_size)

                # Loop over K-iterations (double-buffered in kernel; here sequential)
                for t in range(num_iterations):
                    kt0 = k0 + t * k_iter_size
                    kt1 = min(k1, kt0 + k_iter_size)
                    if kt1 <= kt0:
                        continue

                    # Split the 128-wide K-iteration into 8 × 16 fragments
                    for fk in range(num_k_frags):
                        frag_k0 = kt0 + fk * k_frag
                        frag_k1 = min(kt1, frag_k0 + k_frag)
                        if frag_k1 <= frag_k0:
                            continue

                        # A := X tile (M_i × K_sub), convert to fp32 for accumulation
                        A = x[row0:row1, frag_k0:frag_k1].to(torch.float32)   # [M_i, k_sub]

                        # For each 16-wide output block (N tile)
                        for j in range(num_blocks_n):
                            n0 = 16 * j
                            n1 = n0 + 16
                            # W_sub: [16, k_sub] in T; B := W_sub^T is [k_sub, 16]
                            W_sub = W[n0:n1, frag_k0:frag_k1].to(torch.float32)  # [16, k_sub]
                            B = W_sub.transpose(0, 1)                              # [k_sub, 16]

                            # Y_frag[:, n0:n1] += A @ B  (fp32 accumulate)
                            Y_frag[:, n0:n1] += A @ B

                # End of all K-iterations for this chunk
                chunk_contribs.append(Y_frag)

            # Cross-chunk reduction (like grid-wide tmp sum); includes y_init from bx==0 exactly once.
            Y_tile_fp32 = torch.stack(chunk_contribs, dim=0).sum(dim=0)  # [M_i, d_out], fp32

            # Store back to y in dtype T
            y[row0:row1, :] = Y_tile_fp32.to(dtype=T_dtype)

    return y

if __name__ == "__main__":
    # Example sizes (keep small for clarity)
    P = 3                       # num_problems
    d_in = 640
    d_out = 64                  # multiple of 16 (required)
    row_counts = [37, 128, 19]  # different segment heights

    # Build s offsets
    s_list = [0]
    for rc in row_counts:
        s_list.append(s_list[-1] + rc)
    s = torch.tensor(s_list, dtype=torch.int64)

    total_rows = s[-1].item()

    # Dtype T (the kernel runs on fp16/bf16; accumulate in fp32)
    T = torch.float32

    # Random test data
    torch.manual_seed(0)
    x = torch.randn(total_rows, d_in, dtype=T)
    y = torch.randn(total_rows, d_out, dtype=T)

    # Per-problem weights: either [d_out, d_in] or [num_layers, d_out, d_in]
    w_list = [torch.randn(d_out, d_in, dtype=T) for _ in range(P)]

    # Make reference result: Y_ref = Y_in + X W^T per segment
    y_ref = y.clone()
    for p in range(P):
        a, b = s[p].item(), s[p+1].item()
        y_ref[a:b] = (y_ref[a:b].to(torch.float32) + x[a:b].to(torch.float32) @ w_list[p].to(torch.float32).t()).to(T)

    # Run the sequential analogue (choose a chunk_size to force multiple K-chunks)
    y_out = sgmv_shrink_torch(
        y=y.clone(),
        x=x,
        w_list=w_list,
        s=s,
        layer_idx=0,
        chunk_size=256,      # e.g., split K across chunks (d_in=640 -> 3 chunks: 256, 256, 128)
        num_warps=2,         # row-step = 32 rows per step (2 warps * 16 rows)
        cooperative=True     # sum across chunks like the kernel's cooperative mode
    )

    # Compare
    max_abs_err = (y_out.to(torch.float32) - y_ref.to(torch.float32)).abs().max().item()
    print("max_abs_err:", max_abs_err)   # should be ~1e-2 to 1e-3 for fp16/bf16 scale
