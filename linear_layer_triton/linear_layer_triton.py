from typing import Optional
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
from torch.cuda.amp import custom_fwd

from activation_fns import tanh_triton, sigmoid_triton, relu_triton, leaky_relu_triton, gelu_triton, fast_gelu_triton

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, "SPLIT_K": 1}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, "SPLIT_K": 1}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
    prune_configs_by={"early_config_prune": early_config_prune, "perf_model": estimate_matmul_time, "top_k": 10},
)
@triton.jit
def linear_layer_triton_kernel(
        # Pointers to matrices
        A, B, bias_ptr, C,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `A`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
        ADD_BIAS: tl.constexpr
):
    """
    Kernel for computing (C = A @ B.T + bias)
    A (inp) has shape (M, K), B (Weight) has shape (N, K), bias has shape (N,) and C (Output) has shape (M, N)
    Note - A column-major traversal of B is essentially a row-major traversal of B_T which is what we want
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # Change ordering of ids to promote L2 data reuse
    # Refer "L2 Cache Optimizations" (https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_M, BLOCK_K] pointers
    # `b_ptrs` is a block of [BLOCK_K, BLOCK_N] pointers
    # Refer "Pointer Arithmetics" (https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#pointer-arithmetics)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) # Row-major
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn) # Column-major

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_M, BLOCK_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if ADD_BIAS:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "tanh":
        accumulator = tanh_triton(accumulator)
    if ACTIVATION == "sigmoid":
        accumulator = sigmoid_triton(accumulator)
    if ACTIVATION == "relu":
        accumulator = relu_triton(accumulator)
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu_triton(accumulator)
    if ACTIVATION == "gelu":
        accumulator = gelu_triton(accumulator)
    if ACTIVATION == "fast_gelu":
        accumulator = fast_gelu_triton(accumulator)

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def compute_linear_layer_triton(inp: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], activation: str="") -> torch.Tensor:
    # Check constraints
    ## Dimension
    inp_flat = inp if inp.ndim == 2 else inp.flatten(0, 1)
    add_bias = bias is not None
    assert inp_flat.shape[1] == weight.shape[1], "Dimension mismatch"
    if bias is not None:
        assert bias.shape[0] == weight.shape[0], "Dimension mismatch"
    ## Dtype
    assert inp.dtype == weight.dtype, "Dtype mismatch"
    if add_bias:
        assert inp.dtype == bias.dtype
    ## Contiguous
    assert inp.is_contiguous(), "Matrix A must be contiguous"
    assert weight.is_contiguous(), "Matrix B must be contiguous"
    if add_bias:
        assert bias.is_contiguous(), "Matrix B must be contiguous"

    M, K = inp_flat.shape
    N, K = weight.shape

    # Allocates output
    out = torch.empty((M, N), device=inp.device, dtype=inp.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    linear_layer_triton_kernel[grid](
        inp_flat, weight, bias, out,
        M, N, K,
        inp_flat.stride(0), inp_flat.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        ACTIVATION=activation,
        ADD_BIAS=add_bias
    )
    out = out if inp.ndim == 2 else out.reshape(inp.shape[0], -1, N)

    return out

class LinearLayerTriton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: torch.autograd.function.FunctionCtx, inp: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], activation: str="") -> torch.Tensor:
        ctx.save_for_backward(inp, weight, bias)
        out = compute_linear_layer_triton(inp, weight, bias, activation)
        return out
    
def linear_layer_triton(inp: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], activation: str="") -> torch.Tensor:
    return LinearLayerTriton.apply(inp, weight, bias, activation)