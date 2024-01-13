import torch
import triton
import triton.language as tl
from src.activation_fns import tanh_triton_bwd, sigmoid_triton_bwd, relu_triton_bwd, leaky_relu_triton_bwd, fast_gelu_triton_bwd
from typing import Optional, Any

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_N": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_stages=3, num_warps=4),
    ],
    key=["N"],
)
@triton.heuristics({
    'EVEN_N': lambda args: args["N"] % (args['BLOCK_N']) == 0,
})
@triton.jit
def linear_layer_triton_bwd_kernel(
        # Pointers to matrices
        GRAD_OUT, GRAD_ACT, ACT_INPUT,
        # Matrix dimensions
        M, N,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `A`
        # by to get the element one row down (A has M rows).
        stride_grad_out_m, stride_grad_out_n,
        stide_act_inp_m, stride_act_inp_n,
        # Meta-parameters
        BLOCK_N: tl.constexpr,
        EVEN_N: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    """
    Compute gradient for each input
    """
    # Per row - pid m
    # Per column - chunk_id of BLOCK_N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    act_input_ptrs = ACT_INPUT + pid_m * stide_act_inp_m + rn

    if EVEN_N:
        act_input = tl.load(act_input_ptrs)
    else:
        act_input = tl.load(act_input_ptrs, mask=rn < N, other=0.0)

    if ACTIVATION == "tanh":
        grad_act = tanh_triton_bwd(act_input)
    elif ACTIVATION == "sigmoid":
        grad_act = sigmoid_triton_bwd(act_input)
    elif ACTIVATION == "relu":
        grad_act = relu_triton_bwd(act_input)
    elif ACTIVATION == "leaky_relu":
        grad_act = leaky_relu_triton_bwd(act_input)
    elif ACTIVATION == "gelu":
        grad_act = fast_gelu_triton_bwd(act_input)
    elif ACTIVATION == "fast_gelu":
        grad_act = fast_gelu_triton_bwd(act_input)
    else:
        grad_act = act_input

    # now read the incoming gradient, the backpropagated one is the multiple of both
    grad_out_ptrs = GRAD_OUT + pid_m * stride_grad_out_m + rn

    if EVEN_N:
        grad_out = tl.load(grad_out_ptrs)
    else:
        grad_out = tl.load(grad_out_ptrs, mask=rn < N)

    grad_act *= grad_out

    grad_act_ptrs = GRAD_ACT + pid_m * stride_grad_out_m + rn
    tl.store(grad_act_ptrs, grad_act, mask=rn < N)

def compute_linear_layer_triton_bwd(grad_out: torch.Tensor, inp: torch.Tensor, act_inp: Optional[torch.Tensor], weight: torch.Tensor, activation: str="") -> Any:
    """
    Compute grad_inp = activation^-1(grad_out) @ weight.transpose()

    Weight is already transposed as is of shape N, K 
    So, grad_inp = grad_act @ weight ((M, N) @ (N, K) -> (M, K))
    """
    ## Make Contiguous
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_out_flat = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, 1) # M, N
    inp_flat = inp if inp.ndim == 2 else inp.flatten(0, 1)

    assert weight.shape[0] == grad_out_flat.shape[1], "Dimension mismatch"
    
    ## Dtype
    assert grad_out.dtype == weight.dtype, "Dtype mismatch"    

    M, N = inp_flat.shape
    N, K = weight.shape

    # Allocates output
    out = torch.empty((M, N), device=inp.device, dtype=inp.dtype)

    if len(activation) > 0:
        grad_act = torch.empty_like(grad_out_flat)
        # Some activations do not require their inputs to
        # know of their grad, the downstream grad is enough
        if act_inp is None:
            act_inp = grad_out_flat
        
        # 2D launch kernel where each block gets its own program.
        grid = lambda META: (M, triton.cdiv(N, META["BLOCK_N"]))

        linear_layer_triton_bwd_kernel[grid](
            grad_out_flat, grad_act, act_inp, 
            M, N,
            grad_act.stride(0), grad_act.stride(1),
            act_inp.stride(0), act_inp.stride(1),
            ACTIVATION=activation
        )

        grad_out_flat = grad_act

    grad_inp = triton.ops.matmul(grad_out_flat, weight)
    grad_weight = grad_out_flat.transpose(1, 0) @ inp_flat
    grad_bias = torch.sum(grad_out_flat, dim=0)

    grad_inp = grad_inp.reshape_as(inp)
    
    return grad_inp, grad_weight, grad_bias