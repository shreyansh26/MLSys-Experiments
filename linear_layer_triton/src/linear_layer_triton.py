import math
from typing import Optional, Any
import torch
import triton
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.nn as nn
from src.kernels.linear_layer_triton_fwd_kernel import compute_linear_layer_triton_fwd
from src.kernels.linear_layer_triton_bwd_kernel import compute_linear_layer_triton_bwd

class linear_layer_triton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: torch.autograd.function.FunctionCtx, inp: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], activation: str="", save_act_inp: bool=True) -> torch.Tensor:
        out, act_inp = compute_linear_layer_triton_fwd(inp, weight, bias, activation, save_act_inp)
        ctx.activation = activation
        ctx.save_for_backward(weight, bias, inp, act_inp)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor) -> Any:
        weight, bias, inp, act_inp = ctx.saved_tensors

        grad_inp, grad_weight, grad_bias = compute_linear_layer_triton_bwd(grad_out, inp, act_inp, weight, ctx.activation)
        if bias is not None:
            return grad_inp, grad_weight, grad_bias, None, None
        return grad_inp, grad_weight, None, None, None
    
# def linear_layer_triton(inp: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], activation: str="") -> torch.Tensor:
#     return LinearLayerTriton.apply(inp, weight, bias, activation)

class LinearLayerTriton(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], activation: str="", save_act_inp: bool=True):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.activation = activation
        self.save_act_inp = save_act_inp

    def forward(self, inp: torch.Tensor):
        return linear_layer_triton.apply(inp, self.weight, self.bias, self.activation, self.save_act_inp)

class LinearLayerTriton2(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=False, activation: str="", **kwargs):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=True
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features), requires_grad=True)
            if bias
            else None
        )
        self.activation = activation
        self.save_act_inp = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp: torch.Tensor):
        return linear_layer_triton.apply(inp, self.weight, self.bias, self.activation, self.save_act_inp)