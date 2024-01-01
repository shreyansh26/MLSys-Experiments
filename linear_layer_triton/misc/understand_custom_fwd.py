import torch
from torch.cuda.amp import custom_fwd, custom_bwd

class LinearLayerTriton(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float64)
    def forward(ctx: torch.autograd.function.FunctionCtx, inp: torch.Tensor, weight: torch.Tensor, activation: str="") -> torch.Tensor:
        ctx.save_for_backward(inp, weight)
        print(inp.dtype)
        return inp

def linear_layer_triton(inp: torch.Tensor, weight: torch.Tensor, activation: str="") -> torch.Tensor:
    print(inp.dtype)
    return LinearLayerTriton.apply(inp, weight, activation)


x = torch.randn((1000, 1024), device='cuda', dtype=torch.float32)
weight = torch.randn((1000, 1024), device='cuda', dtype=torch.float32)

linear_layer_triton(x, weight)

with torch.cuda.amp.autocast():
    linear_layer_triton(x, weight)