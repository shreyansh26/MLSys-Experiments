import torch
import torch.nn as nn
from linear_layer_triton import LinearLayerTriton
from torch.fx import GraphModule
import utils
from typing import Callable

def linear_layer_triton_wrapper(inp: torch.Tensor, ll_layer: nn.Linear, activation=""):
    weight = ll_layer.weight
    bias = ll_layer.bias

    if weight.dtype == torch.float32:
        weight.data = weight.data.half()
    if bias is not None and bias.dtype == torch.float32:
        bias.data = bias.data.half()
    
    llt = LinearLayerTriton(weight, bias, activation)
    return llt(inp)

torch.fx.wrap("linear_layer_triton_wrapper")

def replace_linear_layer1(gm: GraphModule, debug: bool=False):
    if debug:
        print(gm.code)
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, v):
            return self.linear(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, v):
            return linear_layer_triton_wrapper(v, self.linear)

    utils.replace_pattern(gm, Pattern(), Replacement())
    if debug:
        print(gm.code)

def replace_linear_layer2(gm: GraphModule, activation_module: Callable, activation: str, debug: bool=False):
    if debug:
        print(gm.code)
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.activation = activation_module

        def forward(self, v):
            return self.activation(self.linear(v))

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.activation = activation_module

        def forward(self, v):
            return linear_layer_triton_wrapper(v, self.linear, activation=activation)

    utils.replace_pattern(gm, Pattern(), Replacement())
    if debug:
        print(gm.code)

def patch_linear_layer(gm: GraphModule, debug: bool=False):
    replace_linear_layer2(gm, torch.nn.GELU(), "fast_gelu", debug=debug)
    replace_linear_layer2(gm, torch.nn.ReLU(), "relu", debug=debug)
    replace_linear_layer1(gm, debug=debug)