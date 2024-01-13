import torch
import torch.nn as nn
from src.linear_layer_triton import LinearLayerTriton
from torch.fx import GraphModule
import src.utils as utils
from typing import Callable
from transformers.activations import NewGELUActivation

def linear_layer_triton_wrapper(inp: torch.Tensor, ll_layer: nn.Linear, activation=""):
    weight = ll_layer.weight
    bias = ll_layer.bias

    if weight.dtype == torch.float32:
        weight.data = weight.data.half()
    if bias is not None and bias.dtype == torch.float32:
        bias.data = bias.data.half()
    # print(inp.shape, weight.shape)
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

# T5DenseActDense pattern
def replace_linear_layer3(gm: GraphModule, activation_module: Callable, activation: str, debug: bool=False):
    if debug:
        print(gm.code)
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1, 1)
            self.linear2 = torch.nn.Linear(1, 1)
            self.dropout = torch.nn.Dropout()
            self.activation = activation_module

        def forward(self, v):
            v = self.linear1(v)
            v = self.activation(v)
            v = self.dropout(v)
            return self.linear2(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1, 1)
            self.linear2 = torch.nn.Linear(1, 1)
            self.dropout = torch.nn.Dropout()
            self.activation = activation_module

        def forward(self, v):
            v = linear_layer_triton_wrapper(v, self.linear1, activation=activation)
            v = self.dropout(v)
            return self.linear2(v)

    utils.replace_pattern(gm, Pattern(), Replacement())
    if debug:
        print(gm.code)

# T5DenseGatedActDense pattern
def replace_linear_layer4(gm: GraphModule, activation_module: Callable, activation: str, debug: bool=False):
    if debug:
        print(gm.code)
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1, 1)
            self.linear2 = torch.nn.Linear(1, 1)
            self.linear3 = torch.nn.Linear(1, 1)
            self.dropout = torch.nn.Dropout()
            self.activation = activation_module

        def forward(self, v):
            hidden_gelu = self.activation(self.linear1(v))
            hidden_linear = self.linear2(v)
            v = hidden_gelu * hidden_linear
            v = self.dropout(v)
            return self.linear3(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1, 1)
            self.linear2 = torch.nn.Linear(1, 1)
            self.linear3 = torch.nn.Linear(1, 1)
            self.dropout = torch.nn.Dropout()
            self.activation = activation_module

        def forward(self, v):
            hidden_gelu = linear_layer_triton_wrapper(v, self.linear1, activation=activation)
            hidden_linear = self.linear2(v)
            v = hidden_gelu * hidden_linear
            v = self.dropout(v)
            return self.linear3(v)

    utils.replace_pattern(gm, Pattern(), Replacement())
    if debug:
        print(gm.code)
        
def patch_linear_layer(gm: GraphModule, debug: bool=False):
    replace_linear_layer4(gm, NewGELUActivation(), "fast_gelu", debug=debug)
    replace_linear_layer4(gm, torch.nn.GELU(), "fast_gelu", debug=debug)
    replace_linear_layer4(gm, torch.nn.ReLU(), "relu", debug=debug)
    replace_linear_layer3(gm, NewGELUActivation(), "fast_gelu", debug=debug)
    replace_linear_layer3(gm, torch.nn.GELU(), "fast_gelu", debug=debug)
    replace_linear_layer3(gm, torch.nn.ReLU(), "relu", debug=debug)
    replace_linear_layer2(gm, NewGELUActivation(), "fast_gelu", debug=debug)
    replace_linear_layer2(gm, torch.nn.GELU(), "fast_gelu", debug=debug)
    replace_linear_layer2(gm, torch.nn.ReLU(), "relu", debug=debug)
    replace_linear_layer1(gm, debug=debug)