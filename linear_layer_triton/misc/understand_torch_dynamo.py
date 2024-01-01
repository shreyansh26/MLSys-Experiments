from typing import List
import torch
from torch import _dynamo as torchdynamo
import torch._dynamo.config
import logging

torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.output_code = True

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

@torchdynamo.optimize(my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
for _ in range(100):
    o = toy_example(torch.randn(10), torch.randn(10))
    # print(o)