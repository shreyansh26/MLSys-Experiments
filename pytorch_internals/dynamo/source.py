# https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html
# TORCH_LOGS="+dynamo" python source.py
import torch
from torch import Tensor
from typing import List

# @torch.compile
# def foo(x: Tensor, y: List[Tensor]):
#     a = x * y[0]
#     return a * x

# foo(torch.randn(10), [torch.randn(10), torch.randn(20)])

@torch.compile
def fn(x, l):
    return x * len(l[0])

fn(torch.randn(8), ["Hi", "Hello"])