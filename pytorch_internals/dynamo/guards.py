# https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html
# TORCH_LOGS=graph_code,guards,recompiles python guards.py
import torch

@torch.compile
def fn(a, b):
    return a * len(b)

fn(torch.arange(10), "Hello")
fn(torch.arange(10), "Hi")