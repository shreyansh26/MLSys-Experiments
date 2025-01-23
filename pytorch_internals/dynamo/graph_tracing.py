# https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html
# TORCH_LOGS=graph_code python graph_tracing.py
import torch

@torch.compile
def fn(x, n):
    y = x ** 2
    if n >= 0:
        return (n + 1) * y
    else:
        return y / n

x = torch.randn(200)
fn(x, 2)
fn(x, 3)
fn(x, -2)
fn(x, 4)