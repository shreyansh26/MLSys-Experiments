# https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html
# TORCH_LOGS=graph_code python mark_dynamic_3.py
import torch

def fn2(a, b):
    c = torch.cat([a, b], dim=0)
    return c.shape[0] * c

arg1 = torch.randn(4, 3)
arg2 = torch.randn(4, 3)
torch._dynamo.mark_dynamic(arg1, 0)
torch._dynamo.mark_dynamic(arg2, 0)

# Works fine
compiled_fn = torch.compile(fn2)
out = compiled_fn(arg1, arg2)