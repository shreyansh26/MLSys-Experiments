# https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html
# TORCH_LOGS=graph_code python mark_dynamic_2.py
import torch

def fn(a, b):
    return (a.shape[0] * a) @ b.T

arg1 = torch.randn(4, 3)
arg2 = torch.randn(4, 3)
torch._dynamo.mark_dynamic(arg1, 0)
torch._dynamo.mark_dynamic(arg2, 0)

compiled_fn = torch.compile(fn)
out = compiled_fn(arg1, arg2)

new_arg1 = torch.randn(8, 3)
new_arg2 = torch.randn(10, 3)
out = compiled_fn(new_arg1, new_arg2)