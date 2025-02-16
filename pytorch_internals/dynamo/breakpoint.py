import torch
from torch._dynamo.comptime import comptime

def f(x):
    x = x + torch.tensor(1)
    comptime.breakpoint()
    return torch.sin(torch.cos(x))

f_compiled = torch.compile(f)

x = torch.tensor(1)
print(f_compiled(x))
print(f(x))
