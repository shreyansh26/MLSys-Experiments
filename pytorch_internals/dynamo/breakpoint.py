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

'''
ctx.print_bt() to print the user stack trace

ctx.print_locals() to print all current locals

ctx.print_graph() to print the currently traced graph

ctx.disas() to print the currently traced function’s bytecode

Use standard pdb commands, such as bt/u/d/n/s/r, - you can go up the pdb stack to inspect more Dynamo internals
'''