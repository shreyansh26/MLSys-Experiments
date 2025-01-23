# https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html
# TORCH_LOGS=dynamo python guard_origin_on_symbolic_ints.py

import torch

@torch.compile(dynamic=True)
def fn(a):
    if a.shape[0] * 2 < 16:
        return a
    else:
        return a + 1

fn(torch.randn(8))


'''
I0123 11:19:40.477000 140311762486400 torch/fx/experimental/symbolic_shapes.py:5082] [0/0] eval 2*s0 >= 16 [guard added] at nt/ssd1/shreyansh/home_dir/misc_experiments/pytorch_internals/dynamo
guard_origin_on_symbolic_ints.py:8 in fn (_dynamo/variables/tensor.py:1041 in evaluate_expr), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="2*s0 >= 16"
'''