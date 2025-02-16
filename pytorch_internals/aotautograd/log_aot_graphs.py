# TORCH_LOGS=aot_graphs python log.py
import torch

def fn(a, b, c, d):
    x = a + b + c + d
    return x.cos().cos()

# Test that it works
a, b, c, d = [torch.randn(2, 4, requires_grad=True) for _ in range(4)]

fn = torch.compile(fn)
ref = fn(a, b, c, d)
loss = ref.sum()
# loss.backward()