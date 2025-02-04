import torch
from functorch.compile import aot_function
# AOT Autograd has a suite of already integrated backends. Lets import the NNC compiler backend - ts_compile
from functorch.compile import ts_compile

def fn(a, b, c, d):
    x = a + b + c + d
    return x.cos().cos()

# Test that it works
a, b, c, d = [torch.randn(2, 4, requires_grad=True) for _ in range(4)]
ref = fn(a, b, c, d)
loss = ref.sum()
loss.backward()

# Lets compile the forward and backward through ts_compile.
aot_nnc_fn = aot_function(fn, fw_compiler=ts_compile, bw_compiler=ts_compile)

# Correctness checking. Lets clone the input so that we can check grads.
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs

res = aot_nnc_fn(*cloned_inputs)
loss = res.sum()
loss.backward()
assert torch.allclose(ref, res)
assert torch.allclose(a.grad, cloned_a.grad)
assert torch.allclose(b.grad, cloned_b.grad)
assert torch.allclose(c.grad, cloned_c.grad)
assert torch.allclose(d.grad, cloned_d.grad)