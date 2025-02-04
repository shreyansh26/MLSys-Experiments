import torch
from functorch.compile import aot_function

def fn(a, b, c, d):
    x = a + b + c + d
    return x.cos().cos()

# Test that it works
a, b, c, d = [torch.randn(2, 4, requires_grad=True) for _ in range(4)]
ref = fn(a, b, c, d)
loss = ref.sum()
loss.backward()

# The compiler_fn is called after the forward and backward graphs are extracted.
# Here, we just print the code in the compiler_fn. Return of this function is a callable.
def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module

# Pass on the compiler_fn to the aot_function API
aot_print_fn = aot_function(fn, fw_compiler=compiler_fn, bw_compiler=compiler_fn)

# Run the aot_print_fn once to trigger the compilation and print the graphs
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs
res = aot_print_fn(cloned_a, cloned_b, cloned_c, cloned_d)
res.sum().backward()
assert torch.allclose(ref, res)