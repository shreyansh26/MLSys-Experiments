import torch
import time
import statistics
from functorch.compile import aot_function
from functorch.compile import ts_compile
from functorch.compile import min_cut_rematerialization_partition

def fn(a, b, c, d):
    x = a + b + c + d
    return x.cos().cos()

aot_nnc_fn = aot_function(fn, fw_compiler=ts_compile, bw_compiler=ts_compile)

a, b, c, d = [torch.randn(1024, 2048, requires_grad=True) for _ in range(4)]
ref = fn(a, b, c, d)
loss = ref.sum()
loss.backward()

cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs

res = aot_nnc_fn(*cloned_inputs)
loss = res.sum()
loss.backward()

# Zero out the gradients so we can do a comparison later
a.grad, b.grad, c.grad, d.grad = (None,) * 4

# Lets set up the partitioner. Also set the fwd and bwd compilers to the printer function that we used earlier.
# This will show us how the recomputation has modified the graph.
def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module

aot_fn = aot_function(fn, fw_compiler=compiler_fn, bw_compiler=compiler_fn, partition_fn=min_cut_rematerialization_partition)
res = aot_fn(a, b, c, d).sum().backward()

# Lets set up the partitioner and NNC compiler.
aot_recompute_nnc_fn = aot_function(fn, fw_compiler=ts_compile, bw_compiler=ts_compile, partition_fn=min_cut_rematerialization_partition)

# Correctness checking. Lets clone the input so that we can check grads.
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs

res = aot_recompute_nnc_fn(*cloned_inputs)
loss = res.sum()
loss.backward()
assert torch.allclose(ref, res)
assert torch.allclose(a.grad, cloned_a.grad)
assert torch.allclose(b.grad, cloned_b.grad)
assert torch.allclose(c.grad, cloned_c.grad)
assert torch.allclose(d.grad, cloned_d.grad)

def bench(fn, args, prefix):
    warmup = 10
    iterations = 100

    for _ in range(warmup):
        ref = fn(*args)
        ref.sum().backward()
    
    fw_latencies = []
    bw_latencies = []
    for _ in range(iterations):
        for arg in args:
            arg.grad = None

        fw_begin = time.perf_counter()
        ref = fn(*args)
        fw_end = time.perf_counter()

        loss = ref.sum() 

        bw_begin = time.perf_counter()
        loss.backward()
        bw_end = time.perf_counter()

        fw_latencies.append(fw_end - fw_begin)
        bw_latencies.append(bw_end - bw_begin)
    
    avg_fw_latency = statistics.mean(fw_latencies) * 10**6
    avg_bw_latency = statistics.mean(bw_latencies) * 10**6
    print(prefix, "Fwd = " + str(avg_fw_latency) + " us", "Bwd = " + str(avg_bw_latency) + " us", sep=', ')

large_inputs = [torch.randn(1024, 2048, requires_grad=True) for _ in range(4)]

bench(fn, large_inputs, "Eager")
bench(aot_nnc_fn, large_inputs, "AOT")
bench(aot_recompute_nnc_fn, large_inputs, "AOT_Recomp")