import torch
from linear_layer_triton import LinearLayerTriton
import time

torch.manual_seed(0)

ACTIVATION = "fast_gelu"
ADD_BIAS = True

x = torch.randn((1000, 1024), device='cuda', dtype=torch.float16)
ll_layer = torch.nn.Linear(1024, 512, bias=ADD_BIAS, dtype=torch.float16).to("cuda")
act = torch.nn.Identity()

if ACTIVATION == "tanh":
    act = torch.nn.Tanh()
if ACTIVATION == "sigmoid":
    act = torch.nn.Sigmoid()
if ACTIVATION == "relu":
    act = torch.nn.ReLU()
if ACTIVATION == "leaky_relu":
    act = torch.nn.LeakyReLU()
if ACTIVATION == "gelu":
    act = torch.nn.GELU()
if ACTIVATION == "fast_gelu":
    act = torch.nn.GELU(approximate="tanh")

weight = ll_layer.weight
bias = ll_layer.bias

linear_layer_triton = LinearLayerTriton(weight, bias, activation=ACTIVATION)

for i in range(10):
    _ = act(ll_layer(x))
    _ = linear_layer_triton(x)

print("Warmup (for Torch) & Compilation (for Triton) done!")

triton_time = 0
for i in range(10):
    t1 = time.time_ns()
    triton_output = linear_layer_triton(x)
    t2 = time.time_ns()
    triton_time += (t2 - t1)

triton_time = triton_time / 10 / 1_000_000   

torch_time = 0
for i in range(10):
    t1 = time.time_ns()
    torch_output = act(ll_layer(x))
    t2 = time.time_ns()
    torch_time += (t2 - t1)

torch_time = torch_time / 10 / 1_000_000   

print(f"Triton time: {triton_time}ms")
print(f"Torch time: {torch_time}ms")

try:
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=0)
    print("✅ Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ Triton and Torch differ!")