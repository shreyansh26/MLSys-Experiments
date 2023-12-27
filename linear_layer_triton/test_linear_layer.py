import torch
from linear_layer import linear_layer
import time

torch.manual_seed(0)

ACTIVATION = "gelu"
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

for i in range(10):
    _ = act(ll_layer(x))
    _ = linear_layer(x, weight, bias, add_bias=ADD_BIAS)
    _ = linear_layer(x, weight, bias, activation=ACTIVATION, add_bias=ADD_BIAS)

print("Warmup (for Torch) & Compilation (for Triton) done!")

t1 = time.time_ns()
triton_output = linear_layer(x, weight, bias, activation=ACTIVATION, add_bias=ADD_BIAS)
t2 = time.time_ns()
torch_output = act(ll_layer(x))
t3 = time.time_ns()

print(f"Triton time: {(t2 - t1)/1000000}ms")
print(f"Torch time: {(t3 - t2)/1000000}ms")

try:
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=0)
    print("✅ Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ Triton and Torch differ!")