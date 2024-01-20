import torch
from src.linear_layer_triton import LinearLayerTriton
import time

torch.manual_seed(0)

ACTIVATION = "gelu"
ADD_BIAS = True

# Can use torch.float16 as well
x = torch.randn((1000, 1024), device='cuda', dtype=torch.float32)
ll_layer = torch.nn.Linear(1024, 512, bias=ADD_BIAS, dtype=torch.float32).to("cuda")
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

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i in range(10):
    triton_output = linear_layer_triton(x)

start.record()
triton_output = linear_layer_triton(x)
end.record()
torch.cuda.synchronize()
triton_time = start.elapsed_time(end)

start2 = torch.cuda.Event(enable_timing=True)
end2 = torch.cuda.Event(enable_timing=True)

for i in range(10):
    torch_output = act(ll_layer(x))

start2.record()
torch_output = act(ll_layer(x))
end2.record()
torch.cuda.synchronize()
torch_time = start2.elapsed_time(end2)  

print(f"Triton time: {triton_time}ms")
print(f"Torch time: {torch_time}ms")

try:
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=0)
    print("✅ (Forward pass) Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ (Forward pass) Triton and Torch differ!")

triton_loss = triton_output.sum()
triton_loss.backward()
triton_weight_grad = linear_layer_triton.weight.grad

torch_loss = torch_output.sum()
torch_loss.backward()
torch_weight_grad = ll_layer.weight.grad

print(triton_weight_grad)
print(torch_weight_grad)

try:
    torch.testing.assert_close(triton_weight_grad, torch_weight_grad, atol=1e-2, rtol=0)
    print("✅ (Backward pass) Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ (Backward pass) Triton and Torch differ!")