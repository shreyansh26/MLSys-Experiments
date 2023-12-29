import time
import torch
from mlp_models import MLP1, MLP2
from patch_mlp import patch_mlp

mlp = MLP2().to('cuda')
mlp = mlp.half()
new_mlp = patch_mlp(mlp)

x = torch.randn((1000, 1024), device='cuda', dtype=torch.float16)

for i in range(10):
    _ = mlp(x)
    _ = new_mlp(x)

print("Warmup (for Torch) & Compilation (for Triton) done!")

triton_time = 0
for i in range(10):
    t1 = time.time_ns()
    triton_output = new_mlp(x)
    t2 = time.time_ns()
    triton_time += (t2 - t1)

triton_time = triton_time / 10 / 1_000_000   

torch_time = 0
for i in range(10):
    t1 = time.time_ns()
    torch_output = mlp(x)
    t2 = time.time_ns()
    torch_time += (t2 - t1)

torch_time = torch_time / 10 / 1_000_000   

print(f"Triton time: {triton_time}ms")
print(f"Torch time: {torch_time}ms")

try:
    torch.testing.assert_close(mlp(x), new_mlp(x), atol=1e-2, rtol=0)
    print("✅ Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ Triton and Torch differ!")