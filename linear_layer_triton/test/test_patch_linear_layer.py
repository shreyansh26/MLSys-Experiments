import time
import torch
import copy
from src.mlp_models import MLP1, MLP2, MLP3
from src.patch_linear_layer import patch_linear_layer
from torch.fx import symbolic_trace

model = MLP3().to('cuda')
model = model.half()
gm = symbolic_trace(model)
gm_old = copy.deepcopy(gm)
patch_linear_layer(gm, debug=True)

x = torch.randn((1000, 1024), device='cuda', dtype=torch.float16)

for i in range(10):
    _ = gm_old(x)
    _ = gm(x)

print("Warmup (for Torch) & Compilation (for Triton) done!")

triton_time = 0
for i in range(10):
    t1 = time.time_ns()
    triton_output = gm(x)
    t2 = time.time_ns()
    triton_time += (t2 - t1)

triton_time = triton_time / 10 / 1_000_000   

torch_time = 0
for i in range(10):
    t1 = time.time_ns()
    torch_output = gm_old(x)
    t2 = time.time_ns()
    torch_time += (t2 - t1)

torch_time = torch_time / 10 / 1_000_000   

print(f"Triton time: {triton_time}ms")
print(f"Torch time: {torch_time}ms")

try:
    torch.testing.assert_close(gm_old(x), gm(x), atol=1e-2, rtol=0)
    print("✅ Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ Triton and Torch differ!")