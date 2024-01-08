import time
import torch
import copy
from src.mlp_models import MLP1, MLP2, MLP3, MLP4, MLP5
from src.patch_linear_layer import patch_linear_layer
from torch.fx import symbolic_trace

NUM_ITERS = 100
DEBUG = False

model = MLP5().to('cuda')
model = model.half()
model.eval()
gm = symbolic_trace(model)
gm_old = copy.deepcopy(gm)
patch_linear_layer(gm, debug=DEBUG)

# print(gm_old.code)
# print("**"*100)
# print(gm.code)

x = torch.randn((1000, 1024), device='cuda', dtype=torch.float16)

torch.cuda.synchronize()

for i in range(NUM_ITERS):
    _ = gm(x)

print("Compilation (for Triton) done!")

torch.cuda.synchronize()

triton_time = 0
for i in range(NUM_ITERS):
    t1 = time.perf_counter()
    triton_output = gm(x)
    t2 = time.perf_counter()
    triton_time += (t2 - t1)
    torch.cuda.synchronize()

triton_time = triton_time / NUM_ITERS * 1_000_000   

for i in range(NUM_ITERS):
    _ = gm_old(x)

print("Warmup (for Torch) done!")

torch.cuda.synchronize()

torch_time = 0
for i in range(NUM_ITERS):
    t1 = time.perf_counter()
    torch_output = gm_old(x)
    t2 = time.perf_counter()
    torch_time += (t2 - t1)
    torch.cuda.synchronize()

torch_time = torch_time / NUM_ITERS * 1_000_000      

print(f"Triton time: {triton_time}ms")
print(f"Torch time: {torch_time}ms")

try:
    torch.testing.assert_close(gm_old(x), gm(x), atol=1e-2, rtol=0)
    print("✅ Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ Triton and Torch differ!")