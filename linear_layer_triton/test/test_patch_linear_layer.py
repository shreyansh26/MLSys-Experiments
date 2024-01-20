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

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i in range(NUM_ITERS):
    _ = gm(x)

print("Compilation (for Triton) done!")

for i in range(NUM_ITERS):
    triton_output = gm(x)

start.record()
triton_output = gm(x)
end.record()
torch.cuda.synchronize()
triton_time = start.elapsed_time(end)

for i in range(NUM_ITERS):
    _ = gm_old(x)

print("Warmup (for Torch) done!")

start2 = torch.cuda.Event(enable_timing=True)
end2 = torch.cuda.Event(enable_timing=True)

for i in range(NUM_ITERS):
    torch_output = gm_old(x)

start2.record()
torch_output = gm_old(x)
end2.record()
torch.cuda.synchronize()
torch_time = start2.elapsed_time(end2)      

print(f"Triton time: {triton_time}ms")
print(f"Torch time: {torch_time}ms")

try:
    torch.testing.assert_close(gm_old(x), gm(x), atol=1e-2, rtol=0)
    print("✅ Triton and Torch match!")
except Exception as e:
    print(e)
    print("❌ Triton and Torch differ!")