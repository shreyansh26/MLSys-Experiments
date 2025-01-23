from transformers import BertTokenizer, BertModel
import torch.nn as nn
import time
import torch
import torch_tensorrt

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(512, 1024)
        self.l2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l1(x)
        x = self.l2(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().eval()
model = model.to(device)
inputs = [torch_tensorrt.Input(
        min_shape=[64, 512],
        opt_shape=[256, 512],
        max_shape=[512, 512],
    )]

model = torch_tensorrt.compile(model, inputs=inputs)

input_data = torch.randn(256, 512).to(device)

total_time = 0

# Warmup
for i in range(2):
    output = model(input_data)

print(output)

print("Starting...")

for i in range(1000):
    start = time.time_ns()
    output = model(input_data)
    end = time.time_ns()
    total_time += (end - start) / 1_000_000

print(f"Average time: {total_time/1000:.2f}ms")