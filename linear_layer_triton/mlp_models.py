import torch
import torch.nn as nn

class MLP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll = nn.Linear(1024, 512)

    def forward(self, x):
        return self.ll(x)

class MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll = nn.Linear(1024, 512)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        return self.gelu(self.ll(x))