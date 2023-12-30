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

class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 1024)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.ll1(x)
        x = self.gelu(x)
        x = self.ll2(x)
        return x