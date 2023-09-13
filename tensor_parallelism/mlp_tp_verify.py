import torch
import torch.nn as nn
import time

class MLP(nn.Module):
    def __init__(self, d, device="cpu"):
        super().__init__()
        self.A = torch.rand(d, 4*d).to(device)
        self.B = torch.rand(4*d, d).to(device)
        self.gelu = nn.GELU()

    def direct_forward(self, x):
        x = torch.matmul(x, self.A)
        time.sleep(1)
        x = self.gelu(x)
        x = torch.matmul(x, self.B)
        return x

    def split_forward(self, x):
        A1, A2 = torch.chunk(self.A, 2, dim=-1)
        B1, B2 = torch.chunk(self.B, 2, dim=0)
        XA1 = torch.matmul(x, A1) 
        XA2 = torch.matmul(x, A2) 
        Y1 = self.gelu(XA1)
        Y2 = self.gelu(XA2)
        Y1B1 = torch.matmul(Y1, B1)
        Y2B2 = torch.matmul(Y2, B2)
        return Y1B1 + Y2B2

    def forward(self, x):
        ans1 = self.direct_forward(x)
        ans2 = self.split_forward(x)
        return torch.allclose(ans1, ans2)

if __name__ == "__main__":
    n = 1000
    d = 16384
    mlp = MLP(d, "cuda:0")
    x = torch.rand(n, d).to("cuda:0")

    print(mlp.direct_forward(x))
    # print(mlp(x))