import torch
import torch.nn as nn

SPLITS = 4
# X   
x0 = torch.load('artifacts/input_0.pt').to(0)
x1 = torch.load('artifacts/input_1.pt').to(0)
# x2 = torch.load('artifacts/input_2.pt').to(0)
# x3 = torch.load('artifacts/input_3.pt').to(0)

## Same
print(x0)
print(x1)
# print(x2)
# print(x3)

# A
A_chunks = []
for i in range(SPLITS):
    A_sp = torch.load(f'artifacts/A_{i}.pt').t().to(0)
    A_chunks.append(A_sp)

A = torch.hstack(A_chunks)
print(A.shape)

B_chunks = []
for i in range(SPLITS):
    B_sp = torch.load(f'artifacts/B_{i}.pt').t().to(0)
    B_chunks.append(B_sp)
    
B = torch.vstack(B_chunks)
print(B.shape)

x = x0

ans = torch.matmul(x, A)
ans = nn.GELU()(ans)
ans = torch.matmul(ans, B)

print(ans)

ans_code = torch.load('artifacts/out_0.pt')
print(ans_code)
