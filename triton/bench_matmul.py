import torch
from triton.testing import do_bench
from triton.ops import matmul

torch.set_default_device('cuda')

print("A - Row Major, B - Row Major")
for M, K, N in [(2047, 2048, 2048), (2048, 2047, 2048), (2048, 2048, 2047)]:
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)

    print(f"M={M}, K={K}, N={N}")

    print(do_bench(lambda : torch.mm(A, B)))

print("A - Column Major, B - Row Major")
for M, K, N in [(2047, 2048, 2048), (2048, 2047, 2048), (2048, 2048, 2047)]:
    A = torch.randn(K, M, dtype=torch.bfloat16).t()
    B = torch.randn(K, N, dtype=torch.bfloat16)

    print(f"M={M}, K={K}, N={N}")

    print(do_bench(lambda : torch.mm(A, B)))


M = 4096
K = 4096
N = 4096

A = torch.randn(M, K, dtype=torch.bfloat16)
B = torch.randn(K, N, dtype=torch.bfloat16)

print(f"M={M}, K={K}, N={N}")
print("Torch mm", do_bench(lambda : torch.mm(A, B)))
print("Triton mm", do_bench(lambda : matmul(A, B)))

A = torch.randn(M, K, dtype=torch.bfloat16)
B = torch.randn(K, N, dtype=torch.bfloat16)
B = B[:, :4095]

print(f"M={M}, K={K}, N={N}")
print("Torch mm", do_bench(lambda : torch.mm(A, B)))
print("Triton mm", do_bench(lambda : matmul(A, B)))