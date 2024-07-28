from functools import partial
import timeit
import torch
import triton
import triton.language as tl

# Source - https://srush.github.io/annotated-mamba/hard.html

@triton.jit
def plus_fn(a, b):
    return a + b

@triton.jit
def cumsum_kernel(X, H, Y, K: tl.constexpr):
    pid = tl.program_id(0)

    kid = pid * K 
    Ks = tl.arange(0, K)

    x = tl.load(X + kid + Ks)
    h_0 = tl.load(H + Ks*0 + pid, Ks == 0, 0)

    x = plus_fn(x, h_0)

    # Compute scan
    hs = tl.associative_scan(x, 0, plus_fn)
    y = hs

    tl.store(Y + kid + Ks, y)

    tl.store(H + Ks*0 + pid, hs, mask=(Ks == K-1))

def cumsum_block(x, y, K):
    seqlen = y.shape[0]
    BLOCKS = seqlen // K
    h = torch.zeros(2, BLOCKS).float().cuda()
    cumsum_kernel[(BLOCKS,)](x, h[0], y, K)

    # Store cumulative sums of previous blocks
    # print(h[0])
    h[1, 1:] = h[0].cumsum(dim=0)[:-1]
    # print(h[1])

    cumsum_kernel[(BLOCKS,)](x, h[1], y, K)

K = 16
BLOCKS = 8
SEQLEN = K * BLOCKS

h = torch.zeros(BLOCKS).float().cuda()
x = torch.arange(SEQLEN).float().cuda()
y = torch.zeros(SEQLEN).float().cuda()

cumsum_kernel[(BLOCKS,)](x, h, y, K)

print(x)
print(h)
print(y)

cumsum_block(x, y, K)
y_ = x.cumsum(dim=0)
assert torch.allclose(y, y_)

y_large = torch.zeros(2**25).float().cuda()
x_large = torch.arange(2**25).float().cuda()

# Correctness - K>=256 works with Triton==3.0.0 and now with 2.3.1
cumsum_block(x_large, y_large, K=2**10)
y_large_ = x_large.cumsum(dim=0)
print(y_large)
print(y_large_)
torch.testing.assert_close(y_large, y_large_, atol=1e-4, rtol=1e-4)

# Benchmarking

times = timeit.Timer(partial(cumsum_block, x_large, y_large, 2**10)).repeat(3, 1000)
print(f"Triton time: {(min(times)/1000)*1000:.3f}ms")

times = timeit.Timer(partial(torch.cumsum, x_large, 0)).repeat(3, 1000)
print(f"Torch time: {(min(times)/1000)*1000:.3f}ms")
      