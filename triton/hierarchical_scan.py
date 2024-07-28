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


K = 16
BLOCKS = 8
SEQLEN = K * BLOCKS

def cumsum(x):
    y = []
    h = 0
    for k in range(len(x)):
        h = h + x[k]
        y.append(h)
    return h, y

h = torch.zeros(BLOCKS).float().cuda()
x = torch.arange(SEQLEN).float().cuda()
y = torch.zeros(SEQLEN).float().cuda()

cumsum_kernel[(BLOCKS,)](x, h, y, K)

print(x)
print(h)
print(y)

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

cumsum_block(x, y, K)
y_ = torch.cumsum(x, dim=0)
assert torch.allclose(y, y_)


y_large = torch.zeros(2**25).float().cuda()
x_large = torch.arange(2**25).float().cuda()

times = timeit.Timer(partial(cumsum_block, x_large, y_large, K)).repeat(3, 1000)
print(f"{(min(times)/1000)*1000:.3f}ms")

y_large_ = torch.cumsum(x_large, dim=0)
assert torch.allclose(y, y_)