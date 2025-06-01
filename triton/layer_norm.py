import triton
import triton.language as tl
import torch
import torch.nn.functional as F

@triton.jit
def layer_norm_kernel(
    X, Y,              # input and output pointers
    N, stride,         # sequence length (column size) and stride
    G, B,              # gamma and beta pointers
    M, RS,             # mean and rstd (1/std) pointers
    eps,               # epsilon
    BLOCK_SIZE: tl.constexpr):
    
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # Mean calculation
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(0, N, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        mean += x

    mean = tl.sum(mean, axis=0) / N
    tl.store(M + row, mean)

    # Variance calculation
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=mean).to(tl.float32) # other = mean so that it can become 0 when subtracting mean
        x = x - mean
        var += x * x

    var = tl.sum(var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(RS + row, rstd)

    # Normalization step
    for i in range(0, N, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=mean).to(tl.float32)
        y = (x - mean) * rstd
        g = tl.load(G + cols, mask=mask, other=1).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0).to(tl.float32)
        y = y * g + b
        tl.store(Y + cols, y, mask=mask)
    
    
def layer_norm(X, G, B, M, RS, eps=1e-5, BLOCK_SIZE=1024):
    N = X.shape[0]
    Y = torch.empty_like(X)
    layer_norm_kernel[(N,)](X, Y, N, X.stride(0), G, B, M, RS, eps, BLOCK_SIZE)
    return Y

X = torch.randn(1024, 1024).cuda()
G = torch.randn(1024).cuda()
B = torch.randn(1024).cuda()
M = torch.zeros(1024).cuda()
RS = torch.zeros(1024).cuda()

Y = layer_norm(X, G, B, M, RS, 1e-5)
Y_actual = F.layer_norm(X, (1024,), weight=G, bias=B, eps=1e-5)

print(Y)
print(Y_actual)

assert torch.allclose(Y, Y_actual, atol=1e-5, rtol=1e-5)