import triton
import triton.language as tl
import torch
import torch.nn.functional as F

@triton.jit
def layer_norm_fwd_kernel(
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

@triton.jit
def layer_norm_bwd_kernel_dx(
    DX, DY,
    DG, DB,
    X,
    G, B,
    M, RS,
    N, stride,
    LOCK,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr):
    
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    
    X += row * stride
    DX += row * stride
    DY += row * stride

    # Get lock ids
    lock_id = row % GROUP_SIZE_M
    LOCK += lock_id
    COUNT = LOCK + GROUP_SIZE_M

    DG = DG + lock_id * N + cols
    DB = DB + lock_id * N + cols

    # Load data
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    g = tl.load(G + cols, mask=mask, other=1).to(tl.float32)
    
    m = tl.load(M + row)
    rstd = tl.load(RS + row)

    # Compute dx
    xhat = (x - m) * rstd
    gdy = g * dy
    c1 = tl.sum(xhat * gdy, axis=0) / N
    c2 = tl.sum(gdy, axis=0) / N
    dx = (gdy - (xhat * c1 + c2)) * rstd

    # Write dx
    tl.store(DX + cols, dx, mask=mask)

    # Accumulate partial sums for dg and db
    partial_dg = (dy * xhat).to(g.dtype)
    partial_db = dy.to(g.dtype)

    while tl.atomic_cas(LOCK, 0, 1) == 1:
        pass

    count = tl.load(COUNT)

    # No accumulation in first store
    if count == 0:
        tl.atomic_xchg(COUNT, 1)
    else:
        partial_dg += tl.load(DG, mask=mask)
        partial_db += tl.load(DB, mask=mask)

    tl.store(DG, partial_dg, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # barrier to ensure all threads finished before releasing lock
    tl.debug_barrier()

    # Release lock
    tl.atomic_xchg(LOCK, 0)

@triton.jit
def layer_norm_bwd_kernel_dgdb(
    DG_partial, DB_partial,
    DG, DB,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr):
    
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dg_partial = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    db_partial = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offset = rows[:, None] * N + cols[None, :]

        dg_partial += tl.load(DG_partial + offset, mask=mask, other=0)
        db_partial += tl.load(DB_partial + offset, mask=mask, other=0)

    sum_dg = tl.sum(dg_partial, axis=0)
    sum_db = tl.sum(db_partial, axis=0)

    tl.store(DG + cols, sum_dg, mask=cols < N)
    tl.store(DB + cols, sum_db, mask=cols < N)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, gamma, bias, eps=1e-5): # Keep args same as torch.nn.functional.layer_norm
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_inp = x.reshape(-1, x.shape[-1])
        M, N = x_inp.shape

        mean = torch.empty(M, dtype=torch.float32, device=x.device) # Mean and rstd are stored in float32
        rstd = torch.empty(M, dtype=torch.float32, device=x.device) # Mean and rstd are stored in float32

        # Less than 64KB per feature: enqueue kernel
        MAX_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_SIZE, triton.next_power_of_2(N))

        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        # Heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # call kernel
        layer_norm_fwd_kernel[(M,)](x_inp, y, N, x_inp.stride(0), gamma, bias, mean, rstd, eps, BLOCK_SIZE, num_warps=num_warps)
        ctx.save_for_backward(x, gamma, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented")        

def test_layer_norm(M, N, dtype, eps=1e-5, device="cuda"):
    layer_norm = LayerNorm.apply
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    gamma = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    y = layer_norm(x, w_shape, gamma, bias, eps)

    y_ref = torch.nn.functional.layer_norm(x, w_shape, gamma, bias, eps)
    print(y)
    print(y_ref)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-forward-flops',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))
def bench_layer_norm_flops(M, N, dtype, provider, mode='forward', eps=1e-5, device="cuda"):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    gamma = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    layer_norm = LayerNorm.apply

    def y_fwd():
        if provider == "triton":
            return layer_norm(x, w_shape, gamma, bias, eps)

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, gamma, bias, eps)

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-forward-latency',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))
def bench_layer_norm_latency(M, N, dtype, provider, mode='forward', eps=1e-5, device="cuda"):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    gamma = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    layer_norm = LayerNorm.apply

    def y_fwd():
        if provider == "triton":
            return layer_norm(x, w_shape, gamma, bias, eps)

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, gamma, bias, eps)

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    
    return ms, min_ms, max_ms

if __name__ == "__main__":
    test_layer_norm(1151, 8192, torch.float32)
    bench_layer_norm_flops.run(save_path='plots/layer_norm', print_data=True)
    bench_layer_norm_latency.run(save_path='plots/layer_norm', print_data=True)