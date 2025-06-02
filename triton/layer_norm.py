import triton
import triton.language as tl
import torch

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
    DX, DY,         # input and output gradient pointers
    DG, DB,         # partial sums of dgamma and dbias
    X,              # input data pointer
    G, B,           # gamma and beta pointers
    M, RS,          # mean and rstd (1/std) pointers
    N, stride,      # sequence length (column size) and stride
    LOCK,           # lock pointer
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr):
    
    row = tl.program_id(0)
    
    X += row * stride
    DX += row * stride
    DY += row * stride

    # Load mean and rstd for this row
    m = tl.load(M + row)
    rstd = tl.load(RS + row)

    # First pass: compute c1 and c2 across all elements
    c1_sum = tl.zeros([1], dtype=tl.float32)
    c2_sum = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_SIZE_N):
        cols = block_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        g = tl.load(G + cols, mask=mask, other=1).to(tl.float32)
        
        xhat = (x - m) * rstd
        gdy = g * dy
        c1_sum += tl.sum(xhat * gdy, axis=0)
        c2_sum += tl.sum(gdy, axis=0)
    
    c1 = c1_sum / N
    c2 = c2_sum / N

    # Get lock ids
    lock_id = row % GROUP_SIZE_M
    LOCK += lock_id
    COUNT = LOCK + GROUP_SIZE_M

    # Second pass: compute dx and accumulate dg, db
    for block_start in range(0, N, BLOCK_SIZE_N):
        cols = block_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        DG_block = DG + lock_id * N + cols
        DB_block = DB + lock_id * N + cols

        # Load data for this block
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        g = tl.load(G + cols, mask=mask, other=1).to(tl.float32)

        # Compute dx for this block
        xhat = (x - m) * rstd
        gdy = g * dy
        dx = (gdy - (xhat * c1 + c2)) * rstd

        # Write dx for this block
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dg and db
        partial_dg = (dy * xhat).to(g.dtype)
        partial_db = dy.to(g.dtype)

        # Acquire lock
        while tl.atomic_cas(LOCK, 0, 1) == 1:
            pass

        count = tl.load(COUNT)

        # No accumulation in first store
        if count == 0:
            tl.atomic_xchg(COUNT, 1)
        else:
            partial_dg += tl.load(DG_block, mask=mask)
            partial_db += tl.load(DB_block, mask=mask)

        tl.store(DG_block, partial_dg, mask=mask)
        tl.store(DB_block, partial_db, mask=mask)

        # barrier to ensure all threads finished before releasing lock
        tl.debug_barrier()

        # Release lock
        tl.atomic_xchg(LOCK, 0)

@triton.jit
def layer_norm_bwd_kernel_dgdb(
    DG_partial, DB_partial,     # partial sums of dgamma and dbias
    DG, DB,                     # final output pointers to dgamma and dbias
    M, N,                       # M here is GROUP_SIZE_M
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

        # Allocate memory
        mean = torch.empty((M,), dtype=torch.float32, device=x.device) # Mean and rstd are stored in float32
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device) # Mean and rstd are stored in float32

        # Less than 64KB per feature: enqueue kernel
        MAX_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_SIZE, triton.next_power_of_2(N))

        # New implementation loops over the feature dimension to support feature dim >= 64KB as well
        # if N > BLOCK_SIZE:
        #     raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

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
    def backward(ctx, dy):
        x, gamma, bias, mean, rstd = ctx.saved_tensors

        # Heuristics
        N = gamma.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        # Allocate memory
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=gamma.device)
        _dgamma = torch.zeros((GROUP_SIZE_M, N), dtype=gamma.dtype, device=gamma.device)
        _dbias = torch.zeros((GROUP_SIZE_M, N), dtype=bias.dtype, device=bias.device)
        dgamma = torch.empty((N,), dtype=gamma.dtype, device=gamma.device)
        dbeta = torch.empty((N,), dtype=bias.dtype, device=bias.device)
        dx = torch.empty_like(dy)

        # reshape input data into 2D tensor
        x_inp = x.reshape(-1, x.shape[-1])
        M, N = x_inp.shape

        # call kernel
        layer_norm_bwd_kernel_dx[(M,)](dx, dy, _dgamma, _dbias, x_inp, gamma, bias, mean, rstd, N, x_inp.stride(0), locks, GROUP_SIZE_M=GROUP_SIZE_M, BLOCK_SIZE_N=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)

        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        layer_norm_bwd_kernel_dgdb[grid](_dgamma, _dbias, dgamma, dbeta, min(GROUP_SIZE_M, M), N, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128)

        return dx, None, dgamma, dbeta, None        

def test_layer_norm(M, N, dtype, eps=1e-5, device="cuda", mode="both"):
    layer_norm = LayerNorm.apply
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    gamma = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)

    x.requires_grad_(True)

    y = layer_norm(x, w_shape, gamma, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, gamma, bias, eps)
    print(y)
    print(y.shape)
    print(y_ref)
    print(y_ref.shape)
    # Use higher tolerance for float16 - 1e-2, 0
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)

    if mode == "forward":
        return
    
    y.backward(dy, retain_graph=True)
    dx, dg, db = [_.grad.clone() for _ in [x, gamma, bias]]

    x.grad, gamma.grad, bias.grad = None, None, None

    y_ref.backward(dy, retain_graph=True)
    dx_ref, dg_ref, db_ref = [_.grad.clone() for _ in [x, gamma, bias]]

    print(dx)
    print(dx.shape)
    print(dx_ref)
    print(dx_ref.shape)

    # Use higher tolerance for float16 - 1e-2, 0
    assert torch.allclose(dx, dx_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dg, dg_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(db, db_ref, atol=1e-5, rtol=1e-5)

def create_perf_report(mode, bench_type):
    plot_name = f"layer-norm-{mode}-{bench_type}"
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[512 * i for i in range(2, 128)], # till 65k feature length
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            plot_name=plot_name,
            args={'M': 4096, 'dtype': torch.float16, 'mode': mode},
        ))

def bench_layer_norm_generic(M, N, dtype, provider, mode, bench_type, eps=1e-5, device="cuda"):
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    gamma = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    layer_norm = LayerNorm.apply

    def y_fwd():
        if provider == "triton":
            return layer_norm(x, w_shape, gamma, bias, eps)
        elif provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, gamma, bias, eps)

    if bench_type == "flops":
        if mode == "forward":
            gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
            ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
            return gbps(ms), gbps(max_ms), gbps(min_ms)
        elif mode == "backward":
            y = y_fwd()
            gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles, grad_to_none=[x], rep=500)
            return gbps(ms), gbps(max_ms), gbps(min_ms)
    elif bench_type == "latency":
        if mode == "forward":
            ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
            return ms, min_ms, max_ms
        elif mode == "backward":
            y = y_fwd()
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles, grad_to_none=[x], rep=500)
            return ms, min_ms, max_ms


def make_benchmark(bench_type, mode):
    @create_perf_report(mode, bench_type)
    def bench(M, N, dtype, provider, eps=1e-5, device="cuda", **kwargs):
        return bench_layer_norm_generic(M, N, dtype, provider, mode, bench_type, eps, device)
    return bench

if __name__ == "__main__":
    test_layer_norm(1151, 131072, torch.float32, mode="both")
    test_layer_norm(1151, 131071, torch.float32, mode="both")

    bench_layer_norm_flops_forward = make_benchmark("flops", "forward")
    bench_layer_norm_flops_backward = make_benchmark("flops", "backward")
    bench_layer_norm_latency_forward = make_benchmark("latency", "forward")
    bench_layer_norm_latency_backward = make_benchmark("latency", "backward")

    bench_layer_norm_flops_forward.run(save_path='plots/layer_norm', print_data=True)
    bench_layer_norm_flops_backward.run(save_path='plots/layer_norm', print_data=True)
    bench_layer_norm_latency_forward.run(save_path='plots/layer_norm', print_data=True)
    bench_layer_norm_latency_backward.run(save_path='plots/layer_norm', print_data=True)