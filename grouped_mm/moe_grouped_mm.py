import argparse, time
import torch
import torch.nn.functional as F

def get_grouped_mm():
    """
    Prefer torch.nn.functional.grouped_mm if present, else fall back to torch._grouped_mm.
    """
    if hasattr(F, "grouped_mm"):
        return F.grouped_mm
    if hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm
    raise RuntimeError("Neither torch.nn.functional.grouped_mm nor torch._grouped_mm exists in this PyTorch build.")

def sample_expert_assignments(num_tokens: int, num_experts: int, skew: float, device):
    """
    Make MoE-like imbalanced routing.
    - skew > 1.0 => heavier head experts, longer tail.
    """
    # Create a geometric-ish distribution over experts
    probs = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    probs = probs.pow(-skew)
    probs = probs / probs.sum()
    # Sample top-1 expert per token
    return torch.multinomial(probs, num_samples=num_tokens, replacement=True)  # [T]

def pack_by_expert(x: torch.Tensor, expert_idx: torch.Tensor, num_experts: int, align: int = 16):
    """
    Pack tokens so that tokens for expert 0 come first, then expert 1, ...
    
    Args:
      x: [T, K] activation matrix (2D). Typically BF16/FP16 on CUDA, FP32 on CPU.
      expert_idx: [T] integer expert assignment per token (top-1 routing). Must be on the
        same device as x, and values must lie in [0, num_experts).
      num_experts: E, total number of experts. Controls bincount(minlength=E) and the
        length of counts/offs.
      align: Padding multiple used to create x_packed_pad with shape [T + pad, K]. If T is
        already aligned, this function still pads by align (>= 1 row) to satisfy grouped_mm's
        requirement that offs[-1] (=T) is STRICTLY < x_packed_pad.shape[0] (=T+pad).
    
    Returns:
      x_packed_pad: [T + pad, K]
      perm:         [T] original token indices in packed order
      inv_perm:     [T] to unpermute packed outputs back to original order
      counts:       [E] tokens per expert (int32, device)
      offs:         [E] cumsum(counts) (int32, device), offs[-1] = T
    """
    device = x.device
    T, K = x.shape

    # Stable sort by expert id => contiguous segments per expert
    perm = torch.argsort(expert_idx)  # [T]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(T, device=device, dtype=perm.dtype)

    x_packed = x[perm]  # [T, K]

    counts = torch.bincount(expert_idx, minlength=num_experts).to(torch.int32)  # [E]
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)  # [E]

    # grouped_mm requires offs[-1] < mat_a.shape[0] (strictly less) for 2D mat_a,
    # so we add at least 1 padding row.
    pad = (align - (T % align)) % align
    if pad == 0:
        pad = align  # guarantee >= 1
    x_packed_pad = torch.cat([x_packed, x.new_zeros((pad, K))], dim=0)

    return x_packed_pad, perm, inv_perm, counts, offs

def moe_mlp_loop(w1, w2, w3, x_packed, counts_cpu):
    """
    Naive loop: one matmul per expert per projection.
    Shapes:
      x_packed: [T, K]
      w*: [E, N, K] where N=ffn for w1/w3, N=K for w2 (to project back)
    """
    outs = []
    start = 0
    for e, c in enumerate(counts_cpu):
        if c == 0:
            continue
        xe = x_packed[start:start + c]  # [c, K]
        # xe @ w.T => [c, N]
        h1 = F.silu(xe @ w1[e].transpose(-2, -1))
        h3 = xe @ w3[e].transpose(-2, -1)
        h = h1 * h3
        ye = h @ w2[e].transpose(-2, -1)  # [c, K]
        outs.append(ye)
        start += c
    if len(outs) == 0:
        return x_packed[:0, :]  # empty
    return torch.cat(outs, dim=0)  # [T, K]

def moe_mlp_grouped(grouped_mm, w1, w2, w3, x_packed_pad, offs):
    """
    Grouped version: 3 grouped_mm calls matching your screenshot logic.
    We pass weights transposed to (E, K, N) so that grouped_mm computes A @ B.
    """
    # A: [Tpad, K] (2D), B: [E, K, N] (3D), offs: [E]
    xbf = x_packed_pad.to(torch.bfloat16)

    h1 = grouped_mm(xbf, w1.to(torch.bfloat16).transpose(-2, -1), offs=offs)
    h1 = F.silu(h1)

    h3 = grouped_mm(xbf, w3.to(torch.bfloat16).transpose(-2, -1), offs=offs)
    h = h1 * h3

    out = grouped_mm(h, w2.to(torch.bfloat16).transpose(-2, -1), offs=offs)
    return out  # note: out includes padded rows at the end

def time_it(fn, iters=50, warmup=10, is_cuda=False):
    # warmup
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3 / iters  # ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=1024)      # K
    ap.add_argument("--ffn", type=int, default=4096)         # N for w1/w3
    ap.add_argument("--skew", type=float, default=1.2)       # routing imbalance
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    is_cuda = (device.type == "cuda")
    if is_cuda:
        print("CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))

    grouped_mm = None
    try:
        grouped_mm = get_grouped_mm()
    except Exception as e:
        print("NOTE: grouped_mm not available in this build:", e)
        print("      You'll still see correctness + loop baseline; grouped path will be skipped.")
        grouped_mm = None

    T, E, K, N = args.tokens, args.experts, args.hidden, args.ffn

    # Simulate token routing
    expert_idx = sample_expert_assignments(T, E, args.skew, device=device)  # [T]

    # Inputs / weights
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16 if is_cuda else torch.float32)
    # w*: [E, N, K] (so x @ w.T => [*, N])
    w1 = torch.randn(E, N, K, device=device, dtype=torch.bfloat16 if is_cuda else torch.float32)
    w3 = torch.randn(E, N, K, device=device, dtype=torch.bfloat16 if is_cuda else torch.float32)
    # project back to K: [E, K, N] logically, but we store as [E, K, N]? keep consistent: [E, K, N] means x_ffn @ w2.T
    # We want (c,N) @ (N,K) => store w2 as [E, K, N] so w2.T is [E, N, K]
    # To keep the same convention "x @ w.T", store w2 as [E, K, N]
    w2 = torch.randn(E, K, N, device=device, dtype=torch.bfloat16 if is_cuda else torch.float32)

    # Pack inputs by expert
    x_packed_pad, perm, inv_perm, counts, offs = pack_by_expert(x, expert_idx, E, align=16)
    Tpad = x_packed_pad.shape[0]

    # Loop version uses host counts (this is exactly the sync pitfall in many MoE impls)
    counts_cpu = counts.cpu().tolist()  # device->host sync if counts is on CUDA

    def run_loop():
        # operate only on real tokens (exclude padded tail)
        x_packed = x_packed_pad[:T]
        out_packed = moe_mlp_loop(w1, w2, w3, x_packed, counts_cpu)  # [T, K]
        # unpermute back to original token order
        out = torch.empty_like(out_packed)
        out[inv_perm] = out_packed
        return out

    def run_grouped():
        out_packed_pad = moe_mlp_grouped(grouped_mm, w1, w2, w3, x_packed_pad, offs)  # [Tpad, K]
        out_packed = out_packed_pad[:T]  # drop padding rows
        out = torch.empty_like(out_packed)
        out[inv_perm] = out_packed
        return out

    # Correctness check
    out_loop = run_loop()
    if grouped_mm is not None and is_cuda:
        out_grouped = run_grouped()
        max_err = (out_loop.float() - out_grouped.float()).abs().max().item()
        print(f"max |loop - grouped| = {max_err:.6e}")
    else:
        print("Skipping grouped correctness (need grouped_mm + CUDA BF16).")

    # Benchmark
    t_loop = time_it(run_loop, iters=args.iters, warmup=args.warmup, is_cuda=is_cuda)
    print(f"looped MoE MLP:   {t_loop:.4f} ms / iter")

    if grouped_mm is not None and is_cuda:
        t_grp = time_it(run_grouped, iters=args.iters, warmup=args.warmup, is_cuda=is_cuda)
        print(f"grouped_mm MLP:  {t_grp:.4f} ms / iter")
        print(f"speedup:         {t_loop / t_grp:.2f}x")
    else:
        print("grouped_mm benchmark skipped (not available / not CUDA).")

    # Useful debug: token histogram
    with torch.no_grad():
        c = counts.cpu()
        print(f"tokens per expert: min={int(c.min())}, max={int(c.max())}, nonempty={(c>0).sum().item()}/{E}")
        print(f"T={T}, Tpad={Tpad}, pad={Tpad - T}")

if __name__ == "__main__":
    main()
