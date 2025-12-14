import argparse, time
import torch
import torch.nn.functional as F

def get_grouped_mm():
    # Prefer public API if present; else private.
    if hasattr(F, "grouped_mm"):
        return F.grouped_mm
    if hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm
    return None

def pack_by_expert(x, expert_idx, num_experts, pad_multiple=16):
    """
    Pack tokens so expert 0 tokens are contiguous, then expert 1, etc.
    
    Args:
      x: [T, K] activation matrix (2D). Must be on same device as expert_idx.
      expert_idx: [T] integer expert assignment per token. Values must be in [0, num_experts).
      num_experts: E, total number of experts (defines bincount/minlength and offs length).
      pad_multiple: Pad packed tokens so Tpad is a multiple of this. Also forces at least
        one full multiple of padding when T is already aligned, to satisfy grouped_mm's
        requirement that offs[-1] (=T) is STRICTLY < x_pad.shape[0] (=Tpad).
    
    Returns:
      x_pad:   [Tpad, K]
      inv_perm:[T] (to unpermute outputs back to original token order)
      offs:    [E] int32 cumsum(counts), offs[i] = end of group i in packed order
      T:       original token count
    """
    device = x.device
    T, K = x.shape

    perm = torch.argsort(expert_idx)              # [T]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(T, device=device, dtype=perm.dtype)

    x_packed = x[perm]                            # [T, K]
    counts = torch.bincount(expert_idx, minlength=num_experts).to(torch.int32)  # [E]
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)                       # [E]

    # grouped_mm expects offs[-1] to be STRICTLY < sliced dim length,
    # so we add at least 1 padding row (and often align to 16).
    pad = (pad_multiple - (T % pad_multiple)) % pad_multiple
    if pad == 0:
        pad = pad_multiple
    x_pad = torch.cat([x_packed, x.new_zeros((pad, K))], dim=0)  # [Tpad, K]
    return x_pad, inv_perm, offs, T

def time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3 / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)   # T
    ap.add_argument("--experts", type=int, default=64)    # E
    ap.add_argument("--in", dest="K", type=int, default=1024)
    ap.add_argument("--out", dest="N", type=int, default=1024)
    ap.add_argument("--skew", type=float, default=1.2)    # routing imbalance
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))
    grouped_mm = get_grouped_mm()

    T, E, K, N = args.tokens, args.experts, args.K, args.N

    # MoE-ish routing: imbalanced distribution over experts
    probs = torch.arange(1, E + 1, device=device, dtype=torch.float32).pow(-args.skew)
    probs = probs / probs.sum()
    print(probs)
    expert_idx = torch.multinomial(probs, num_samples=T, replacement=True)  # [T]

    # Inputs + per-expert weights
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16 if device.type=="cuda" else torch.float32)
    W = torch.randn(E, N, K, device=device, dtype=x.dtype)  # per-expert linear: y = x @ W[e].T

    # Pack by expert once (like real MoE dispatch does)
    x_pad, inv_perm, offs, T_real = pack_by_expert(x, expert_idx, E, pad_multiple=16)

    # ----- Baseline: loop over experts -----
    # This uses host-driven slicing + many small matmuls (lots of launches).
    counts_t = torch.bincount(expert_idx, minlength=E)  # [E] on device
    counts = counts_t.cpu().tolist()  # (sync if CUDA)

    def run_loop():
        x_packed = x_pad[:T_real]
        outs = []
        s = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            xe = x_packed[s:s+c]                  # [c, K]
            ye = xe @ W[e].transpose(-2, -1)      # [c, N]
            outs.append(ye)
            s += c
        y_packed = torch.cat(outs, dim=0) if outs else x_packed[:0, :N]
        y = torch.empty_like(y_packed)
        y[inv_perm] = y_packed
        return y

    # ----- Optimized: grouped_mm -----
    def run_grouped():
        if grouped_mm is None or device.type != "cuda":
            raise RuntimeError("grouped_mm not available / not CUDA in this env")

        # grouped_mm: A is 2D [Tpad, K], B is 3D [E, K, N] (so A @ B per group)
        y_packed_pad = grouped_mm(x_pad.to(torch.bfloat16), W.to(torch.bfloat16).transpose(-2, -1), offs=offs)
        y_packed = y_packed_pad[:T_real]          # IMPORTANT: drop padded tail
        y = torch.empty_like(y_packed)
        y[inv_perm] = y_packed
        return y

    # Correctness
    with torch.no_grad():
        y_loop = run_loop().float()
        if grouped_mm is not None and device.type == "cuda":
            y_grp = run_grouped().float()
            print("max |loop - grouped| =", (y_loop - y_grp).abs().max().item())
        else:
            print("Skipping grouped correctness (grouped_mm missing or not CUDA).")

    # Timing
    t_loop = time_ms(run_loop, iters=args.iters)
    print(f"loop:      {t_loop:.4f} ms/iter")

    if grouped_mm is not None and device.type == "cuda":
        t_grp = time_ms(run_grouped, iters=args.iters)
        print(f"grouped:   {t_grp:.4f} ms/iter")
        print(f"speedup:   {t_loop/t_grp:.2f}x")
    else:
        print("grouped:   (skipped)")

    # Useful debug: token histogram + padding
    with torch.no_grad():
        c = counts_t.cpu()
        Tpad = int(x_pad.shape[0])
        print(f"tokens per expert: min={int(c.min())}, max={int(c.max())}, nonempty={(c>0).sum().item()}/{E}")
        print(f"T={T_real}, Tpad={Tpad}, pad={Tpad - T_real}")

if __name__ == "__main__":
    main()
