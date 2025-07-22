import torch
from flash_attn import flash_attn_with_kvcache

B, max_seq_len, nheads, headdim = 2, 4096, 16, 64      # max sequence length for cache
dtype = torch.float16

k_cache = torch.empty(B, max_seq_len, nheads, headdim, device='cuda', dtype=dtype)
v_cache = torch.empty_like(k_cache)
cache_lens = torch.zeros(B, dtype=torch.int32, device='cuda')

for step in range(50):                # generate 50 tokens
    q = torch.randn(B, 1, nheads, headdim, device='cuda', dtype=dtype)

    # pretend this step produces new K/V (e.g. via an MHA proj)
    k_new = torch.randn_like(q)
    v_new = torch.randn_like(q)

    out = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        k=k_new, v=v_new,
        cache_seqlens=cache_lens,
        causal=True,
    )

    cache_lens += 1                      # advance sequence lengths
    print(f"step {step} out.shape: {out.shape}")