# For inference - does not support backward pass
import torch
from flash_attn import flash_attn_with_kvcache
from naive_attention import naive_causal_attention

VERIFY = True

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

    if VERIFY:
        batch_idx = torch.arange(B, device=k_cache.device) 
        k_newly_added = k_cache[batch_idx, cache_lens-1].unsqueeze(1)
        v_newly_added = v_cache[batch_idx, cache_lens-1].unsqueeze(1)

        torch.testing.assert_close(k_newly_added, k_new, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(v_newly_added, v_new, atol=1e-3, rtol=1e-2)

        seq_len = cache_lens[0].item()
        k_cache_naive = k_cache[batch_idx, :seq_len]
        v_cache_naive = v_cache[batch_idx, :seq_len]

        out_naive = naive_causal_attention(q, k_cache_naive, v_cache_naive)
        torch.testing.assert_close(out, out_naive, atol=1e-3, rtol=1e-2)
                          
    print(f"step {step} out.shape: {out.shape}")