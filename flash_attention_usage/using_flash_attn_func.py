import torch
from flash_attn import flash_attn_func
import math

# shape: (B, seq_len, nheads, headdim)
B, seq_len, nheads, headdim = 2, 2048, 16, 64
q = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")
k = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")
v = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")

# causal=True for auto-regressive models
out = flash_attn_func(q, k, v, causal=True)
print(out.shape)  # (2, 2048, 16, 64)

# Naive causal attention
def naive_causal_attention(q, k, v):
    B, seq_len, nheads, headdim = q.shape
    attention_weights = torch.einsum("nqhd,nkhd->nhqk", [q, k])    
    attention_weights = attention_weights / math.sqrt(headdim)
    attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
    attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
    attention_weights = torch.softmax(attention_weights, dim=-1)
    out = torch.einsum("nhqk,nkhd->nqhd", [attention_weights, v])
    return out

out_naive = naive_causal_attention(q, k, v)
print(out_naive.shape)  # (2, 2048, 16, 64)

torch.testing.assert_close(out, out_naive, atol=1e-3, rtol=1e-2)