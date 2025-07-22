import torch
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from naive_attention import naive_causal_attention

# shape: (B, seq_len, nheads, headdim)
B, seq_len, nheads, headdim = 2, 2048, 16, 64
q = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")
k = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")
v = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")

qkv = torch.stack([q, k, v], dim=2)
print("stacked qkv shape:\t\t", qkv.shape)

# causal=True for auto-regressive models
out_flash = flash_attn_func(q, k, v, causal=True)
print("out_flash.shape:\t\t", out_flash.shape)  # (2, 2048, 16, 64)

out_flash_qkvpacked = flash_attn_qkvpacked_func(qkv, causal=True)
print("out_flash_qkvpacked.shape:\t", out_flash_qkvpacked.shape)  # (2, 2048, 16, 64)

out_naive = naive_causal_attention(q, k, v)
print("out_naive.shape:\t\t", out_naive.shape)  # (2, 2048, 16, 64)

torch.testing.assert_close(out_flash, out_naive, atol=1e-3, rtol=1e-2)
torch.testing.assert_close(out_flash_qkvpacked, out_naive, atol=1e-3, rtol=1e-2)