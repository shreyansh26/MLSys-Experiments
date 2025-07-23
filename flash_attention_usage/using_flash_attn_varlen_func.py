import torch, flash_attn
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis
from flash_attn import flash_attn_varlen_func

# batch with different lengths
x      = torch.randn(3, 512, 8, 128, device='cuda', dtype=torch.bfloat16)
attn_m = torch.tensor([[1]*37  + [0]*475,     # 37‑token prompt
                       [1]*512,
                       [1]*128 + [0]*384], device='cuda', dtype=torch.bool)

# 1) strip padding
x_flat, idx, cu_lens, max_s = unpad_input(x, ~attn_m)   # mask is “padding” mask
q = k = v = x_flat                                       # toy example

# 2) call varlen
y_flat = flash_attn_varlen_func(q, k, v,
                                cu_seqlens_q=cu_lens,
                                cu_seqlens_k=cu_lens,
                                max_seqlen_q=max_s,
                                max_seqlen_k=max_s,
                                causal=True)

# 3) re‑pad to (B,S,…) shape
y = pad_input(y_flat, idx, *x.shape[:2])
