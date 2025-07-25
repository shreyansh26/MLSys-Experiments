import torch
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from naive_attention import naive_causal_attention

# shape: (B, seq_len, nheads, headdim)
B, q_seq_len, nheads, headdim = 2, 2048, 16, 64
kv_seq_len = 2048 # 2048

assert q_seq_len <= kv_seq_len

q = torch.randn(B, q_seq_len, nheads, headdim, dtype=torch.float16, device="cuda", requires_grad=True)
k = torch.randn(B, kv_seq_len, nheads, headdim, dtype=torch.float16, device="cuda", requires_grad=True)
v = torch.randn(B, kv_seq_len, nheads, headdim, dtype=torch.float16, device="cuda", requires_grad=True)

# causal=True for auto-regressive models
out_flash = flash_attn_func(q, k, v, causal=True)
print("out_flash.shape:\t\t", out_flash.shape)  # (2, 2048, 16, 64)

out_naive = naive_causal_attention(q, k, v)
print("out_naive.shape:\t\t", out_naive.shape)  # (2, 2048, 16, 64)

torch.testing.assert_close(out_flash, out_naive, atol=1e-3, rtol=1e-2)
print('forward OK')

# gradient check
loss = out_flash.square().mean()
loss.backward()
grad_naive = torch.autograd.grad(out_naive.square().mean(), q, retain_graph=True)[0]
torch.testing.assert_close(q.grad, grad_naive, atol=1e-3, rtol=1e-2)
print('backward OK')

if q_seq_len == kv_seq_len:
    q.grad = k.grad = v.grad = None
    qkv = torch.stack([q, k, v], dim=2)
    print("stacked qkv shape:\t\t", qkv.shape)

    out_flash_qkvpacked = flash_attn_qkvpacked_func(qkv, causal=True)
    print("out_flash_qkvpacked.shape:\t", out_flash_qkvpacked.shape)  # (2, 2048, 16, 64)

    torch.testing.assert_close(out_flash_qkvpacked, out_naive, atol=1e-3, rtol=1e-2)
    print('forward OK')

    # gradient check
    loss = out_flash_qkvpacked.square().mean()
    loss.backward()
    grad_naive = torch.autograd.grad(out_naive.square().mean(), q, retain_graph=True)[0]
    torch.testing.assert_close(q.grad, grad_naive, atol=1e-3, rtol=1e-2)
    print('backward OK')