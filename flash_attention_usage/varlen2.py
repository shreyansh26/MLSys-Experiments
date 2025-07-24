import torch, math
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_varlen_func
from naive_attention import naive_varlen

torch.manual_seed(0)
dtype   = torch.float16         # stay in fp32 for tight tolerances
B, S, H, D = 3, 512, 8, 64

# --- create a ragged batch ---------------------------------------------------
x        = torch.randn(B, S, H, D, device='cuda', dtype=dtype, requires_grad=True)
lengths  = torch.tensor([37, 512, 128], device='cuda', dtype=torch.int32)
valid_mask = torch.arange(S, device='cuda')[None] < lengths[:, None]   # False for PAD

# --- 1) strip padding --------------------------------------------------------
x_flat, idx, cu_lens, max_s = unpad_input(x, valid_mask)
q = k = v = x_flat

# --- 2) FlashAttention‑varlen -----------------------------------------------
out_flat = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_lens,
    cu_seqlens_k=cu_lens,
    max_seqlen_q=max_s,
    max_seqlen_k=max_s,
    causal=True,
)
out_flash = pad_input(out_flat, idx, B, S)                      # back to (B,S,H,D)

# --- 3) naïve reference ------------------------------------------------------
y_ref = naive_varlen(x, lengths)

# --- 4) compare --------------------------------------------------------------
torch.testing.assert_close(out_flash, y_ref, rtol=1e-2, atol=1e-3)
print('forward OK')

# gradient check
loss = out_flash.square().mean()
loss.backward()
grad_ref = torch.autograd.grad(y_ref.square().mean(), x, retain_graph=True)[0]
torch.testing.assert_close(x.grad, grad_ref, rtol=1e-2, atol=1e-3)
print('backward OK')
