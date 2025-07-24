import torch, math
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_varlen_func
from naive_attention import naive_varlen, naive_varlen_vectorized

torch.manual_seed(0)
dtype   = torch.float16         # stay in fp32 for tight tolerances
B, seq_len, nheads, headdim = 2, 2048, 16, 64

# create a ragged batch
lengths  = torch.tensor([37, 512], device='cuda', dtype=torch.int32)
q        = torch.randn(B, seq_len, nheads, headdim, device='cuda', dtype=dtype, requires_grad=True)
k        = torch.randn(B, seq_len, nheads, headdim, device='cuda', dtype=dtype, requires_grad=True)
v        = torch.randn(B, seq_len, nheads, headdim, device='cuda', dtype=dtype, requires_grad=True)
valid_mask = torch.arange(seq_len, device='cuda')[None] < lengths[:, None]   # False for PAD

# strip padding
q_flat, idx, cu_lens, max_s = unpad_input(q, valid_mask)
k_flat, _, _, _ = unpad_input(k, valid_mask)
v_flat, _, _, _ = unpad_input(v, valid_mask)

out_flat = flash_attn_varlen_func(
    q_flat, k_flat, v_flat,
    cu_seqlens_q=cu_lens,
    cu_seqlens_k=cu_lens,
    max_seqlen_q=max_s,
    max_seqlen_k=max_s,
    causal=True,
)
out_flash = pad_input(out_flat, idx, B, seq_len)                      # back to (B,S,H,D)
print("out_flash.shape:\t\t", out_flash.shape)      # (2, 2048, 16, 64)

# out_naive = naive_varlen(q, k, v, lengths) # Works correctly
out_naive = naive_varlen_vectorized(q, k, v, valid_mask)
print("out_naive.shape:\t\t", out_naive.shape)      # (2, 2048, 16, 64)

torch.testing.assert_close(out_flash, out_naive, atol=1e-3, rtol=1e-2)
print('forward OK')

# gradient check
loss = out_flash.square().mean()
loss.backward()
grad_naive = torch.autograd.grad(out_naive.square().mean(), q, retain_graph=True)[0]
torch.testing.assert_close(q.grad, grad_naive, atol=1e-3, rtol=1e-2)
print('backward OK')

'''
Think in terms of the attention computation itself.

Attention for one head is

    αqk = softmax_k ( (q · kᵀ) / √d )              # over the key dimension k
    y_q = Σ_k αqk v_k

1. Masking keys (columns) BEFORE soft-max  
   • Objective: guarantee that padded tokens never receive probability mass, no matter which query is attending.  
   • Implementation: set the corresponding logits to −∞ for all queries; soft-max then yields αqk = 0 for every padded key.  
   • This must happen prior to soft-max because probabilities are produced per-row; once the soft-max is done you cannot “take mass away” cleanly.

2. Masking queries (rows) AFTER the weighted sum  
   • Padded queries should produce no output and should not contribute gradients, but they do not affect the probability distribution produced for real queries.  
   • If you tried to mask them by setting their whole logit row to −∞ *before* soft-max you'd end up with a row of −∞ only:  
     exp(−∞)=0 ⇒ row-sum=0 ⇒ division-by-zero ⇒ NaNs / undefined gradients.  
   • Zeroing the post-soft-max output for those rows is numerically safe and leaves the valid rows untouched.

So:

• key-side padding → mask at the logit stage (because it’s about “who can be attended to”).  
• query-side padding → mask at the output stage (because it’s about “who is doing the attending”).

That ordering keeps the mathematics correct and the implementation numerically stable.
'''