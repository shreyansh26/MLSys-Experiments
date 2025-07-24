import torch, math
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_varlen_func
from naive_attention import naive_varlen, naive_causal_varlen_vectorized_attention

torch.manual_seed(0)
dtype   = torch.float16         # stay in fp32 for tight tolerances
B, q_seq_len, nheads, headdim = 2, 2048, 16, 64
kv_seq_len = 4096

# create a ragged batch
lengths  = torch.tensor([1783, 512], device='cuda', dtype=torch.int32)
q        = torch.randn(B, q_seq_len, nheads, headdim, device='cuda', dtype=dtype, requires_grad=True)
k        = torch.randn(B, kv_seq_len, nheads, headdim, device='cuda', dtype=dtype, requires_grad=True)
v        = torch.randn(B, kv_seq_len, nheads, headdim, device='cuda', dtype=dtype, requires_grad=True)

# Padding masks
# Query-side mask has q_seq_len columns, key/value mask has kv_seq_len columns.
q_valid_mask  = torch.arange(q_seq_len,  device='cuda')[None] < lengths[:, None]
k_valid_mask  = torch.arange(kv_seq_len, device='cuda')[None] < lengths[:, None]

# Strip padding
q_flat, idx_q, cu_lens_q, max_q = unpad_input(q, q_valid_mask)
k_flat, idx_k, cu_lens_k, max_k = unpad_input(k, k_valid_mask)
v_flat, _,          _,      _   = unpad_input(v, k_valid_mask)

out_flat = flash_attn_varlen_func(
    q_flat, k_flat, v_flat,
    cu_seqlens_q=cu_lens_q,
    cu_seqlens_k=cu_lens_k,
    max_seqlen_q=max_q,
    max_seqlen_k=max_k,
    causal=True,
)

out_flash = pad_input(out_flat, idx_q, B, q_seq_len)                      # back to (B,S,H,D)
print("out_flash.shape:\t\t", out_flash.shape)      # (2, 2048, 16, 64)

# out_naive = naive_varlen(q, k, v, lengths) # Works correctly
out_naive = naive_causal_varlen_vectorized_attention(q, k, v, q_valid_mask, k_valid_mask)
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

• key-side padding → mask at the logit stage (because it's about "who can be attended to").  
• query-side padding → mask at the output stage (because it's about "who is doing the attending").

That ordering keeps the mathematics correct and the implementation numerically stable.
------------------------------------------------------------

`flash_attn_func` and `flash_attn_varlen_func` implement slightly different
definitions of “causal” when the key sequence is longer than the query
sequence.

1. `flash_attn_func` (used in `using_flash_attn_func.py`)  
   • Works on *dense* tensors of shape `(B, Sq, H, D)` and `(B, Sk, H, D)`.  
   • When `Sq < Sk` it assumes the extra `Sk-Sq` keys are a
     “prefix-memory” that is always visible.  
   • Internally it therefore keeps every element on or **below**
     the diagonal offset by `Sk-Sq`:  

     ```
     causal_mask[i, j] = j > i + (Sk - Sq)   # masked out
     ```
     which is exactly the rule you coded in
     `naive_causal_attention` ( `torch.tril(..., diagonal=Sk-Sq)` ).
   • Hence the assertion passes.

2. `flash_attn_varlen_func` (used in `using_flash_attn_varlen_func.py`)  
   • Works on *unpadded* ragged batches: each sequence i has its own
     length `Li`, you pass those via `cu_seqlens_{q,k}`.  
   • Here the library assumes **each individual sequence has `Li_q =
     Li_k`** (same query and key length after unpadding).  
     Once the padding is stripped there is *no longer* a global
     rectangular situation where `Sk > Sq`; every per-sequence block is
     square.  
   • Consequently it applies the *standard* causal mask

     ```
     causal_mask[i, j] = j > i      # masked out
     ```
     i.e. `torch.triu(..., diagonal=1)` with **no offset**.
   • That matches the behaviour of the version you called
     `naive_varlen_vectorized` (strict triangular mask) and is why the
     assertion succeeds only with that version.

Why `naive_varlen_vectorized_v2` differs  
----------------------------------------
`v2` keeps the “diagonal offset” logic from `naive_causal_attention`, but
after unpadding every per-sequence block is already square, so the
offset lets queries see future tokens that `flash_attn_varlen_func`
correctly masks.  The resulting 12-13 % element mismatch is exactly those
extra attentions.

What to do
----------

• If you want to replicate `flash_attn_varlen_func`, keep the
  strict-triangular mask (your original `naive_varlen_vectorized`).  

• If you need the “prefix-memory” semantics for ragged batches you would
  have to implement it yourself for both the Flash-Attention path and
  your naïve reference, because the stock
  `flash_attn_varlen_func` does not provide that option.

So the apparent inconsistency comes from two different APIs using two
different causal conventions; each of your reference implementations is
accurate for the particular Flash-Attention function it is paired with.
'''