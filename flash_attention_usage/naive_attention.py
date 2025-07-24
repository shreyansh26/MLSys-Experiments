import torch
import math

def naive_causal_attention(q, k, v):
    B, q_seq_len, nheads, headdim = q.shape
    _, k_seq_len, _, _ = k.shape
    diag = k_seq_len - q_seq_len
    attention_weights = torch.einsum("nqhd,nkhd->nhqk", [q, k])    
    attention_weights = attention_weights / math.sqrt(headdim)
    attention_mask = torch.tril(torch.ones(q_seq_len, k_seq_len, device=q.device), diagonal=diag)
    attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
    attention_weights = torch.softmax(attention_weights, dim=-1)
    out = torch.einsum("nhqk,nkhd->nqhd", [attention_weights, v])
    return out

def naive_varlen(q_t, k_t, v_t, lens):
    B, S, H, D = q_t.shape
    scale = 1 / math.sqrt(D)
    outs = []
    for b, L in enumerate(lens.tolist()):
        q, k, v = q_t[b, :L] , k_t[b, :L] , v_t[b, :L]                      # (L, H, D)
        sim  = torch.einsum('qhd,khd->hqk', q, k) * scale
        sim  = sim.masked_fill(torch.triu(torch.ones_like(sim, dtype=torch.bool), 1), -float('inf'))
        attn = sim.softmax(-1)
        out  = torch.einsum('hqk,khd->qhd', attn, v)     # (L, H, D)
        outs.append(torch.cat([out, out.new_zeros(S - L, H, D)], dim=0))
    return torch.stack(outs, dim=0)                      # (B, S, H, D)

def naive_varlen_vectorized(q, k, v, valid_mask):
    B, q_seq_len, nheads, headdim = q.shape
    _, k_seq_len, _, _ = k.shape
    diag = k_seq_len - q_seq_len
    attention_weights = torch.einsum("nqhd,nkhd->nhqk", [q, k])    
    attention_weights = attention_weights / math.sqrt(headdim)
    attention_mask = torch.tril(torch.ones(q_seq_len, k_seq_len, device=q.device), diagonal=diag)
    attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
    # Mask out key positions that correspond to padding tokens. We broadcast the mask over
    # the heads and query dimensions: (B, 1, 1, k_seq_len)
    key_pad = (~valid_mask).bool()                          # (B, k_seq_len)
    attention_weights = attention_weights.masked_fill(key_pad[:, None, None, :], float('-inf'))
    attention_weights = torch.softmax(attention_weights, dim=-1)
    out = torch.einsum("nhqk,nkhd->nqhd", [attention_weights, v])
    # Zero-out the outputs that correspond to padded query positions
    out = out * valid_mask[:, :, None, None]
    return out           # (B, S, H, D)