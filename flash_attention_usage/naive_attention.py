import torch
import math

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