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