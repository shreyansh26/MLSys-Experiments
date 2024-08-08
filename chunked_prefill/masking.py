import torch

def compute_attn(q, k, v, q_start_idx=None, q_end_idx=None):
    d = q.shape[-1]
    if q_start_idx is not None:
        q_idx = torch.arange(q_start_idx, q_end_idx).reshape(-1, 1)
    else:
        q_idx = torch.arange(q.shape[0]).reshape(-1, 1)
    k_idx = torch.arange(k.shape[0])
    bool_mask = q_idx < k_idx
    mask = torch.ones(q.shape[0], k.shape[0])
    mask = mask.masked_fill(bool_mask, float('-inf'))

    att = (q @ k.T)/d**0.5
    att = att + mask
    att = torch.nn.functional.softmax(att, dim=-1)
    att = att @ v
    return att

def compute_chunked_prefill_attn(q, k, v, num_chunks, chunk_width):
    att_chunk_list = []

    for chunk in range(num_chunks):
        q_chunk = q[chunk*chunk_width:(chunk+1)*chunk_width,:]
        k_chunk = k[0:(chunk+1)*chunk_width,:]
        v_chunk = v[0:(chunk+1)*chunk_width,:]

        att_chunk = compute_attn(q_chunk, k_chunk, v_chunk, chunk*chunk_width, (chunk+1)*chunk_width)
        att_chunk_list.append(att_chunk)
        
    return torch.vstack(att_chunk_list)

if __name__ == "__main__":
    SEQ_LEN = 12
    HIDDEN_DIM = 32
    NUM_CHUNKS = 3
    assert SEQ_LEN % NUM_CHUNKS == 0

    q = torch.randn(SEQ_LEN, HIDDEN_DIM)
    k = torch.randn(SEQ_LEN, HIDDEN_DIM)
    v = torch.randn(SEQ_LEN, HIDDEN_DIM)

    orig_attn = compute_attn(q, k, v)

    CHUNK_WIDTH = SEQ_LEN // NUM_CHUNKS

    chunked_prefill_attn = compute_chunked_prefill_attn(q, k, v, NUM_CHUNKS, CHUNK_WIDTH)

    print(orig_attn.shape)
    print(chunked_prefill_attn.shape)

    torch.testing.assert_close(orig_attn, chunked_prefill_attn)