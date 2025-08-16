## Llama Inference (FlashAttention experiments)

### Overview
This folder contains minimal Llama-3.2 inference scripts showing three variants:
- Baseline, padding-free batched generation (`generation.py` + `model_llama.py`)
- KV-cache based generation (`generation_kv_cache.py` + `model_llama_kv_cache.py`)
- Paged KV-cache generation (`generation_kv_cache_paged.py` + `model_llama_kv_cache_paged.py`)

All three share a simple driver that:
- Builds prompts (optionally via `chat_format.py` for Llama-3 chat format)
- Tokenizes with `tokenizer_llama.py`
- Generates tokens up to a target length using a padding-free scheme: the prompt is left-aligned; a boolean mask protects prompt tokens while newly generated tokens are written in-place to the output buffer.

### Requirements
- Python 3.10+
- CUDA-capable GPU and a CUDA-enabled PyTorch install
- FlashAttention 2 (v2.5.8 in my experiments) for KV-cache variants
- `jinja2` (used by `chat_format.py`)
- Hugging Face CLI for downloading weights

### Download model weights
Run one or both (requires a valid HF token):

```bash
export HF_TOKEN=<hf_token>
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --include "original/*" --local-dir llama_3b_instruct
huggingface-cli download meta-llama/Llama-3.2-3B          --include "original/*" --local-dir llama_3b
```

The scripts expect a directory structure like:
```
llama_3b_instruct/
  original/
    tokenizer.model
    params.json
    consolidated.00.pth
llama_3b/
  original/
    tokenizer.model
    params.json
    consolidated.00.pth
```

### Run
- Baseline, padding-free:
```bash
python generation.py
```

- KV cache:
```bash
python generation_kv_cache.py
```

- Paged KV cache:
```bash
python generation_kv_cache_paged.py
```

Each script defines `model_name`, `model_path`, and a `prompt` list in `__main__`. Adjust as needed (e.g., switch between base and instruct). Outputs are decoded and printed for each batch item.

### Notes
- Padding-free batching: We allocate an output tensor of shape `(batch, max_output_len)` filled with PAD. Prompt tokens are copied to the left; a boolean mask keeps them fixed while sampling fills subsequent positions.
- KV cache variant preallocates per-layer K/V tensors sized to `max_output_len` to avoid reallocation during decoding.
- Paged KV cache variant keeps fixed-size pages (e.g., 256 tokens) and updates a per-request block table to reduce fragmentation and improve memory reuse across requests.

### Troubleshooting
- Ensure `tokenizer.model`, `params.json`, and `consolidated.00.pth` exist under the selected `model_path`.
- Use a GPU with sufficient memory, especially for larger `max_output_len` and batch sizes.
- FlashAttention must be correctly installed for the KV-cache scripts.