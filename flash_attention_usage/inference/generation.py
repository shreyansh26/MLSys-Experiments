import json
import time
from typing import Optional
import torch
from model_llama import Transformer
from tokenizer_llama import Tokenizer
from dataclasses import dataclass

def multinomial_sample_one(
    probs: torch.Tensor, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    curr_idx: int,
) -> torch.Tensor:
    logits = model(x, curr_idx)  # (B, T, vocab_size)
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # ensure batch dimension (T,) --> (B, T)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    generated_tokens = input_ids.clone()
    prompt_len = input_ids.shape[1]
    curr_idx = prompt_len

    for curr_idx in range(prompt_len, prompt_len + max_new_tokens):
        next_token = generate_next_token(
            model,
            x=generated_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
            curr_idx=curr_idx,
        )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    return generated_tokens

def load_model(model_path, model_args):
    with torch.device("meta"):
        model = Transformer(model_args)
    
    model = model.to_empty(device="cpu")
    state_dict = torch.load(f"{model_path}/consolidated.00.pth", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, assign=True)

    # Load freqs_cis separately
    with torch.no_grad():
        model.freqs_cis = model._precompute_freqs_cis()
    return model

def load_model2(model_path, model_args):
    model = Transformer(model_args)
    
    state_dict = torch.load(f"{model_path}/consolidated.00.pth", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, assign=True)

    return model

@dataclass
class ModelArgs:
    dim: int = None
    n_layers: int = None
    n_heads: int = None
    n_kv_heads: int = None
    vocab_size: int = None
    ffn_dim_multiplier: float = None
    multiple_of: int = None
    norm_eps: float = None
    rope_theta: float = None
    use_scaled_rope: bool = None
    max_seq_len: int = None

if __name__ == "__main__":
    model_name = "llama_3b_instruct"
    model_path = f"./{model_name}/original"
    model_config = f"{model_path}/params.json"
    with open(model_config, "r") as f:
        params = json.load(f)

    params['max_seq_len'] = 131072
    model_args = ModelArgs(**params)
    print(model_args)

    time_start = time.time()
    model = load_model(model_path, model_args)
    time_end = time.time()
    print(f"Model loading time: {time_end - time_start} seconds")

    time_start = time.time()
    model.to("cuda")
    time_end = time.time()
    print(f"Model to GPU transfer time: {time_end - time_start} seconds")

    tokenizer = Tokenizer(f"{model_path}/tokenizer.model")
    time_end = time.time()
    print(f"Tokenizer loading time: {time_end - time_start} seconds")

    prompt = "Hello, who are you?"
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    input_ids = torch.tensor(tokens).unsqueeze(0).to("cuda")

    time_start = time.time()
    output = generate(model, input_ids, max_new_tokens=100)
    time_end = time.time()
    print(f"Generation time: {time_end - time_start} seconds")
    print(tokenizer.decode(output[0].tolist()))