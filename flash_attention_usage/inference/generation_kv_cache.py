import json
import time
from typing import Optional
import torch
from model_llama_kv_cache import Transformer
from tokenizer_llama import Tokenizer
from dataclasses import dataclass
from chat_format import render

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
    kv_cache: list,
    *,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    curr_idx: int,
) -> torch.Tensor:
    logits = model(x, kv_cache=kv_cache, curr_idx=curr_idx)  # (B, T, vocab_size)
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

    head_dim = model.model_args.dim // model.model_args.n_heads

    # Allocate enough space for prompt + generated tokens
    prompt_len = input_ids.shape[1]
    cache_len = prompt_len + max_new_tokens
    kv_cache = [(
        torch.zeros((input_ids.shape[0], cache_len, model.model_args.n_kv_heads, head_dim), dtype=torch.bfloat16, device="cuda"),
        torch.zeros((input_ids.shape[0], cache_len, model.model_args.n_kv_heads, head_dim), dtype=torch.bfloat16, device="cuda"),
    ) for _ in range(model.model_args.n_layers)]

    generated_tokens = input_ids.clone()
    next_token = input_ids.clone()

    prompt_len = input_ids.shape[1]
    curr_idx = 0

    for idx in range(prompt_len, prompt_len + max_new_tokens):
        next_token = generate_next_token(
            model,
            x=next_token,
            kv_cache=kv_cache,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
            curr_idx=curr_idx,
        )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        curr_idx = idx

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

def convert_to_chat_template(user_prompt: str, system_prompt: str = ""):
    converted_message = render(system_prompt, user_prompt)
    return converted_message

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

    prompt = ["Hello, who are you?", "What is the capital of France? And what is the capital of Germany?"]
    inp_list = []
    inp_lens = []
    for p in prompt:
        converted_message = convert_to_chat_template(p)
        tokens = tokenizer.encode(converted_message, bos=True, eos=False)
        inp_list.append(torch.tensor(tokens))
        inp_lens.append(len(tokens))
    
    max_len = max(inp_lens)
    for i in range(len(inp_list)):
        inp_list[i] = torch.nn.functional.pad(inp_list[i], (0, max_len - inp_lens[i]), value=tokenizer.eot_id)
    inp_list = torch.stack(inp_list).to("cuda")
    inp_lens = torch.tensor(inp_lens).to("cuda")
    print(inp_list.shape)
    print(inp_list)

    time_start = time.time()
    output = generate(model, inp_list, max_new_tokens=100)
    time_end = time.time()
    print(f"Generation time: {time_end - time_start} seconds")
    print(tokenizer.decode(output[0].tolist()))
    print("="*100)
    print(tokenizer.decode(output[1].tolist()))