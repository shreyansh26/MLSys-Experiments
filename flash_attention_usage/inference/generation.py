import json
import time
from typing import Optional
import torch
from model_llama import Transformer
from tokenizer_llama import Tokenizer
from dataclasses import dataclass
from chat_format import render

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
    input_text_mask: torch.Tensor,
    min_prompt_len: int,
    max_output_len: int,
    *,
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

    stop_tokens = torch.tensor(list(tokenizer.stop_tokens), device="cuda")
    eos_reached = torch.tensor([False] * input_ids.shape[0], device="cuda")

    for curr_idx in range(min_prompt_len, max_output_len):
        next_token = generate_next_token(
            model,
            x=input_ids[:, :curr_idx],
            temperature=temperature,
            top_k=top_k,
            rng=rng,
            curr_idx=curr_idx,
        )
        next_token = torch.where(input_text_mask[:, curr_idx], input_ids[:, curr_idx], next_token.squeeze(-1))
        input_ids[:, curr_idx] = next_token
        eos_reached |= (~input_text_mask[:, curr_idx]) & (torch.isin(next_token, stop_tokens))

        if all(eos_reached):
            print("EOS reached")
            break

    return input_ids

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
    model_name = "llama_3b"
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

    prompt = ["This is the story of", "Once upon a time in a land far, far away"]
    max_output_len = 150

    inp_list = []
    inp_lens = []
    for p in prompt:
        # converted_message = convert_to_chat_template(p)
        converted_message = p
        tokens = tokenizer.encode(converted_message, bos=True, eos=False)
        inp_list.append(torch.tensor(tokens))
        inp_lens.append(len(tokens))
    
    max_len = max(inp_lens)
    min_prompt_len = min(inp_lens)

    print(inp_lens)
    print(min_prompt_len)

    batch_size = len(prompt)

    model_input = torch.full((batch_size, max_output_len), tokenizer.pad_id, dtype=torch.long, device="cuda")
    for i in range(batch_size):
        model_input[i, :inp_lens[i]] = inp_list[i]

    input_text_mask = model_input != tokenizer.pad_id
    print(input_text_mask)
    
    time_start = time.time()
    output = generate(model, model_input, input_text_mask, min_prompt_len, max_output_len)
    time_end = time.time()
    print(f"Generation time: {time_end - time_start} seconds")
    print(tokenizer.decode(output[0].tolist()))
    print("="*100)
    print(tokenizer.decode(output[1].tolist()))