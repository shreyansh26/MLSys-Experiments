import json
import time
from typing import Optional
import torch
from model_llama_kv_cache_paged import Transformer
from tokenizer_llama import Tokenizer
from dataclasses import dataclass
import math
import random
from chat_format import render

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

# From - https://github.com/tspeterkim/paged-attention-minimal/blob/main/llama3-paged.py
class PagedKVCache:
    def __init__(self, inp_tokens, model_args: ModelArgs, batch_size: int, max_len: int, tokenizer):
        self.model_args = model_args
        self.block_size = 256

        self.max_len = max_len if max_len is not None else model_args.max_seq_len
        self.head_dim = model_args.dim // model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.batch_size = batch_size
        self.num_blocks = ((self.max_len + self.block_size - 1) // self.block_size) * 4 # heuristic
        self.block_table = {i: [] for i in range(self.batch_size)}
        self.free_blocks = set(range(self.num_blocks))
        self.k_cache_paged = torch.zeros((self.num_blocks, self.block_size, model_args.n_kv_heads, self.head_dim), dtype=torch.bfloat16, device="cuda")
        self.v_cache_paged = torch.zeros((self.num_blocks, self.block_size, model_args.n_kv_heads, self.head_dim), dtype=torch.bfloat16, device="cuda")

        seq_lens = (inp_tokens != tokenizer.pad_id).sum(-1)
        for b, seq_len in enumerate(seq_lens.tolist()):
            num_blocks_to_reserve = math.ceil(seq_len / self.block_size)
            num_filled_positions = seq_len % self.block_size
            num_blocks_to_reserve = math.ceil(num_filled_positions / self.block_size)
            for i in range(num_blocks_to_reserve):
                index = self.get_free_block()
                if i == num_blocks_to_reserve - 1:
                    self.block_table[b].append((index, num_filled_positions))
                else:
                    self.block_table[b].append((index, self.block_size))

    def get_free_block(self) -> int:
        if len(self.free_blocks) == 0:
            raise Exception('No more free blocks. Implement scheduling and preemption.')
        index = random.choice(list(self.free_blocks))
        self.free_blocks.remove(index)
        return index
    
    def get_block_table(self):
        max_len = max(len(b) for b in self.block_table.values())
        block_table = [[-1] * max_len for _ in range(self.batch_size)]
        for i, b in self.block_table.items():
            for j, (index, _) in enumerate(b):
                block_table[i][j] = index
        return torch.tensor(block_table, dtype=torch.int32, device="cuda")

    def get_kv_cache(self):
        return self.k_cache_paged, self.v_cache_paged

    def get_last_pos(self):
        last_pos = [(len(b)-1)*self.block_size + b[len(b)-1][1]-1 for b in self.block_table.values()]         # First term for filled positions, second term for the last block
        return torch.tensor(last_pos, dtype=torch.int32, device="cuda")

    def free_memory(self, index):
        blocks = self.block_table[index]
        if len(blocks) == 1:
            return
        for i, _ in blocks[1:]:
            self.free_blocks.add(i)
        self.block_table[index] = blocks[:1]

    def update(self, eos_reached, input_text_mask):
        for i, (eos, is_prompt) in enumerate(zip(eos_reached, input_text_mask)):
            if is_prompt: # if the token is part of the original prompt, we skip
                continue
            if eos: # free the request's blocks since we have generated the complete answer
                self.free_memory(i)
                continue

            old_index, n = self.block_table[i][-1]
            if n == self.block_size: # allocate new block if necessary
                new_index = self.get_free_block()
                self.block_table[i].append((new_index, 1))
            else: # otherwise, just use the next available slot in the block
                self.block_table[i][-1] = (old_index, n+1)

    def get_fragmented_memory_size(self):
        size = 0
        for b in self.block_table.values():
            _, filled = b[-1] # only the last block has fragmentation
            size += (self.block_size - filled) * self.n_kv_heads * self.head_dim * 2 * 2
        return size

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
    kv_cache: PagedKVCache,
    eos_reached: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    curr_idx: int,
) -> torch.Tensor:
    logits = model(x, kv_cache=kv_cache, curr_idx=curr_idx, eos_reached=eos_reached)  # (B, T, vocab_size)
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

    # Allocate enough space for prompt + generated tokens
    kv_cache = [PagedKVCache(input_ids, model.model_args, input_ids.shape[0], max_output_len, tokenizer) for _ in range(model.model_args.n_layers)]

    stop_tokens = torch.tensor(list(tokenizer.stop_tokens), device="cuda")
    eos_reached = torch.tensor([False] * input_ids.shape[0], device="cuda")

    curr_idx = 0

    for idx in range(min_prompt_len, max_output_len):
        next_token = generate_next_token(
            model,
            x=input_ids[:, curr_idx:idx],
            kv_cache=kv_cache,
            eos_reached=eos_reached,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
            curr_idx=curr_idx,
        )
        next_token = torch.where(input_text_mask[:, idx], input_ids[:, idx], next_token.squeeze(-1))
        input_ids[:, idx] = next_token
        
        for layer in range(model.model_args.n_layers):
            kv_cache[layer].update(eos_reached.tolist(), input_text_mask[:, idx].tolist())

        eos_reached |= (~input_text_mask[:, idx]) & (torch.isin(next_token, stop_tokens))

        if all(eos_reached):
            print("EOS reached")
            break
        
        curr_idx = idx

    fragmented_memory = sum(kv_cache[layer].get_fragmented_memory_size() for layer in range(model.model_args.n_layers))
    fragmented_ratio = fragmented_memory / torch.cuda.get_device_properties(0).total_memory
    print(f'Fragmented Memory: {fragmented_memory / 1e9:.2f} GB ({fragmented_ratio * 100:.2f}%)')

    return input_ids

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

    # prompt = ["This is the story of", "Once upon a time in a land far, far away"]
    prompt = ["Hello, who are you?", "What is the capital of India? What is the capital of Germany?"]
    max_output_len = 500

    inp_list = []
    inp_lens = []
    for p in prompt:
        converted_message = convert_to_chat_template(p)
        # converted_message = p
        tokens = tokenizer.encode(converted_message, bos=True, eos=False)
        inp_list.append(torch.tensor(tokens))
        inp_lens.append(len(tokens))
    
    max_len = max(inp_lens)
    min_prompt_len = min(inp_lens)
    batch_size = len(prompt)

    model_input = torch.full((batch_size, max_output_len), tokenizer.pad_id, dtype=torch.long, device="cuda")
    for i in range(batch_size):
        model_input[i, :inp_lens[i]] = inp_list[i]

    input_text_mask = model_input != tokenizer.pad_id
    # print(input_text_mask)

    time_start = time.time()
    output = generate(model, model_input, input_text_mask, min_prompt_len, max_output_len)
    time_end = time.time()
    print(f"Generation time: {time_end - time_start} seconds")
    print(tokenizer.decode(output[0].tolist()))
    print("="*100)
    print(tokenizer.decode(output[1].tolist()))