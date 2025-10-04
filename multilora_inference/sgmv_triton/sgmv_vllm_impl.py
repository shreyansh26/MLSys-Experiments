import torch

from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.lora.layers import LoRAMapping

# ----- Shapes and basic config -----
device = "cuda"
num_tokens = 128
input_dim = 4096
out_dim = 4096
rank = 16

num_loras = 7                 # number of adapters
max_loras = 8                 # capacity; can be >= num_loras
batch_size = num_tokens       # treat each token as a "row" to map to a LoRA
max_batches = 1               # for logits/prompt metadata; not used here
vocab_size = 0                # not used for linear path; safe to set 0
extra_vocab_size = 0          # not used for linear path; safe to set 0
scale = 1.0                   # typical LoRA uses alpha/r; set as needed

# ----- Inputs/Outputs -----
# x: [num_tokens, input_dim] fp16/bf16 (must match lora_a dtype)
x = torch.randn(num_tokens, input_dim, device=device, dtype=torch.float16)
# y: [num_tokens, out_dim] (accumulate result in-place)
y = torch.zeros(num_tokens, out_dim, device=device, dtype=torch.float16)

# ----- Prepare LoRA weights (single slice) -----
# A_i: [rank, input_dim], B_i: [out_dim, rank]
A_list = [torch.randn(rank, input_dim, device=device, dtype=torch.float16) for _ in range(num_loras)]
B_list = [torch.randn(out_dim, rank, device=device, dtype=torch.float16) for _ in range(num_loras)]

# Stack adapters along dim 0: [num_loras, rank, input_dim] and [num_loras, out_dim, rank]
A = torch.stack(A_list, dim=0).contiguous()
B = torch.stack(B_list, dim=0).contiguous()

# Each “slice” is one weight matrix. Single-slice case => length-1 tuples.
lora_a_stacked = (A,)  # [num_loras, rank, input_dim]
lora_b_stacked = (B,)  # [num_loras, out_dim, rank]
output_slices = (out_dim,)  # single slice outputs whole out_dim

# ----- Token→LoRA mapping -----
# LoRA IDs must be > 0 to be considered active; -1 means no LoRA.
# Define IDs [1..num_loras], and map tokens to those IDs (or -1).
lora_ids = [i + 1 for i in range(num_loras)]  # IDs: 1..7
# Map index->id for all slots up to max_loras; keep trailing None if capacity > num_loras
lora_index_to_id = lora_ids + [None] * (max_loras - len(lora_ids))

# Example: round-robin assign tokens to adapters (change as needed)
token_lora_ids = [lora_ids[t % num_loras] for t in range(num_tokens)]
# If you want some tokens to use no LoRA, set to -1 for those positions:
# token_lora_ids[k] = -1

# Build mapping; for generation/SGMV path use is_prefill=False and
# set prompt_mapping same as index_mapping.
mapping = LoRAMapping(
    index_mapping=tuple(token_lora_ids),
    prompt_mapping=(token_lora_ids[0],),
    is_prefill=False,
)

# ----- Construct wrapper and prepare metadata -----
wrapper = PunicaWrapperGPU(
    max_num_batched_tokens=num_tokens,
    max_batches=max_batches,
    device=device,
    max_loras=max_loras,
)

wrapper.update_metadata(
    mapping=mapping,
    lora_index_to_id=lora_index_to_id,
    max_loras=max_loras,
    vocab_size=vocab_size,
    extra_vocab_size=extra_vocab_size,
)

y_orig = y.clone()
# ----- Run SGMV for a single linear weight -----
# This internally performs:
#  - shrink: buffer = (x @ A) * scale  -> [1, num_tokens, rank] in fp32
#  - expand: y += buffer @ B           -> [num_tokens, out_dim], add_inputs=True adds residual if provided
wrapper.add_lora_linear(
    y=y,
    x=x,
    lora_a_stacked=lora_a_stacked,
    lora_b_stacked=lora_b_stacked,
    lora_bias_stacked=None,            # or a tuple like (bias_tensor,) shaped [num_loras, out_dim]
    scale=scale,
    output_slices=output_slices,
    buffer=None,                       # optional; let it allocate [1, num_tokens, rank] fp32
)

print(y)

print(y - y_orig)
# y now contains the LoRA contribution added in-place.
# If you want the pure LoRA delta, initialize y=zeros and keep add_inputs=True (default).