# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.configuration_llama import LlamaConfig
from bgmv_cuda import lora_bgmv_cuda as lora_bgmv_cuda_impl
from bgmv_triton import lora_bgmv_triton as lora_bgmv_triton_impl
from sgmv_triton import lora_sgmv_triton as lora_sgmv_triton_impl


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, layer_idx, lora_indices, lora_inference_mode):
        gate_proj = self.gate_proj(x)
        lora_A_weights = getattr(self, "lora_A_weights", None)
        lora_B_weights = getattr(self, "lora_B_weights", None)
        if lora_A_weights is not None and lora_B_weights is not None and layer_idx is not None:
            if lora_inference_mode == "gbmm":
                A_gate = lora_A_weights["gate_proj"][layer_idx][lora_indices, :, :].to(gate_proj)
                B_gate = lora_B_weights["gate_proj"][layer_idx][lora_indices, :, :].to(gate_proj)
                gate_proj = gate_proj + torch.einsum("bfi,bni->bnf", B_gate, torch.einsum("bfi,bni->bnf", A_gate, x)) * getattr(self, "lora_scale", 1.0)
            elif lora_inference_mode == "bgmv_cuda":
                A_gate = lora_A_weights["gate_proj"].to(gate_proj)
                B_gate = lora_B_weights["gate_proj"].to(gate_proj)
                lora_bgmv_cuda_impl(gate_proj, x, A_gate, B_gate, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "bgmv_triton":
                A_gate = lora_A_weights["gate_proj"].to(gate_proj)
                B_gate = lora_B_weights["gate_proj"].to(gate_proj)
                lora_bgmv_triton_impl(gate_proj, x, A_gate, B_gate, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "sgmv_triton":
                A_gate = lora_A_weights["gate_proj"][layer_idx]
                B_gate = lora_B_weights["gate_proj"][layer_idx]
                lora_sgmv_triton_impl(gate_proj, x, A_gate, B_gate, lora_indices, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            else:
                raise ValueError(f"Invalid LoRA inference mode: {lora_inference_mode}")

        up_proj = self.up_proj(x)
        if lora_A_weights is not None and lora_B_weights is not None and layer_idx is not None:
            if lora_inference_mode == "gbmm":
                A_up = lora_A_weights["up_proj"][layer_idx][lora_indices, :, :].to(up_proj)
                B_up = lora_B_weights["up_proj"][layer_idx][lora_indices, :, :].to(up_proj)
                up_proj = up_proj + torch.einsum("bfi,bni->bnf", B_up, torch.einsum("bfi,bni->bnf", A_up, x)) * getattr(self, "lora_scale", 1.0)
            elif lora_inference_mode == "bgmv_cuda":
                A_up = lora_A_weights["up_proj"].to(up_proj)
                B_up = lora_B_weights["up_proj"].to(up_proj)
                lora_bgmv_cuda_impl(up_proj, x, A_up, B_up, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "bgmv_triton":
                A_up = lora_A_weights["up_proj"].to(up_proj)
                B_up = lora_B_weights["up_proj"].to(up_proj)
                lora_bgmv_triton_impl(up_proj, x, A_up, B_up, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "sgmv_triton":
                A_up = lora_A_weights["up_proj"][layer_idx]
                B_up = lora_B_weights["up_proj"][layer_idx]
                lora_sgmv_triton_impl(up_proj, x, A_up, B_up, lora_indices, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            else:
                raise ValueError(f"Invalid LoRA inference mode: {lora_inference_mode}")
        act = self.act_fn(gate_proj) * up_proj
        down_proj = self.down_proj(act)
        if lora_A_weights is not None and lora_B_weights is not None and layer_idx is not None:
            if lora_inference_mode == "gbmm":
                A_down = lora_A_weights["down_proj"][layer_idx][lora_indices, :, :].to(down_proj)
                B_down = lora_B_weights["down_proj"][layer_idx][lora_indices, :, :].to(down_proj)
                down_proj = down_proj + torch.einsum("bfi,bni->bnf", B_down, torch.einsum("bfi,bni->bnf", A_down, act)) * getattr(self, "lora_scale", 1.0)
            elif lora_inference_mode == "bgmv_cuda":
                A_down = lora_A_weights["down_proj"].to(down_proj)
                B_down = lora_B_weights["down_proj"].to(down_proj)
                lora_bgmv_cuda_impl(down_proj, act, A_down, B_down, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "bgmv_triton":
                A_down = lora_A_weights["down_proj"].to(down_proj)
                B_down = lora_B_weights["down_proj"].to(down_proj)
                lora_bgmv_triton_impl(down_proj, act, A_down, B_down, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "sgmv_triton":
                A_down = lora_A_weights["down_proj"][layer_idx]
                B_down = lora_B_weights["down_proj"][layer_idx]
                lora_sgmv_triton_impl(down_proj, act, A_down, B_down, lora_indices, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            else:
                raise ValueError(f"Invalid LoRA inference mode: {lora_inference_mode}")
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idx: Optional[int] = None,
        lora_indices: Optional[torch.Tensor] = None,
        lora_inference_mode: Optional[str] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        lora_A_weights = getattr(self, "lora_A_weights", None)
        lora_B_weights = getattr(self, "lora_B_weights", None)
        if lora_A_weights is not None and lora_B_weights is not None and layer_idx is not None:
            if lora_inference_mode == "gbmm":
                A_q = lora_A_weights["q_proj"][layer_idx][lora_indices, :, :].to(query_states)
                B_q = lora_B_weights["q_proj"][layer_idx][lora_indices, :, :].to(query_states)
                query_states = query_states + torch.einsum("bfi,bni->bnf", B_q, torch.einsum("bfi,bni->bnf", A_q, hidden_states)) * getattr(self, "lora_scale", 1.0)
            elif lora_inference_mode == "bgmv_cuda":
                A_q = lora_A_weights["q_proj"].to(query_states)
                B_q = lora_B_weights["q_proj"].to(query_states)
                lora_bgmv_cuda_impl(query_states, hidden_states, A_q, B_q, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "bgmv_triton":
                A_q = lora_A_weights["q_proj"].to(query_states)
                B_q = lora_B_weights["q_proj"].to(query_states)
                lora_bgmv_triton_impl(query_states, hidden_states, A_q, B_q, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "sgmv_triton":
                A_q = lora_A_weights["q_proj"][layer_idx]
                B_q = lora_B_weights["q_proj"][layer_idx]
                lora_sgmv_triton_impl(query_states, hidden_states, A_q, B_q, lora_indices, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            else:
                raise ValueError(f"Invalid LoRA inference mode: {lora_inference_mode}")
        query_states = query_states.view(hidden_shape).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        if lora_A_weights is not None and lora_B_weights is not None and layer_idx is not None:
            if lora_inference_mode == "gbmm":
                A_k = lora_A_weights["k_proj"][layer_idx][lora_indices, :, :].to(key_states)
                B_k = lora_B_weights["k_proj"][layer_idx][lora_indices, :, :].to(key_states)
                key_states = key_states + torch.einsum("bfi,bni->bnf", B_k, torch.einsum("bfi,bni->bnf", A_k, hidden_states)) * getattr(self, "lora_scale", 1.0)
            elif lora_inference_mode == "bgmv_cuda":
                A_k = lora_A_weights["k_proj"].to(key_states)
                B_k = lora_B_weights["k_proj"].to(key_states)
                lora_bgmv_cuda_impl(key_states, hidden_states, A_k, B_k, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "bgmv_triton":
                A_k = lora_A_weights["k_proj"].to(key_states)
                B_k = lora_B_weights["k_proj"].to(key_states)
                lora_bgmv_triton_impl(key_states, hidden_states, A_k, B_k, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "sgmv_triton":
                A_k = lora_A_weights["k_proj"][layer_idx]
                B_k = lora_B_weights["k_proj"][layer_idx]
                lora_sgmv_triton_impl(key_states, hidden_states, A_k, B_k, lora_indices, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            else:
                raise ValueError(f"Invalid LoRA inference mode: {lora_inference_mode}")
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        
        value_states = self.v_proj(hidden_states)
        if lora_A_weights is not None and lora_B_weights is not None and layer_idx is not None:
            if lora_inference_mode == "gbmm":
                A_v = lora_A_weights["v_proj"][layer_idx][lora_indices, :, :].to(value_states)
                B_v = lora_B_weights["v_proj"][layer_idx][lora_indices, :, :].to(value_states)
                value_states = value_states + torch.einsum("bfi,bni->bnf", B_v, torch.einsum("bfi,bni->bnf", A_v, hidden_states)) * getattr(self, "lora_scale", 1.0)
            elif lora_inference_mode == "bgmv_cuda":
                A_v = lora_A_weights["v_proj"].to(value_states)
                B_v = lora_B_weights["v_proj"].to(value_states)
                lora_bgmv_cuda_impl(value_states, hidden_states, A_v, B_v, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "bgmv_triton":
                A_v = lora_A_weights["v_proj"].to(value_states)
                B_v = lora_B_weights["v_proj"].to(value_states)
                lora_bgmv_triton_impl(value_states, hidden_states, A_v, B_v, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "sgmv_triton":
                A_v = lora_A_weights["v_proj"][layer_idx]
                B_v = lora_B_weights["v_proj"][layer_idx]
                lora_sgmv_triton_impl(value_states, hidden_states, A_v, B_v, lora_indices, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            else:
                raise ValueError(f"Invalid LoRA inference mode: {lora_inference_mode}")
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output_out = self.o_proj(attn_output)
        if lora_A_weights is not None and lora_B_weights is not None and layer_idx is not None:
            if lora_inference_mode == "gbmm":
                A_o = lora_A_weights["o_proj"][layer_idx][lora_indices, :, :].to(attn_output)
                B_o = lora_B_weights["o_proj"][layer_idx][lora_indices, :, :].to(attn_output)
                attn_output_out = attn_output_out + torch.einsum("bfi,bni->bnf", B_o, torch.einsum("bfi,bni->bnf", A_o, attn_output)) * getattr(self, "lora_scale", 1.0)
            elif lora_inference_mode == "bgmv_cuda":
                A_o = lora_A_weights["o_proj"].to(attn_output)
                B_o = lora_B_weights["o_proj"].to(attn_output)
                lora_bgmv_cuda_impl(attn_output_out, attn_output, A_o, B_o, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "bgmv_triton":
                A_o = lora_A_weights["o_proj"].to(attn_output)
                B_o = lora_B_weights["o_proj"].to(attn_output)
                lora_bgmv_triton_impl(attn_output_out, attn_output, A_o, B_o, lora_indices, num_layers=self.num_hidden_layers, layer_idx=layer_idx, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            elif lora_inference_mode == "sgmv_triton":
                A_o = lora_A_weights["o_proj"][layer_idx]
                B_o = lora_B_weights["o_proj"][layer_idx]
                lora_sgmv_triton_impl(attn_output_out, attn_output, A_o, B_o, lora_indices, num_lora_adapters=self.num_lora_adapters, scale=getattr(self, "lora_scale", 1.0))
            else:
                raise ValueError(f"Invalid LoRA inference mode: {lora_inference_mode}")
        return attn_output_out, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        layer_idx: Optional[int] = None,
        lora_indices: Optional[torch.Tensor] = None,
        lora_inference_mode: Optional[str] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            layer_idx=layer_idx,
            lora_indices=lora_indices,
            lora_inference_mode=lora_inference_mode,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, layer_idx, lora_indices, lora_inference_mode)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # LoRA: storage for weights; when set via the property, we propagate
        # them down to all relevant submodules so they can access without
        # explicit forward args.
        self._lora_A_weights = None
        self._lora_B_weights = None
        self._lora_scale = 1.0
        self._num_lora_adapters = 0
        self._num_hidden_layers = 0

    def _propagate_lora_to_submodules(self) -> None:
        """
        Attach the LoRA weight dictionaries to all decoder submodules that need them.
        This lets modules like `LlamaAttention` and `LlamaMLP` access the weights via
        getattr(self, "lora_A_weights", None) without having to thread them through forwards.
        """
        for layer in self.layers:
            # Attach on the layer itself (optional) and required leaf submodules
            layer.lora_A_weights = self._lora_A_weights
            layer.lora_B_weights = self._lora_B_weights
            layer.self_attn.lora_A_weights = self._lora_A_weights
            layer.self_attn.lora_B_weights = self._lora_B_weights
            layer.mlp.lora_A_weights = self._lora_A_weights
            layer.mlp.lora_B_weights = self._lora_B_weights
            layer.self_attn.lora_scale = self._lora_scale
            layer.mlp.lora_scale = self._lora_scale
            layer.self_attn.num_lora_adapters = self._num_lora_adapters
            layer.mlp.num_lora_adapters = self._num_lora_adapters
            layer.self_attn.num_hidden_layers = self._num_hidden_layers
            layer.mlp.num_hidden_layers = self._num_hidden_layers

    @property
    def lora_A_weights(self):
        return self._lora_A_weights

    @lora_A_weights.setter
    def lora_A_weights(self, value):
        self._lora_A_weights = value
        self._propagate_lora_to_submodules()

    @property
    def lora_B_weights(self):
        return self._lora_B_weights

    @lora_B_weights.setter
    def lora_B_weights(self, value):
        self._lora_B_weights = value
        self._propagate_lora_to_submodules()

    @property
    def lora_scale(self):
        return self._lora_scale

    @lora_scale.setter
    def lora_scale(self, value):
        self._lora_scale = value
        self._propagate_lora_to_submodules()

    @property
    def num_lora_adapters(self):
        return self._num_lora_adapters

    @num_lora_adapters.setter
    def num_lora_adapters(self, value):
        self._num_lora_adapters = value
        self._propagate_lora_to_submodules()

    @property
    def num_hidden_layers(self):
        return self._num_hidden_layers

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        self._num_hidden_layers = value
        self._propagate_lora_to_submodules()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        lora_indices: Optional[torch.Tensor] = None,
        lora_inference_mode: Optional[str] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                layer_idx=idx,
                lora_indices=lora_indices,
                lora_inference_mode=lora_inference_mode,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        lora_indices: Optional[torch.Tensor] = None,
        lora_inference_mode: Optional[str] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```

        Args:
            lora_indices (`torch.LongTensor`, *optional*):
                Per-sample LoRA adapter indices to use during generation.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            lora_indices=lora_indices,
            lora_inference_mode=lora_inference_mode,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        lora_indices: Optional[torch.Tensor] = None,
        lora_inference_mode: Optional[str] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            lora_inference_mode=lora_inference_mode,
            **kwargs,
        )
        if lora_indices is not None:
            model_inputs["lora_indices"] = lora_indices
        if lora_inference_mode is not None:
            model_inputs["lora_inference_mode"] = lora_inference_mode
        return model_inputs


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]