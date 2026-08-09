"""A compact Llama-style decoder block used by the checkpointing examples."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DecoderConfig:
    dim: int = 512
    num_heads: int = 8
    hidden_dim: int = 1536
    dropout: float = 0.1
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if (self.dim // self.num_heads) % 2:
            raise ValueError("the attention head dimension must be even for RoPE")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        positions = torch.arange(
            sequence_length, device=query.device, dtype=self.inverse_frequency.dtype
        )
        frequencies = torch.outer(positions, self.inverse_frequency)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cosine = angles.cos().to(query.dtype)[None, None, :, :]
        sine = angles.sin().to(query.dtype)[None, None, :, :]
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


class CausalSelfAttention(nn.Module):
    """Explicit attention math keeps SAC's per-operation choices observable."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.output = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        query, key = self.rope(query, key)

        scores = query @ key.transpose(-2, -1)
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        causal_mask = torch.ones(
            sequence, sequence, dtype=torch.bool, device=x.device
        ).triu(1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1)
        probabilities = F.dropout(
            probabilities, p=self.dropout, training=self.training
        )
        attended = probabilities @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    """Pre-norm decoder: RoPE attention, residuals, and a SwiGLU MLP."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = SwiGLU(config)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + F.dropout(
            self.attention(self.attention_norm(x)),
            p=self.dropout,
            training=self.training,
        )
        return x + F.dropout(
            self.mlp(self.mlp_norm(x)),
            p=self.dropout,
            training=self.training,
        )
