from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BabyVNextConfig


class ExplicitLayerNorm(nn.LayerNorm):
    """ROCm-stable LayerNorm equation preserved from Research Baby."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        centered = inputs - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias
        return normalized


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BabyVNextConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = float(config.attention_dropout)
        self.backend: Literal["sdpa", "reference"] = config.attention_backend  # type: ignore[assignment]
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.qkv_bias)
        self.projection = nn.Linear(
            config.d_model, config.d_model, bias=config.attention_output_bias
        )

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, time, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        return q, k, v

    def _reference_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        time = q.shape[-2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal = torch.ones((time, time), dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(x)
        if self.backend == "sdpa":
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attended = self._reference_attention(q, k, v)
        batch, _, time, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(batch, time, self.d_model)
        return self.projection(merged)


class FeedForward(nn.Module):
    def __init__(self, config: BabyVNextConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp, bias=config.mlp_bias),
            nn.ReLU(),
            nn.Linear(config.d_mlp, config.d_model, bias=config.mlp_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: BabyVNextConfig) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = ExplicitLayerNorm(config.d_model, eps=config.norm_epsilon)
        self.norm2 = ExplicitLayerNorm(config.d_model, eps=config.norm_epsilon)
        self.attention_residual_dropout = nn.Dropout(config.residual_dropout)
        self.feed_forward_residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_residual_dropout(self.attention(self.norm1(x)))
        return x + self.feed_forward_residual_dropout(self.feed_forward(self.norm2(x)))


class BabyVNextLM(nn.Module):
    def __init__(self, config: BabyVNextConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = ExplicitLayerNorm(config.d_model, eps=config.norm_epsilon)
        self.language_head = nn.Linear(
            config.d_model, config.vocab_size, bias=config.lm_head_bias
        )

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        _, time = tokens.shape
        if time > self.config.context_length:
            raise ValueError(
                f"Baby vNext context is {self.config.context_length}, received {time}"
            )
        positions = torch.arange(time, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.language_head(self.forward_hidden(tokens))

    def set_attention_backend(self, backend: Literal["sdpa", "reference"]) -> None:
        if backend not in {"sdpa", "reference"}:
            raise ValueError("unsupported attention backend")
        for block in self.blocks:
            block.attention.backend = backend

