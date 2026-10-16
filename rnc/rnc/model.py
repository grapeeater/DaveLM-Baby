"""RNC-1 model: decoder-only pre-norm transformer.

RMSNorm, rotary positions, QK-normalised causal attention, SwiGLU MLP, no biases, tied embedding/readout.
See docs/RNC_ARCHITECTURE.md for the reasoning behind every choice. There are no auxiliary heads, no task
routes and no hard-coded answers anywhere in this file.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, base: float, device, dtype=torch.float32):
    """cos/sin tables of shape (T, head_dim) for rotate-half RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D) -> broadcast over B, H. Computed in fp32 for accuracy.
    xf = x.float()
    out = xf * cos + rotate_half(xf) * sin
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.backend = "auto"  # "auto" -> SDPA; "math" -> SDPA math kernel; "reference" -> explicit fp32 softmax

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.backend == "reference":
            y = self._reference_attention(q, k, v)
        elif self.backend == "math":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    @staticmethod
    def _reference_attention(q, k, v):
        T = q.shape[-2]
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        return (F.softmax(scores, dim=-1) @ v.float()).to(q.dtype)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_mlp, bias=False)
        self.down = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)
        self.d_mlp = cfg.d_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).split(self.d_mlp, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class RNCModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None  # readout uses embed.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()
        self.register_buffer("_dummy", torch.zeros(0), persistent=False)

    # --- init -----------------------------------------------------------------------------------------------
    def _init_weights(self) -> None:
        std = self.cfg.init_std
        residual_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                s = residual_std if name.endswith(("attn.proj", "mlp.down")) else std
                nn.init.normal_(module.weight, mean=0.0, std=s)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    # --- utilities ------------------------------------------------------------------------------------------
    def set_attention_backend(self, backend: str) -> None:
        if backend not in ("auto", "math", "reference"):
            raise ValueError(backend)
        for b in self.blocks:
            b.attn.backend = backend

    def num_parameters(self) -> int:
        """Unique trainable parameters (tied readout counted once)."""
        return sum(p.numel() for p in self.parameters())

    def parameter_groups(self, weight_decay: float):
        """Decay 2-D matrices; no decay for embeddings, norms and any 1-D tensor."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and not name.startswith("embed."):
                decay.append(p)
            else:
                no_decay.append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    # --- forward --------------------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(f"sequence length {T} exceeds context_length {self.cfg.context_length}")
        x = self.embed(idx)
        positions = torch.arange(T, device=idx.device)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_base, idx.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        if self.lm_head is None:
            logits = F.linear(x, self.embed.weight)
        else:
            logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    # --- sampling -------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0,
                 eos_id: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        """Plain autoregressive sampling. No KV cache (context is short); nothing is injected or post-edited."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.context_length:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k < logits.size(-1):
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and idx.shape[0] == 1 and int(nxt.item()) == eos_id:
                break
        return idx


def analytic_parameter_count(cfg: ModelConfig) -> int:
    """Independent closed-form count used by the tests to cross-check the module."""
    d, L, V, M, H = cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.d_mlp, cfg.head_dim
    per_layer = 4 * d * d + 3 * d * M + 2 * d + (2 * H if cfg.qk_norm else 0)
    total = V * d + L * per_layer + d
    if not cfg.tie_embeddings:
        total += V * d
    return total
