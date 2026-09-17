"""Identity-value gated residual overwrite (P5)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityResidualOverwrite(nn.Module):
    """read = attn @ x (identity V); x <- x + gate * (read - x)."""

    def __init__(self, d_model: int, gate_bias: float = -4.0, qk_init: str = "zero") -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, 1)
        if qk_init == "zero":
            nn.init.zeros_(self.q.weight)
            nn.init.zeros_(self.k.weight)
        elif qk_init == "xavier":
            nn.init.xavier_uniform_(self.q.weight)
            nn.init.xavier_uniform_(self.k.weight)
        else:
            raise ValueError(f"unknown qk_init {qk_init}")
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))
        self.last_attn: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_hidden: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, time, width = hidden.shape
        scores = torch.matmul(self.q(hidden), self.k(hidden).transpose(-2, -1)) * (width ** -0.5)
        causal = torch.ones((time, time), dtype=torch.bool, device=hidden.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        read = torch.matmul(attn, hidden)
        gate = torch.sigmoid(self.gate(hidden))
        self.last_attn = attn
        self.last_gate = gate.squeeze(-1)
        rewritten = hidden + gate * (read - hidden)
        self.last_hidden = rewritten
        return rewritten


def attach_overwrite(model, module: IdentityResidualOverwrite):
    bucket: dict = {}

    def hook(_block, _inputs, output):
        rewritten = module(output)
        bucket["h0"] = rewritten
        return rewritten

    handle = model.blocks[0].register_forward_hook(hook)
    return handle, bucket


def pointer_aux(attn, gate, specs: list[tuple[int, int, int]], *, use_gate: bool, eps: float = 1e-8):
    if not specs:
        zero = attn.new_zeros(())
        return zero, zero
    ptr = []
    gate_terms = []
    for batch_i, gen_pos, query_pos in specs:
        ptr.append(-torch.log(attn[batch_i, gen_pos, query_pos] + eps))
        gate_terms.append((1.0 - gate[batch_i, gen_pos]) ** 2)
    ptr_loss = torch.stack(ptr).mean()
    gate_loss = torch.stack(gate_terms).mean() if use_gate else ptr_loss.new_zeros(())
    return ptr_loss, gate_loss
