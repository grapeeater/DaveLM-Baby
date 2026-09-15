"""T32 causal ordered source-memory prefix model.

This module is a versioned extension of Baby vNext.  It leaves the historical
model untouched and adds only a zero-initialized 640-channel memory gate.  The
ordered identity mention states are exposed as an auxiliary K/V attention
branch of the final Transformer block; normal token-to-token attention is
unchanged and the auxiliary branch is masked to answer positions.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_vnext.binding import BabyVNextWithBinding


class T32MemoryModel(BabyVNextWithBinding):
    def __init__(self, config):
        super().__init__(config)
        # Exact parent neutrality: at zero this branch contributes identically
        # zero to the final block residual.  It remains differentiable because
        # the pre-gate memory signal is nonzero on QA rows.
        self.memory_gate = nn.Parameter(torch.zeros(config.d_model))

    @staticmethod
    def pointer_route(hidden: torch.Tensor, spans: list[list[int]], query_pos: int):
        q = F.normalize(hidden[0, query_pos], dim=-1)
        keys = []
        for span in spans:
            idx = torch.tensor(span, device=hidden.device, dtype=torch.long)
            keys.append(F.normalize(hidden[0, idx].mean(0), dim=-1))
        key = torch.stack(keys)
        sims = (key * q.unsqueeze(0)).sum(-1)
        weights = F.softmax(sims, dim=-1)
        retrieved = (weights.unsqueeze(-1) * key).sum(0)
        return sims, weights, retrieved

    def _memory_entries(self, x: torch.Tensor,
                        mention_spans: Sequence[Sequence[Sequence[int]]],
                        source_weights: torch.Tensor):
        """Gather ordered mention-token states and source-selection weights."""
        batch, _, dim = x.shape
        entries, weights = [], []
        for b in range(batch):
            row_entries, row_weights = [], []
            spans_b = mention_spans[b]
            for source_i, span in enumerate(spans_b):
                w = source_weights[b, source_i]
                for pos in span:
                    row_entries.append(x[b, int(pos)])
                    row_weights.append(w)
            entries.append(row_entries)
            weights.append(row_weights)
        kmax = max((len(r) for r in entries), default=0)
        if kmax == 0:
            return x.new_zeros((batch, 0, dim)), x.new_zeros((batch, 0)), torch.zeros((batch, 0), dtype=torch.bool, device=x.device)
        mem = x.new_zeros((batch, kmax, dim))
        mw = x.new_zeros((batch, kmax))
        valid = torch.zeros((batch, kmax), dtype=torch.bool, device=x.device)
        for b, row in enumerate(entries):
            if row:
                mem[b, :len(row)] = torch.stack(row)
                mw[b, :len(row)] = torch.stack(weights[b])
                valid[b, :len(row)] = True
        return mem, mw, valid

    def forward_hidden_with_memory(
        self,
        tokens: torch.Tensor,
        fact_spans: Sequence[Sequence[Sequence[int]]],
        mention_spans: Sequence[Sequence[Sequence[int]]],
        query_positions: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward with ordered source K/V available in the final block.

        `fact_spans` and `mention_spans` are TRAIN/protocol metadata.  The
        memory is built from block-10 source hidden states, then read by the
        final block only at positions marked by `answer_mask`.
        """
        if tokens.ndim != 2 or answer_mask.shape != tokens.shape:
            raise ValueError("tokens and answer_mask must be [batch,time]")
        batch, time = tokens.shape
        if time > self.config.context_length:
            raise ValueError("sequence exceeds context")
        pos = torch.arange(time, device=tokens.device)
        x = self.base_model.token_embedding(tokens) + self.base_model.position_embedding(pos)
        x = self.base_model.embedding_dropout(x)
        for block in self.base_model.blocks[:-1]:
            x = block(x)
        # Source states are intentionally taken before the final block.  This
        # makes the memory an input to native causal computation rather than a
        # post-head logit correction.
        source_hidden = x
        final_block = self.base_model.blocks[-1]
        normed = final_block.norm1(x)
        attn = final_block.attention
        q, _, _ = attn._split(normed)
        normal = attn(normed)

        pointer_weights = []
        for b in range(batch):
            spans = [list(map(int, s)) for s in fact_spans[b]]
            _, w, _ = self.pointer_route(source_hidden[b:b+1], spans, int(query_positions[b]))
            pointer_weights.append(w)
        pw = torch.stack(pointer_weights, dim=0)
        mem, mw, valid = self._memory_entries(source_hidden, mention_spans, pw)
        memory_stats = {
            "pointer_weights": pw.detach(),
            "memory_valid": valid.detach(),
        }
        if mem.shape[1] == 0:
            memory_out = torch.zeros_like(normal)
            memory_mass = torch.zeros((batch, time), device=tokens.device)
        else:
            mk, mv = attn.qkv(mem).chunk(3, dim=-1)[1:]
            shape = (batch, mem.shape[1], attn.n_heads, attn.head_dim)
            mk = mk.view(shape).transpose(1, 2)
            mv = mv.view(shape).transpose(1, 2)
            scores = torch.matmul(q, mk.transpose(-2, -1)) * (attn.head_dim ** -0.5)
            # Soft pointer selection is a log prior over source-span tokens.
            scores = scores + torch.log(mw.clamp_min(1e-9)).unsqueeze(1).unsqueeze(1)
            allow = answer_mask.unsqueeze(-1) & valid.unsqueeze(1)
            scores = scores.masked_fill(~allow.unsqueeze(1), float("-inf"))
            probs = F.softmax(scores, dim=-1)
            probs = torch.where(allow.unsqueeze(1), probs, torch.zeros_like(probs))
            attended = torch.matmul(probs, mv)
            merged = attended.transpose(1, 2).contiguous().view(batch, time, self.config.d_model)
            memory_out = attn.projection(merged)
            memory_out = memory_out * answer_mask.unsqueeze(-1).to(memory_out.dtype)
            memory_mass = probs.sum(dim=1).sum(dim=-1)
            memory_stats["memory_attention"] = probs.detach()
        gated = memory_out * self.memory_gate.view(1, 1, -1)
        x = x + final_block.attention_residual_dropout(normal + gated)
        x = x + final_block.feed_forward_residual_dropout(final_block.feed_forward(final_block.norm2(x)))
        return self.base_model.final_norm(x), {**memory_stats, "memory_mass": memory_mass.detach(), "memory_pre_gate_norm": memory_out.detach().norm(dim=-1)}

    def forward(self, input_ids: torch.Tensor, layout=None):
        # Preserve the historical binding interface for ordinary callers.
        return super().forward(input_ids, layout)
