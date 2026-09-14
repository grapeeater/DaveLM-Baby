from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_vnext.binding import BabyVNextWithBinding


class T30RoutedModel(BabyVNextWithBinding):
    """Baby vNext with one minimal persistent pointer-source route.

    The parent model and all inherited binding/localizer parameters are unchanged.
    ``route_gate`` is the only new trainable parameter: a zero-initialized
    per-channel gate that scales the existing query-weighted fact-clause
    representation before adding it to answer-generation hidden states.
    """

    def __init__(self, config):
        super().__init__(config)
        self.route_gate = nn.Parameter(torch.zeros(config.d_model))

    @staticmethod
    def pointer_route(hidden: torch.Tensor, spans: list[list[int]], query_pos: int):
        """Return pointer logits, weights, and the weighted source representation."""
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

    def logits_with_route(
        self,
        hidden: torch.Tensor,
        retrieved: torch.Tensor,
        answer_start: int,
    ) -> torch.Tensor:
        """Apply the persistent route at all answer prediction positions."""
        if hidden.ndim != 3 or retrieved.ndim != 1:
            raise ValueError("hidden must be [B,T,D] and retrieved must be [D]")
        if not 0 <= answer_start < hidden.shape[1]:
            raise ValueError("answer_start outside hidden sequence")
        mask = torch.zeros(
            (hidden.shape[0], hidden.shape[1], 1), device=hidden.device, dtype=hidden.dtype
        )
        mask[:, answer_start:, :] = 1
        routed = hidden + mask * retrieved.view(1, 1, -1) * self.route_gate.view(1, 1, -1)
        return self.base_model.language_head(routed)

    def route_logits_for_row(self, hidden, row, spans, prompt_len, answer_start):
        sims, weights, retrieved = self.pointer_route(
            hidden, spans, prompt_len - 1
        )
        logits = self.logits_with_route(hidden, retrieved, answer_start)
        return logits, sims, weights, retrieved
