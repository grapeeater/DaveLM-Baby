from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BabyVNextConfig
from .model import BabyVNextLM


def orthogonal_basis(u: torch.Tensor) -> torch.Tensor:
    """Householder basis used by the authoritative Pilot-1 OrthoLocalizer."""
    unit = u / torch.linalg.vector_norm(u)
    pivot = int(torch.argmax(torch.abs(unit)))
    e = torch.zeros_like(unit)
    e[pivot] = 1.0 if unit[pivot] >= 0 else -1.0
    w = unit - e
    denominator = torch.dot(w, w)
    if float(denominator.detach().cpu()) == 0.0:
        raise RuntimeError("orthogonal localizer encountered a degenerate Householder vector")
    h = torch.eye(unit.numel(), device=u.device, dtype=u.dtype) - 2 * torch.outer(w, w) / denominator
    keep = [index for index in range(unit.numel()) if index != pivot]
    return h[:, keep]


class OrthogonalTwoSlotLocalizer(nn.Module):
    """Dimension-parameterized form of the proven two-slot localizer."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.u = nn.Parameter(torch.empty(d_model))
        self.q = nn.Parameter(torch.empty(d_model - 1))
        self.bs = nn.Parameter(torch.zeros(()))
        self.ba = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.u, mean=0.0, std=d_model ** -0.5)
        nn.init.normal_(self.q, mean=0.0, std=d_model ** -0.5)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        v = orthogonal_basis(self.u) @ self.q
        shared = hidden @ self.u + self.bs
        antisymmetric = hidden @ v + self.ba
        return torch.stack((shared + antisymmetric, shared - antisymmetric), dim=-1)


@dataclass(frozen=True)
class BindingLayout:
    """Explicit task-to-model geometry; no row offset is embedded in the model."""

    query_positions: torch.Tensor
    answer_positions: torch.Tensor
    key_positions: torch.Tensor
    value_positions: torch.Tensor
    valid_candidates: torch.Tensor

    def validate(self, batch: int, time: int) -> None:
        if self.query_positions.shape != (batch,) or self.answer_positions.shape != (batch,):
            raise ValueError("query_positions and answer_positions must have shape [batch]")
        if self.key_positions.ndim != 2 or self.key_positions.shape[0] != batch:
            raise ValueError("key_positions must have shape [batch, candidates]")
        if self.value_positions.shape != self.key_positions.shape:
            raise ValueError("value_positions must match key_positions")
        if self.valid_candidates.shape != self.key_positions.shape:
            raise ValueError("valid_candidates must match key_positions")
        if self.valid_candidates.dtype != torch.bool:
            raise ValueError("valid_candidates must be boolean")
        if not bool(self.valid_candidates.any(dim=1).all()):
            raise ValueError("every document needs at least one valid candidate")
        for positions in (self.query_positions, self.answer_positions, self.key_positions, self.value_positions):
            if int(positions.min()) < 0 or int(positions.max()) >= time:
                raise ValueError("binding positions are outside the sequence")


def legacy_t13_layout(
    query_positions: torch.Tensor,
    answer_positions: torch.Tensor,
    time: int,
    *,
    value_offset: int = 4,
    candidate_start: int = 1,
) -> BindingLayout:
    """Historical synthetic-format adapter; the core model never assumes offset=4."""
    batch = query_positions.shape[0]
    candidate_count = time - candidate_start - value_offset
    if candidate_count <= 0:
        raise ValueError("sequence is too short for the requested legacy layout")
    base = torch.arange(
        candidate_start,
        candidate_start + candidate_count,
        device=query_positions.device,
    )
    keys = base.unsqueeze(0).expand(batch, -1)
    values = keys + value_offset
    valid = (keys < query_positions.unsqueeze(1)) & (values < time)
    return BindingLayout(query_positions, answer_positions, keys, values, valid)


class BabyVNextWithBinding(nn.Module):
    def __init__(self, config: BabyVNextConfig) -> None:
        super().__init__()
        self.config = config
        self.base_model = BabyVNextLM(config)
        self.localizer = OrthogonalTwoSlotLocalizer(config.d_model)
        retrieval_dim = config.binding.retrieval_dim
        self.wq = nn.Linear(config.d_model, retrieval_dim, bias=False)
        self.wk = nn.Linear(config.d_model, retrieval_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        self.scale = retrieval_dim ** -0.5

    @staticmethod
    def _gather(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_index = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
        return hidden[batch_index, positions]

    def forward(
        self, input_ids: torch.Tensor, layout: BindingLayout | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.base_model.forward_hidden(input_ids)
        base_logits = self.base_model.language_head(hidden)
        if layout is None:
            return base_logits, {}
        batch, time, _ = hidden.shape
        layout.validate(batch, time)
        key_hidden = self._gather(hidden, layout.key_positions)
        value_hidden = self._gather(hidden, layout.value_positions)
        localizer_logits = self.localizer(key_hidden)
        localizer_logits = localizer_logits.masked_fill(
            ~layout.valid_candidates.unsqueeze(-1), float("-inf")
        )
        localization_attention = F.softmax(localizer_logits, dim=1)
        attention_t = localization_attention.transpose(1, 2)
        source_slots = torch.matmul(attention_t, key_hidden)
        value_slots = torch.matmul(attention_t, value_hidden)
        rows = torch.arange(batch, device=hidden.device)
        query = hidden[rows, layout.query_positions]
        query_projection = self.wq(query)
        key_projection = self.wk(source_slots)
        slot_scores = (query_projection.unsqueeze(1) * key_projection).sum(-1) * self.scale
        row_weights = F.softmax(slot_scores, dim=-1)
        projected_values = self.wv(value_slots)
        retrieved = (row_weights.unsqueeze(-1) * projected_values).sum(dim=1)
        answer_hidden = hidden[rows, layout.answer_positions] + self.wo(retrieved)
        answer_logits = self.base_model.language_head(answer_hidden)
        output = base_logits.clone()
        output[rows, layout.answer_positions] = answer_logits
        return output, {
            "localization_scores": localizer_logits,
            "localization_attention": localization_attention,
            "slot_scores": slot_scores,
            "row_weights": row_weights,
            "retrieved": retrieved,
            "valid_candidate_counts": layout.valid_candidates.sum(-1),
        }

