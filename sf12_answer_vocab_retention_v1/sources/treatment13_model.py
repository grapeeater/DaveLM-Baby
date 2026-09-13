r"""Treatment-13 model skeleton: learned mapping-row localization + retrieval.

T13 removes the T12 source-row scaffold. It is still handed the final query-key
position and the answer prediction position (grammar public). It is NOT handed
mapping-row source/value positions.

Mechanism (smallest clean intervention):
1. A tiny learned localizer scores every token position up to (but not at/after)
   the query as a potential mapping source-key position, producing two row-slot
   attention distributions (slot 0, slot 1) over candidate positions.
2. For each row slot the source-key representation is the slot-weighted mean of
   hidden states at the localized positions; the row's value representation is
   the slot-weighted mean of hidden states at position + ROW_VALUE_OFFSET (a
   public grammar constant), which pairs a localized key with its row-local value
   without revealing which row matches the query.
3. T12-style query-conditioned retrieval then scores the two localized row-slot
   keys against the final query key and retrieves the weighted value mixture into
   the answer hidden state.

No correct-row label, target token id, or candidate identity is supplied to the
forward pass. Preflight/smoke only - no training.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from treatment13_config import EMBED_SIZE, RETRIEVAL_DIM, ROW_VALUE_OFFSET, VOCAB_SIZE

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from v0_8_2.model import build_model  # authoritative base builder


class LearnedLocalizer(nn.Module):
    """Learned source-key position scorer for two row slots."""

    def __init__(self) -> None:
        super().__init__()
        self.scorer = nn.Linear(EMBED_SIZE, 2, bias=True)

    def forward(self, hidden):
        # hidden: [B,T,H]
        return self.scorer(hidden)  # [B,T,2]


class Treatment13Model(nn.Module):
    def __init__(self, base_model: Optional[nn.Module] = None) -> None:
        super().__init__()
        if base_model is None:
            base_model = build_model("untied")
        self.base_model = base_model
        self.localizer = LearnedLocalizer()
        self.wq = nn.Linear(EMBED_SIZE, RETRIEVAL_DIM, bias=False)
        self.wk = nn.Linear(EMBED_SIZE, RETRIEVAL_DIM, bias=False)
        self.wv = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.wo = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.scale = RETRIEVAL_DIM ** -0.5
        self._captured: Optional[torch.Tensor] = None
        self._handle = base_model.final_norm.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._captured = output
        return None

    def forward(self, input_ids, qpos, anspos, localizer_logits: Optional[torch.Tensor] = None):
        """input_ids [B,T]; qpos/anspos int tensors [B].

        Returns (logits[B,T,V] with answer position replaced by the
        localized-retrieval answer, extras dict).
        """
        self._captured = None
        base_logits = self.base_model(input_ids)
        hidden = self._captured
        if hidden is None:
            raise RuntimeError("final_norm hook did not capture hidden states")
        hidden = hidden.float()
        B, T, H = hidden.shape
        device = hidden.device
        rows = torch.arange(B, device=device)

        n_cand = T - 1 - ROW_VALUE_OFFSET  # candidates whose +offset value stays in-range
        pos = torch.arange(1, 1 + n_cand, device=device)  # candidate hidden positions
        valid = (pos.view(1, -1).expand(B, n_cand) < qpos.unsqueeze(1)) & \
                (pos.view(1, -1).expand(B, n_cand) + ROW_VALUE_OFFSET < T)

        if localizer_logits is None:
            loc_logits = self.localizer(hidden[:, 1:1 + n_cand, :])  # [B,n_cand,2]
        else:
            loc_logits = localizer_logits
        loc_logits = loc_logits.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        loc_att = F.softmax(loc_logits, dim=1)  # [B,n_cand,2]

        h_cand = hidden[:, 1:1 + n_cand, :]                # [B,n_cand,H]
        h_value = hidden[:, 1 + ROW_VALUE_OFFSET:1 + ROW_VALUE_OFFSET + n_cand, :]
        loc_att_t = loc_att.transpose(1, 2)                # [B,2,n_cand]
        src_slots = torch.matmul(loc_att_t, h_cand)        # [B,2,H]
        val_slots = torch.matmul(loc_att_t, h_value)       # [B,2,H]

        q = hidden[rows, qpos]               # [B,H]
        q_proj = self.wq(q)                  # [B,d]
        k_proj = self.wk(src_slots)          # [B,2,d]
        scores = (q_proj.unsqueeze(1) * k_proj).sum(-1) * self.scale  # [B,2]
        alpha = F.softmax(scores, dim=-1)    # [B,2]
        v_proj = self.wv(val_slots)          # [B,2,H]
        retrieved = (alpha.unsqueeze(-1) * v_proj).sum(dim=1)  # [B,H]

        ans_hidden = hidden[rows, anspos]
        augmented = ans_hidden + self.wo(retrieved)
        answer_logits = self.base_model.language_head(augmented)  # [B,V]

        out_logits = base_logits.clone()
        out_logits[rows, anspos] = answer_logits
        extras = {
            "localization_scores": loc_logits,
            "localization_attention": loc_att,
            "slot_scores": scores,
            "row_weights": alpha,
            "retrieved": retrieved,
            "valid_candidate_counts": valid.sum(-1),
        }
        return out_logits, extras

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def base_model_parameter_count(self) -> int:
        return sum(p.numel() for p in self.base_model.parameters())

    def new_parameter_count(self) -> int:
        count = sum(p.numel() for p in self.localizer.parameters())
        count += (self.wq.weight.numel() + self.wk.weight.numel()
                  + self.wv.weight.numel() + self.wo.weight.numel())
        return count
