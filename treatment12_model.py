r"""Treatment-12 model: Baby + minimal query-conditioned mapping retrieval.

Architecture-only intervention over T11. The base DaveLMV082("untied") model is
unchanged. A small retrieval module reads structural token positions from the
document grammar (allowed) and, at the answer prediction position, scores the
two mapping-row source keys against the final query key, retrieves the weighted
row-value representation, and injects it into the answer hidden state with one
residual path before the existing language head.

Anti-cheating: the module never receives the target row identity, orientation,
correct candidate index, or target token id. Row/value position tokens are read
for the metric "correct row" only after the forward pass, never into the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from treatment12_config import EMBED_SIZE, RETRIEVAL_DIM

import sys
from pathlib import Path

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from v0_8_2.model import build_model  # authoritative frozen base builder


class QueryConditionedMappingRetrieval(nn.Module):
    """Row retrieval head: scores the two source keys against the query key.

    Params: Wq, Wk (query/source projection to d), Wv (value projection), Wo
    (residual readout into hidden size). No row label is used.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wq = nn.Linear(EMBED_SIZE, RETRIEVAL_DIM, bias=False)
        self.wk = nn.Linear(EMBED_SIZE, RETRIEVAL_DIM, bias=False)
        self.wv = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.wo = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.scale = RETRIEVAL_DIM ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same init distribution as the rest of the model; no pretrained value.
        for module in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, q, k0, k1, v0, v1):
        """q/k/v are [B, EMBED] final-norm hidden vectors at structural positions."""
        q_proj = self.wq(q)
        k0_proj = self.wk(k0)
        k1_proj = self.wk(k1)
        score0 = (q_proj * k0_proj).sum(-1) * self.scale
        score1 = (q_proj * k1_proj).sum(-1) * self.scale
        scores = torch.stack([score0, score1], dim=-1)  # [B,2]
        alpha = F.softmax(scores, dim=-1)  # [B,2]
        retrieved = alpha[:, 0:1] * self.wv(v0) + alpha[:, 1:2] * self.wv(v1)
        return scores, alpha, retrieved


def doc_retrieval_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve retrieval positions/tokens for one document from the grammar.

    Returns positions and expected token ids; used by trainer/audit and by the
    preflight. Row indices follow mapping-line order (row0 = first line).
    """
    ids = doc["full_document_token_ids"]
    query_slot = int(doc["query_slot"])
    query_key_token = int(doc["query_key_token"])
    qdp = int(doc["qdp"])
    ans_index = int(doc["answer_token_index"])
    ans_causal = ans_index - 1
    q_row_src = int(doc["query_key_clause_pos"])
    q_row_val = int(doc["target_clause_pos"])
    o_row_src = int(doc["other_key_pos"])
    o_row_val = int(doc["distractor_value_pos"])

    if query_slot == 0:
        row0_src, row0_val = q_row_src, q_row_val
        row1_src, row1_val = o_row_src, o_row_val
    else:
        row0_src, row0_val = o_row_src, o_row_val
        row1_src, row1_val = q_row_src, q_row_val

    return {
        "doc_id": doc["doc_id"],
        "query_slot": query_slot,
        "qdp": qdp,
        "answer_causal_pos": ans_causal,
        "answer_token_index": ans_index,
        "row0_src_pos": row0_src,
        "row0_val_pos": row0_val,
        "row1_src_pos": row1_src,
        "row1_val_pos": row1_val,
        "row0_src_token": int(ids[row0_src]),
        "row0_val_token": int(ids[row0_val]),
        "row1_src_token": int(ids[row1_src]),
        "row1_val_token": int(ids[row1_val]),
        "query_token_at_qdp": int(ids[qdp]),
        "answer_token_at_ans_index": int(ids[ans_index]),
        "query_key_token": query_key_token,
        "target_value_token": int(doc["target_value_token"]),
        "distractor_value_token": int(doc["distractor_value_token"]),
    }


def correct_row_index(meta: Dict[str, Any]) -> int:
    if meta["row0_src_token"] == meta["query_key_token"]:
        return 0
    if meta["row1_src_token"] == meta["query_key_token"]:
        return 1
    raise ValueError("query key does not match either mapping-row source token")


class Treatment12Model(nn.Module):
    """Untied Baby plus the single answer-position retrieval injection."""

    def __init__(self, base_model: Optional[nn.Module] = None) -> None:
        super().__init__()
        if base_model is None:
            base_model = build_model("untied")
        self.base_model = base_model
        self.retrieval = QueryConditionedMappingRetrieval()
        self._captured_hidden: Optional[torch.Tensor] = None
        self._handle = base_model.final_norm.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._captured_hidden = output
        return None

    def forward(self, input_ids, positions):
        """positions: dict of int tensors [B] with retrieval/answer positions.

        Returns (logits[B,T,V] with augmented answer position,
                 extras dict for logging only).
        """
        self._captured_hidden = None
        base_logits = self.base_model(input_ids)
        hidden = self._captured_hidden
        if hidden is None:
            raise RuntimeError("final_norm hook did not capture hidden states")
        hidden = hidden.float()
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)

        def gather(key: str) -> torch.Tensor:
            return hidden[rows, positions[key].long()]  # [B, EMBED]

        q = gather("q")
        k0 = gather("k0")
        k1 = gather("k1")
        v0 = gather("v0")
        v1 = gather("v1")
        answer_hidden = gather("answer")  # hidden at answer causal position

        scores, alpha, retrieved = self.retrieval(q, k0, k1, v0, v1)
        augmented = answer_hidden + self.retrieval.wo(retrieved)
        answer_logits = self.base_model.language_head(augmented)  # [B,V]

        out_logits = base_logits.clone()
        out_logits[rows, positions["answer"].long()] = answer_logits

        extras = {
            "retrieval_scores": scores,
            "retrieval_weights": alpha,
            "retrieved": retrieved,
        }
        return out_logits, extras

    def remove_hook(self) -> None:
        self._handle.remove()
