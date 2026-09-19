"""S2 paired query-contrast unit tests. No training, no protected eval."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.baby_v010.selection_s2 import (
    CONTRAST_MARGIN,
    contrast_loss_from_candidate_logits,
    futility_at_200,
    inventory_rank,
)


def test_inventory_rank_counts_strictly_greater():
    assert inventory_rank([1.0, 4.0, 3.0], 0) == 3
    assert inventory_rank([1.0, 4.0, 3.0], 1) == 1
    assert inventory_rank([4.0, 4.0, 3.0], 0) == 1


def test_query_flip_has_near_zero_contrast_loss():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    query = torch.tensor([0, 1])
    loss = contrast_loss_from_candidate_logits(logits, query, margin=CONTRAST_MARGIN)
    assert float(loss) < 0.05


def test_query_invariant_collapse_keeps_high_contrast_loss():
    logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    query = torch.tensor([0, 1])
    loss = contrast_loss_from_candidate_logits(logits, query, margin=CONTRAST_MARGIN)
    expected = float(F.softplus(torch.tensor(CONTRAST_MARGIN)))
    assert abs(float(loss) - expected) < 1e-5


def test_parent_sized_residue_still_has_loss_at_margin_two():
    logits = torch.tensor([[0.4, 0.0], [0.0, 0.4]])
    query = torch.tensor([0, 1])
    loss = contrast_loss_from_candidate_logits(logits, query, margin=2.0)
    # effect = 0.8, softplus(2 - 0.8)
    expected = float(F.softplus(torch.tensor(1.2)))
    assert abs(float(loss) - expected) < 1e-5
    assert float(loss) > 1.0


def test_k4_all_pairs_average():
    logits = torch.eye(4) * 8.0
    query = torch.arange(4)
    loss = contrast_loss_from_candidate_logits(logits, query, margin=2.0)
    assert float(loss) < 0.05


def test_futility_requires_all_three_mechanism_failures():
    parent = {
        "summary": {
            "query_logit_effect": 0.78,
            "same_first_token_rate": 0.82,
            "queried_inventory_rank1": 0.38,
        }
    }
    dead = {
        "summary": {
            "query_logit_effect": 0.80,
            "same_first_token_rate": 0.81,
            "queried_inventory_rank1": 0.39,
        }
    }
    alive = {
        "summary": {
            "query_logit_effect": 1.00,
            "same_first_token_rate": 0.81,
            "queried_inventory_rank1": 0.39,
        }
    }
    assert futility_at_200(parent, dead) is True
    assert futility_at_200(parent, alive) is False


def test_s2_treatment_is_not_s1_first_token_ce():
    source = Path(__file__).resolve().parents[1].joinpath("src/baby_v010/selection_s2.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "")
    assert "keyed_contrast_loss" in source
    assert "CONTRAST_MARGIN = 2.0" in source
    assert "loss=loss+F.cross_entropy(z[list(ii),list(ss)]" not in compact
    assert "first-token vocabulary CE" not in source
