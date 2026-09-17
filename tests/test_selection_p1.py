from __future__ import annotations

import pytest
import torch

from src.baby_v010.config import BabyVNextConfig
from src.baby_v010.model import BabyVNextLM
from src.baby_v010.selection_p1 import keyed_pointers, pointer_loss


def test_pointer_loss_is_zero_when_mass_is_one() -> None:
    batch, heads, time = 2, 4, 8
    weights = [torch.zeros(batch, heads, time, time) for _ in range(3)]
    for layer in weights:
        layer[0, 2, 7, 3] = 1.0
        layer[1, 0, 5, 1] = 1.0
    loss = pointer_loss(weights, [(0, 7, 3), (1, 5, 1)], eps=1e-8)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_pointer_loss_uses_max_over_heads_and_layers() -> None:
    weights = [torch.full((1, 2, 4, 4), 0.01), torch.full((1, 2, 4, 4), 0.01)]
    weights[1][0, 1, 3, 0] = 0.5
    loss = pointer_loss(weights, [(0, 3, 0)], eps=1e-8)
    assert float(loss) == pytest.approx(-torch.log(torch.tensor(0.5 + 1e-8)).item())


def test_keyed_pointers_skip_induction() -> None:
    items = [
        {"kind": "keyed", "input": [1, 2, 3, 4], "query_position": 1},
        {"kind": "induction", "input": [1, 2, 3, 4], "query_position": 1},
    ]
    assert keyed_pointers(items) == [(0, 3, 1)]


def test_attention_capture_matches_reference_logits() -> None:
    torch.manual_seed(0)
    model = BabyVNextLM(BabyVNextConfig()).eval()
    model.set_attention_backend("reference")
    tokens = torch.randint(0, model.config.vocab_size, (2, 12))
    baseline = model(tokens)
    captured = []
    model.set_attention_capture(captured)
    captured_out = model(tokens)
    model.set_attention_capture(None)
    assert torch.allclose(baseline, captured_out, atol=1e-5)
    assert len(captured) == model.config.n_layers
    assert captured[0].shape[0] == 2


def test_sdpa_path_ignores_empty_capture() -> None:
    torch.manual_seed(1)
    model = BabyVNextLM(BabyVNextConfig()).eval()
    model.set_attention_backend("sdpa")
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    a = model(tokens)
    model.set_attention_capture(None)
    b = model(tokens)
    assert torch.allclose(a, b, atol=1e-5)
