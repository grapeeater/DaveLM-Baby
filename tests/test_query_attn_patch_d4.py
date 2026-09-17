from __future__ import annotations

import torch

from src.baby_v010.query_attn_patch_d4 import cosine_writes, decide


def _base(**overrides):
    payload = dict(
        full_as_hits=74,
        full_q0r_hits=127,
        steer_on_miss=0.96,
        n=215,
        g_as=74 / 215,
        g_q=74 / 215,
        g_c=74 / 215,
        g_p=74 / 215,
        s_as=0.30,
        s_c=0.30,
        c_as=0.04,
        c_q=0.04,
    )
    payload.update(overrides)
    return decide(**payload)


def test_decide_invalid_replication() -> None:
    assert _base(full_as_hits=73) == "INVALID"
    assert _base(steer_on_miss=0.5) == "INVALID"


def test_decide_pointer_sufficient() -> None:
    assert (
        _base(g_q=0.55, s_c=0.45, c_q=0.45, g_c=0.35, g_p=0.34) == "POINTER_SUFFICIENT"
    )


def test_decide_ov_dead() -> None:
    assert _base() == "OV_DEAD"


def test_decide_prev_token_precedence() -> None:
    assert _base(g_p=0.55, g_q=0.55, c_q=0.5) == "PREV_TOKEN"


def test_decide_writes_only() -> None:
    assert _base(c_q=0.5) == "POINTER_WRITES_ONLY"


def test_cosine_writes_gain() -> None:
    assert cosine_writes(0.15, 0.04)
    assert not cosine_writes(0.08, 0.04)


def test_onehot_row_is_delta() -> None:
    weights = torch.zeros(1, 2, 5, 5)
    weights[..., :] = 0.2
    gen, source = 4, 1
    weights[:, :, gen, :] = 0.0
    weights[:, :, gen, source] = 1.0
    assert float(weights[0, 0, gen].sum()) == 1.0
    assert float(weights[0, 0, gen, source]) == 1.0
    assert float(weights[0, 0, 3].sum()) == 1.0
