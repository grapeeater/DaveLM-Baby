from __future__ import annotations

from src.baby_v010.query_splice_d3b import decide, filler_position, pair_index_at_start


def _item() -> dict:
    return {
        "kind": "keyed",
        "input": [2, 701, 210, 211, 90, 702, 220, 221, 90, 701, 800],
        "query_key": 701,
        "query_index": 0,
        "query_position": 9,
        "pair_count": 2,
        "candidate_heads": [210, 220],
        "source": [701, 210, 211, 702, 220, 221],
        "target": [210, 211, 90, 3],
        "target_span": [210, 211],
        "value_length": 2,
        "variant": "keyed_0",
    }


def test_pair_index_and_filler() -> None:
    item = _item()
    sources = {"query": 9, "match_key": 1, "competitor_key": 5}
    assert pair_index_at_start(item, 1) == 0
    assert pair_index_at_start(item, 5) == 1
    fill = filler_position(item, sources)
    assert fill is not None
    assert fill not in {1, 5, 9, 10}


def test_decide_replication_invalid() -> None:
    out = decide(
        full_as_hits=73,
        full_q0r_hits=127,
        n_shared=200,
        g_as=0.34,
        g_q=0.59,
        g_c=0.34,
        g_r=0.34,
        s_as=0.30,
        s_c=0.30,
    )
    assert out["verdict"] == "INVALID"


def test_decide_query_identity() -> None:
    out = decide(
        full_as_hits=74,
        full_q0r_hits=127,
        n_shared=200,
        g_as=0.34,
        g_q=0.59,
        g_c=0.35,
        g_r=0.34,
        s_as=0.30,
        s_c=0.31,
    )
    assert out["verdict"] == "QUERY_IDENTITY"


def test_decide_spliced_identity() -> None:
    out = decide(
        full_as_hits=74,
        full_q0r_hits=127,
        n_shared=200,
        g_as=0.34,
        g_q=0.59,
        g_c=0.35,
        g_r=0.34,
        s_as=0.30,
        s_c=0.55,
    )
    assert out["verdict"] == "SPLICED_IDENTITY"


def test_decide_key_subspace() -> None:
    out = decide(
        full_as_hits=74,
        full_q0r_hits=127,
        n_shared=200,
        g_as=0.34,
        g_q=0.59,
        g_c=0.55,
        g_r=0.35,
        s_as=0.30,
        s_c=0.32,
    )
    assert out["verdict"] == "KEY_SUBSPACE"


def test_decide_overwrite_precedes() -> None:
    out = decide(
        full_as_hits=74,
        full_q0r_hits=127,
        n_shared=200,
        g_as=0.34,
        g_q=0.59,
        g_c=0.35,
        g_r=0.50,
        s_as=0.30,
        s_c=0.30,
    )
    assert out["verdict"] == "OVERWRITE"
