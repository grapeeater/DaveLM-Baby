from __future__ import annotations

import pytest

from src.baby_v010.query_presence import (
    adjudicate,
    assert_query_only_twins,
    attention_index_masses,
    cosine_l2,
    directed_pairs,
    flip_toward_donor,
    loc_class,
    skip_identity,
    twin_diffs,
)


def _item(query_index: int, last: str = "marker") -> dict:
    body = [701, 210, 211, 90, 702, 220, 221, 90]
    if last == "query":
        inp = [2, *body, 701 + query_index]
        query_position = len(inp) - 1
    elif last == "prev":
        inp = [2, *body, 701 + query_index, 802]
        query_position = len(inp) - 2
    else:
        inp = [2, *body, 701 + query_index, 750, 751]
        query_position = len(inp) - 3
    return {
        "input": inp,
        "query_key": 701 + query_index,
        "query_index": query_index,
        "query_position": query_position,
        "pair_count": 2,
        "body_id": "T0",
        "candidate_heads": [210, 220],
        "target_span": [210, 211] if query_index == 0 else [220, 221],
        "source": [701, 210, 211, 702, 220, 221],
        "target": [210, 211, 90, 3] if query_index == 0 else [220, 221, 90, 3],
    }


def test_loc_class_last_prev_deeper() -> None:
    assert loc_class(_item(0, "query")) == "last_is_query"
    assert loc_class(_item(0, "prev")) == "prev_is_query"
    assert loc_class(_item(0, "deeper")) == "query_deeper"


def test_twins_differ_only_at_query_position() -> None:
    a, b = _item(0, "prev"), _item(1, "prev")
    a["input"] = list(a["input"])
    b["input"] = list(a["input"])
    b["input"][a["query_position"]] = 702
    b["query_key"] = 702
    b["query_index"] = 1
    assert twin_diffs(a, b) == [a["query_position"]]
    assert_query_only_twins([a, b])


def test_assert_twins_rejects_extra_diff() -> None:
    a, b = _item(0, "prev"), _item(1, "prev")
    b["input"] = list(a["input"])
    b["input"][a["query_position"]] = 702
    b["input"][1] = 999
    b["query_key"] = 702
    b["query_index"] = 1
    with pytest.raises(RuntimeError):
        assert_query_only_twins([a, b])


def test_directed_pairs_are_k_times_k_minus_one() -> None:
    group = [_item(0, "prev"), _item(1, "prev")]
    group[1]["query_index"] = 1
    pairs = directed_pairs(group)
    assert len(pairs) == 2
    assert { (p[0]["query_index"], p[1]["query_index"]) for p in pairs } == {(0, 1), (1, 0)}


def test_attention_mass_tracks_query_at_five_times_uniform() -> None:
    gen_pos = 9
    uniform = 1.0 / 10
    row = [uniform] * 10
    row[3] = 5 * uniform
    masses = attention_index_masses(
        row, gen_pos=gen_pos, query_pos=3, body_keys=[1, 5], queried_body=1
    )
    assert masses["tracks_query"] is True
    assert masses["query_slot"] == pytest.approx(5 * uniform)


def test_attention_mass_does_not_track_at_uniform() -> None:
    row = [0.1] * 10
    masses = attention_index_masses(
        row, gen_pos=9, query_pos=3, body_keys=[1], queried_body=1
    )
    assert masses["tracks_query"] is False


def test_cosine_identical_and_orthogonal() -> None:
    same = cosine_l2([1.0, 0.0], [1.0, 0.0])
    assert same["cosine"] == pytest.approx(1.0)
    assert same["max_abs"] == pytest.approx(0.0)
    ortho = cosine_l2([1.0, 0.0], [0.0, 1.0])
    assert ortho["cosine"] == pytest.approx(0.0)


def test_skip_identity_and_flip_helpers() -> None:
    assert skip_identity(1e-6) is True
    assert skip_identity(1e-4) is False
    assert flip_toward_donor(1, 0, 0) is True
    assert flip_toward_donor(0, 0, 0) is False
    assert flip_toward_donor(1, 2, 0) is False


def _base_report(**overrides) -> dict:
    long_flip_block = [0.0] * 12
    long_flip_attn = [0.0] * 12
    report = {
        "preflight": {
            "parent_sha_match": True,
            "diagnostic_sha_match": True,
            "protected_material_opened": False,
            "twins_query_only": True,
            "sdpa_reference_max_abs": 1e-5,
        },
        "positive_control": {
            "median_full_logit_cosine": 0.80,
            "flip_toward_donor_final": 0.40,
            "flip_toward_donor_best_block": 0.40,
        },
        "long_gap": {
            "median_final_residual_cosine": 0.9999,
            "median_full_logit_cosine": 0.9995,
            "flip_toward_donor_final": 0.0,
            "flip_toward_donor_block": long_flip_block,
            "flip_toward_donor_attn": long_flip_attn,
            "fraction_rows_any_query_tracking_head": 0.0,
        },
    }
    report.update(overrides)
    return report


def test_adjudicate_supports_B() -> None:
    decision = adjudicate(_base_report())
    assert decision["valid"] is True
    assert decision["verdict"] == "B"
    assert decision["hypothesis"]["H_B"] is True


def test_adjudicate_supports_A() -> None:
    report = _base_report()
    report["long_gap"]["flip_toward_donor_final"] = 0.40
    report["long_gap"]["median_full_logit_cosine"] = 0.70
    decision = adjudicate(report)
    assert decision["verdict"] == "A"


def test_adjudicate_supports_composition() -> None:
    report = _base_report()
    report["long_gap"]["fraction_rows_any_query_tracking_head"] = 0.80
    report["long_gap"]["flip_toward_donor_final"] = 0.02
    decision = adjudicate(report)
    assert decision["verdict"] == "COMPOSITION"


def test_identity_final_patch_is_a_noop_on_cpu() -> None:
    import torch

    from src.baby_v010.config import BabyVNextConfig
    from src.baby_v010.model import BabyVNextLM
    from src.baby_v010.query_presence_trace import patched_logits

    torch.manual_seed(0)
    model = BabyVNextLM(BabyVNextConfig()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 16))
    baseline = model(tokens)
    hidden = model.forward_hidden(tokens)
    patched = patched_logits(model, tokens, 15, "final", None, hidden[0, 15])
    assert torch.allclose(baseline[0, 15], patched[0, 15], atol=1e-5)


def test_adjudicate_invalid_without_positive_control() -> None:
    report = _base_report()
    report["positive_control"]["flip_toward_donor_final"] = 0.01
    report["positive_control"]["flip_toward_donor_best_block"] = 0.01
    decision = adjudicate(report)
    assert decision["valid"] is False
    assert decision["verdict"] == "INVALID"
    assert "positive_flip" in decision["invalid_reasons"]
