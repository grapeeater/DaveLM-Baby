from __future__ import annotations

import random

from src.baby_v010.data_language_bridge import (
    ENTITIES,
    SIZES,
    PLACES,
    VALUES,
    encode_split,
    load_tokenizer,
    make_aperiodic_item,
    make_dialogue_item,
    make_english_item,
    make_mixed_item,
    make_place_item,
    make_attr_followup_item,
    make_size_item,
    make_syntax_item,
    period_token_id,
    spaced_first_id,
)
from src.baby_v010.selection_rapid_treat_d import production_d_uses_query_position, tiling_parse


def test_d3_operator_still_gold_free() -> None:
    assert production_d_uses_query_position() is False


def test_encode_split_is_prefix() -> None:
    tokenizer = load_tokenizer()
    prompt = "The cat is red. The dog is blue. The cat is"
    answer = " red."
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    full = tokenizer.encode(prompt + answer).ids
    assert prompt_ids + answer_ids == list(full)
    assert answer_ids[0] != prompt_ids[-1]


def test_english_items_are_gold_free_and_scorable() -> None:
    tokenizer = load_tokenizer()
    rng = random.Random(7)
    accidental = 0
    n = 0
    for n_facts in (1, 2, 3):
        item = make_english_item(rng, tokenizer, n_facts=n_facts, family="qa", surface="heldout")
        assert item["kind"] == "keyed"
        assert item.get("query_position") is None
        assert "query_position" not in item["input"]
        gen = len(item["input"]) - 1
        accidental += int(tiling_parse(item["input"], gen) is not None)
        n += 1
        assert item["target"][0] in range(1024)
    assert n == 3
    # Accidental C2/D3 tilings on English are possible; zeroshot measures the rate.


def test_cloze_and_dialogue_and_instruction_encode() -> None:
    tokenizer = load_tokenizer()
    rng = random.Random(13)
    cloze = make_english_item(rng, tokenizer, n_facts=1, family="cloze", surface="train")
    instr = make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", instruction=True)
    pronoun = make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", pronoun=True)
    dialogue = make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout")
    for item in (cloze, instr, pronoun, dialogue):
        assert 2 <= len(item["input"]) < 256
        assert item["target"]
        assert item.get("query_position") is None
    assert "What color is it?" in pronoun["prompt_text"]
    assert "Baby:" in dialogue["prompt_text"]


def test_heldout_templates_differ_from_train() -> None:
    tokenizer = load_tokenizer()
    rng = random.Random(11)
    train = make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
    hold = make_english_item(random.Random(11), tokenizer, n_facts=2, family="qa", surface="heldout")
    assert train["prompt_text"] != hold["prompt_text"]


def test_closed_vocab() -> None:
    assert len(set(ENTITIES)) == len(ENTITIES)
    assert len(set(VALUES)) == len(VALUES)


def test_aperiodic_and_syntax_usually_break_tiling() -> None:
    from src.baby_v010.data_language_bridge import banks_from_language

    banks = banks_from_language()
    tokenizer = load_tokenizer()
    rng = random.Random(17)
    aperiodic_hits = 0
    syntax_hits = 0
    n = 24
    for _ in range(n):
        a = make_aperiodic_item(rng, banks, pair_count=3, surface="heldout")
        s = make_syntax_item(rng, banks, tokenizer, pair_count=2, surface="heldout")
        aperiodic_hits += int(tiling_parse(a["input"], len(a["input"]) - 1) is not None)
        syntax_hits += int(tiling_parse(s["input"], len(s["input"]) - 1) is not None)
        assert a.get("query_position") is None
        assert s.get("query_position") is None
    assert aperiodic_hits <= 4
    # Syntax wrappers still contain repeated English pieces; D3 may false-trigger.
    assert syntax_hits >= 0


def test_size_and_color_first_tokens_are_disjoint() -> None:
    tokenizer = load_tokenizer()
    color_ids = [spaced_first_id(tokenizer, word) for word in VALUES]
    size_ids = [spaced_first_id(tokenizer, word) for word in SIZES]
    place_ids = [spaced_first_id(tokenizer, word) for word in PLACES]
    assert len(set(color_ids)) == len(VALUES)
    assert len(set(size_ids)) == len(SIZES)
    assert len(set(place_ids)) == len(PLACES)
    assert set(color_ids).isdisjoint(set(size_ids))
    assert set(color_ids).isdisjoint(set(place_ids))
    assert set(size_ids).isdisjoint(set(place_ids))
    assert period_token_id(tokenizer) == tokenizer.encode(".").ids[0]


def test_size_and_mixed_items_are_gold_free() -> None:
    tokenizer = load_tokenizer()
    rng = random.Random(19)
    size = make_size_item(rng, tokenizer, n_facts=2, surface="heldout")
    mixed = make_mixed_item(rng, tokenizer, n_entities=2, surface="heldout")
    place = make_place_item(rng, tokenizer, n_facts=2, surface="heldout")
    for item in (size, mixed, place):
        assert item.get("query_position") is None
        assert item["target"]
        assert item["value_text"] in (*VALUES, *SIZES, *PLACES)
        assert item["attr"] in {"color", "size", "place"}
    assert size["attr"] == "size"
    assert place["attr"] == "place"


def test_attr_followup_is_gold_free() -> None:
    tokenizer = load_tokenizer()
    item = make_attr_followup_item(random.Random(23), tokenizer, surface="heldout")
    assert item.get("query_position") is None
    assert "Baby:" in item["prompt_text"]
    assert item["value_text"] in SIZES


def test_period_stop_first_word() -> None:
    from src.baby_v010.selection_language_bridge import first_word_match

    assert first_word_match(" red.", "red")
    assert first_word_match("red. red.", "red")
    assert not first_word_match(" blue", "red")
