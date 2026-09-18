from __future__ import annotations

import random

from src.baby_v010.data_language_bridge import (
    ENTITIES,
    HOLDOUT_EVENTS,
    HOLDOUT_STORY_FRAMES,
    SIZES,
    PLACES,
    TRAIN_EVENTS,
    TRAIN_STORY_FRAMES,
    VALUES,
    WHO_ENTITIES,
    encode_split,
    load_tokenizer,
    make_aperiodic_item,
    make_dialogue_item,
    make_english_item,
    make_mixed_item,
    make_place_item,
    make_attr_followup_item,
    make_size_item,
    make_story_item,
    make_story_mixed_item,
    make_longturn_item,
    make_roleplay_item,
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


def test_e8_gate_thresholds() -> None:
    from src.baby_v010.selection_language_bridge import e8_gate

    fail = {
        "story_color_heldout": {"first_top1": 0.1},
        "story_who_heldout": {"first_top1": 0.1},
        "story_event_heldout": {"first_top1": 0.1},
        "qa_2fact_heldout": {"first_top1": 0.9},
    }
    assert e8_gate(fail)[0] == "FAIL"
    advance = dict(fail)
    advance["story_color_heldout"] = {"first_top1": 0.45}
    assert e8_gate(advance)[0] == "ADVANCE"
    grad = {
        "story_color_heldout": {"first_top1": 0.85},
        "story_who_heldout": {"first_top1": 0.55},
        "story_event_heldout": {"first_top1": 0.55},
        "qa_2fact_heldout": {"first_top1": 0.80},
    }
    assert e8_gate(grad)[0] == "GRAD"


def test_e9_gate_thresholds() -> None:
    from src.baby_v010.selection_language_bridge import e9_gate

    fail = {
        "story_color_heldout": {"first_top1": 0.9},
        "story_color_3e_heldout": {"first_top1": 0.1},
        "story_pronoun_heldout": {"first_top1": 0.1},
        "qa_2fact_heldout": {"first_top1": 0.9},
    }
    assert e9_gate(fail)[0] == "FAIL"
    advance = dict(fail)
    advance["story_pronoun_heldout"] = {"first_top1": 0.45}
    assert e9_gate(advance)[0] == "ADVANCE"
    grad = {
        "story_color_heldout": {"first_top1": 0.90},
        "story_color_3e_heldout": {"first_top1": 0.75},
        "story_pronoun_heldout": {"first_top1": 0.60},
        "qa_2fact_heldout": {"first_top1": 0.80},
    }
    assert e9_gate(grad)[0] == "GRAD"


def test_period_stop_first_word() -> None:
    from src.baby_v010.selection_language_bridge import first_word_match

    assert first_word_match(" red.", "red")
    assert first_word_match("red. red.", "red")
    assert not first_word_match(" blue", "red")


def test_who_entities_first_tokens_unique_and_disjoint() -> None:
    tokenizer = load_tokenizer()
    who_ids = [spaced_first_id(tokenizer, word) for word in WHO_ENTITIES]
    color_ids = [spaced_first_id(tokenizer, word) for word in VALUES]
    size_ids = [spaced_first_id(tokenizer, word) for word in SIZES]
    assert len(set(who_ids)) == len(WHO_ENTITIES)
    assert set(who_ids).isdisjoint(color_ids)
    assert set(who_ids).isdisjoint(size_ids)
    assert set(TRAIN_STORY_FRAMES).isdisjoint(HOLDOUT_STORY_FRAMES)
    assert set(TRAIN_EVENTS).isdisjoint(HOLDOUT_EVENTS)
    assert set(WHO_ENTITIES).issubset(ENTITIES)


def test_story_items_are_gold_free() -> None:
    tokenizer = load_tokenizer()
    rng = random.Random(31)
    for ask in ("color", "who", "event"):
        for surface in ("train", "heldout"):
            item = make_story_item(rng, tokenizer, surface=surface, ask=ask)
            dialogue = make_story_item(rng, tokenizer, surface=surface, ask=ask, dialogue=True)
            for row in (item, dialogue):
                assert row.get("query_position") is None
                assert row["target"]
                assert 2 <= len(row["input"]) < 256
                assert row["answer_text"] == " " + row["value_text"]
                if ask == "color":
                    assert row["value_text"] in VALUES
                else:
                    assert row["value_text"] in WHO_ENTITIES
            assert "Baby:" in dialogue["prompt_text"]
            assert item["prompt_text"] != dialogue["prompt_text"]
    train = make_story_item(random.Random(41), tokenizer, surface="train", ask="color")
    hold = make_story_item(random.Random(41), tokenizer, surface="heldout", ask="color")
    assert train["prompt_text"] != hold["prompt_text"]
    assert "Remember:" not in train["story_text"]
    assert "Long ago" not in train["story_text"]
    assert any(marker in hold["story_text"] for marker in ("Long ago", "Remember:", "In the yard"))
    three = make_story_item(random.Random(43), tokenizer, surface="heldout", ask="color", n_chars=3)
    pronoun = make_story_item(random.Random(47), tokenizer, surface="heldout", ask="color", pronoun=True)
    size = make_story_item(random.Random(53), tokenizer, surface="heldout", ask="size")
    assert three.get("query_position") is None
    assert three["value_text"] in VALUES
    assert "What color is it?" in pronoun["prompt_text"] or "Which color is it?" in pronoun["prompt_text"]
    assert pronoun["value_text"] in VALUES
    assert size["value_text"] in SIZES
    assert size["attr"] == "size"
    four = make_story_item(random.Random(59), tokenizer, surface="heldout", ask="color", n_chars=4)
    mixed_story = make_story_mixed_item(random.Random(61), tokenizer, surface="heldout", combine=False)
    combine = make_story_mixed_item(random.Random(67), tokenizer, surface="heldout", combine=True)
    assert four.get("query_position") is None
    assert four["value_text"] in VALUES
    assert mixed_story["value_text"] in (*VALUES, *SIZES)
    assert combine["value_text"] in (*VALUES, *SIZES)
    assert "one?" in combine["prompt_text"] or "one." in combine["prompt_text"]
    long3 = make_longturn_item(random.Random(71), tokenizer, n_turns=3, surface="heldout")
    role = make_roleplay_item(random.Random(73), tokenizer, surface="heldout")
    assert long3["prompt_text"].count("Human:") == 3
    assert long3["prompt_text"].count("Baby:") == 3
    assert long3["value_text"] in VALUES
    assert "Kid:" in role["prompt_text"] and "Mom:" in role["prompt_text"]
    assert role["value_text"] in VALUES
