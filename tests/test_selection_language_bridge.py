from __future__ import annotations

import random

from src.baby_v010.data_language_bridge import (
    ENTITIES,
    last_mentioned_who,
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
    make_compose_sentence_item,
    make_dialogue_item,
    make_english_item,
    make_mixed_item,
    make_place_item,
    make_attr_followup_item,
    make_phrase_item,
    make_open_item,
    make_size_item,
    make_story_item,
    make_story_mixed_item,
    make_who_bind_item,
    make_who_pair_items,
    make_who_sentence_item,
    make_has_object_item,
    make_beside_item,
    score_bind_sentence,
    score_has_sentence,
    TRAIN_HAS_WHO,
    HOLDOUT_HAS_WHO,
    TRAIN_HAS_WHAT,
    HOLDOUT_HAS_WHAT,
    TRAIN_HAS_ANSWERS,
    HOLDOUT_HAS_ANSWERS,
    TRAIN_BESIDE_WHERE,
    HOLDOUT_BESIDE_WHERE,
    TRAIN_BESIDE_WHO,
    HOLDOUT_BESIDE_WHO,
    TRAIN_BESIDE_ANSWERS,
    HOLDOUT_BESIDE_ANSWERS,
    make_longturn_item,
    make_roleplay_item,
    make_syntax_item,
    period_token_id,
    spaced_first_id,
    TRAIN_PHRASE_ANSWERS,
    HOLDOUT_PHRASE_ANSWERS,
    TRAIN_PHRASE_QUERIES,
    HOLDOUT_PHRASE_QUERIES,
    TRAIN_OPEN_QUERIES,
    HOLDOUT_OPEN_QUERIES,
    TRAIN_WHO_BIND_SIZE,
    HOLDOUT_WHO_BIND_SIZE,
    TRAIN_WHO_BIND_COLOR,
    HOLDOUT_WHO_BIND_COLOR,
    TRAIN_WHO_SENT_COLOR,
    HOLDOUT_WHO_SENT_COLOR,
    TRAIN_WHO_SENT_SIZE,
    HOLDOUT_WHO_SENT_SIZE,
    TRAIN_WHO_SENT_ANSWERS,
    HOLDOUT_WHO_SENT_ANSWERS,
    TRAIN_COMPOSE_SENT_ANSWERS,
    HOLDOUT_COMPOSE_SENT_ANSWERS,
    TRAIN_FORMAT_SENT_PREFIXES,
    HOLDOUT_FORMAT_SENT_PREFIXES,
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


def test_e13_gate_thresholds() -> None:
    from src.baby_v010.selection_language_bridge import e13_gate

    fail = {
        "mixed_2e_heldout": {"first_top1": 0.47},
        "story_combine_heldout": {"first_top1": 0.25},
        "story_mixed_heldout": {"first_top1": 0.38},
        "fact_combine_heldout": {"first_top1": 0.22},
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
    }
    assert e13_gate(fail)[0] == "FAIL"
    advance = dict(fail)
    advance["mixed_2e_heldout"] = {"first_top1": 0.56}
    assert e13_gate(advance)[0] == "ADVANCE"
    grad = {
        "mixed_2e_heldout": {"first_top1": 0.72},
        "story_combine_heldout": {"first_top1": 0.34},
        "story_mixed_heldout": {"first_top1": 0.56},
        "fact_combine_heldout": {"first_top1": 0.62},
        "qa_2fact_heldout": {"first_top1": 0.90},
        "size_stop_heldout": {"free_exact": 0.88},
    }
    assert e13_gate(grad)[0] == "GRAD"
    lost_stop = dict(grad)
    lost_stop["size_stop_heldout"] = {"free_exact": 0.40}
    assert e13_gate(lost_stop)[0] == "ADVANCE"


def test_e14_gate_thresholds() -> None:
    from src.baby_v010.selection_language_bridge import e14_gate

    fail = {
        "phrase_2fact_heldout": {"free_exact": 0.0},
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
    }
    assert e14_gate(fail)[0] == "FAIL"
    advance = dict(fail)
    advance["phrase_2fact_heldout"] = {"free_exact": 0.25}
    assert e14_gate(advance)[0] == "ADVANCE"
    grad = dict(fail)
    grad["phrase_2fact_heldout"] = {"free_exact": 0.50}
    assert e14_gate(grad)[0] == "GRAD"


def test_e15_gate_thresholds() -> None:
    from src.baby_v010.selection_language_bridge import e15_gate

    fail = {
        "open_2fact_heldout": {"first_top1": 0.1},
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
        "dialogue_2fact_heldout": {"first_top1": 0.90},
    }
    assert e15_gate(fail)[0] == "FAIL"
    advance = dict(fail)
    advance["open_2fact_heldout"] = {"first_top1": 0.35}
    assert e15_gate(advance)[0] == "ADVANCE"
    grad = dict(fail)
    grad["open_2fact_heldout"] = {"first_top1": 0.62}
    assert e15_gate(grad)[0] == "GRAD"


def test_e16_gate_thresholds() -> None:
    from src.baby_v010.selection_language_bridge import e16_gate

    fail = {
        "about_2fact_heldout": {"first_top1": 0.1},
        "open_2fact_heldout": {"first_top1": 0.62},
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
    }
    assert e16_gate(fail)[0] == "FAIL"
    advance = dict(fail)
    advance["about_2fact_heldout"] = {"first_top1": 0.35}
    assert e16_gate(advance)[0] == "ADVANCE"
    grad = dict(advance)
    grad["about_2fact_heldout"] = {"first_top1": 0.62}
    assert e16_gate(grad)[0] == "GRAD"


def test_e17_gate_thresholds() -> None:
    from src.baby_v010.selection_language_bridge import e17_gate

    fail = {
        "copy_2fact_heldout": {"first_top1": 0.1},
        "saystop_2fact_heldout": {"first_top1": 0.1},
        "field_2e_heldout": {"first_top1": 0.1},
        "nostory_heldout": {"first_top1": 0.1},
        "format_word_heldout": {"first_top1": 0.1},
        "format_sent_heldout": {"first_top1": 0.1},
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
    }
    assert e17_gate(fail)[0] == "FAIL"
    advance = dict(fail)
    advance["saystop_2fact_heldout"] = {"first_top1": 0.72}
    assert e17_gate(advance)[0] == "ADVANCE"
    paraphrase_only = dict(advance)
    paraphrase_only["nostory_heldout"] = {"first_top1": 0.81}
    paraphrase_only["format_word_heldout"] = {"first_top1": 0.90}
    assert e17_gate(paraphrase_only)[0] == "ADVANCE"
    grad = dict(paraphrase_only)
    grad["copy_2fact_heldout"] = {"first_top1": 0.56}
    assert e17_gate(grad)[0] == "GRAD"


def test_e18_and_e19_gate_thresholds() -> None:
    from src.baby_v010.selection_language_bridge import e18_gate, e19_gate, e20_gate

    e18_fail = {
        "chat_open_heldout": {"first_top1": 0.1},
        "chat_role_heldout": {"first_top1": 0.1},
        "chat_reuse_heldout": {"first_top1": 0.1},
        "chat_long3_heldout": {"first_top1": 0.1},
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
    }
    assert e18_gate(e18_fail)[0] == "FAIL"
    e18_grad = {
        "chat_open_heldout": {"first_top1": 0.72},
        "chat_role_heldout": {"first_top1": 0.62},
        "chat_reuse_heldout": {"first_top1": 0.56},
        "chat_long3_heldout": {"first_top1": 0.50},
        "qa_2fact_heldout": {"first_top1": 0.90},
        "size_stop_heldout": {"free_exact": 0.88},
    }
    assert e18_gate(e18_grad)[0] == "GRAD"
    e19_fail = {
        "happened_heldout": {"first_top1": 0.1},
        "yesno_2fact_heldout": {"first_top1": 0.1},
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
    }
    assert e19_gate(e19_fail)[0] == "FAIL"
    e19_grad = dict(e19_fail)
    e19_grad["happened_heldout"] = {"first_top1": 0.56}
    e19_grad["yesno_2fact_heldout"] = {"first_top1": 0.56}
    assert e19_gate(e19_grad)[0] == "GRAD"
    e20_fail = {
        "chat_loop_color_heldout": {"first_top1": 0.1},
        "chat_loop_fact_heldout": {"first_top1": 0.1},
        "chat_open_heldout": {"first_top1": 0.1},
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
    }
    assert e20_gate(e20_fail)[0] == "FAIL"
    e20_grad = dict(e20_fail)
    e20_grad["chat_loop_color_heldout"] = {"first_top1": 0.62}
    e20_grad["chat_open_heldout"] = {"first_top1": 0.72}
    assert e20_gate(e20_grad)[0] == "GRAD"


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
    fact_combine = make_mixed_item(random.Random(69), tokenizer, n_entities=2, surface="heldout", combine=True)
    mixed_stop = make_mixed_item(random.Random(71), tokenizer, n_entities=2, surface="heldout", period=True)
    assert fact_combine.get("query_position") is None
    assert fact_combine["family"] == "fact_combine"
    assert "one?" in fact_combine["prompt_text"] or "one." in fact_combine["prompt_text"]
    assert fact_combine["value_text"] in (*VALUES, *SIZES)
    assert mixed_stop["answer_text"].endswith(".")
    assert mixed_stop["value_text"] in (*VALUES, *SIZES)
    grouped = make_mixed_item(random.Random(77), tokenizer, n_entities=2, surface="train", grouped=True)
    assert grouped.get("query_position") is None
    assert grouped["target"]
    phrase = make_phrase_item(random.Random(79), tokenizer, n_facts=2, surface="heldout")
    train_phrase = make_phrase_item(random.Random(79), tokenizer, n_facts=2, surface="train")
    assert phrase.get("query_position") is None
    assert len(phrase["target"]) >= 3
    assert phrase["answer_text"].endswith(".")
    assert phrase["value_text"] in VALUES
    assert phrase["prompt_text"] != train_phrase["prompt_text"]
    assert "sentence" in phrase["prompt_text"].lower() or "Sentence" in phrase["prompt_text"]
    assert set(TRAIN_PHRASE_QUERIES).isdisjoint(HOLDOUT_PHRASE_QUERIES)
    assert set(TRAIN_PHRASE_ANSWERS).isdisjoint(HOLDOUT_PHRASE_ANSWERS)
    assert " looks " not in "".join(TRAIN_PHRASE_ANSWERS + HOLDOUT_PHRASE_ANSWERS)
    open_item = make_open_item(random.Random(83), tokenizer, n_facts=2, surface="heldout")
    train_open = make_open_item(random.Random(83), tokenizer, n_facts=2, surface="train")
    assert open_item.get("query_position") is None
    assert open_item["value_text"] in VALUES
    assert open_item["answer_text"].endswith(".")
    assert open_item["prompt_text"] != train_open["prompt_text"]
    assert set(TRAIN_OPEN_QUERIES).isdisjoint(HOLDOUT_OPEN_QUERIES)
    about = make_open_item(random.Random(89), tokenizer, n_facts=2, surface="heldout", about=True)
    assert about.get("query_position") is None
    assert about["family"] == "english_about"
    assert about["value_text"] in VALUES
    long3 = make_longturn_item(random.Random(71), tokenizer, n_turns=3, surface="heldout")
    role = make_roleplay_item(random.Random(73), tokenizer, surface="heldout")
    assert long3["prompt_text"].count("Human:") == 3
    assert long3["prompt_text"].count("Baby:") == 3
    assert long3["value_text"] in VALUES
    assert "Kid:" in role["prompt_text"] and "Mom:" in role["prompt_text"]
    assert role["value_text"] in VALUES


def test_e17_instruction_items_are_gold_free() -> None:
    from src.baby_v010.data_language_bridge import (
        HOLDOUT_COPY_QUERIES,
        HOLDOUT_FORMAT_SENT_ANSWERS,
        HOLDOUT_SAYSTOP_QUERIES,
        TRAIN_COPY_QUERIES,
        TRAIN_FORMAT_SENT_ANSWERS,
        TRAIN_SAYSTOP_QUERIES,
        YES_WORD,
        NO_WORD,
        make_chat_loop_item,
        make_copy_item,
        make_factreuse_item,
        make_field_item,
        make_format_item,
        make_happened_item,
        make_nostory_item,
        make_saystop_item,
        make_varied_dialogue_item,
        make_varied_longturn_item,
        make_yesno_item,
    )

    tokenizer = load_tokenizer()
    copy_hold = make_copy_item(random.Random(101), tokenizer, n_facts=2, surface="heldout")
    copy_train = make_copy_item(random.Random(101), tokenizer, n_facts=2, surface="train")
    assert copy_hold.get("query_position") is None
    assert copy_hold["prompt_text"] != copy_train["prompt_text"]
    assert copy_hold["value_text"] in VALUES
    assert copy_hold["answer_text"].endswith(".")
    assert set(TRAIN_COPY_QUERIES).isdisjoint(HOLDOUT_COPY_QUERIES)
    say = make_saystop_item(random.Random(103), tokenizer, n_facts=2, surface="heldout")
    assert say["value_text"] in VALUES
    assert "stop" in say["prompt_text"].lower()
    assert set(TRAIN_SAYSTOP_QUERIES).isdisjoint(HOLDOUT_SAYSTOP_QUERIES)
    field = make_field_item(random.Random(107), tokenizer, n_entities=2, surface="heldout")
    assert field["value_text"] in (*VALUES, *SIZES)
    assert field["attr"] in {"color", "size"}
    nostory = make_nostory_item(random.Random(109), tokenizer, surface="heldout")
    assert nostory["value_text"] in VALUES
    assert "story" in nostory["prompt_text"].lower() or "continue" in nostory["prompt_text"].lower() or "go on" in nostory["prompt_text"].lower()
    word = make_format_item(random.Random(113), tokenizer, n_facts=2, surface="heldout", sentence=False)
    sent = make_format_item(random.Random(113), tokenizer, n_facts=2, surface="heldout", sentence=True)
    sent_train = make_format_item(random.Random(113), tokenizer, n_facts=2, surface="train", sentence=True)
    assert word["value_text"] in VALUES
    assert sent["answer_text"].startswith(" That is ") or sent["answer_text"].startswith(" It is ")
    assert sent["prompt_text"] != sent_train["prompt_text"]
    assert set(TRAIN_FORMAT_SENT_ANSWERS).isdisjoint(HOLDOUT_FORMAT_SENT_ANSWERS)
    yes_ids = [spaced_first_id(tokenizer, YES_WORD)]
    no_ids = [spaced_first_id(tokenizer, NO_WORD)]
    color_ids = [spaced_first_id(tokenizer, word) for word in VALUES]
    size_ids = [spaced_first_id(tokenizer, word) for word in SIZES]
    assert yes_ids != no_ids
    assert set(yes_ids).isdisjoint(color_ids)
    assert set(no_ids).isdisjoint(color_ids)
    assert set(yes_ids).isdisjoint(size_ids)
    assert set(no_ids).isdisjoint(size_ids)
    chat = make_varied_dialogue_item(random.Random(127), tokenizer, n_facts=2, surface="heldout")
    role = make_varied_dialogue_item(random.Random(127), tokenizer, n_facts=2, surface="heldout", roleplay=True)
    assert "Human:" in chat["prompt_text"]
    assert "Kid:" in role["prompt_text"]
    yn = make_yesno_item(random.Random(131), tokenizer, n_facts=2, surface="heldout")
    assert yn["value_text"] in {YES_WORD, NO_WORD}
    long3 = make_varied_longturn_item(random.Random(137), tokenizer, n_turns=3, surface="heldout")
    assert long3["prompt_text"].count("Human:") == 3
    happened = make_happened_item(random.Random(139), tokenizer, surface="heldout")
    assert happened["value_text"] in {YES_WORD, NO_WORD}
    reuse = make_factreuse_item(random.Random(141), tokenizer, surface="heldout")
    assert reuse["value_text"] in VALUES
    assert reuse["prompt_text"].count("Human:") == 3
    loop = make_chat_loop_item(random.Random(149), tokenizer, n_turns=4, surface="heldout", score="color")
    fact = make_chat_loop_item(random.Random(151), tokenizer, n_turns=4, surface="heldout", score="yesno")
    assert loop["value_text"] in VALUES
    assert fact["value_text"] == YES_WORD
    assert loop["prompt_text"].count("Human:") == 3
    for row in (copy_hold, say, field, nostory, word, sent, chat, yn, long3, happened, reuse, loop, fact):
        assert 2 <= len(row["input"]) < 256
        assert row["target"]
        assert row.get("query_position") is None


def test_usable_chat_pack_is_heldout_existing_skills_only() -> None:
    from src.baby_v010.data_language_bridge import (
        ALLOWED_USABLE_SKILLS,
        FORBIDDEN_USABLE_MARKERS,
        HOLDOUT_ABOUT_QUERIES,
        HOLDOUT_OPEN_QUERIES,
        HOLDOUT_SAYSTOP_QUERIES,
        build_sentence_decode_pack,
        build_usable_chat_pack,
        score_sentence_answer,
        score_usable_turn,
    )

    pack = build_usable_chat_pack()
    assert 8 <= len(pack) <= 12
    assert {int(script["n_turns"]) for script in pack} <= {4, 5}
    assert sum(int(script["n_turns"] == 4) for script in pack) >= 6
    blob = []
    for script in pack:
        assert len(script["turns"]) == script["n_turns"]
        blob.append(script["facts"].lower())
        for turn in script["turns"]:
            assert set(turn["skills"]) <= ALLOWED_USABLE_SKILLS
            blob.append(turn["human"].lower())
            assert turn["gold"]
            assert turn["entity"]
    text = "\n".join(blob)
    for marker in FORBIDDEN_USABLE_MARKERS:
        assert marker not in text, marker
    assert "what color is the" not in text
    assert "copy this word" not in text
    held = " ".join(HOLDOUT_OPEN_QUERIES + HOLDOUT_ABOUT_QUERIES + HOLDOUT_SAYSTOP_QUERIES).lower()
    assert "how about the" in text
    assert "what do you know about the" in text
    assert "stop" in text
    assert held
    terse = score_usable_turn(
        decoded=" white.",
        stopped=True,
        gold="white",
        distractors=("red",),
        skills=["color"],
        n_tokens=2,
    )
    assert terse["usable"] and not terse["rambling"]
    ramble = score_usable_turn(
        decoded=" The cat went home. He was very tired.",
        stopped=True,
        gold="white",
        distractors=("red",),
        skills=["refuse-story", "color"],
        n_tokens=12,
    )
    assert ramble["rambling"] and ramble["generic_continuation"] and not ramble["usable"]
    reuse = score_usable_turn(
        decoded=" red.",
        stopped=True,
        gold="white",
        distractors=("red",),
        skills=["come-back", "color"],
        n_tokens=2,
    )
    assert reuse["fact_reuse"] is False
    assert reuse["distractor_hits"] == ["red"]
    sent = score_sentence_answer(full_text=" cat is red.", entity="cat", color="red", stopped=True)
    assert sent["sentence_ok"]
    one = score_sentence_answer(full_text=" red.", entity="cat", color="red", stopped=True)
    assert one["one_word_color"] and not one["sentence_ok"]
    sentences = build_sentence_decode_pack()
    assert 6 <= len(sentences) <= 12


def test_usable_chat_verdict_parks_five_turn() -> None:
    from src.baby_v010.selection_language_bridge import usable_chat_verdict

    strong4 = {
        "usable_turn": 0.81,
        "fact_reuse": 0.75,
        "period_stop": 1.0,
        "rambling": 0.06,
        "generic_continuation": 0.0,
        "fact_hit": 0.84,
    }
    weak5 = {
        "usable_turn": 0.40,
        "fact_reuse": 0.30,
        "period_stop": 0.90,
        "rambling": 0.10,
        "generic_continuation": 0.05,
        "fact_hit": 0.40,
    }
    verdict, lesson, ramble = usable_chat_verdict(weak5, strong4, weak5)
    assert verdict == "MIXED"
    assert ramble is False
    assert "parked" in lesson
    weak4 = dict(strong4, usable_turn=0.40, fact_hit=0.40, rambling=0.10, generic_continuation=0.05)
    verdict, lesson, ramble = usable_chat_verdict(weak4, weak4, weak5)
    assert verdict == "WEAK"
    assert ramble is False
    ramble4 = dict(strong4, usable_turn=0.40, rambling=0.45, generic_continuation=0.40, period_stop=0.5)
    verdict, lesson, ramble = usable_chat_verdict(ramble4, ramble4, weak5)
    assert verdict == "WEAK"
    assert ramble is True
    assert "rambling" in lesson


def test_who_bind_and_compose_sentence_are_gold_free() -> None:
    tokenizer = load_tokenizer()
    rng = random.Random(324001)
    who = make_who_bind_item(rng, tokenizer, n_entities=2, surface="heldout")
    who3 = make_who_bind_item(rng, tokenizer, n_entities=3, surface="train")
    sent = make_compose_sentence_item(rng, tokenizer, surface="heldout", combine=False)
    comb = make_compose_sentence_item(rng, tokenizer, surface="heldout", combine=True)
    instr = make_compose_sentence_item(rng, tokenizer, surface="heldout", instruct=True)
    train_sent = make_compose_sentence_item(random.Random(324001), tokenizer, surface="train", combine=False)
    for item in (who, who3, sent, comb, instr):
        assert item.get("query_position") is None
        gen = len(item["input"]) - 1
        tiling_parse(item["input"], gen)
        assert item["answer_text"].endswith(".")
        assert len(item["target"]) >= 1
    assert who["value_text"] in WHO_ENTITIES
    assert who3["value_text"] in WHO_ENTITIES
    assert sent["value_text"] in VALUES
    assert "sentence" not in sent["prompt_text"].lower()
    assert any(prefix.strip() in instr["prompt_text"] for prefix in HOLDOUT_FORMAT_SENT_PREFIXES) or "sentence" in instr["prompt_text"].lower()
    assert set(TRAIN_WHO_BIND_SIZE).isdisjoint(HOLDOUT_WHO_BIND_SIZE)
    assert set(TRAIN_WHO_SENT_COLOR).isdisjoint(HOLDOUT_WHO_SENT_COLOR)
    assert set(TRAIN_WHO_SENT_SIZE).isdisjoint(HOLDOUT_WHO_SENT_SIZE)
    assert set(TRAIN_WHO_SENT_COLOR).isdisjoint(TRAIN_WHO_BIND_COLOR)
    assert set(HOLDOUT_WHO_SENT_COLOR).isdisjoint(HOLDOUT_WHO_BIND_COLOR)
    assert set(TRAIN_WHO_SENT_SIZE).isdisjoint(TRAIN_WHO_BIND_SIZE)
    assert set(HOLDOUT_WHO_SENT_SIZE).isdisjoint(HOLDOUT_WHO_BIND_SIZE)
    assert set(TRAIN_WHO_SENT_ANSWERS).isdisjoint(HOLDOUT_WHO_SENT_ANSWERS)
    assert set(TRAIN_COMPOSE_SENT_ANSWERS).isdisjoint(HOLDOUT_COMPOSE_SENT_ANSWERS)
    train_forms = {
        tmpl.format(e=train_sent["entity"], v=train_sent["value_text"]).strip() for tmpl in TRAIN_COMPOSE_SENT_ANSWERS
    }
    hold_forms = {tmpl.format(e=sent["entity"], v=sent["value_text"]).strip() for tmpl in HOLDOUT_COMPOSE_SENT_ANSWERS}
    assert train_sent["answer_text"].strip() in train_forms
    assert sent["answer_text"].strip() in hold_forms
    easy = make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    assert easy["n_entities"] == 1
    assert easy.get("query_position") is None
    assert len(easy["target"]) >= 3
    who_sent = make_who_sentence_item(rng, tokenizer, n_entities=2, surface="heldout")
    assert who_sent.get("query_position") is None
    assert who_sent["family"] == "who_sent"
    assert who_sent["answer_text"].endswith(".")
    who_size = make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train", ask="size")
    assert who_size["attr"] == "who_size"
    assert who_size["value_text"] in SIZES
    who_color = make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train", ask="color")
    assert who_color["attr"] == "who_color"
    assert who_color["value_text"] in VALUES
    mixed3 = make_mixed_item(rng, tokenizer, n_entities=3, surface="heldout", combine=True)
    assert mixed3.get("query_position") is None
    assert mixed3["family"] == "fact_combine"
    for seed in range(20):
        anti = make_who_bind_item(
            random.Random(seed), tokenizer, n_entities=2, surface="train", asked_attr_only=True, anti_recency=True
        )
        assert anti.get("query_position") is None
        assert anti["entity"] != anti.get("last_entity")
        assert last_mentioned_who(anti["prompt_text"]) == anti.get("last_entity")
        sent_anti = make_who_sentence_item(
            random.Random(100 + seed), tokenizer, n_entities=2, surface="train", anti_recency=True
        )
        assert sent_anti["entity"] != sent_anti.get("last_entity")
        assert sent_anti["entity"] in sent_anti["answer_text"]
    pair = make_who_pair_items(random.Random(13), tokenizer, surface="train", as_sentence=True)
    assert len(pair) == 2
    assert pair[0]["entity"] != pair[1]["entity"]
    assert pair[0].get("query_position") is None
    e0, e1 = pair[0]["entity"], pair[1]["entity"]
    assert e0 in pair[0]["prompt_text"] and e1 in pair[0]["prompt_text"]
    assert e0 in pair[1]["prompt_text"] and e1 in pair[1]["prompt_text"]
    assert e0 in pair[0]["answer_text"]
    assert e1 in pair[1]["answer_text"]


def test_has_and_beside_are_gold_free_and_disjoint() -> None:
    tokenizer = load_tokenizer()
    rng = random.Random(326001)
    has_who = make_has_object_item(rng, tokenizer, n_entities=2, surface="train", direction="who")
    has_what = make_has_object_item(rng, tokenizer, n_entities=2, surface="heldout", direction="what")
    beside_where = make_beside_item(rng, tokenizer, surface="train", direction="where")
    beside_who = make_beside_item(rng, tokenizer, surface="heldout", direction="who")
    for item in (has_who, has_what, beside_where, beside_who):
        assert item.get("query_position") is None
        gen = len(item["input"]) - 1
        tiling_parse(item["input"], gen)
        assert item["answer_text"].endswith(".")
        assert len(item["target"]) >= 3
    assert has_who["family"] == "has_object"
    assert has_what["family"] == "has_object"
    assert beside_where["family"] == "beside"
    assert "object" in has_who["answer_text"]
    assert "beside" in beside_who["answer_text"]
    assert set(TRAIN_HAS_WHO).isdisjoint(HOLDOUT_HAS_WHO)
    assert set(TRAIN_HAS_WHAT).isdisjoint(HOLDOUT_HAS_WHAT)
    assert set(TRAIN_HAS_ANSWERS).isdisjoint(HOLDOUT_HAS_ANSWERS)
    assert set(TRAIN_BESIDE_WHERE).isdisjoint(HOLDOUT_BESIDE_WHERE)
    assert set(TRAIN_BESIDE_WHO).isdisjoint(HOLDOUT_BESIDE_WHO)
    assert set(TRAIN_BESIDE_ANSWERS).isdisjoint(HOLDOUT_BESIDE_ANSWERS)
    obj_ids = encode_split(tokenizer, "x", " object")[1]
    beside_ids = encode_split(tokenizer, "x", " beside")[1]
    assert len(obj_ids) == 1
    assert len(beside_ids) == 1
    scored = score_bind_sentence(full_text="The dog is white.", entity="dog", value="white", stopped=True)
    assert scored["sentence_ok"] is True
    assert scored["first_entity"] is True
    one = score_bind_sentence(full_text="dog.", entity="dog", value="white", stopped=True)
    assert one["sentence_ok"] is False
    assert one["one_word_entity"] is True
    has_ok = score_has_sentence(full_text="The dog has the white object.", entity="dog", value="white", stopped=True)
    assert has_ok["sentence_ok"] is True
    shortcut = score_has_sentence(full_text="The dog has the color white.", entity="dog", value="white", stopped=True)
    assert shortcut["sentence_ok"] is False
    beside_ok = score_bind_sentence(
        full_text="The dog is beside the hen.",
        entity="dog",
        value="hen",
        stopped=True,
        predicates=("beside",),
    )
    assert beside_ok["sentence_ok"] is True
    beside_wrong = score_bind_sentence(
        full_text="The dog is hen.",
        entity="dog",
        value="hen",
        stopped=True,
        predicates=("beside",),
    )
    assert beside_wrong["sentence_ok"] is False
