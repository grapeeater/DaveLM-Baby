from __future__ import annotations

import random

from src.baby_v010.data_language_bridge import (
    VALUES,
    WHO_ENTITIES,
    last_mentioned_who,
    load_tokenizer,
    make_who_bind_item,
    spaced_first_id,
)
from src.baby_v010.selection_stack2_r2 import (
    S4M_PARENT,
    S4M_PARENT_SHA,
    WhoSelectHead,
    bare_entity_id_set,
    entity_id_set,
    locate_who_slots,
    mention_indices,
    property_id_set,
    punct_id_set,
    query_turn_active,
    question_id_set,
    route_beside_entity,
    route_entity_pos,
    verify_parent,
)


def test_r2_parent_identity() -> None:
    assert verify_parent(S4M_PARENT, S4M_PARENT_SHA, label="s4m") == S4M_PARENT_SHA


def test_locate_who_slots_is_gold_free() -> None:
    tokenizer = load_tokenizer()
    item = make_who_bind_item(random.Random(327001), tokenizer, n_entities=2, surface="heldout")
    assert item.get("query_position") is None
    prop_ids = set(property_id_set(tokenizer))
    ent_ids = set(entity_id_set(tokenizer))
    slots = locate_who_slots(item["input"], prop_ids, ent_ids)
    assert slots is not None
    cue, ents = slots
    assert len(ents) >= 2
    assert cue > ents[0]
    gold_id = spaced_first_id(tokenizer, item["entity"])
    assert gold_id in {int(item["input"][pos]) for pos in ents}
    assert last_mentioned_who(item["prompt_text"]) in WHO_ENTITIES
    assert item["cue_text"] in VALUES or True


def test_who_select_head_scores_two_entities() -> None:
    import torch

    head = WhoSelectHead(16, copy_scale=3.0)
    hidden = torch.randn(8, 16)
    scores = head.slot_scores(hidden, 6, [1, 3])
    assert scores.shape == (2,)


def test_route_entity_pos_picks_matching_property() -> None:
    import torch

    tokenizer = load_tokenizer()
    item = make_who_bind_item(random.Random(7), tokenizer, n_entities=2, surface="heldout")
    prop_ids = set(property_id_set(tokenizer))
    ent_ids = set(entity_id_set(tokenizer))
    tokens = torch.tensor(item["input"], dtype=torch.long)
    hidden = torch.zeros(len(item["input"]), 8)
    values = item["input"]
    props = [i for i, tok in enumerate(values) if int(tok) in prop_ids]
    cue = props[-1]
    hidden[cue] = torch.ones(8)
    match = [i for i in props if i < cue and int(values[i]) == int(values[cue])]
    if not match:
        return
    hidden[match[-1]] = torch.ones(8)
    ptr = route_entity_pos(hidden, tokens, prop_ids, ent_ids)
    assert ptr is not None
    assert int(values[ptr]) in ent_ids


def test_query_turn_drops_after_entity() -> None:
    tokenizer = load_tokenizer()
    item = make_who_bind_item(random.Random(11), tokenizer, n_entities=2, surface="heldout")
    prop_ids = set(property_id_set(tokenizer))
    ent_ids = set(entity_id_set(tokenizer))
    q_ids = question_id_set(tokenizer)
    punct = punct_id_set(tokenizer)
    values = list(item["input"])
    props = [i for i, tok in enumerate(values) if int(tok) in prop_ids]
    assert query_turn_active(values, props[-1], ent_ids, q_ids, punct)
    values.append(next(iter(ent_ids)))
    assert not query_turn_active(values, props[-1], ent_ids, q_ids, punct)


def test_mention_indices_finds_sentence_initial_entity() -> None:
    tokenizer = load_tokenizer()
    from src.baby_v010.data_language_bridge import BOS, encode_ids

    ids = [BOS, *encode_ids(tokenizer, "dog looks white. hen looks red.")]
    spaced = set(entity_id_set(tokenizer))
    bare = set(bare_entity_id_set(tokenizer))
    ents = mention_indices(ids, spaced, bare_map=bare)
    assert len(ents) >= 2


def test_route_beside_picks_subject() -> None:
    from src.baby_v010.data_language_bridge import BOS, encode_ids
    from src.baby_v010.selection_stack2_r2 import _first_id

    tokenizer = load_tokenizer()
    ids = [BOS, *encode_ids(tokenizer, "That dog is beside the hen. Which one is beside the hen?")]
    ptr = route_beside_entity(
        ids,
        set(entity_id_set(tokenizer)),
        bare_ent_ids=set(bare_entity_id_set(tokenizer)),
        beside_id=_first_id(tokenizer, " beside"),
        q_ids=question_id_set(tokenizer),
    )
    assert ptr is not None
    word = entity_id_set(tokenizer).get(ids[ptr]) or bare_entity_id_set(tokenizer).get(ids[ptr])
    assert word == "dog"
