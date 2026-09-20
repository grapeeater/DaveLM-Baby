from __future__ import annotations

import torch

from src.baby_v010.selection_stack2_r3 import PropMatchHead, WhoFactHead, WhoFactRuntime, attach_prop_match_lexicon


def test_position_scores_cover_only_earlier_tokens():
    head = WhoFactHead(16, dim=8, suffix_k=4, copy_scale=8.0)
    hidden = torch.randn(12, 16)
    scores = head.position_scores(hidden)
    assert scores.shape == (8,)
    ptr = head.pointed_index(hidden)
    assert 0 <= ptr < 8


def test_apply_copy_only_boosts_last_logit():
    head = WhoFactHead(8, dim=4, suffix_k=3, copy_scale=5.0)
    hidden = torch.randn(1, 10, 8)
    tokens = torch.randint(1, 20, (1, 10))
    logits = torch.zeros(1, 10, 32)
    head.eval()
    with torch.no_grad():
        out = head.apply_copy(logits.clone(), hidden, tokens)
    ptr = head.pointed_index(hidden[0])
    src = int(tokens[0, ptr])
    boosted = (out[0, -1] - logits[0, -1]).abs().argmax().item()
    assert boosted == src
    assert float((out[0, :-1] - logits[0, :-1]).abs().max()) == 0.0


def test_runtime_has_no_property_scanner_state():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_head = torch.nn.Linear(4, 7, bias=False)

        def forward_hidden(self, tokens):
            return torch.randn(tokens.shape[0], tokens.shape[1], 4)

        def forward(self, tokens):
            return self.language_head(self.forward_hidden(tokens))

    model = Tiny()
    head = WhoFactHead(4, dim=4, suffix_k=2, copy_scale=3.0)
    runtime = WhoFactRuntime(model, head).install()
    assert not hasattr(runtime, "prop_ids")
    assert not hasattr(runtime, "ent_ids")
    runtime.enabled = True
    logits = model(torch.randint(1, 6, (1, 5)))
    assert logits.shape[-1] == 7
    runtime.uninstall()


def test_prop_match_cue_is_in_suffix_and_match_is_earlier():
    head = PropMatchHead(8, suffix_k=4, copy_scale=8.0)
    hidden = torch.randn(10, 8)
    cue = head.cue_index(hidden)
    assert 6 <= cue <= 9
    match = head.match_index(hidden)
    assert 0 <= match < cue
    hop = head.entity_index(hidden)
    if hop is not None:
        assert 0 <= hop <= match


def test_bind_piece_map_canonicalizes_hen_first_piece():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer, spaced_first_id
    from src.baby_v010.selection_stack2_r2 import entity_piece_seqs, next_entity_finish_id
    from src.baby_v010.selection_stack2_r3 import bind_piece_map

    tokenizer = load_tokenizer()
    mapping = bind_piece_map(tokenizer)
    spaced = spaced_first_id(tokenizer, "hen")
    bare = int(encode_ids(tokenizer, "hen")[0])
    assert mapping[spaced] == spaced
    assert mapping[bare] == spaced
    seqs = entity_piece_seqs(tokenizer)
    assert next_entity_finish_id([spaced], seqs) == encode_ids(tokenizer, " hen")[1]
    assert next_entity_finish_id([spaced, encode_ids(tokenizer, " hen")[1]], seqs) is None
    dog_d = int(encode_ids(tokenizer, "dog")[0])
    assert next_entity_finish_id([1, 2, dog_d], seqs) is None
    from src.baby_v010.selection_stack2_r2 import canonicalize_entity_src, entity_spellings

    spells = entity_spellings(tokenizer)
    spaced_ent = {word: spaced_first_id(tokenizer, word) for word in ("dog", "hen", "duck", "bear", "cat", "frog")}
    dog_bare = encode_ids(tokenizer, "dog")
    assert canonicalize_entity_src(dog_bare, len(dog_bare) - 1, spells, spaced_ent) == spaced_first_id(tokenizer, "dog")


def test_query_kind_aliases_and_who_finish_boosts_is():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer, spaced_first_id
    from src.baby_v010.selection_stack2_r2 import entity_spellings, value_spellings
    from src.baby_v010.data_language_bridge import SIZES, VALUES

    tokenizer = load_tokenizer()
    head = PropMatchHead(8, suffix_k=4, copy_scale=8.0)
    head.has_seq = [int(x) for x in encode_ids(tokenizer, " has")]
    head.have_seq = [int(x) for x in encode_ids(tokenizer, " have")]
    head.belong_seq = [int(x) for x in encode_ids(tokenizer, " belong")]
    head.next_seq = [int(x) for x in encode_ids(tokenizer, " next")]
    head.where_seqs = [
        [int(x) for x in encode_ids(tokenizer, stem)]
        for stem in (" where", " Where", "Where")
        if encode_ids(tokenizer, stem)
    ]
    head.beside_id = int(encode_ids(tokenizer, " beside")[0])
    head.is_id = int(encode_ids(tokenizer, " is")[0])
    head.period_id = int(encode_ids(tokenizer, ".")[0])
    head.entity_spells = entity_spellings(tokenizer)
    head.value_spells = value_spellings(tokenizer)
    head.spaced_ent = {"dog": spaced_first_id(tokenizer, "dog")}
    head.spaced_value = {word: spaced_first_id(tokenizer, word) for word in tuple(VALUES) + tuple(SIZES)}
    assert head._query_kind(encode_ids(tokenizer, "What object does the dog have?")) == "has"
    assert head._query_kind(encode_ids(tokenizer, "Which object belongs to the hen?")) == "has"
    assert head._query_kind(encode_ids(tokenizer, "Who is next to the cat?")) == "beside"
    assert head._query_kind(encode_ids(tokenizer, "Where is dog?")) == "beside"
    assert head._query_kind(encode_ids(tokenizer, "Tell me where the bear is.")) == "beside"
    assert head._query_kind(encode_ids(tokenizer, "Who is white?")) == "who"
    assert head._beside_role(encode_ids(tokenizer, "Where is dog?")) == "where"
    assert head._beside_role(encode_ids(tokenizer, "Who is beside the hen?")) == "who"


def test_beside_partner_is_the_other_entity_either_order():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer, spaced_first_id
    from src.baby_v010.selection_stack2_r2 import entity_spellings

    tokenizer = load_tokenizer()
    head = PropMatchHead(8, suffix_k=4, copy_scale=8.0)
    head.entity_spells = entity_spellings(tokenizer)
    head.spaced_ent = {word: spaced_first_id(tokenizer, word) for word in ("dog", "cat", "hen")}
    head.beside_id = int(encode_ids(tokenizer, " beside")[0])
    head.period_id = int(encode_ids(tokenizer, ".")[0])
    head.punct_ids = {head.period_id, int(encode_ids(tokenizer, "?")[0])}
    last = encode_ids(tokenizer, "The cat is beside the dog.")
    first = encode_ids(tokenizer, "The dog is beside the cat.")
    dog = spaced_first_id(tokenizer, "dog")
    cat = spaced_first_id(tokenizer, "cat")
    assert head._partner_src(dog, last, len(last)) == cat
    assert head._partner_src(cat, last, len(last)) == dog
    assert head._partner_src(dog, first, len(first)) == cat
    assert head._partner_src(cat, first, len(first)) == dog


def test_subject_of_value_uses_clause_not_first_piece():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer, spaced_first_id
    from src.baby_v010.selection_stack2_r2 import entity_spellings, value_spellings
    from src.baby_v010.data_language_bridge import SIZES, VALUES

    tokenizer = load_tokenizer()
    head = PropMatchHead(8, suffix_k=4, copy_scale=8.0)
    head.entity_spells = entity_spellings(tokenizer)
    head.value_spells = value_spellings(tokenizer)
    head.spaced_ent = {word: spaced_first_id(tokenizer, word) for word in ("dog", "hen", "bear", "cat")}
    head.spaced_value = {word: spaced_first_id(tokenizer, word) for word in tuple(VALUES) + tuple(SIZES)}
    head.period_id = int(encode_ids(tokenizer, ".")[0])
    head.punct_ids = {head.period_id}
    facts = encode_ids(tokenizer, "Remember: the dog is huge in size. That hen is tiny in size.")
    tiny = spaced_first_id(tokenizer, "tiny")
    hen = spaced_first_id(tokenizer, "hen")
    assert head._subject_of_value(tiny, facts, len(facts)) == hen
    query = encode_ids(tokenizer, "Tell me who looks tiny.")
    assert head._query_value_src(query) == tiny
    wide_q = encode_ids(tokenizer, "Who looks wide?")
    assert head._query_value_src(wide_q) == spaced_first_id(tokenizer, "wide")


def _bind_combine_lexicon(head, tokenizer):
    from src.baby_v010.data_language_bridge import SIZES, VALUES, encode_ids, spaced_first_id
    from src.baby_v010.selection_stack2_r2 import bridge_entity_spellings, entity_spellings, value_spellings

    head.entity_spells = entity_spellings(tokenizer)
    head.bridge_spells = bridge_entity_spellings(tokenizer)
    head.value_spells = value_spellings(tokenizer)
    head.spaced_value = {word: spaced_first_id(tokenizer, word) for word in tuple(VALUES) + tuple(SIZES)}
    head.color_ask_seq = [int(x) for x in encode_ids(tokenizer, " color")]
    head.size_ask_seq = [int(x) for x in encode_ids(tokenizer, " size")]
    head.about_seqs = [[int(x) for x in encode_ids(tokenizer, " about")]]
    head.describe_seqs = [[int(x) for x in encode_ids(tokenizer, "Describe")]]
    head.and_seq = [int(x) for x in encode_ids(tokenizer, " and")]
    head.period_id = int(encode_ids(tokenizer, ".")[0])
    head.punct_ids = {head.period_id, int(encode_ids(tokenizer, "?")[0]), int(encode_ids(tokenizer, ":")[0])}
    head.is_id = int(encode_ids(tokenizer, " is")[0])
    return head


def test_query_kind_combine_and_about_do_not_steal_who():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer

    tokenizer = load_tokenizer()
    head = _bind_combine_lexicon(PropMatchHead(8, suffix_k=4, copy_scale=8.0), tokenizer)
    assert head._query_kind(encode_ids(tokenizer, "Who is white?")) == "who"
    assert head._query_kind(encode_ids(tokenizer, "Which color is the cat?")) == "direct_color"
    assert head._query_kind(encode_ids(tokenizer, "Tell me the size of the bird?")) == "direct_size"
    assert head._query_kind(encode_ids(tokenizer, "Which color is the tiny one?")) == "combine_color"
    assert head._query_kind(encode_ids(tokenizer, "Tell me the size of the red one?")) == "combine_size"
    assert head._query_kind(encode_ids(tokenizer, "What do you know about the cat?")) == "about"
    assert head._query_kind(encode_ids(tokenizer, "Describe the dog.")) == "about"
    assert head._query_kind(encode_ids(tokenizer, "Tell me the color of the tiny one?")) == "combine_color"


def test_combine_uses_other_attribute_not_cue_or_recency():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer, spaced_first_id

    tokenizer = load_tokenizer()
    head = _bind_combine_lexicon(PropMatchHead(8, suffix_k=4, copy_scale=8.0), tokenizer)
    facts = encode_ids(tokenizer, "The dog is red. The cat is blue. The dog is huge.")
    q_color = encode_ids(tokenizer, "Which color is the huge one?")
    q_size = encode_ids(tokenizer, "Which size is the red one?")
    assert head._subject_word_of_value("huge", facts, len(facts)) == "dog"
    assert head._subject_word_of_value("red", facts, len(facts)) == "dog"
    assert head._combine_target_src("combine_color", q_color, facts, len(facts)) == spaced_first_id(tokenizer, "red")
    assert head._combine_target_src("combine_size", q_size, facts, len(facts)) == spaced_first_id(tokenizer, "huge")
    adj = encode_ids(tokenizer, "Long ago a tiny pig ate. That pig was white.")
    assert head._subject_word_of_value("tiny", adj, len(adj)) == "pig"
    q_story = encode_ids(tokenizer, "Which color is the tiny one?")
    assert head._combine_target_src("combine_color", q_story, adj, len(adj)) == spaced_first_id(tokenizer, "white")


def test_about_plan_keeps_both_facts_for_requested_entity():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer

    tokenizer = load_tokenizer()
    head = _bind_combine_lexicon(PropMatchHead(8, suffix_k=4, copy_scale=8.0), tokenizer)
    facts = encode_ids(tokenizer, "The dog is red. The cat is blue. The dog is huge.")
    query = encode_ids(tokenizer, "Tell me about the dog.")
    plan = head._about_plan(query, facts, len(facts))
    decoded = tokenizer.decode(plan, skip_special_tokens=True).lower()
    assert "dog" in decoded
    assert "red" in decoded
    assert "huge" in decoded
    assert "and" in decoded
    assert "blue" not in decoded
    assert "cat" not in decoded


def test_direct_size_uses_named_bridge_entity():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer

    tokenizer = load_tokenizer()
    head = _bind_combine_lexicon(PropMatchHead(8, suffix_k=4, copy_scale=8.0), tokenizer)
    facts = encode_ids(tokenizer, "Remember: the frog is wide in size. That bird is tiny in size.")
    query = encode_ids(tokenizer, "Tell me the size of the bird?")
    assert head._query_kind(query) == "direct_size"
    assert head._direct_target_word("direct_size", query, facts, len(facts)) == "tiny"


def test_role_only_treats_baby_cue_as_empty_answer():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer

    tokenizer = load_tokenizer()
    head = _bind_combine_lexicon(PropMatchHead(8, suffix_k=4, copy_scale=8.0), tokenizer)
    role_ids = set()
    for stem in ("\nBaby:", "Baby:", ":", "\n", "Baby"):
        role_ids.update(int(x) for x in encode_ids(tokenizer, stem))
    head.role_ids = role_ids
    assert head._role_only(encode_ids(tokenizer, "\nBaby:"))
    assert not head._role_only(encode_ids(tokenizer, " red."))
    role_then = encode_ids(tokenizer, "\nBaby:") + encode_ids(tokenizer, " red")
    assert head._content_tail(role_then) == encode_ids(tokenizer, " red")
    assert head._query_kind(encode_ids(tokenizer, "Tell me the size of the bird.")) == "direct_size"
    head.qmark_id = int(encode_ids(tokenizer, "?")[0])
    convo = encode_ids(tokenizer, "How about the pig? pig is red.\nHuman: Tell me the size of the bird.\nBaby:")
    bound = head._query_bound(convo)
    assert bound is not None
    span = head._question_span(convo, bound)
    assert head._query_kind(span) == "direct_size"
    assert head._query_entity_word(span) == "bird"


def _bind_r7_lexicon(head, tokenizer):
    return attach_prop_match_lexicon(head, tokenizer)


def test_inverse_bind_uses_full_bridge_and_value_before_entity():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer, spaced_first_id

    tokenizer = load_tokenizer()
    head = _bind_r7_lexicon(PropMatchHead(8, suffix_k=4, copy_scale=8.0), tokenizer)
    leftward = encode_ids(tokenizer, "The cow is blue. The pig is red. The bird is green.")
    assert head._subject_word_of_value("red", leftward, len(leftward)) == "pig"
    assert head._subject_of_value(spaced_first_id(tokenizer, "red"), leftward, len(leftward)) == spaced_first_id(
        tokenizer, "pig"
    )
    adj = encode_ids(tokenizer, "A pink fox hid. A tiny cow sat.")
    assert head._subject_word_of_value("pink", adj, len(adj)) == "fox"
    assert head._subject_word_of_value("tiny", adj, len(adj)) == "cow"
    mid = encode_ids(tokenizer, "The cat is blue. The pig is red. The dog is white.")
    assert head._subject_word_of_value("red", mid, len(mid)) == "pig"
    q_who = encode_ids(tokenizer, "Who is red?")
    assert head._query_kind(q_who) == "who"
    assert head._answer_src("who", q_who, mid, len(mid), None) == spaced_first_id(tokenizer, "pig")
    plan = head._who_plan(q_who, mid, len(mid))
    decoded = tokenizer.decode(plan, skip_special_tokens=True).lower().strip()
    assert decoded.startswith("pig")
    assert "red" in decoded
    assert "cat" not in decoded


def test_has_value_vs_has_entity_and_relation_without_beside_stem():
    from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer, spaced_first_id

    tokenizer = load_tokenizer()
    head = _bind_r7_lexicon(PropMatchHead(8, suffix_k=4, copy_scale=8.0), tokenizer)
    facts = encode_ids(tokenizer, "The pig has the red object. The cow has the white object. The fox is beside the duck.")
    q_has_value = encode_ids(tokenizer, "What does the pig have?")
    q_has_entity = encode_ids(tokenizer, "Which one has the white object?")
    q_belong = encode_ids(tokenizer, "Which object belongs to the pig?")
    q_near = encode_ids(tokenizer, "Who is near the duck?")
    q_about = encode_ids(tokenizer, "Talk about the fox.")
    q_then = encode_ids(tokenizer, "the cow then?")
    assert head._query_kind(q_has_value) == "has"
    assert head._query_kind(q_has_entity) == "has"
    assert head._query_kind(q_belong) == "has"
    assert head._query_kind(q_near) == "who"
    assert head._query_kind(q_about) == "about"
    assert head._query_kind(q_then) == "about"
    assert head._speech_kind("who", q_near, facts, len(facts)) == "beside"
    assert head._answer_src("has", q_has_value, facts, len(facts), None) == spaced_first_id(tokenizer, "pig")
    assert head._answer_src("has", q_has_entity, facts, len(facts), None) == spaced_first_id(tokenizer, "cow")
    assert head._answer_src("has", q_belong, facts, len(facts), None) == spaced_first_id(tokenizer, "pig")
    has_value_plan = tokenizer.decode(head._has_plan(q_has_value, facts, len(facts)), skip_special_tokens=True).lower()
    has_entity_plan = tokenizer.decode(head._has_plan(q_has_entity, facts, len(facts)), skip_special_tokens=True).lower()
    assert has_value_plan.strip().startswith("pig") and "red" in has_value_plan and "has" in has_value_plan
    assert has_entity_plan.strip().startswith("cow") and "white" in has_entity_plan
    near_plan = tokenizer.decode(head._beside_plan(q_near, facts, len(facts)), skip_special_tokens=True).lower()
    assert near_plan.strip().startswith("fox")
    assert "duck" in near_plan
    assert not near_plan.startswith("duck")
    where_q = encode_ids(tokenizer, "Where is the fox?")
    where_plan = tokenizer.decode(head._beside_plan(where_q, facts, len(facts)), skip_special_tokens=True).lower()
    assert where_plan.strip().startswith("fox")
    assert "duck" in where_plan
