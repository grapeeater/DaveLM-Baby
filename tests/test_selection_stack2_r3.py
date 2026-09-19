from __future__ import annotations

import torch

from src.baby_v010.selection_stack2_r3 import PropMatchHead, WhoFactHead, WhoFactRuntime


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
