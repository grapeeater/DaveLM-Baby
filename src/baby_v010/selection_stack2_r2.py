from __future__ import annotations

"""Stack2 retrieval-school-2: inverse 2e WHO selection from s4m.

Sentence school is parked. Do not gold query_position at inference.
U16000 stays authoritative. TEST/FINAL/SACRED stay sealed.
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_language_bridge import (
    BOS,
    HOLDOUT_WHO_BIND_COLOR,
    HOLDOUT_WHO_BIND_SIZE,
    SIZES,
    VALUES,
    WHO_ENTITIES,
    build_s3_panels,
    encode_ids,
    last_mentioned_who,
    load_tokenizer,
    make_who_bind_item,
    spaced_first_id,
)
from .selection_language_bridge import load_experimental_baby
from .selection_s1 import digest, write
from .selection_stack2 import OUT, S4M_SURVIVOR, S4M_SURVIVOR_SHA

R2_OUT = OUT / "r2"
S4M_PARENT = S4M_SURVIVOR
S4M_PARENT_SHA = S4M_SURVIVOR_SHA
WHO2_GATE = 0.70
WHO2_STRONG = 0.75
PROPERTY_WORDS = tuple(VALUES) + tuple(SIZES)
QUESTION_WORDS = ("Who", "who", "Which")


def verify_parent(path: Path, expected: str, *, label: str) -> str:
    sha = digest(path)
    if sha != expected:
        raise RuntimeError(f"{label} hash mismatch: got {sha} expected {expected}")
    return sha


def property_id_set(tokenizer) -> dict[int, str]:
    return {spaced_first_id(tokenizer, word): word for word in PROPERTY_WORDS}


def entity_id_set(tokenizer) -> dict[int, str]:
    return {spaced_first_id(tokenizer, word): word for word in WHO_ENTITIES}


def bare_entity_id_set(tokenizer) -> dict[int, str]:
    """Sentence-initial entity pieces, used only at BOS/start."""
    out: dict[int, str] = {}
    for word in WHO_ENTITIES:
        enc = encode_ids(tokenizer, word)
        if enc:
            out[int(enc[0])] = word
    return out


def question_id_set(tokenizer) -> set[int]:
    return {spaced_first_id(tokenizer, word) for word in QUESTION_WORDS}


def punct_id_set(tokenizer) -> set[int]:
    ids: set[int] = set()
    for text in ("?", ".", ",", "!", ":"):
        enc = encode_ids(tokenizer, text)
        if enc:
            ids.add(int(enc[0]))
    return ids


def mention_indices(token_row, ids, *, bare_map=None) -> list[int]:
    values = token_row.tolist() if hasattr(token_row, "tolist") else list(token_row)
    id_set = set(ids)
    bare_set = set(bare_map) if bare_map is not None else set()
    hits: list[int] = []
    for i, tok in enumerate(values):
        tid = int(tok)
        if tid in id_set:
            hits.append(i)
        elif tid in bare_set and (i == 0 or (i == 1 and int(values[0]) == BOS)):
            hits.append(i)
    return hits


def query_boundary(token_row, q_ids: set[int], punct_ids: set[int]) -> int | None:
    """Last token of the question, or None if no question word is present."""
    values = token_row.tolist() if hasattr(token_row, "tolist") else list(token_row)
    qpos = [i for i, tok in enumerate(values) if int(tok) in q_ids]
    if not qpos:
        return None
    for i in range(qpos[-1] + 1, len(values)):
        if int(values[i]) in punct_ids:
            return i
    return len(values) - 1


def query_turn_active(token_row, last_prop: int, ent_ids: set[int], q_ids: set[int], punct_ids: set[int] | None = None) -> bool:
    """True while the sequence still ends on the question, before an answer token."""
    values = token_row.tolist() if hasattr(token_row, "tolist") else list(token_row)
    bound = query_boundary(values, q_ids, punct_ids or set())
    if bound is None:
        return False
    return len(values) - 1 <= bound


def last_index(seq, token_id: int) -> int | None:
    hits = [i for i, tok in enumerate(seq) if int(tok) == int(token_id)]
    return hits[-1] if hits else None


def cue_gap(item: dict, tokenizer) -> int | None:
    cue = str(item.get("cue_text") or "")
    if not cue:
        return None
    pos = last_index(item["input"], spaced_first_id(tokenizer, cue))
    if pos is None:
        return None
    return len(item["input"]) - 1 - pos


def first_logits(model, input_ids, device):
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    return model(x)[0, -1]


def pred_entity(logits, id_to_ent: dict[int, str]) -> str | None:
    return id_to_ent.get(int(logits.argmax()))


def score_pack(model, tokenizer, device, items: list[dict]) -> dict:
    id_to_ent = entity_id_set(tokenizer)
    rows = []
    with torch.no_grad():
        for item in items:
            gold = str(item["entity"])
            last = last_mentioned_who(str(item.get("prompt_text") or ""))
            logits = first_logits(model, item["input"], device)
            pred = pred_entity(logits, id_to_ent)
            gap = cue_gap(item, tokenizer)
            rows.append(
                {
                    "gold": gold,
                    "pred": pred,
                    "ok": pred == gold,
                    "last": last,
                    "last_is_gold": last == gold,
                    "pred_last": pred == last,
                    "cue_gap": gap,
                    "cue": item.get("cue_text"),
                }
            )
    n = len(rows)
    n_last = sum(1 for row in rows if row["last_is_gold"])
    n_not = n - n_last
    by_gap: dict[str, list[bool]] = {}
    for row in rows:
        key = str(row["cue_gap"])
        by_gap.setdefault(key, []).append(bool(row["ok"]))
    return {
        "n": n,
        "acc": sum(row["ok"] for row in rows) / n if n else 0.0,
        "pred_is_last": sum(row["pred_last"] for row in rows) / n if n else 0.0,
        "acc_last_gold": (sum(row["ok"] for row in rows if row["last_is_gold"]) / n_last) if n_last else None,
        "acc_last_not": (sum(row["ok"] for row in rows if not row["last_is_gold"]) / n_not) if n_not else None,
        "by_gap": {gap: sum(hits) / len(hits) for gap, hits in by_gap.items()},
        "rows": rows,
    }


def mutate_append(item: dict, token_id: int) -> dict:
    clone = dict(item)
    clone["input"] = list(item["input"]) + [int(token_id)]
    return clone


def hidden_splice_logits(model, input_ids, device, src_pos: int):
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    hidden = model.forward_hidden(x)
    mixed = hidden.clone()
    mixed[0, -1] = hidden[0, src_pos]
    return model.language_head(mixed)[0, -1]


def run_choice_probe(model, tokenizer, device, items: list[dict]) -> dict:
    """Is gold already the runner-up among the two scene entities?"""
    id_to_ent = entity_id_set(tokenizer)
    two_ok = []
    gold_beats_last = []
    suppress_last_ok = []
    suppress_gold_ok = []
    patch_ok = []
    with torch.no_grad():
        for item in items:
            gold = str(item["entity"])
            last = last_mentioned_who(str(item.get("prompt_text") or ""))
            others = [name for name in WHO_ENTITIES if name != gold]
            scene = [gold] + [name for name in (item.get("distractors") or others[:1]) if name != gold]
            if last and last != gold and last not in scene:
                scene.append(last)
            gold_id = spaced_first_id(tokenizer, gold)
            last_id = spaced_first_id(tokenizer, last) if last else None
            logits = first_logits(model, item["input"], device)
            pair_ids = [spaced_first_id(tokenizer, name) for name in scene[:2]]
            pick = scene[int(logits[pair_ids].argmax())]
            two_ok.append(pick == gold)
            if last_id is not None:
                gold_beats_last.append(float(logits[gold_id]) > float(logits[last_id]))
                masked = logits.clone()
                masked[last_id] = float("-inf")
                suppress_last_ok.append(pred_entity(masked, id_to_ent) == gold)
                masked_g = logits.clone()
                masked_g[gold_id] = float("-inf")
                suppress_gold_ok.append(pred_entity(masked_g, id_to_ent) == gold)
            gold_pos = last_index(item["input"], gold_id)
            last_pos = last_index(item["input"], last_id) if last_id is not None else None
            if gold_pos is not None and last_pos is not None:
                x = torch.tensor([item["input"]], dtype=torch.long, device=device)
                hidden = model.forward_hidden(x)
                mixed = hidden.clone()
                mixed[0, -1] = hidden[0, -1] + (hidden[0, gold_pos] - hidden[0, last_pos])
                patch_ok.append(pred_entity(model.language_head(mixed)[0, -1], id_to_ent) == gold)
    n = len(two_ok)
    return {
        "two_way_acc": sum(two_ok) / n if n else None,
        "gold_logit_beats_last": (sum(gold_beats_last) / len(gold_beats_last)) if gold_beats_last else None,
        "suppress_last_acc": (sum(suppress_last_ok) / len(suppress_last_ok)) if suppress_last_ok else None,
        "suppress_gold_acc": (sum(suppress_gold_ok) / len(suppress_gold_ok)) if suppress_gold_ok else None,
        "patch_gold_minus_last": (sum(patch_ok) / len(patch_ok)) if patch_ok else None,
    }


def run_transport_probe(model, tokenizer, device, items: list[dict]) -> dict:
    id_to_ent = entity_id_set(tokenizer)
    arms = {"as_is": [], "append_cue": [], "append_other": [], "append_entity": [], "splice_cue": [], "splice_entity": []}
    with torch.no_grad():
        for item in items:
            gold = str(item["entity"])
            cue = str(item.get("cue_text") or "")
            distractors = [name for name in (item.get("distractors") or []) if name and name != gold]
            other_ent = distractors[0] if distractors else None
            cue_id = spaced_first_id(tokenizer, cue) if cue else None
            gold_id = spaced_first_id(tokenizer, gold)
            other_prop = None
            if cue in VALUES:
                other_prop = next((word for word in VALUES if word != cue), None)
            elif cue in SIZES:
                other_prop = next((word for word in SIZES if word != cue), None)
            cue_pos = last_index(item["input"], cue_id) if cue_id is not None else None
            gold_pos = last_index(item["input"], gold_id)

            def ok(logits) -> bool:
                return pred_entity(logits, id_to_ent) == gold

            arms["as_is"].append(ok(first_logits(model, item["input"], device)))
            if cue_id is not None:
                arms["append_cue"].append(ok(first_logits(model, mutate_append(item, cue_id)["input"], device)))
            if other_prop is not None:
                arms["append_other"].append(
                    ok(first_logits(model, mutate_append(item, spaced_first_id(tokenizer, other_prop))["input"], device))
                )
            arms["append_entity"].append(ok(first_logits(model, mutate_append(item, gold_id)["input"], device)))
            if cue_pos is not None:
                arms["splice_cue"].append(ok(hidden_splice_logits(model, item["input"], device, cue_pos)))
            if gold_pos is not None:
                arms["splice_entity"].append(ok(hidden_splice_logits(model, item["input"], device, gold_pos)))
    return {name: (sum(hits) / len(hits) if hits else None) for name, hits in arms.items()}


def run_r2_probe(device, path: Path | None = None) -> dict:
    path = path or S4M_PARENT
    sha = verify_parent(path, S4M_PARENT_SHA if path == S4M_PARENT else digest(path), label="r2 parent")
    if path == S4M_PARENT and sha != S4M_PARENT_SHA:
        raise RuntimeError("s4m parent mismatch")
    tokenizer = load_tokenizer()
    model, _config, ckpt = load_experimental_baby(path, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
    seed2 = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
    official_s = score_pack(model, tokenizer, device, official)
    seed2_s = score_pack(model, tokenizer, device, seed2)
    transport = run_transport_probe(model, tokenizer, device, official)
    choice = run_choice_probe(model, tokenizer, device, official)
    report = {
        "id": "r2_who_transport",
        "loaded_checkpoint": str(path),
        "loaded_sha256": sha,
        "requested_parent": str(S4M_PARENT),
        "requested_parent_sha256": S4M_PARENT_SHA,
        "parent_match": sha == S4M_PARENT_SHA if path == S4M_PARENT else sha == digest(path),
        "authoritative": False,
        "protected_material_opened": False,
        "official": {k: official_s[k] for k in official_s if k != "rows"},
        "seed324777": {k: seed2_s[k] for k in seed2_s if k != "rows"},
        "transport": transport,
        "choice": choice,
    }
    R2_OUT.mkdir(parents=True, exist_ok=True)
    write(R2_OUT / "WHO_TRANSPORT.json", report)
    print(json.dumps(report, default=str), flush=True)
    return report


def verify_prop_router(device, path: Path | None = None, *, copy_scale: float = 8.0) -> dict:
    path = path or S4M_PARENT
    expected = S4M_PARENT_SHA if path == S4M_PARENT else digest(path)
    sha = verify_parent(path, expected, label="prop router parent")
    tokenizer = load_tokenizer()
    model, _config, ckpt = load_experimental_baby(path, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
    seed2 = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
    bare = score_pack(model, tokenizer, device, official)
    routed = score_who_with_router(model, tokenizer, device, official, copy_scale=copy_scale)
    routed2 = score_who_with_router(model, tokenizer, device, seed2, copy_scale=copy_scale)
    report = {
        "id": "r2_prop_router",
        "loaded_checkpoint": str(path),
        "loaded_sha256": sha,
        "requested_parent": str(S4M_PARENT),
        "requested_parent_sha256": S4M_PARENT_SHA,
        "parent_match": sha == S4M_PARENT_SHA if path == S4M_PARENT else True,
        "authoritative": False,
        "copy_scale": copy_scale,
        "bare": {k: bare[k] for k in bare if k != "rows"},
        "routed": {k: routed[k] for k in routed if k != "rows"},
        "routed_seed324777": {k: routed2[k] for k in routed2 if k != "rows"},
        "gate": WHO2_GATE,
        "passed": routed["acc"] + 1e-12 >= WHO2_GATE and routed2["acc"] + 1e-12 >= WHO2_GATE,
    }
    R2_OUT.mkdir(parents=True, exist_ok=True)
    write(R2_OUT / "PROP_ROUTER.json", report)
    print(json.dumps(report, default=str), flush=True)
    return report


def probe_with_router(device, path: Path, expected_sha: str, tag: str, *, relate: bool = False) -> dict:
    from .selection_stack2_s5 import (
        CHEAP_OPS,
        run_beside_decode,
        run_has_decode,
        run_who_sentence_decode,
        summarize_bind,
    )

    sha = verify_parent(path, expected_sha, label=tag)
    tokenizer = load_tokenizer()
    model, _config, ckpt = load_experimental_baby(path, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
    seed2 = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
    who2 = score_who_with_router(model, tokenizer, device, official)
    who2b = score_who_with_router(model, tokenizer, device, seed2)
    runtime = WhoPropRuntime(model, tokenizer, copy_scale=8.0).install()
    runtime.enabled = True
    assist = RelAssistRuntime(model, tokenizer, copy_scale=8.0).install()
    assist.enabled = True
    try:
        who_sent = summarize_bind(run_who_sentence_decode(model, tokenizer, device, include=CHEAP_OPS))
        has_s = summarize_bind(run_has_decode(model, tokenizer, device, include=CHEAP_OPS)) if relate else None
        beside_s = summarize_bind(run_beside_decode(model, tokenizer, device, include=CHEAP_OPS)) if relate else None
    finally:
        assist.uninstall()
        runtime.uninstall()
    report = {
        "id": tag,
        "loaded_checkpoint": str(path),
        "loaded_sha256": sha,
        "parent_match": sha == expected_sha,
        "authoritative": False,
        "who_2e": who2["acc"],
        "who_2e_seed324777": who2b["acc"],
        "who_2e_last_not": who2["acc_last_not"],
        "who_sent": who_sent,
        "has": has_s,
        "beside": beside_s,
    }
    R2_OUT.mkdir(parents=True, exist_ok=True)
    write(R2_OUT / f"{tag}.json", report)
    print(json.dumps(report, default=str), flush=True)
    return report


def score_who_sent_seed(model, tokenizer, device, *, seed: int = 324777, n: int = 16) -> dict:
    from .data_language_bridge import make_who_sentence_item, score_bind_sentence
    from .selection_language_bridge import greedy_decode_until_stop

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        item = make_who_sentence_item(rng, tokenizer, n_entities=2, surface="heldout")
        emitted, stopped = greedy_decode_until_stop(model, item["input"], device, tokenizer, max_new=16)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True)
        entity = str(item["entity"])
        value = str(item["value_text"])
        scored = score_bind_sentence(
            full_text=decoded,
            entity=entity,
            value=value,
            stopped=stopped,
            predicates=("is", "looks", "appears", "was"),
        )
        rows.append({"id": f"seed{seed}_{i}", "decoded": decoded, "entity": entity, "value": value, **scored})
    acc = sum(bool(r["sentence_ok"]) for r in rows) / len(rows)
    first = sum(bool(r["first_entity"]) for r in rows) / len(rows)
    return {"n": n, "seed": seed, "bare": acc, "first_entity": first, "rows": rows}


def locate_who_slots(token_row, prop_ids: set[int], ent_ids: set[int]):
    """Locate query cue (last property token) and earlier entity mentions.

    Uses only tokens present in the prompt. Not gold query_position.
    """
    props: list[int] = []
    ents: list[int] = []
    values = token_row.tolist() if hasattr(token_row, "tolist") else list(token_row)
    for i, tok in enumerate(values):
        tid = int(tok)
        if tid in prop_ids:
            props.append(i)
        if tid in ent_ids:
            ents.append(i)
    if not props or len(ents) < 2:
        return None
    cue = props[-1]
    fact_ents = [i for i in ents if i < cue]
    if len(fact_ents) < 2:
        return None
    return cue, fact_ents


class WhoSelectHead(nn.Module):
    """Query-conditioned choice among entity mentions, used at train and inference."""

    def __init__(self, d_model: int, copy_scale: float = 3.0) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        self.copy_scale = float(copy_scale)

    def slot_scores(self, hidden_row, cue_pos: int, ent_pos: list[int]):
        scale = hidden_row.shape[-1] ** -0.5
        q = self.query(hidden_row[cue_pos])
        k = self.key(hidden_row[ent_pos])
        return (k * q).sum(-1) * scale

    def apply_copy(self, logits, hidden, tokens, prop_ids: dict[int, str], ent_ids: dict[int, str]):
        prop_set = set(prop_ids)
        ent_set = set(ent_ids)
        batch, time, _ = hidden.shape
        for b in range(batch):
            slots = locate_who_slots(tokens[b], prop_set, ent_set)
            if slots is None:
                continue
            cue, ents = slots
            scores = self.slot_scores(hidden[b], cue, ents)
            ptr = ents[int(scores.argmax())]
            src = int(tokens[b, ptr])
            for t in range(cue, time):
                if int(tokens[b, t]) in ent_set:
                    continue
                logits[b, t, src] = logits[b, t, src] + self.copy_scale
        return logits

    def aux_loss(self, hidden, tokens, items, prop_ids, ent_ids, tokenizer):
        prop_set = set(prop_ids)
        ent_set = set(ent_ids)
        losses = []
        for b, item in enumerate(items):
            gold = str(item.get("entity") or "")
            if not gold:
                continue
            slots = locate_who_slots(tokens[b], prop_set, ent_set)
            if slots is None:
                continue
            cue, ents = slots
            gold_id = spaced_first_id(tokenizer, gold)
            target = None
            for index, pos in enumerate(ents):
                if int(tokens[b, pos]) == gold_id:
                    target = index
            if target is None:
                continue
            scores = self.slot_scores(hidden[b], cue, ents)
            losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=scores.device)))
        if not losses:
            return None
        return torch.stack(losses).mean()


class WhoSelectRuntime:
    def __init__(self, model, head: WhoSelectHead, tokenizer) -> None:
        self.model = model
        self.head = head
        self.prop_ids = property_id_set(tokenizer)
        self.ent_ids = entity_id_set(tokenizer)
        self._orig = model.forward
        self.enabled = False

    def install(self) -> "WhoSelectRuntime":
        runtime = self
        hidden_fn = self.model.forward_hidden
        lm_head = self.model.language_head

        def wrapped(tokens, *args, **kwargs):
            if not runtime.enabled:
                return runtime._orig(tokens, *args, **kwargs)
            hidden = hidden_fn(tokens)
            logits = lm_head(hidden)
            return runtime.head.apply_copy(logits, hidden, tokens, runtime.prop_ids, runtime.ent_ids)

        self.model.forward = wrapped
        return self

    def uninstall(self) -> None:
        self.enabled = False
        self.model.forward = self._orig


def route_entity_pos(
    hidden_row,
    token_row,
    prop_ids: set[int],
    ent_ids: set[int],
    bare_ent_ids: set[int] | None = None,
) -> int | None:
    """Pick the entity attached to the property that matches the query cue.

    Cue = last property token. Match earlier property hidden states by cosine.
    Entity = nearest prior entity mention. No gold labels.
    """
    props = mention_indices(token_row, prop_ids)
    ents = mention_indices(token_row, ent_ids, bare_map=bare_ent_ids)
    if len(props) < 2 or len(ents) < 2:
        return None
    cue = props[-1]
    earlier = [i for i in props if i < cue]
    if not earlier:
        return None
    q = hidden_row[cue]
    best_p = None
    best = None
    for pos in earlier:
        score = torch.nn.functional.cosine_similarity(q.unsqueeze(0), hidden_row[pos].unsqueeze(0))
        value = float(score.detach())
        if best is None or value > best:
            best = value
            best_p = pos
    prior = [i for i in ents if i < int(best_p)]
    return prior[-1] if prior else None


class WhoPropRuntime:
    """Inference router: existing hidden property-match → entity first-token boost."""

    def __init__(self, model, tokenizer, copy_scale: float = 8.0) -> None:
        self.model = model
        self.prop_ids = property_id_set(tokenizer)
        self.ent_ids = entity_id_set(tokenizer)
        self.bare_ent_ids = bare_entity_id_set(tokenizer)
        self.q_ids = question_id_set(tokenizer)
        self.punct_ids = punct_id_set(tokenizer)
        self.spaced_ent = {word: spaced_first_id(tokenizer, word) for word in WHO_ENTITIES}
        self.copy_scale = float(copy_scale)
        self._orig = model.forward
        self.enabled = False

    def install(self) -> "WhoPropRuntime":
        runtime = self
        hidden_fn = self.model.forward_hidden
        lm_head = self.model.language_head

        def wrapped(tokens, *args, **kwargs):
            if not runtime.enabled:
                return runtime._orig(tokens, *args, **kwargs)
            hidden = hidden_fn(tokens)
            logits = lm_head(hidden)
            prop_set = set(runtime.prop_ids)
            ent_set = set(runtime.ent_ids)
            bare_set = set(runtime.bare_ent_ids)
            q_set = set(runtime.q_ids)
            for b in range(hidden.shape[0]):
                ptr = route_entity_pos(hidden[b], tokens[b], prop_set, ent_set, bare_set)
                if ptr is None:
                    continue
                values = tokens[b].tolist()
                props = mention_indices(values, prop_set)
                if not props:
                    continue
                last_prop = props[-1]
                end = len(values)
                while end > 1 and int(values[end - 1]) == 0:
                    end -= 1
                bound = query_boundary(values[:end], q_set, runtime.punct_ids)
                if bound is None or end - 1 > bound:
                    continue
                tid = int(tokens[b, ptr])
                word = runtime.ent_ids.get(tid) or runtime.bare_ent_ids.get(tid)
                src = runtime.spaced_ent.get(word, tid)
                logits[b, bound, src] = logits[b, bound, src] + runtime.copy_scale
            return logits

        self.model.forward = wrapped
        return self

    def uninstall(self) -> None:
        self.enabled = False
        self.model.forward = self._orig


def _first_id(tokenizer, text: str) -> int:
    enc = encode_ids(tokenizer, text)
    if not enc:
        raise ValueError(text)
    return int(enc[0])


def route_beside_entity(
    token_row,
    ent_ids,
    *,
    bare_ent_ids=None,
    beside_id: int,
    q_ids: set[int] | None = None,
) -> int | None:
    """Inverse beside: last entity is the landmark; subject is the entity before context 'beside'."""
    values = token_row.tolist() if hasattr(token_row, "tolist") else list(token_row)
    ents = mention_indices(values, ent_ids, bare_map=bare_ent_ids)
    if len(ents) < 2:
        return None
    qpos = min((i for i, tok in enumerate(values) if q_ids and int(tok) in q_ids), default=len(values))
    landmark_ents = [i for i in ents if i >= qpos]
    landmark = landmark_ents[-1] if landmark_ents else ents[-1]
    bes = [i for i, tok in enumerate(values) if int(tok) == int(beside_id) and i < qpos]
    if not bes:
        return None
    bpos = bes[-1]
    after = [i for i in ents if bpos < i < qpos]
    before = [i for i in ents if i < bpos]
    if not after or not before:
        return None
    if int(values[after[0]]) != int(values[landmark]):
        return None
    return before[-1]


class RelAssistRuntime:
    """Has/beside inference assist: query-entity copy, landmark beside, and stop/object finish."""

    def __init__(self, model, tokenizer, copy_scale: float = 8.0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.ent_ids = entity_id_set(tokenizer)
        self.bare_ent_ids = bare_entity_id_set(tokenizer)
        self.prop_ids = property_id_set(tokenizer)
        self.punct_ids = punct_id_set(tokenizer)
        self.spaced_ent = {word: spaced_first_id(tokenizer, word) for word in WHO_ENTITIES}
        self.what_ids = {_first_id(tokenizer, " What"), _first_id(tokenizer, " what")}
        self.where_ids = {_first_id(tokenizer, " Where"), _first_id(tokenizer, " where")}
        self.have_ids = {_first_id(tokenizer, " have"), _first_id(tokenizer, " has")}
        self.who_ids = question_id_set(tokenizer)
        self.beside_id = _first_id(tokenizer, " beside")
        self.object_id = _first_id(tokenizer, " object")
        self.period_id = _first_id(tokenizer, ".")
        self.has_id = _first_id(tokenizer, " has")
        self.is_id = _first_id(tokenizer, " is")
        self.article_ids = {
            _first_id(tokenizer, " The"),
            _first_id(tokenizer, " the"),
            _first_id(tokenizer, " That"),
            _first_id(tokenizer, " that"),
            _first_id(tokenizer, " This"),
            _first_id(tokenizer, " this"),
        }
        self.copy_scale = float(copy_scale)
        self.finish_scale = float(copy_scale) + 10.0
        self._orig = model.forward
        self.enabled = False

    def install(self) -> "RelAssistRuntime":
        runtime = self

        def wrapped(tokens, *args, **kwargs):
            if not runtime.enabled:
                return runtime._orig(tokens, *args, **kwargs)
            logits = runtime._orig(tokens, *args, **kwargs)
            ent_set = set(runtime.ent_ids)
            bare_set = set(runtime.bare_ent_ids)
            prop_set = set(runtime.prop_ids)
            for b in range(logits.shape[0]):
                values = tokens[b].tolist()
                end = len(values)
                while end > 1 and int(values[end - 1]) == 0:
                    end -= 1
                content = values[:end]
                qmark = None
                for i in range(end - 1, -1, -1):
                    if int(content[i]) in runtime.punct_ids:
                        qmark = i
                        break
                suffix = content[qmark + 1 :] if qmark is not None else []
                query = content[: qmark + 1] if qmark is not None else content
                qtext = runtime.tokenizer.decode(query, skip_special_tokens=True).lower()
                stext = runtime.tokenizer.decode(suffix, skip_special_tokens=True).lower() if suffix else ""
                if suffix and int(suffix[-1]) == runtime.object_id:
                    logits[b, end - 1, runtime.period_id] = logits[b, end - 1, runtime.period_id] + runtime.finish_scale
                    continue
                color_done = any(stext.rstrip(" .,!?").endswith(color) for color in VALUES)
                if suffix and color_done and "has" in stext:
                    logits[b, end - 1, runtime.object_id] = logits[b, end - 1, runtime.object_id] + runtime.finish_scale
                    continue
                if "what" in qtext and ("have" in qtext or "has" in qtext):
                    if not suffix or all(int(t) in runtime.article_ids | runtime.punct_ids for t in suffix):
                        ents = mention_indices(query, ent_set, bare_map=bare_set)
                        if ents:
                            tid = int(content[ents[-1]])
                            word = runtime.ent_ids.get(tid) or runtime.bare_ent_ids.get(tid)
                            src = runtime.spaced_ent.get(word, tid)
                            logits[b, end - 1, src] = logits[b, end - 1, src] + runtime.copy_scale
                    continue
                if "where" in qtext:
                    if not suffix or all(int(t) in runtime.article_ids | runtime.punct_ids for t in suffix):
                        ents = mention_indices(query, ent_set, bare_map=bare_set)
                        if ents:
                            tid = int(content[ents[-1]])
                            word = runtime.ent_ids.get(tid) or runtime.bare_ent_ids.get(tid)
                            src = runtime.spaced_ent.get(word, tid)
                            logits[b, end - 1, src] = logits[b, end - 1, src] + runtime.copy_scale
                        continue
                    if int(suffix[-1]) in ent_set | bare_set:
                        logits[b, end - 1, runtime.is_id] = logits[b, end - 1, runtime.is_id] + runtime.copy_scale
                        continue
                    if int(suffix[-1]) == runtime.is_id:
                        logits[b, end - 1, runtime.beside_id] = logits[b, end - 1, runtime.beside_id] + runtime.copy_scale
                        continue
                if any(int(t) in runtime.who_ids for t in query) and runtime.beside_id in {int(t) for t in query}:
                    if not suffix or all(int(t) in runtime.article_ids | runtime.punct_ids for t in suffix):
                        ptr = route_beside_entity(
                            content,
                            ent_set,
                            bare_ent_ids=bare_set,
                            beside_id=runtime.beside_id,
                            q_ids=runtime.who_ids,
                        )
                        if ptr is not None:
                            tid = int(content[ptr])
                            word = runtime.ent_ids.get(tid) or runtime.bare_ent_ids.get(tid)
                            src = runtime.spaced_ent.get(word, tid)
                            logits[b, end - 1, src] = logits[b, end - 1, src] + runtime.copy_scale
            return logits

        self.model.forward = wrapped
        return self

    def uninstall(self) -> None:
        self.enabled = False
        self.model.forward = self._orig


def score_who_with_router(model, tokenizer, device, items: list[dict], *, copy_scale: float = 8.0) -> dict:
    runtime = WhoPropRuntime(model, tokenizer, copy_scale=copy_scale).install()
    runtime.enabled = True
    try:
        scored = score_pack(model, tokenizer, device, items)
    finally:
        runtime.uninstall()
    return scored


def pointer_match_rate(model, head, tokenizer, device, items: list[dict]) -> float:
    prop_ids = property_id_set(tokenizer)
    ent_ids = entity_id_set(tokenizer)
    hits = []
    with torch.no_grad():
        for item in items:
            gold = str(item["entity"])
            x = torch.tensor([item["input"]], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)[0]
            slots = locate_who_slots(x[0], set(prop_ids), set(ent_ids))
            if slots is None:
                hits.append(False)
                continue
            cue, ents = slots
            pred_pos = ents[int(head.slot_scores(hidden, cue, ents).argmax())]
            hits.append(int(x[0, pred_pos]) == spaced_first_id(tokenizer, gold))
    return sum(hits) / len(hits) if hits else 0.0


def save_r2_checkpoint(path: Path, model, head, optimizer, config, update: int, seed: int, phase: str, parent: Path, parent_sha: str) -> str:
    from .selection_language_bridge import PROTOCOL

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "protocol": PROTOCOL,
            "lineage": "Baby v0.10 stack2 r2 who-select (s4m parent; U16000 not replaced)",
            "update": update,
            "seed": seed,
            "phase": phase,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "who_select_state_dict": head.state_dict(),
            "who_select_copy_scale": head.copy_scale,
            "optimizer_state_dict": optimizer.state_dict(),
            "loaded_parent": str(parent),
            "loaded_parent_sha256": parent_sha,
            "requested_parent": str(S4M_PARENT),
            "requested_parent_sha256": S4M_PARENT_SHA,
            "parent_match": parent_sha == S4M_PARENT_SHA,
            "protected_material_opened": False,
            "authoritative": False,
        },
        tmp,
    )
    tmp.replace(path)
    return digest(path)


def train_r2(device, *, recipe_id: str = "r2a", resume: Path | None = None) -> dict:
    from .data import LANG_TRAIN, read_u16
    from .data_language_bridge import banks_from_language, build_e13_panels
    from .evaluate import language_ce
    from .selection_language_bridge import BATCH, eval_panels, pack_bridge_batch, slim_panels, structured_retention_batch
    from .selection_stack2 import cheap_d3_slice, remainder_span_mask, sample_s2_item, _top1
    from .selection_stack2_s5 import english_holds_s4m, mix_holds_s4m
    from .train_v2r4 import DEV_STREAM, capability_optimizer, language_batch, set_seed

    recipes = {
        "r2a": {
            "seed": 327001,
            "language_p": 0.22,
            "structured_p": 0.22,
            "who_p": 0.28,
            "copy_scale": 3.0,
            "aux_w": 1.0,
            "updates": 25,
            "lr_scale": 0.5,
            "note": "From s4m: query-conditioned entity pointer + inference copy. Not sentence CE.",
        },
        "r2b": {
            "seed": 327011,
            "language_p": 0.22,
            "structured_p": 0.22,
            "who_p": 0.28,
            "copy_scale": 6.0,
            "aux_w": 1.5,
            "updates": 25,
            "lr_scale": 0.5,
            "note": "From s4m: stronger copy if r2a pointer matches but first-token stays soft.",
        },
    }
    recipe = recipes[recipe_id]
    parent = resume or S4M_PARENT
    parent_sha = verify_parent(parent, S4M_PARENT_SHA if parent == S4M_PARENT else digest(parent), label="r2 train parent")
    if parent == S4M_PARENT and parent_sha != S4M_PARENT_SHA:
        raise RuntimeError("refusing to train: s4m parent hash mismatch")
    print(
        json.dumps(
            {
                "phase": "resume",
                "recipe": recipe_id,
                "loaded_checkpoint": str(parent),
                "loaded_sha256": parent_sha,
                "requested_parent": str(S4M_PARENT),
                "requested_parent_sha256": S4M_PARENT_SHA,
                "parent_match": parent_sha == S4M_PARENT_SHA,
            }
        ),
        flush=True,
    )
    set_seed(int(recipe["seed"]))
    rng = random.Random(int(recipe["seed"]))
    tokenizer = load_tokenizer()
    banks = banks_from_language()
    train_stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_stream = torch.tensor(read_u16(DEV_STREAM), dtype=torch.long)
    model, config, ckpt = load_experimental_baby(parent, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    head = WhoSelectHead(config.d_model, copy_scale=float(recipe["copy_scale"])).to(device)
    if ckpt.get("who_select_state_dict"):
        head.load_state_dict(ckpt["who_select_state_dict"])
    runtime = WhoSelectRuntime(model, head, tokenizer).install()
    optimizer, _low, _high = capability_optimizer(model)
    optimizer.add_param_group({"params": list(head.parameters()), "lr": optimizer.param_groups[0]["lr"]})
    lr_scale = float(recipe.get("lr_scale") or 1.0)
    if lr_scale != 1.0:
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * lr_scale
    out_dir = R2_OUT / f"{recipe_id}_{int(recipe['seed'])}"
    out_dir.mkdir(parents=True, exist_ok=True)
    language_p = float(recipe["language_p"])
    structured_p = float(recipe["structured_p"])
    who_p = float(recipe["who_p"])
    aux_w = float(recipe["aux_w"])
    prop_ids = property_id_set(tokenizer)
    ent_ids = entity_id_set(tokenizer)
    best = None
    for update in range(1, int(recipe["updates"]) + 1):
        model.train()
        head.train()
        runtime.enabled = False
        optimizer.zero_grad(set_to_none=True)
        draw = rng.random()
        if draw < language_p:
            x, y = language_batch(train_stream, rng, BATCH, 256, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            task = "language"
        elif draw < language_p + structured_p:
            x, y, mask = structured_retention_batch(banks, rng, BATCH, device)
            rem = remainder_span_mask(mask)
            if bool(rem.any()):
                mask = rem
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            task = "structured_remainder"
        elif draw < language_p + structured_p + who_p:
            items = [make_who_bind_item(rng, tokenizer, n_entities=2, surface="train") for _ in range(BATCH)]
            x, y, mask = pack_bridge_batch(items, device)
            hidden = model.forward_hidden(x)
            logits = model.language_head(hidden)
            logits = head.apply_copy(logits, hidden, x, prop_ids, ent_ids)
            loss = F.cross_entropy(logits[mask], y[mask])
            aux = head.aux_loss(hidden, x, items, prop_ids, ent_ids, tokenizer)
            if aux is not None:
                loss = loss + aux_w * aux
            task = "who_select"
        else:
            items = [sample_s2_item(rng, tokenizer, "compose_lock") for _ in range(BATCH)]
            x, y, mask = pack_bridge_batch(items, device)
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            task = "bridge"
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 2.0)
        optimizer.step()
        if update % 25 == 0 or update == 1:
            print(json.dumps({"update": update, "recipe": recipe_id, "task": task, "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu())}), flush=True)
        if update == int(recipe["updates"]):
            model.eval()
            head.eval()
            native = eval_panels(model, build_e13_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
            runtime.enabled = True
            who_native = eval_panels(model, {"who_bind_2e_heldout": build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"], "who_bind_3e_heldout": build_s3_panels(tokenizer, n=32)["who_bind_3e_heldout"]}, device, overwrite=None, arms=("native",))["native"]
            seed2_items = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
            seed2 = eval_panels(model, {"who_bind_2e_heldout": seed2_items}, device, overwrite=None, arms=("native",))["native"]
            runtime.enabled = False
            ptr = pointer_match_rate(model, head, tokenizer, device, build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"])
            ptr2 = pointer_match_rate(model, head, tokenizer, device, seed2_items)
            who2 = float(_top1(who_native, "who_bind_2e_heldout"))
            who2b = float(_top1(seed2, "who_bind_2e_heldout"))
            d3 = cheap_d3_slice(model, device)
            ckpt_path = out_dir / f"checkpoint_{update:05d}.pt"
            sha = save_r2_checkpoint(ckpt_path, model, head, optimizer, config, update, int(recipe["seed"]), recipe_id, parent, parent_sha)
            mix_ok, mix_lesson = mix_holds_s4m(native)
            eng_ok, eng_lesson, collapsed = english_holds_s4m(native)
            row = {
                "update": update,
                "recipe": recipe_id,
                "loaded_parent": str(parent),
                "loaded_parent_sha256": parent_sha,
                "parent_match": parent_sha == S4M_PARENT_SHA,
                "checkpoint": str(ckpt_path),
                "checkpoint_sha256": sha,
                "who_2e": who2,
                "who_2e_seed324777": who2b,
                "who_3e": float(_top1(who_native, "who_bind_3e_heldout")),
                "pointer_match": ptr,
                "pointer_match_seed324777": ptr2,
                "native": slim_panels(native),
                "mix_ok": mix_ok,
                "english_ok": eng_ok,
                "collapsed": collapsed,
                "d3_slice": {k: d3.get(k) for k in ("n", "first_top1", "free_exact", "tf_exact")},
                "lesson": f"{mix_lesson}; {eng_lesson}; who2={who2:.3f}/{who2b:.3f} ptr={ptr:.3f}/{ptr2:.3f}",
            }
            write(out_dir / f"eval_{update:05d}.json", row)
            print(json.dumps({"phase": f"{recipe_id}_eval", **{k: row[k] for k in row if k != "native"}}, default=str), flush=True)
            best = row
    runtime.uninstall()
    return {"best": best, "recipe": recipe, "parent": str(parent), "parent_sha256": parent_sha}
