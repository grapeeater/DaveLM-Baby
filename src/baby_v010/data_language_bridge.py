from __future__ import annotations

"""Controlled language-bridge items for Baby v0.10.

Same bind/retrieval job as (key, value, SEP) tiling, with progressively less
rigid surfaces. Vocab stays the frozen 1024-piece tokenizer. Never reads gold
query_position. Never opens TEST/FINAL/SACRED.
"""

import hashlib
import random
from pathlib import Path

from .data import BOS, EOS, LANG_TRAIN, Banks, build_banks, sample_span
from .data_v2 import _filler, _reserved_surface_tokens, _sample, _surface
from .selection_s1 import digest

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
EXPECTED_TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"

ENTITIES = ("cat", "dog", "bird", "frog", "bear", "fox", "hen", "pig", "cow", "duck")
VALUES = ("red", "green", "yellow", "pink", "white", "blue")
# First-token IDs of " {word}" must be unique and disjoint from VALUES.
SIZES = ("small", "short", "wide", "thin", "tiny", "huge")
# First-token IDs of " {word}" must be unique and disjoint from VALUES and SIZES.
PLACES = ("nest", "lake", "cave", "town", "park", "farm")

TRAIN_FACTS = (
    "The color of the {e} is {v}.",
    "The {e} is {v}.",
    "{e} has the color {v}.",
    "The {e} appears {v}.",
    "Note the {e} is {v}.",
    "This {e} is {v}.",
    "{e}: {v}.",
    "Keep in mind the {e} is {v}.",
    "See the {e}. It is {v}.",
    "Here the {e} is {v}.",
)
HOLDOUT_FACTS = (
    "{e} looks {v}.",
    "Remember: the {e} is {v}.",
    "That {e} is {v}.",
)
TRAIN_QUERIES = (
    "What color is the {e}?",
    "What color is {e}?",
    "What is {e}'s color?",
    "Do you know the color of the {e}?",
    "Can you say the color of the {e}?",
    "Please tell the color of the {e}.",
    "Name the color of the {e}.",
    "The {e} has what color?",
)
HOLDOUT_QUERIES = (
    "Which color is the {e}?",
    "Tell me the color of the {e}.",
    "What is the color of the {e}?",
)
INSTRUCTION_PREFIXES = (
    "Answer with one word. ",
    "Reply with only the color. ",
)
HOLDOUT_INSTRUCTIONS = (
    "Give just the color. ",
    "Name the color only. ",
)
TRAIN_SIZE_FACTS = (
    "The size of the {e} is {v}.",
    "The {e} is {v} in size.",
    "Note the {e} is {v} in size.",
    "This {e} is {v} in size.",
    "Keep in mind the {e} is {v} in size.",
)
HOLDOUT_SIZE_FACTS = (
    "Remember: the {e} is {v} in size.",
    "That {e} is {v} in size.",
    "{e} looks {v} in size.",
)
TRAIN_SIZE_QUERIES = (
    "What size is the {e}?",
    "What size is {e}?",
    "Please tell the size of the {e}.",
    "Name the size of the {e}.",
)
HOLDOUT_SIZE_QUERIES = (
    "Which size is the {e}?",
    "Tell me the size of the {e}.",
    "What is the size of the {e}?",
)
TRAIN_PLACE_FACTS = (
    "The {e} lives at the {v}.",
    "The {e} is at the {v}.",
    "Note the {e} lives at the {v}.",
    "This {e} lives at the {v}.",
    "Keep in mind the {e} is at the {v}.",
)
HOLDOUT_PLACE_FACTS = (
    "Remember: the {e} lives at the {v}.",
    "That {e} is at the {v}.",
    "{e} lives at the {v}.",
)
TRAIN_PLACE_QUERIES = (
    "Where does the {e} live?",
    "Where is the {e}?",
    "Please tell where the {e} lives.",
    "Name the home of the {e}.",
)
HOLDOUT_PLACE_QUERIES = (
    "Where does {e} live?",
    "Tell me where the {e} lives.",
    "What is the home of the {e}?",
)

_TOKENIZER = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    if digest(TOKENIZER_PATH) != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("tokenizer hash mismatch")
    from tokenizers import Tokenizer

    _TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
    return _TOKENIZER


def encode_ids(tokenizer, text: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(text).ids]


def encode_split(tokenizer, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    """Split prompt/answer on a verified BPE prefix of the concatenated string."""
    prompt_ids = encode_ids(tokenizer, prompt)
    full_ids = encode_ids(tokenizer, prompt + answer)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"BPE prefix split failed for {prompt!r} + {answer!r}")
    answer_ids = full_ids[len(prompt_ids) :]
    if not answer_ids:
        raise ValueError(f"empty answer ids for {prompt!r} + {answer!r}")
    return prompt_ids, answer_ids


def _item(
    *,
    prompt_ids: list[int],
    answer_ids: list[int],
    kind: str,
    family: str,
    surface: str,
    variant: str,
    extra: dict | None = None,
) -> dict:
    row = [BOS, *prompt_ids]
    target = list(answer_ids)
    if len(row) + len(target) > 255:
        raise RuntimeError("language-bridge item exceeds context")
    payload = {
        "input": row,
        "target": target,
        "target_span": list(answer_ids),
        "kind": kind,
        "family": family,
        "surface": surface,
        "variant": variant,
        "query_position": None,
    }
    if extra:
        payload.update(extra)
    return payload


def spaced_first_id(tokenizer, word: str) -> int:
    ids = encode_ids(tokenizer, " " + word)
    if not ids:
        raise ValueError(word)
    return int(ids[0])


def period_token_id(tokenizer) -> int:
    ids = encode_ids(tokenizer, ".")
    if len(ids) != 1:
        raise RuntimeError(f"period is not a single token: {ids}")
    return int(ids[0])


def _facts_and_query(
    rng: random.Random,
    n_facts: int,
    *,
    fact_pool: tuple[str, ...],
    query_pool: tuple[str, ...],
    value_bank: tuple[str, ...] = VALUES,
    instruction: str | None = None,
    pronoun: bool = False,
) -> tuple[str, str, str, str]:
    entities = list(_sample(rng, ENTITIES, n_facts))
    colors = list(_sample(rng, value_bank, n_facts))
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    query_e, query_v = rng.choice(pairs)
    fact_t = rng.choice(fact_pool)
    query_t = rng.choice(query_pool)
    if pronoun:
        others = [pair for pair in pairs if pair[0] != query_e]
        ordered = others + [(query_e, query_v)]
        fact_text = " ".join(fact_t.format(e=e, v=v) for e, v in ordered)
        query_text = "What color is it?"
    else:
        fact_text = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
        query_text = query_t.format(e=query_e)
    prompt = fact_text + " " + query_text
    if instruction:
        prompt = instruction + prompt
    answer = " " + query_v if not query_text.endswith(" is") else " " + query_v
    if query_text.endswith(" is"):
        answer = " " + query_v + "."
    return prompt, answer, query_e, query_v


def make_english_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
    family: str = "qa",
    instruction: bool = False,
    pronoun: bool = False,
) -> dict:
    if n_facts < 1 or n_facts > len(ENTITIES):
        raise ValueError(n_facts)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    if family == "cloze":
        query_pool = ("The {e} is", "The color of the {e} is")
    elif surface == "train":
        query_pool = TRAIN_QUERIES
    else:
        query_pool = HOLDOUT_QUERIES
    instr = None
    if instruction:
        instr = rng.choice(INSTRUCTION_PREFIXES if surface == "train" else HOLDOUT_INSTRUCTIONS)
    prompt, answer, entity, value = _facts_and_query(
        rng,
        n_facts,
        fact_pool=fact_pool,
        query_pool=query_pool,
        instruction=instr,
        pronoun=pronoun,
    )
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    variant = f"{family}_{n_facts}f"
    if instruction:
        variant += "_instr"
    if pronoun:
        variant += "_pronoun"
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family=f"english_{family}",
        surface=surface,
        variant=variant,
        extra={"entity": entity, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": "color"},
    )


def make_size_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
    family: str = "qa",
) -> dict:
    if n_facts < 1 or n_facts > len(ENTITIES):
        raise ValueError(n_facts)
    fact_pool = TRAIN_SIZE_FACTS if surface == "train" else HOLDOUT_SIZE_FACTS
    if family == "cloze":
        query_pool = ("The {e} is", "The size of the {e} is")
    elif surface == "train":
        query_pool = TRAIN_SIZE_QUERIES
    else:
        query_pool = HOLDOUT_SIZE_QUERIES
    prompt, answer, entity, value = _facts_and_query(
        rng,
        n_facts,
        fact_pool=fact_pool,
        query_pool=query_pool,
        value_bank=SIZES,
    )
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_size",
        surface=surface,
        variant=f"size_{family}_{n_facts}f",
        extra={"entity": entity, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": "size"},
    )


def make_place_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
    family: str = "qa",
) -> dict:
    if n_facts < 1 or n_facts > len(ENTITIES):
        raise ValueError(n_facts)
    fact_pool = TRAIN_PLACE_FACTS if surface == "train" else HOLDOUT_PLACE_FACTS
    if family == "cloze":
        query_pool = ("The {e} lives at the", "The {e} is at the")
    elif surface == "train":
        query_pool = TRAIN_PLACE_QUERIES
    else:
        query_pool = HOLDOUT_PLACE_QUERIES
    prompt, answer, entity, value = _facts_and_query(
        rng,
        n_facts,
        fact_pool=fact_pool,
        query_pool=query_pool,
        value_bank=PLACES,
    )
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_place",
        surface=surface,
        variant=f"place_{family}_{n_facts}f",
        extra={"entity": entity, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": "place"},
    )


def make_mixed_item(
    rng: random.Random,
    tokenizer,
    *,
    n_entities: int = 2,
    surface: str = "train",
) -> dict:
    """Color and size facts; query one attribute. Distractor entity when n_entities>1."""
    n_entities = max(1, min(n_entities, len(ENTITIES)))
    entities = list(_sample(rng, ENTITIES, n_entities))
    colors = list(_sample(rng, VALUES, n_entities))
    sizes = list(_sample(rng, SIZES, n_entities))
    color_facts = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    size_facts = TRAIN_SIZE_FACTS if surface == "train" else HOLDOUT_SIZE_FACTS
    color_queries = TRAIN_QUERIES if surface == "train" else HOLDOUT_QUERIES
    size_queries = TRAIN_SIZE_QUERIES if surface == "train" else HOLDOUT_SIZE_QUERIES
    rows = list(zip(entities, colors, sizes))
    rng.shuffle(rows)
    ask_color = rng.random() < 0.5
    query_e, query_c, query_s = rng.choice(rows)
    parts: list[str] = []
    for entity, color, size in rows:
        parts.append(rng.choice(color_facts).format(e=entity, v=color))
        parts.append(rng.choice(size_facts).format(e=entity, v=size))
    rng.shuffle(parts)
    if ask_color:
        query = rng.choice(color_queries).format(e=query_e)
        value = query_c
        attr = "color"
    else:
        query = rng.choice(size_queries).format(e=query_e)
        value = query_s
        attr = "size"
    prompt = " ".join(parts) + " " + query
    answer = " " + value
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_mixed",
        surface=surface,
        variant=f"mixed_{n_entities}e",
        extra={
            "entity": query_e,
            "value_text": value,
            "prompt_text": prompt,
            "answer_text": answer,
            "attr": attr,
        },
    )


def make_aperiodic_item(
    rng: random.Random,
    banks: Banks,
    *,
    pair_count: int = 3,
    value_len: int = 2,
    surface: str = "train",
) -> dict:
    """(key, value, SEP) records with *uneven* filler so C2/D3 tiling should fail."""
    heldout = surface != "train"
    markers, sep = _surface(banks, heldout, rng)
    key_pool = [x for x in banks.key if x not in _reserved_surface_tokens(banks)]
    keys = _sample(rng, key_pool, pair_count)
    records = [(key, sample_span(rng, banks, value_len, low_prior=False)) for key in keys]
    query_key, answer = rng.choice(records)
    rendered = list(records)
    rng.shuffle(rendered)
    used = set(markers) | {sep} | set(keys)
    used.update(tok for _, value in rendered for tok in value)
    body: list[int] = []
    for i, (key, value) in enumerate(rendered):
        body.extend([key, *value, sep])
        if i < len(rendered) - 1:
            gap = rng.choice((1, 2, 3, 5, 8, 12) if surface == "train" else (2, 4, 7, 11))
            body.extend(_filler(rng, banks, gap, used))
    row = [BOS, *body, markers[0], query_key, markers[1]]
    target = list(answer)
    if len(row) + len(target) > 255:
        raise RuntimeError("aperiodic item exceeds context")
    return {
        "input": row,
        "target": target,
        "target_span": list(answer),
        "kind": "keyed",
        "family": "aperiodic",
        "surface": surface,
        "variant": f"aperiodic_{pair_count}",
        "query_key": query_key,
        "query_position": None,
    }


def make_syntax_item(
    rng: random.Random,
    banks: Banks,
    tokenizer,
    *,
    pair_count: int = 2,
    value_len: int = 1,
    surface: str = "train",
) -> dict:
    """English wrapper words around raw key/value token IDs."""
    heldout = surface != "train"
    markers, sep = _surface(banks, heldout, rng)
    key_pool = [x for x in banks.key if x not in _reserved_surface_tokens(banks)]
    keys = _sample(rng, key_pool, pair_count)
    records = [(key, sample_span(rng, banks, value_len, low_prior=False)) for key in keys]
    query_key, answer = rng.choice(records)
    rendered = list(records)
    rng.shuffle(rendered)
    if surface == "train":
        left, mid, between, ask = "Note ", " is ", ". Also ", "What is "
    else:
        left, mid, between, ask = "Record ", " equals ", ". Then ", "Name "
    ids: list[int] = []
    for i, (key, value) in enumerate(rendered):
        if i == 0:
            ids.extend(encode_ids(tokenizer, left))
        else:
            ids.extend(encode_ids(tokenizer, between))
        ids.append(int(key))
        ids.extend(encode_ids(tokenizer, mid))
        ids.extend(int(x) for x in value)
    ids.extend(encode_ids(tokenizer, ". " + ask))
    ids.append(int(query_key))
    ids.extend(encode_ids(tokenizer, "?"))
    return _item(
        prompt_ids=ids,
        answer_ids=[int(x) for x in answer],
        kind="keyed",
        family="syntax_wrap",
        surface=surface,
        variant=f"syntax_{pair_count}",
        extra={"query_key": query_key, "sep": sep, "markers": list(markers)},
    )


def make_rigid_item(rng: random.Random, banks: Banks, *, difficulty: str = "primitive") -> dict:
    from .data_v2 import make_item

    item = make_item(rng, banks, difficulty=difficulty, kind="keyed")
    item["family"] = "rigid"
    item["query_position"] = None
    return item


def build_probe_panels(banks: Banks, tokenizer, seed: int = 310001, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    panels = {
        "rigid_primitive": [make_rigid_item(rng, banks, difficulty="primitive") for _ in range(n)],
        "aperiodic": [make_aperiodic_item(rng, banks, pair_count=3, surface="heldout") for _ in range(n)],
        "syntax_wrap": [make_syntax_item(rng, banks, tokenizer, pair_count=2, surface="heldout") for _ in range(n)],
        "cloze_1fact": [make_english_item(rng, tokenizer, n_facts=1, family="cloze", surface="heldout") for _ in range(n)],
        "cloze_2fact": [make_english_item(rng, tokenizer, n_facts=2, family="cloze", surface="heldout") for _ in range(n)],
        "qa_2fact_train": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train") for _ in range(n)],
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "qa_3fact_heldout": [make_english_item(rng, tokenizer, n_facts=3, family="qa", surface="heldout") for _ in range(n)],
    }
    return panels


def make_dialogue_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "heldout",
) -> dict:
    """Short two-role context: facts, then Human/Baby turn."""
    entities = list(_sample(rng, ENTITIES, n_facts))
    colors = list(_sample(rng, VALUES, n_facts))
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    query_e, query_v = rng.choice(pairs)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
    if surface == "train":
        human = f"what color is the {query_e}?"
    else:
        human = f"which color is the {query_e}?"
    prompt = f"{facts}\nHuman: {human}\nBaby:"
    answer = " " + query_v
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="dialogue",
        surface=surface,
        variant=f"dialogue_{n_facts}f",
        extra={"entity": query_e, "value_text": query_v, "prompt_text": prompt, "answer_text": answer},
    )


def make_multiturn_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "heldout",
) -> dict:
    """Two Human/Baby turns; second question is the scored target."""
    if n_facts < 2:
        raise ValueError("multiturn needs at least two facts")
    entities = list(_sample(rng, ENTITIES, n_facts))
    colors = list(_sample(rng, VALUES, n_facts))
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    (e1, v1), (e2, v2) = pairs[0], pairs[1]
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
    if surface == "train":
        h1 = f"what color is the {e1}?"
        h2 = f"what color is the {e2}?"
    else:
        h1 = f"which color is the {e1}?"
        h2 = f"tell me the color of the {e2}."
    prompt = f"{facts}\nHuman: {h1}\nBaby: {v1}\nHuman: {h2}\nBaby:"
    answer = " " + v2
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="multiturn",
        surface=surface,
        variant=f"multiturn_{n_facts}f",
        extra={"entity": e2, "value_text": v2, "prompt_text": prompt, "answer_text": answer, "prev_entity": e1, "prev_value": v1},
    )


def make_attr_followup_item(
    rng: random.Random,
    tokenizer,
    *,
    surface: str = "heldout",
) -> dict:
    """Turn 1 asks color; turn 2 asks size of a (possibly other) entity."""
    n_facts = 2
    entities = list(_sample(rng, ENTITIES, n_facts))
    colors = list(_sample(rng, VALUES, n_facts))
    sizes = list(_sample(rng, SIZES, n_facts))
    rows = list(zip(entities, colors, sizes))
    rng.shuffle(rows)
    (e1, c1, _s1), (e2, _c2, s2) = rows[0], rows[1]
    color_facts = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    size_facts = TRAIN_SIZE_FACTS if surface == "train" else HOLDOUT_SIZE_FACTS
    parts = []
    for entity, color, size in rows:
        parts.append(rng.choice(color_facts).format(e=entity, v=color))
        parts.append(rng.choice(size_facts).format(e=entity, v=size))
    rng.shuffle(parts)
    facts = " ".join(parts)
    if surface == "train":
        h1 = f"what color is the {e1}?"
        h2 = f"what size is the {e2}?"
    else:
        h1 = f"which color is the {e1}?"
        h2 = f"tell me the size of the {e2}."
    prompt = f"{facts}\nHuman: {h1}\nBaby: {c1}\nHuman: {h2}\nBaby:"
    answer = " " + s2
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="attr_followup",
        surface=surface,
        variant="attr_followup_2e",
        extra={"entity": e2, "value_text": s2, "prompt_text": prompt, "answer_text": answer, "prev_entity": e1, "prev_value": c1, "attr": "size"},
    )


def build_e2_panels(tokenizer, seed: int = 310101, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_4fact_heldout": [make_english_item(rng, tokenizer, n_facts=4, family="qa", surface="heldout") for _ in range(n)],
        "instr_2fact_heldout": [
            make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", instruction=True)
            for _ in range(n)
        ],
        "pronoun_2fact_heldout": [
            make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", pronoun=True)
            for _ in range(n)
        ],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e3_panels(tokenizer, seed: int = 310401, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "multiturn_2fact_heldout": [make_multiturn_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "multiturn_3fact_heldout": [make_multiturn_item(rng, tokenizer, n_facts=3, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "pronoun_2fact_heldout": [
            make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", pronoun=True)
            for _ in range(n)
        ],
    }


def build_e4_panels(tokenizer, seed: int = 310501, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_2fact_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_3fact_heldout": [make_size_item(rng, tokenizer, n_facts=3, family="qa", surface="heldout") for _ in range(n)],
        "mixed_2e_heldout": [make_mixed_item(rng, tokenizer, n_entities=2, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "multiturn_2fact_heldout": [make_multiturn_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e6_panels(tokenizer, seed: int = 310601, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_2fact_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "place_2fact_heldout": [make_place_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "place_3fact_heldout": [make_place_item(rng, tokenizer, n_facts=3, family="qa", surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "multiturn_2fact_heldout": [make_multiturn_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e7_panels(tokenizer, seed: int = 310701, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_2fact_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "instr_2fact_heldout": [
            make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", instruction=True)
            for _ in range(n)
        ],
        "attr_followup_heldout": [make_attr_followup_item(rng, tokenizer, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "multiturn_2fact_heldout": [make_multiturn_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def sample_train_item(
    rng: random.Random,
    banks: Banks,
    tokenizer,
    *,
    phase: str = "e1",
) -> dict:
    """Curriculum draw. phase=e1 is syntax+english bind; e2 adds instruction/pronoun/4-fact."""
    roll = rng.random()
    if phase == "e1":
        if roll < 0.08:
            return make_english_item(rng, tokenizer, n_facts=1, family="cloze", surface="train")
        if roll < 0.22:
            return make_english_item(rng, tokenizer, n_facts=2, family="cloze", surface="train")
        if roll < 0.55:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.72:
            return make_english_item(rng, tokenizer, n_facts=3, family="qa", surface="train")
        if roll < 0.86:
            return make_syntax_item(rng, tokenizer=tokenizer, banks=banks, pair_count=2, surface="train")
        return make_aperiodic_item(rng, banks, pair_count=3, surface="train")
    if phase == "e3":
        if roll < 0.15:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.28:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.48:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", pronoun=True)
        if roll < 0.80:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.92:
            return make_multiturn_item(rng, tokenizer, n_facts=3, surface="train")
        return make_english_item(rng, tokenizer, n_facts=3, family="qa", surface="train")
    if phase == "e4":
        if roll < 0.18:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.28:
            return make_english_item(rng, tokenizer, n_facts=3, family="qa", surface="train")
        if roll < 0.40:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.50:
            return make_size_item(rng, tokenizer, n_facts=3, family="qa", surface="train")
        if roll < 0.68:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.78:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.86:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.94:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", pronoun=True)
    if phase == "e5":
        if roll < 0.14:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.24:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.36:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.48:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.60:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.82:
            return make_multiturn_item(rng, tokenizer, n_facts=3, surface="train")
        if roll < 0.92:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", pronoun=True)
        return make_english_item(rng, tokenizer, n_facts=3, family="qa", surface="train")
    if phase == "e6":
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.22:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.32:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.50:
            return make_place_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.62:
            return make_place_item(rng, tokenizer, n_facts=3, family="qa", surface="train")
        if roll < 0.74:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.86:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.94:
            return make_multiturn_item(rng, tokenizer, n_facts=3, surface="train")
        return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", pronoun=True)
    if phase == "e7":
        if roll < 0.14:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.26:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.36:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.56:
            return make_attr_followup_item(rng, tokenizer, surface="train")
        if roll < 0.68:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.80:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.90:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", pronoun=True)
    if roll < 0.12:
        return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
    if roll < 0.22:
        return make_english_item(rng, tokenizer, n_facts=3, family="qa", surface="train")
    if roll < 0.32:
        return make_english_item(rng, tokenizer, n_facts=4, family="qa", surface="train")
    if roll < 0.45:
        return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
    if roll < 0.62:
        return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", pronoun=True)
    if roll < 0.90:
        return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
    if roll < 0.96:
        return make_syntax_item(rng, tokenizer=tokenizer, banks=banks, pair_count=3, surface="train")
    return make_aperiodic_item(rng, banks, pair_count=4, surface="train")


def banks_from_language() -> Banks:
    from .data import read_u16

    return build_banks(read_u16(LANG_TRAIN))
