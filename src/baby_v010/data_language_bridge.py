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
# Who/event answers use this subset: unique spaced first-tokens, disjoint from
# VALUES/SIZES (cat/cow, frog/fox, bird/blue, pig/pink collide on first token).
WHO_ENTITIES = ("dog", "hen", "duck", "bear", "cat", "frog")
TRAIN_EVENTS = ("sat down", "ran by", "looked up", "came out")
HOLDOUT_EVENTS = ("went home", "came by", "walked along", "went out")
TRAIN_STORY_FRAMES = (
    "Once upon a time a {e1} {a1}. The {e1} was {v1}. Then a {e2} {a2}. The {e2} was {v2}.",
    "One day a {e1} {a1}. The {e1} was {v1}. A {e2} {a2}. The {e2} was {v2}.",
    "A {e1} {a1}. The {e1} was {v1}. Then a {e2} {a2}. The {e2} was {v2}.",
    "There was a {e1} in the sun. The {e1} {a1}. The {e1} was {v1}. A {e2} {a2}. The {e2} was {v2}.",
    "Once upon a time there was a {e1}. The {e1} {a1}. The {e1} was {v1}. Then a {e2} {a2}. The {e2} was {v2}.",
)
HOLDOUT_STORY_FRAMES = (
    "Long ago a {e1} {a1}. That {e1} was {v1}. A {e2} {a2}. That {e2} was {v2}.",
    "A {e1} {a1}. Remember: the {e1} is {v1}. Then a {e2} {a2}. Remember: the {e2} is {v2}.",
    "In the yard a {e1} {a1}. The {e1} looks {v1}. A {e2} {a2}. The {e2} looks {v2}.",
)
TRAIN_WHO_QUERIES = (
    "Who is {v}?",
    "Who was {v}?",
)
HOLDOUT_WHO_QUERIES = (
    "Which one is {v}?",
    "Tell me who is {v}.",
    "Who looks {v}?",
)
TRAIN_EVENT_QUERIES = (
    "Who {act}?",
    "What {act}?",
)
HOLDOUT_EVENT_QUERIES = (
    "Tell me who {act}.",
    "Which one {act}?",
)
TRAIN_STORY_FRAMES_3 = tuple(
    frame + " Then a {e3} {a3}. The {e3} was {v3}." for frame in TRAIN_STORY_FRAMES
)
HOLDOUT_STORY_FRAMES_3 = (
    HOLDOUT_STORY_FRAMES[0] + " A {e3} {a3}. That {e3} was {v3}.",
    HOLDOUT_STORY_FRAMES[1] + " Then a {e3} {a3}. Remember: the {e3} is {v3}.",
    HOLDOUT_STORY_FRAMES[2] + " A {e3} {a3}. The {e3} looks {v3}.",
)
TRAIN_STORY_FRAMES_4 = tuple(
    frame + " Then a {e4} {a4}. The {e4} was {v4}." for frame in TRAIN_STORY_FRAMES_3
)
HOLDOUT_STORY_FRAMES_4 = (
    HOLDOUT_STORY_FRAMES_3[0] + " A {e4} {a4}. That {e4} was {v4}.",
    HOLDOUT_STORY_FRAMES_3[1] + " Then a {e4} {a4}. Remember: the {e4} is {v4}.",
    HOLDOUT_STORY_FRAMES_3[2] + " A {e4} {a4}. The {e4} looks {v4}.",
)
TRAIN_MIXED_STORY_FRAMES = (
    "Once upon a time a {s1} {e1} {a1}. The {e1} was {c1}. Then a {s2} {e2} {a2}. The {e2} was {c2}.",
    "One day a {e1} {a1}. The {e1} was {s1} in size. The {e1} was {c1}. A {e2} {a2}. The {e2} was {s2} in size. The {e2} was {c2}.",
    "A {s1} {e1} {a1}. The {e1} was {c1}. Then a {s2} {e2} {a2}. The {e2} was {c2}.",
    "Once upon a time a {e1} {a1}. The {e1} was {s1} in size. The {e1} was {c1}. Then a {e2} {a2}. The {e2} was {s2} in size. The {e2} was {c2}.",
)
HOLDOUT_MIXED_STORY_FRAMES = (
    "Long ago a {s1} {e1} {a1}. That {e1} was {c1}. A {s2} {e2} {a2}. That {e2} was {c2}.",
    "A {s1} {e1} {a1}. Remember: the {e1} is {c1}. Then a {s2} {e2} {a2}. Remember: the {e2} is {c2}.",
    "In the yard a {s1} {e1} {a1}. The {e1} looks {c1}. A {s2} {e2} {a2}. The {e2} looks {c2}.",
)
TRAIN_COMBINE_QUERIES_COLOR = (
    "What color is the {s} one?",
    "Name the color of the {s} one.",
)
HOLDOUT_COMBINE_QUERIES_COLOR = (
    "Which color is the {s} one?",
    "Tell me the color of the {s} one.",
)
TRAIN_COMBINE_QUERIES_SIZE = (
    "What size is the {c} one?",
    "Name the size of the {c} one.",
)
HOLDOUT_COMBINE_QUERIES_SIZE = (
    "Which size is the {c} one?",
    "Tell me the size of the {c} one.",
)

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
TRAIN_PHRASE_ANSWERS = (
    " The {e} is {v}.",
)
HOLDOUT_PHRASE_ANSWERS = (
    " {e} is {v}.",
)
TRAIN_PHRASE_QUERIES = (
    "Say a sentence about the color of the {e}.",
    "In a sentence, what color is the {e}?",
)
HOLDOUT_PHRASE_QUERIES = (
    "Give a sentence for the color of the {e}.",
    "Please answer in a sentence. Which color is the {e}?",
)
TRAIN_OPEN_QUERIES = (
    "What about the {e}?",
    "And the {e}?",
)
HOLDOUT_OPEN_QUERIES = (
    "How about the {e}?",
    "The {e} then?",
)
TRAIN_ABOUT_QUERIES = (
    "Tell me about the {e}.",
    "Describe the {e}.",
)
HOLDOUT_ABOUT_QUERIES = (
    "What do you know about the {e}?",
    "Talk about the {e}.",
)
# E17: instruction types beyond one-word color QA. Train/hold templates differ.
TRAIN_COPY_QUERIES = (
    "Copy this word: {w}.",
    "Say {w} then stop.",
)
HOLDOUT_COPY_QUERIES = (
    "Write the word {w}.",
    "Please say {w} and stop.",
)
TRAIN_SAYSTOP_QUERIES = (
    "Say the color of the {e} then stop.",
    "Name the color of the {e} then stop.",
)
HOLDOUT_SAYSTOP_QUERIES = (
    "Please say the color of the {e} and stop.",
    "Give the color of the {e} then stop.",
)
TRAIN_FIELD_COLOR_PREFIXES = (
    "Answer only the color. ",
)
HOLDOUT_FIELD_COLOR_PREFIXES = (
    "Give just the color. ",
)
TRAIN_FIELD_SIZE_PREFIXES = (
    "Answer only the size. ",
)
HOLDOUT_FIELD_SIZE_PREFIXES = (
    "Give just the size. ",
)
TRAIN_FIELD_QUERIES = (
    "Now the {e}.",
)
HOLDOUT_FIELD_QUERIES = (
    "Next, the {e}.",
)
TRAIN_NOSTORY_PREFIXES = (
    "Do not go on. ",
    "No story now. ",
)
HOLDOUT_NOSTORY_PREFIXES = (
    "Stop the story. ",
    "Do not continue. ",
)
TRAIN_FORMAT_SENT_PREFIXES = (
    "Answer in one sentence. ",
    "Use a sentence. ",
)
HOLDOUT_FORMAT_SENT_PREFIXES = (
    "Please answer in a sentence. ",
    "Give a short sentence. ",
)
TRAIN_FORMAT_SENT_ANSWERS = (
    " It is {v}.",
)
HOLDOUT_FORMAT_SENT_ANSWERS = (
    " That is {v}.",
)
# First tokens of " true" / " wrong" are disjoint from VALUES/SIZES/ENTITIES.
YES_WORD = "true"
NO_WORD = "wrong"
TRAIN_CHAT_QUERIES = (
    "how about the {e}?",
    "what about the {e}?",
    "and the {e}?",
)
HOLDOUT_CHAT_QUERIES = (
    "the {e} then?",
    "tell me about the {e}.",
    "how is the {e}?",
)
TRAIN_YESNO_QUERIES = (
    "is the {e} {c}?",
    "is that {c} for the {e}?",
)
HOLDOUT_YESNO_QUERIES = (
    "was the {e} {c}?",
    "does the {e} look {c}?",
)
TRAIN_HAPPENED_QUERIES = (
    "Did the {e} {act}?",
    "Did that {e} {act}?",
)
HOLDOUT_HAPPENED_QUERIES = (
    "Did the {e} {act} too?",
    "Did {e} {act}?",
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


def make_phrase_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
) -> dict:
    """Same color bind, but the target is a short period-stopped sentence."""
    if n_facts < 1 or n_facts > len(ENTITIES):
        raise ValueError(n_facts)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    query_pool = TRAIN_PHRASE_QUERIES if surface == "train" else HOLDOUT_PHRASE_QUERIES
    prompt, _one, entity, value = _facts_and_query(
        rng,
        n_facts,
        fact_pool=fact_pool,
        query_pool=query_pool,
    )
    answer = rng.choice(TRAIN_PHRASE_ANSWERS if surface == "train" else HOLDOUT_PHRASE_ANSWERS).format(e=entity, v=value)
    if not answer.startswith(" "):
        answer = " " + answer
    if not answer.endswith("."):
        answer = answer + "."
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    if len(answer_ids) < 3:
        raise RuntimeError(f"phrase answer too short: {answer!r}")
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_phrase",
        surface=surface,
        variant=f"phrase_{n_facts}f",
        extra={"entity": entity, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": "color"},
    )


def make_open_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
    about: bool = False,
) -> dict:
    """Color bind with a less-scripted follow-up instead of Human/Baby or Which-color."""
    if n_facts < 1 or n_facts > len(ENTITIES):
        raise ValueError(n_facts)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    if about:
        query_pool = TRAIN_ABOUT_QUERIES if surface == "train" else HOLDOUT_ABOUT_QUERIES
        family = "english_about"
        variant = f"about_{n_facts}f"
    else:
        query_pool = TRAIN_OPEN_QUERIES if surface == "train" else HOLDOUT_OPEN_QUERIES
        family = "english_open"
        variant = f"open_{n_facts}f"
    prompt, answer, entity, value = _facts_and_query(
        rng,
        n_facts,
        fact_pool=fact_pool,
        query_pool=query_pool,
    )
    if not answer.endswith("."):
        answer = answer + "."
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family=family,
        surface=surface,
        variant=variant,
        extra={"entity": entity, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": "color"},
    )


def _period(word: str) -> str:
    text = word if word.startswith(" ") else " " + word
    return text if text.endswith(".") else text + "."


def make_copy_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
) -> dict:
    """Copy a listed token that is not the bound color (instruction override)."""
    if n_facts < 1 or n_facts > len(ENTITIES):
        raise ValueError(n_facts)
    entities = list(_sample(rng, ENTITIES, n_facts))
    colors = list(_sample(rng, VALUES, n_facts))
    unused = [color for color in VALUES if color not in colors]
    listed = rng.choice(unused)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
    query_pool = TRAIN_COPY_QUERIES if surface == "train" else HOLDOUT_COPY_QUERIES
    query = rng.choice(query_pool).format(w=listed)
    prompt = facts + " " + query
    answer = _period(listed)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_copy",
        surface=surface,
        variant=f"copy_{n_facts}f",
        extra={"entity": listed, "value_text": listed, "prompt_text": prompt, "answer_text": answer, "attr": "copy"},
    )


def make_saystop_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
) -> dict:
    """Color bind with an explicit say-then-stop instruction."""
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    query_pool = TRAIN_SAYSTOP_QUERIES if surface == "train" else HOLDOUT_SAYSTOP_QUERIES
    prompt, answer, entity, value = _facts_and_query(
        rng,
        n_facts,
        fact_pool=fact_pool,
        query_pool=query_pool,
    )
    answer = _period(value)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_saystop",
        surface=surface,
        variant=f"saystop_{n_facts}f",
        extra={"entity": entity, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": "color"},
    )


def make_field_item(
    rng: random.Random,
    tokenizer,
    *,
    n_entities: int = 2,
    surface: str = "train",
) -> dict:
    """Instruction names color vs size; the query does not name the attribute."""
    n_entities = max(1, min(n_entities, len(ENTITIES)))
    entities = list(_sample(rng, ENTITIES, n_entities))
    colors = list(_sample(rng, VALUES, n_entities))
    sizes = list(_sample(rng, SIZES, n_entities))
    color_facts = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    size_facts = TRAIN_SIZE_FACTS if surface == "train" else HOLDOUT_SIZE_FACTS
    rows = list(zip(entities, colors, sizes))
    rng.shuffle(rows)
    query_e, query_c, query_s = rng.choice(rows)
    ask_color = rng.random() < 0.5
    parts: list[str] = []
    for entity, color, size in rows:
        parts.append(rng.choice(color_facts).format(e=entity, v=color))
        parts.append(rng.choice(size_facts).format(e=entity, v=size))
    rng.shuffle(parts)
    facts = " ".join(parts)
    if ask_color:
        prefix_pool = TRAIN_FIELD_COLOR_PREFIXES if surface == "train" else HOLDOUT_FIELD_COLOR_PREFIXES
        value = query_c
        attr = "color"
    else:
        prefix_pool = TRAIN_FIELD_SIZE_PREFIXES if surface == "train" else HOLDOUT_FIELD_SIZE_PREFIXES
        value = query_s
        attr = "size"
    query_pool = TRAIN_FIELD_QUERIES if surface == "train" else HOLDOUT_FIELD_QUERIES
    prompt = rng.choice(prefix_pool) + facts + " " + rng.choice(query_pool).format(e=query_e)
    answer = _period(value)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_field",
        surface=surface,
        variant=f"field_{n_entities}e_{attr}",
        extra={"entity": query_e, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": attr},
    )


def make_nostory_item(
    rng: random.Random,
    tokenizer,
    *,
    surface: str = "train",
) -> dict:
    """Story context, then an instruction to answer the QA instead of continuing."""
    row = make_story_item(rng, tokenizer, surface=surface, ask="color")
    prefix_pool = TRAIN_NOSTORY_PREFIXES if surface == "train" else HOLDOUT_NOSTORY_PREFIXES
    prefix = rng.choice(prefix_pool)
    story = str(row["story_text"])
    # Question is the text after the story in prompt_text.
    suffix = str(row["prompt_text"])[len(story) :].lstrip()
    prompt = story + " " + prefix + suffix
    answer = _period(str(row["value_text"]))
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_nostory",
        surface=surface,
        variant="nostory_color",
        extra={
            "entity": row["entity"],
            "value_text": row["value_text"],
            "prompt_text": prompt,
            "answer_text": answer,
            "attr": "color",
            "story_text": story,
        },
    )


def make_format_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
    sentence: bool = False,
) -> dict:
    """Same color bind; instruction selects one-word vs a short sentence."""
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    query_pool = TRAIN_QUERIES if surface == "train" else HOLDOUT_QUERIES
    prompt, _one, entity, value = _facts_and_query(
        rng,
        n_facts,
        fact_pool=fact_pool,
        query_pool=query_pool,
    )
    if sentence:
        prefix = rng.choice(TRAIN_FORMAT_SENT_PREFIXES if surface == "train" else HOLDOUT_FORMAT_SENT_PREFIXES)
        answer = rng.choice(TRAIN_FORMAT_SENT_ANSWERS if surface == "train" else HOLDOUT_FORMAT_SENT_ANSWERS).format(v=value)
        if not answer.startswith(" "):
            answer = " " + answer
        if not answer.endswith("."):
            answer = answer + "."
        family = "english_format_sent"
        variant = f"format_sent_{n_facts}f"
    else:
        prefix = rng.choice(INSTRUCTION_PREFIXES if surface == "train" else HOLDOUT_INSTRUCTIONS)
        answer = _period(value)
        family = "english_format_word"
        variant = f"format_word_{n_facts}f"
    prompt = prefix + prompt
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family=family,
        surface=surface,
        variant=variant,
        extra={"entity": entity, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": "color"},
    )


def make_varied_dialogue_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "heldout",
    roleplay: bool = False,
) -> dict:
    """Human/Baby or Kid/Mom with open follow-ups instead of only which-color."""
    entities = list(_sample(rng, ENTITIES, n_facts))
    colors = list(_sample(rng, VALUES, n_facts))
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    query_e, query_v = rng.choice(pairs)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
    qpool = TRAIN_CHAT_QUERIES if surface == "train" else HOLDOUT_CHAT_QUERIES
    human = rng.choice(qpool).format(e=query_e)
    if roleplay:
        prompt = f"{facts}\nKid: {human}\nMom:"
        family = "roleplay_varied"
    else:
        prompt = f"{facts}\nHuman: {human}\nBaby:"
        family = "dialogue_varied"
    answer = _period(query_v)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family=family,
        surface=surface,
        variant=f"{family}_{n_facts}f",
        extra={"entity": query_e, "value_text": query_v, "prompt_text": prompt, "answer_text": answer, "attr": "color"},
    )


def make_yesno_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "heldout",
    dialogue: bool = False,
) -> dict:
    """Yes/no over a prior color fact. Answers are true/wrong for first-token scoring."""
    if n_facts < 2:
        raise ValueError("yesno needs at least two facts")
    entities = list(_sample(rng, ENTITIES, n_facts))
    colors = list(_sample(rng, VALUES, n_facts))
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    query_e, query_v = rng.choice(pairs)
    truth = rng.random() < 0.5
    if truth:
        asked = query_v
        value = YES_WORD
    else:
        others = [color for _, color in pairs if color != query_v]
        asked = rng.choice(others)
        value = NO_WORD
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
    qpool = TRAIN_YESNO_QUERIES if surface == "train" else HOLDOUT_YESNO_QUERIES
    query = rng.choice(qpool).format(e=query_e, c=asked)
    if dialogue:
        prompt = f"{facts}\nHuman: {query}\nBaby:"
        family = "yesno_dialogue"
    else:
        prompt = facts + " " + query[0].upper() + query[1:]
        family = "english_yesno"
    answer = _period(value)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family=family,
        surface=surface,
        variant=f"{family}_{n_facts}f",
        extra={"entity": query_e, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": "yesno", "truth": truth},
    )


def make_varied_longturn_item(
    rng: random.Random,
    tokenizer,
    *,
    n_turns: int = 3,
    surface: str = "heldout",
    roleplay: bool = False,
) -> dict:
    """3-4 mixed open turns; last answer is the color bind."""
    if n_turns < 3 or n_turns > 4:
        raise ValueError(n_turns)
    entities = list(_sample(rng, ENTITIES, n_turns))
    colors = list(_sample(rng, VALUES, n_turns))
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
    qpool = TRAIN_CHAT_QUERIES if surface == "train" else HOLDOUT_CHAT_QUERIES
    speaker, responder = ("Kid", "Mom") if roleplay else ("Human", "Baby")
    chunks = [facts]
    for i, (entity, value) in enumerate(pairs[:-1]):
        chunks.append(f"{speaker}: {qpool[i % len(qpool)].format(e=entity)}")
        chunks.append(f"{responder}: {value}.")
    last_e, last_v = pairs[-1]
    chunks.append(f"{speaker}: {qpool[(n_turns - 1) % len(qpool)].format(e=last_e)}")
    chunks.append(f"{responder}:")
    prompt = "\n".join(chunks)
    answer = _period(last_v)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="longturn_varied",
        surface=surface,
        variant=f"longturn_varied_{n_turns}",
        extra={"entity": last_e, "value_text": last_v, "prompt_text": prompt, "answer_text": answer, "attr": "color"},
    )


def make_factreuse_item(
    rng: random.Random,
    tokenizer,
    *,
    surface: str = "heldout",
    roleplay: bool = False,
) -> dict:
    """Ask e1, then e2, then e1 again. Last answer reuses the earlier color fact."""
    e1, e2 = _sample(rng, ENTITIES, 2)
    v1, v2 = _sample(rng, VALUES, 2)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in ((e1, v1), (e2, v2)))
    qpool = TRAIN_CHAT_QUERIES if surface == "train" else HOLDOUT_CHAT_QUERIES
    speaker, responder = ("Kid", "Mom") if roleplay else ("Human", "Baby")
    q1 = qpool[0].format(e=e1)
    q2 = qpool[1 % len(qpool)].format(e=e2)
    q3 = qpool[2 % len(qpool)].format(e=e1)
    prompt = "\n".join(
        (
            facts,
            f"{speaker}: {q1}",
            f"{responder}: {v1}.",
            f"{speaker}: {q2}",
            f"{responder}: {v2}.",
            f"{speaker}: {q3}",
            f"{responder}:",
        )
    )
    answer = _period(v1)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="fact_reuse",
        surface=surface,
        variant="fact_reuse_3t",
        extra={"entity": e1, "value_text": v1, "prompt_text": prompt, "answer_text": answer, "attr": "color", "prev_entity": e2, "prev_value": v2},
    )


def make_happened_item(
    rng: random.Random,
    tokenizer,
    *,
    surface: str = "heldout",
) -> dict:
    """Yes/no over an event without asking who. Not inverted bind."""
    e1, e2 = _sample(rng, ENTITIES, 2)
    v1, v2 = _sample(rng, VALUES, 2)
    events = TRAIN_EVENTS if surface == "train" else HOLDOUT_EVENTS
    a1, a2 = _sample(rng, events, 2)
    frames = TRAIN_STORY_FRAMES if surface == "train" else HOLDOUT_STORY_FRAMES
    story = rng.choice(frames).format(e1=e1, e2=e2, v1=v1, v2=v2, a1=a1, a2=a2)
    truth = rng.random() < 0.5
    if truth:
        query_e, query_act = e1, a1
        value = YES_WORD
    else:
        query_e, query_act = e1, a2
        value = NO_WORD
    qpool = TRAIN_HAPPENED_QUERIES if surface == "train" else HOLDOUT_HAPPENED_QUERIES
    query = rng.choice(qpool).format(e=query_e, act=query_act)
    prompt = story + " " + query
    answer = _period(value)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_happened",
        surface=surface,
        variant="happened_yesno",
        extra={
            "entity": query_e,
            "value_text": value,
            "prompt_text": prompt,
            "answer_text": answer,
            "attr": "yesno",
            "story_text": story,
            "truth": truth,
        },
    )


def make_chat_loop_item(
    rng: random.Random,
    tokenizer,
    *,
    n_turns: int = 4,
    surface: str = "heldout",
    score: str = "color",
) -> dict:
    """Tiny synthetic chat: open color, yes/no on that fact, then the other entity."""
    if n_turns not in {3, 4}:
        raise ValueError(n_turns)
    if score not in {"color", "yesno"}:
        raise ValueError(score)
    e1, e2 = _sample(rng, ENTITIES, 2)
    v1, v2 = _sample(rng, VALUES, 2)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in ((e1, v1), (e2, v2)))
    chat_q = TRAIN_CHAT_QUERIES if surface == "train" else HOLDOUT_CHAT_QUERIES
    yes_q = TRAIN_YESNO_QUERIES if surface == "train" else HOLDOUT_YESNO_QUERIES
    q1 = chat_q[0].format(e=e1)
    q2 = yes_q[0].format(e=e1, c=v1)
    q3 = chat_q[1 % len(chat_q)].format(e=e2)
    chunks = [
        facts,
        f"Human: {q1}",
        f"Baby: {v1}.",
        f"Human: {q2}",
        f"Baby: {YES_WORD}.",
        f"Human: {q3}",
        "Baby:",
    ]
    if score == "yesno":
        # Reuse e1 color after talking about e2.
        q4 = yes_q[min(1, len(yes_q) - 1)].format(e=e1, c=v1)
        chunks = [
            facts,
            f"Human: {q1}",
            f"Baby: {v1}.",
            f"Human: {q3}",
            f"Baby: {v2}.",
            f"Human: {q4}",
            "Baby:",
        ]
        answer = _period(YES_WORD)
        value = YES_WORD
        entity = e1
        attr = "yesno"
    else:
        answer = _period(v2)
        value = v2
        entity = e2
        attr = "color"
    prompt = "\n".join(chunks)
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="chat_loop",
        surface=surface,
        variant=f"chat_loop_{score}_{n_turns}",
        extra={"entity": entity, "value_text": value, "prompt_text": prompt, "answer_text": answer, "attr": attr},
    )


def make_size_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "train",
    family: str = "qa",
    period: bool = False,
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
    if period and not answer.endswith("."):
        answer = answer + "."
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    variant = f"size_{family}_{n_facts}f"
    if period:
        variant += "_stop"
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="english_size",
        surface=surface,
        variant=variant,
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
    combine: bool = False,
    period: bool = False,
    grouped: bool = False,
) -> dict:
    """Color and size facts; query one attribute, or the other via combine."""
    n_entities = max(1, min(n_entities, len(ENTITIES)))
    entities = list(_sample(rng, ENTITIES, n_entities))
    colors = list(_sample(rng, VALUES, n_entities))
    sizes = list(_sample(rng, SIZES, n_entities))
    color_facts = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    size_facts = TRAIN_SIZE_FACTS if surface == "train" else HOLDOUT_SIZE_FACTS
    rows = list(zip(entities, colors, sizes))
    rng.shuffle(rows)
    ask_color = rng.random() < 0.5
    query_e, query_c, query_s = rng.choice(rows)
    if grouped:
        blocks: list[str] = []
        for entity, color, size in rows:
            pair = [
                rng.choice(color_facts).format(e=entity, v=color),
                rng.choice(size_facts).format(e=entity, v=size),
            ]
            rng.shuffle(pair)
            blocks.append(" ".join(pair))
        rng.shuffle(blocks)
        fact_text = " ".join(blocks)
    else:
        parts: list[str] = []
        for entity, color, size in rows:
            parts.append(rng.choice(color_facts).format(e=entity, v=color))
            parts.append(rng.choice(size_facts).format(e=entity, v=size))
        rng.shuffle(parts)
        fact_text = " ".join(parts)
    if combine:
        if ask_color:
            query_pool = TRAIN_COMBINE_QUERIES_COLOR if surface == "train" else HOLDOUT_COMBINE_QUERIES_COLOR
            query = rng.choice(query_pool).format(s=query_s)
            value = query_c
            attr = "color"
        else:
            query_pool = TRAIN_COMBINE_QUERIES_SIZE if surface == "train" else HOLDOUT_COMBINE_QUERIES_SIZE
            query = rng.choice(query_pool).format(c=query_c)
            value = query_s
            attr = "size"
        family = "fact_combine"
        variant = f"fact_combine_{n_entities}e"
    else:
        color_queries = TRAIN_QUERIES if surface == "train" else HOLDOUT_QUERIES
        size_queries = TRAIN_SIZE_QUERIES if surface == "train" else HOLDOUT_SIZE_QUERIES
        if ask_color:
            query = rng.choice(color_queries).format(e=query_e)
            value = query_c
            attr = "color"
        else:
            query = rng.choice(size_queries).format(e=query_e)
            value = query_s
            attr = "size"
        family = "english_mixed"
        variant = f"mixed_{n_entities}e"
    prompt = fact_text + " " + query
    answer = " " + value
    if period and not answer.endswith("."):
        answer = answer + "."
        variant += "_stop"
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family=family,
        surface=surface,
        variant=variant,
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


def make_longturn_item(
    rng: random.Random,
    tokenizer,
    *,
    n_turns: int = 3,
    surface: str = "heldout",
) -> dict:
    """Three or four Human/Baby color turns; last answer is scored."""
    if n_turns < 3 or n_turns > 5:
        raise ValueError(n_turns)
    if n_turns > len(VALUES):
        raise ValueError("not enough colors for longturn")
    entities = list(_sample(rng, ENTITIES, n_turns))
    colors = list(_sample(rng, VALUES, n_turns))
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
    train_qs = (
        "what color is the {e}?",
        "what is {e}'s color?",
        "please tell the color of the {e}.",
        "name the color of the {e}.",
    )
    hold_qs = (
        "which color is the {e}?",
        "tell me the color of the {e}.",
        "what is the color of the {e}?",
        "which color is {e}?",
    )
    qpool = train_qs if surface == "train" else hold_qs
    chunks = [facts]
    for i, (entity, value) in enumerate(pairs[:-1]):
        chunks.append(f"Human: {qpool[i % len(qpool)].format(e=entity)}")
        chunks.append(f"Baby: {value}")
    last_e, last_v = pairs[-1]
    chunks.append(f"Human: {qpool[(n_turns - 1) % len(qpool)].format(e=last_e)}")
    chunks.append("Baby:")
    prompt = "\n".join(chunks)
    answer = " " + last_v
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="longturn",
        surface=surface,
        variant=f"longturn_{n_turns}",
        extra={"entity": last_e, "value_text": last_v, "prompt_text": prompt, "answer_text": answer},
    )


def make_roleplay_item(
    rng: random.Random,
    tokenizer,
    *,
    n_facts: int = 2,
    surface: str = "heldout",
) -> dict:
    """Same bind as dialogue, but Kid/Mom instead of Human/Baby."""
    entities = list(_sample(rng, ENTITIES, n_facts))
    colors = list(_sample(rng, VALUES, n_facts))
    pairs = list(zip(entities, colors))
    rng.shuffle(pairs)
    query_e, query_v = rng.choice(pairs)
    fact_pool = TRAIN_FACTS if surface == "train" else HOLDOUT_FACTS
    fact_t = rng.choice(fact_pool)
    facts = " ".join(fact_t.format(e=e, v=v) for e, v in pairs)
    if surface == "train":
        kid = f"what color is the {query_e}?"
    else:
        kid = f"which color is the {query_e}?"
    prompt = f"{facts}\nKid: {kid}\nMom:"
    answer = " " + query_v
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family="roleplay",
        surface=surface,
        variant=f"roleplay_{n_facts}f",
        extra={"entity": query_e, "value_text": query_v, "prompt_text": prompt, "answer_text": answer},
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


def _humanize_query(query: str) -> str:
    if not query:
        return query
    return query[0].lower() + query[1:]


def make_story_item(
    rng: random.Random,
    tokenizer,
    *,
    surface: str = "heldout",
    ask: str = "color",
    dialogue: bool = False,
    n_chars: int = 2,
    pronoun: bool = False,
) -> dict:
    """Short TinyStories-ish scene. Color/size answers keep first-token scoring clean."""
    if ask not in {"color", "who", "event", "size"}:
        raise ValueError(ask)
    if pronoun and ask != "color":
        raise ValueError("pronoun stories are color-answer only")
    if ask in {"who", "event"}:
        n_chars = 2
        ent_pool = WHO_ENTITIES
        val_pool = VALUES
    else:
        n_chars = max(2, min(int(n_chars), 4))
        ent_pool = ENTITIES
        val_pool = SIZES if ask == "size" else VALUES
    ents = _sample(rng, ent_pool, n_chars)
    vals = _sample(rng, val_pool, n_chars)
    events = TRAIN_EVENTS if surface == "train" else HOLDOUT_EVENTS
    acts = _sample(rng, events, n_chars)
    if n_chars == 4:
        frames = TRAIN_STORY_FRAMES_4 if surface == "train" else HOLDOUT_STORY_FRAMES_4
    elif n_chars == 3:
        frames = TRAIN_STORY_FRAMES_3 if surface == "train" else HOLDOUT_STORY_FRAMES_3
    else:
        frames = TRAIN_STORY_FRAMES if surface == "train" else HOLDOUT_STORY_FRAMES
    fmt = {}
    for i, (ent, val, act) in enumerate(zip(ents, vals, acts), start=1):
        fmt[f"e{i}"] = ent
        fmt[f"v{i}"] = val
        fmt[f"a{i}"] = act
    story = rng.choice(frames).format(**fmt)
    pairs = list(zip(ents, vals, acts))
    if pronoun:
        query_e, query_v, _act = pairs[-1]
        query = "What color is it?" if surface == "train" else rng.choice(("What color is it?", "Which color is it?"))
        answer_word = query_v
        extra_entity = query_e
        ask_label = "pronoun"
    elif ask == "color":
        query_e, query_v, _act = rng.choice(pairs)
        query_pool = TRAIN_QUERIES if surface == "train" else HOLDOUT_QUERIES
        query = rng.choice(query_pool).format(e=query_e)
        answer_word = query_v
        extra_entity = query_e
        ask_label = "color"
    elif ask == "size":
        query_e, query_v, _act = rng.choice(pairs)
        query_pool = TRAIN_SIZE_QUERIES if surface == "train" else HOLDOUT_SIZE_QUERIES
        query = rng.choice(query_pool).format(e=query_e)
        answer_word = query_v
        extra_entity = query_e
        ask_label = "size"
    elif ask == "who":
        query_e, query_v, _act = rng.choice(pairs)
        query_pool = TRAIN_WHO_QUERIES if surface == "train" else HOLDOUT_WHO_QUERIES
        query = rng.choice(query_pool).format(v=query_v)
        answer_word = query_e
        extra_entity = query_e
        ask_label = "who"
    else:
        query_e, _query_v, query_act = rng.choice(pairs)
        query_pool = TRAIN_EVENT_QUERIES if surface == "train" else HOLDOUT_EVENT_QUERIES
        query = rng.choice(query_pool).format(act=query_act)
        answer_word = query_e
        extra_entity = query_e
        ask_label = "event"
    if dialogue:
        prompt = f"{story}\nHuman: {_humanize_query(query)}\nBaby:"
        family = "story_dialogue"
        variant = f"story_dialogue_{ask_label}_{n_chars}e"
    else:
        prompt = story + " " + query
        family = f"story_{ask_label}"
        variant = f"story_{ask_label}_{n_chars}e"
    answer = " " + answer_word
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family=family,
        surface=surface,
        variant=variant,
        extra={
            "entity": extra_entity,
            "value_text": answer_word,
            "prompt_text": prompt,
            "answer_text": answer,
            "attr": ask_label,
            "story_text": story,
        },
    )


def make_story_mixed_item(
    rng: random.Random,
    tokenizer,
    *,
    surface: str = "heldout",
    combine: bool = False,
    dialogue: bool = False,
    period: bool = False,
) -> dict:
    """Story with color and size; query one attribute, optionally via the other."""
    e1, e2 = _sample(rng, ENTITIES, 2)
    c1, c2 = _sample(rng, VALUES, 2)
    s1, s2 = _sample(rng, SIZES, 2)
    events = TRAIN_EVENTS if surface == "train" else HOLDOUT_EVENTS
    a1, a2 = _sample(rng, events, 2)
    frames = TRAIN_MIXED_STORY_FRAMES if surface == "train" else HOLDOUT_MIXED_STORY_FRAMES
    story = rng.choice(frames).format(e1=e1, e2=e2, c1=c1, c2=c2, s1=s1, s2=s2, a1=a1, a2=a2)
    rows = ((e1, c1, s1), (e2, c2, s2))
    query_e, query_c, query_s = rng.choice(rows)
    ask_color = rng.random() < 0.5
    if combine:
        if ask_color:
            query_pool = TRAIN_COMBINE_QUERIES_COLOR if surface == "train" else HOLDOUT_COMBINE_QUERIES_COLOR
            query = rng.choice(query_pool).format(s=query_s)
            answer_word = query_c
            attr = "color"
        else:
            query_pool = TRAIN_COMBINE_QUERIES_SIZE if surface == "train" else HOLDOUT_COMBINE_QUERIES_SIZE
            query = rng.choice(query_pool).format(c=query_c)
            answer_word = query_s
            attr = "size"
        family = "story_combine"
    else:
        if ask_color:
            query_pool = TRAIN_QUERIES if surface == "train" else HOLDOUT_QUERIES
            query = rng.choice(query_pool).format(e=query_e)
            answer_word = query_c
            attr = "color"
        else:
            query_pool = TRAIN_SIZE_QUERIES if surface == "train" else HOLDOUT_SIZE_QUERIES
            query = rng.choice(query_pool).format(e=query_e)
            answer_word = query_s
            attr = "size"
        family = "story_mixed"
    if dialogue:
        prompt = f"{story}\nHuman: {_humanize_query(query)}\nBaby:"
        family = family + "_dialogue"
    else:
        prompt = story + " " + query
    answer = " " + answer_word
    variant = f"{family}_2e"
    if period and not answer.endswith("."):
        answer = answer + "."
        variant += "_stop"
    prompt_ids, answer_ids = encode_split(tokenizer, prompt, answer)
    return _item(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        kind="keyed",
        family=family,
        surface=surface,
        variant=variant,
        extra={
            "entity": query_e,
            "value_text": answer_word,
            "prompt_text": prompt,
            "answer_text": answer,
            "attr": attr,
            "story_text": story,
        },
    )


def build_e8_panels(tokenizer, seed: int = 310801, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "story_who_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="who") for _ in range(n)],
        "story_event_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="event") for _ in range(n)],
        "story_dialogue_heldout": [
            make_story_item(rng, tokenizer, surface="heldout", ask="color", dialogue=True) for _ in range(n)
        ],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e9_panels(tokenizer, seed: int = 310901, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "story_color_3e_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color", n_chars=3) for _ in range(n)],
        "story_pronoun_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color", pronoun=True) for _ in range(n)],
        "story_size_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="size") for _ in range(n)],
        "mixed_2e_heldout": [make_mixed_item(rng, tokenizer, n_entities=2, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e10_panels(tokenizer, seed: int = 311001, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "story_color_4e_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color", n_chars=4) for _ in range(n)],
        "story_mixed_heldout": [make_story_mixed_item(rng, tokenizer, surface="heldout", combine=False) for _ in range(n)],
        "story_combine_heldout": [make_story_mixed_item(rng, tokenizer, surface="heldout", combine=True) for _ in range(n)],
        "mixed_2e_heldout": [make_mixed_item(rng, tokenizer, n_entities=2, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e11_panels(tokenizer, seed: int = 311101, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "longturn_3_heldout": [make_longturn_item(rng, tokenizer, n_turns=3, surface="heldout") for _ in range(n)],
        "longturn_4_heldout": [make_longturn_item(rng, tokenizer, n_turns=4, surface="heldout") for _ in range(n)],
        "roleplay_2fact_heldout": [make_roleplay_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "instr_2fact_heldout": [
            make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", instruction=True)
            for _ in range(n)
        ],
    }


def build_e12_panels(tokenizer, seed: int = 311201, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_2fact_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "longturn_4_heldout": [make_longturn_item(rng, tokenizer, n_turns=4, surface="heldout") for _ in range(n)],
        "longturn_5_heldout": [make_longturn_item(rng, tokenizer, n_turns=5, surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e13_panels(tokenizer, seed: int = 311301, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "story_mixed_heldout": [make_story_mixed_item(rng, tokenizer, surface="heldout", combine=False) for _ in range(n)],
        "story_combine_heldout": [make_story_mixed_item(rng, tokenizer, surface="heldout", combine=True) for _ in range(n)],
        "mixed_2e_heldout": [make_mixed_item(rng, tokenizer, n_entities=2, surface="heldout") for _ in range(n)],
        "fact_combine_heldout": [make_mixed_item(rng, tokenizer, n_entities=2, surface="heldout", combine=True) for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e14_panels(tokenizer, seed: int = 311401, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "phrase_2fact_heldout": [make_phrase_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "instr_2fact_heldout": [
            make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", instruction=True)
            for _ in range(n)
        ],
    }


def build_e15_panels(tokenizer, seed: int = 311501, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "open_2fact_heldout": [make_open_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "instr_2fact_heldout": [
            make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", instruction=True)
            for _ in range(n)
        ],
    }


def build_e16_panels(tokenizer, seed: int = 311601, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "about_2fact_heldout": [make_open_item(rng, tokenizer, n_facts=2, surface="heldout", about=True) for _ in range(n)],
        "open_2fact_heldout": [make_open_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
    }


def build_e17_panels(tokenizer, seed: int = 311701, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "copy_2fact_heldout": [make_copy_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "saystop_2fact_heldout": [make_saystop_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "field_2e_heldout": [make_field_item(rng, tokenizer, n_entities=2, surface="heldout") for _ in range(n)],
        "nostory_heldout": [make_nostory_item(rng, tokenizer, surface="heldout") for _ in range(n)],
        "format_word_heldout": [make_format_item(rng, tokenizer, n_facts=2, surface="heldout", sentence=False) for _ in range(n)],
        "format_sent_heldout": [make_format_item(rng, tokenizer, n_facts=2, surface="heldout", sentence=True) for _ in range(n)],
        "instr_2fact_heldout": [
            make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", instruction=True)
            for _ in range(n)
        ],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
    }


def build_e18_panels(tokenizer, seed: int = 311801, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "chat_open_heldout": [make_varied_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "chat_role_heldout": [make_varied_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout", roleplay=True) for _ in range(n)],
        "chat_yesno_heldout": [make_yesno_item(rng, tokenizer, n_facts=2, surface="heldout", dialogue=True) for _ in range(n)],
        "chat_long3_heldout": [make_varied_longturn_item(rng, tokenizer, n_turns=3, surface="heldout") for _ in range(n)],
        "chat_long4_heldout": [make_varied_longturn_item(rng, tokenizer, n_turns=4, surface="heldout") for _ in range(n)],
        "chat_reuse_heldout": [make_factreuse_item(rng, tokenizer, surface="heldout") for _ in range(n)],
        "longturn_3_heldout": [make_longturn_item(rng, tokenizer, n_turns=3, surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
    }


def build_e19_panels(tokenizer, seed: int = 311901, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
        "happened_heldout": [make_happened_item(rng, tokenizer, surface="heldout") for _ in range(n)],
        "yesno_2fact_heldout": [make_yesno_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "open_2fact_heldout": [make_open_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "about_2fact_heldout": [make_open_item(rng, tokenizer, n_facts=2, surface="heldout", about=True) for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
    }


def build_e20_panels(tokenizer, seed: int = 312001, n: int = 32) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "qa_2fact_heldout": [make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout") for _ in range(n)],
        "size_stop_heldout": [make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout", period=True) for _ in range(n)],
        "chat_loop_color_heldout": [make_chat_loop_item(rng, tokenizer, n_turns=4, surface="heldout", score="color") for _ in range(n)],
        "chat_loop_fact_heldout": [make_chat_loop_item(rng, tokenizer, n_turns=4, surface="heldout", score="yesno") for _ in range(n)],
        "chat_open_heldout": [make_varied_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "chat_long3_heldout": [make_varied_longturn_item(rng, tokenizer, n_turns=3, surface="heldout") for _ in range(n)],
        "dialogue_2fact_heldout": [make_dialogue_item(rng, tokenizer, n_facts=2, surface="heldout") for _ in range(n)],
        "story_color_heldout": [make_story_item(rng, tokenizer, surface="heldout", ask="color") for _ in range(n)],
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
    if phase == "e8":
        # Inverted who/event bind spends D3 if it dominates the bridge mix
        # (e8_story_310801: 52% inverted, D3 196). Keep most mass on E5 skills.
        if roll < 0.14:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.24:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.34:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.44:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.54:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.66:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.82:
            return make_story_item(rng, tokenizer, surface="train", ask="who")
        if roll < 0.96:
            return make_story_item(rng, tokenizer, surface="train", ask="event")
        return make_story_item(rng, tokenizer, surface="train", ask="who", dialogue=True)
    if phase == "e9":
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.20:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.30:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.40:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.50:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.60:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.78:
            return make_story_item(rng, tokenizer, surface="train", ask="color", n_chars=3)
        if roll < 0.92:
            return make_story_item(rng, tokenizer, surface="train", ask="color", pronoun=True)
        return make_story_item(rng, tokenizer, surface="train", ask="color", dialogue=True)
    if phase == "e10":
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.22:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.30:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.38:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.46:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.58:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.82:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=False)
        if roll < 0.94:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=True)
        return make_story_item(rng, tokenizer, surface="train", ask="color", dialogue=True)
    if phase == "e11":
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.20:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.28:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.36:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.44:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.54:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.70:
            return make_longturn_item(rng, tokenizer, n_turns=3, surface="train")
        if roll < 0.86:
            return make_longturn_item(rng, tokenizer, n_turns=4, surface="train")
        if roll < 0.94:
            return make_roleplay_item(rng, tokenizer, n_facts=2, surface="train")
        return make_multiturn_item(rng, tokenizer, n_facts=3, surface="train")
    if phase == "e12":
        # e12_stop_311201 mixed size-stop + 5-turn and D3 183. Size-stop only from E5.
        if roll < 0.14:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.26:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.36:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.46:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.56:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.88:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
    if phase == "e13":
        # E10 dumped story-mixed+story-combine and stalled (combine 0.438, mixed_story ~0.40).
        # E7 color-then-size follow-up spent D3. Who/event inverted bind spent D3.
        # From E12: mixed_2e (entity+attr, not inverted) + light fact-combine on known
        # facts + light story-mixed. Keep size-stop. No follow-up, no who/event.
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.24:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.32:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.40:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.72:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.84:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
        if roll < 0.96:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
    if phase == "e13b":
        # e13_mix_311301: 12% fact-combine + 4% story-combine, D3 199. Drop inverted.
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.24:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.34:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.44:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.82:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
    if phase == "e13c":
        # e13 inverted D3 199; e13b 38% mixed D3 193 and no English move.
        # Light grouped mixed + tiny fact-combine. Do not raise structured.
        if roll < 0.16:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.32:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.44:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.56:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.66:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.76:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.88:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", grouped=True)
        if roll < 0.96:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=False)
        return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, grouped=True)
    if phase == "e14":
        # Mix from E12 either spends D3 or does not learn. Longer period-stopped
        # color replies on sentence cues, without inverted combine / 5-turn / who.
        # e14b_sent_311411: phrase free_exact 0.188, D3 200, one-word QA kept.
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.24:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.34:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.44:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.52:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.60:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        return make_phrase_item(rng, tokenizer, n_facts=2, surface="train")
    if phase == "e15":
        # Less-scripted color follow-up. No mix/combine, no who/event, no 5-turn.
        if roll < 0.16:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.32:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.44:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.56:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.66:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.76:
            return make_multiturn_item(rng, tokenizer, n_facts=2, surface="train")
        return make_open_item(rng, tokenizer, n_facts=2, surface="train")
    if phase == "e16":
        if roll < 0.16:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.32:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.44:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.56:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.68:
            return make_open_item(rng, tokenizer, n_facts=2, surface="train")
        return make_open_item(rng, tokenizer, n_facts=2, surface="train", about=True)
    if phase == "e17":
        # Instruction types beyond one-word. No mix/combine/who/place/5-turn.
        # Copy is the override skill. Do not dump sentence-format (e14 overfit).
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.24:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.32:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.40:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.48:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.76:
            return make_copy_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.86:
            return make_saystop_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.94:
            return make_nostory_item(rng, tokenizer, surface="train")
        return make_field_item(rng, tokenizer, n_entities=2, surface="train")
    if phase == "e17b":
        # e17_copy_311701: 28% copy learned 0.719 and D3 199. Lighter copy, no field.
        if roll < 0.16:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.32:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.44:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.56:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.66:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.80:
            return make_copy_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.90:
            return make_saystop_item(rng, tokenizer, n_facts=2, surface="train")
        return make_nostory_item(rng, tokenizer, surface="train")
    if phase == "e17c":
        # e17 28% copy D3 199 at 0.719; e17b 14% copy D3 202 at 0.406. Middle dose.
        if roll < 0.14:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.28:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.38:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.48:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.58:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.78:
            return make_copy_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.88:
            return make_saystop_item(rng, tokenizer, n_facts=2, surface="train")
        return make_nostory_item(rng, tokenizer, surface="train")
    if phase == "e18":
        # Varied Human/Baby and Kid/Mom. Keep scripted dialogue. No 5-turn.
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.24:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.34:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.44:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.54:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train", instruction=True)
        if roll < 0.68:
            return make_varied_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.78:
            return make_varied_dialogue_item(rng, tokenizer, n_facts=2, surface="train", roleplay=True)
        if roll < 0.88:
            return make_varied_longturn_item(rng, tokenizer, n_turns=3, surface="train")
        if roll < 0.96:
            return make_factreuse_item(rng, tokenizer, surface="train")
        return make_varied_longturn_item(rng, tokenizer, n_turns=4, surface="train")
    if phase == "e19":
        # Broader language: yes/no happened, not who/event. Keep color/size-stop.
        # true/wrong is a new lexeme; keep the dose light.
        if roll < 0.16:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.32:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.44:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.56:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.68:
            return make_open_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.80:
            return make_open_item(rng, tokenizer, n_facts=2, surface="train", about=True)
        if roll < 0.92:
            return make_happened_item(rng, tokenizer, surface="train")
        return make_yesno_item(rng, tokenizer, n_facts=2, surface="train")
    if phase == "e20":
        if roll < 0.14:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.26:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.36:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.46:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.58:
            return make_varied_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.70:
            return make_varied_longturn_item(rng, tokenizer, n_turns=3, surface="train")
        if roll < 0.82:
            return make_chat_loop_item(rng, tokenizer, n_turns=4, surface="train", score="color")
        if roll < 0.92:
            return make_chat_loop_item(rng, tokenizer, n_turns=4, surface="train", score="yesno")
        return make_yesno_item(rng, tokenizer, n_facts=2, surface="train", dialogue=True)
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


ALLOWED_USABLE_SKILLS = frozenset(
    {
        "color",
        "size",
        "how-about",
        "about",
        "say-stop",
        "refuse-story",
        "come-back",
    }
)
STORY_CONTINUATION_MARKERS = (
    "once upon",
    "long ago",
    "went home",
    "came by",
    "walked along",
    "went out",
    "sat down",
    "ran by",
    "looked up",
    "came out",
    "one day",
    "in the yard",
    "very tired",
    "the little",
)
FORBIDDEN_USABLE_MARKERS = (
    "who is",
    "who was",
    "who looks",
    "which one",
    "who sat",
    "who went",
    "copy this",
    "write the word",
    "true",
    "wrong",
    "did the",
    "the tiny one",
    "the huge one",
    "the red one",
    "lives at",
    "where does",
    "please answer in a sentence",
    "give a short sentence",
    "give a sentence",
)


def _usable_words(text: str) -> list[str]:
    return [part.strip(".,!?;:\"'").lower() for part in text.split() if part.strip(".,!?;:\"'")]


def score_usable_turn(
    *,
    decoded: str,
    stopped: bool,
    gold: str,
    distractors: tuple[str, ...] | list[str],
    skills: list[str],
    n_tokens: int,
) -> dict:
    """Fact hit, on-topic, period-stop, rambling. One-word color/size is on-topic, not rambling."""
    words = _usable_words(decoded)
    gold_l = gold.strip().lower()
    fact_hit = gold_l in words
    distractor_hits = sorted(
        {item.strip().lower() for item in distractors if item.strip().lower() in words and item.strip().lower() != gold_l}
    )
    lower = decoded.lower()
    storyish = any(marker in lower for marker in STORY_CONTINUATION_MARKERS)
    n_periods = decoded.count(".")
    rambling = bool(
        storyish
        or n_periods > 1
        or (len(words) >= 10 and not fact_hit)
        or (not stopped and n_tokens >= 12)
    )
    on_topic = fact_hit and not distractor_hits
    if "refuse-story" in skills:
        on_topic = fact_hit and not storyish
        rambling = rambling or storyish
    period_stop = bool(stopped) and ("." in decoded)
    usable = bool(on_topic and period_stop and fact_hit)
    return {
        "fact_hit": fact_hit,
        "fact_reuse": fact_hit if "come-back" in skills else None,
        "on_topic": on_topic,
        "period_stop": period_stop,
        "rambling": rambling,
        "usable": usable,
        "distractor_hits": distractor_hits,
        "n_words": len(words),
        "generic_continuation": (not fact_hit) and (storyish or len(words) >= 6),
    }


def score_sentence_answer(*, full_text: str, entity: str, color: str, stopped: bool) -> dict:
    words = _usable_words(full_text)
    has_entity = entity.strip().lower() in words
    has_color = color.strip().lower() in words
    period = "." in full_text
    sentence_like = has_entity and has_color and period and len(words) >= 3 and ("is" in words or "looks" in words)
    return {
        "has_entity": has_entity,
        "has_color": has_color,
        "period": period,
        "stopped": bool(stopped),
        "n_words": len(words),
        "sentence_ok": sentence_like,
        "one_word_color": has_color and not has_entity and len(words) <= 2,
    }


def build_usable_chat_pack() -> list[dict]:
    """Held-out 4–5 turn chats mixing only skills E12 already has. Eval-only; never trained."""
    return [
        {
            "id": "color_howabout_reuse",
            "n_turns": 4,
            "facts": "fox looks white. pig looks red.",
            "turns": [
                {"human": "Which color is the fox?", "gold": "white", "entity": "fox", "skills": ["color"], "distractors": ["red"]},
                {"human": "How about the pig?", "gold": "red", "entity": "pig", "skills": ["how-about"], "distractors": ["white"]},
                {"human": "Tell me the color of the fox.", "gold": "white", "entity": "fox", "skills": ["color", "come-back"], "distractors": ["red"]},
                {"human": "The pig then?", "gold": "red", "entity": "pig", "skills": ["how-about", "come-back"], "distractors": ["white"]},
            ],
        },
        {
            "id": "size_then_color",
            "n_turns": 4,
            "facts": "Remember: the frog is wide in size. That bird is tiny in size. fox looks white. pig looks red.",
            "turns": [
                {"human": "Which size is the frog?", "gold": "wide", "entity": "frog", "skills": ["size"], "distractors": ["tiny", "white", "red"]},
                {"human": "Tell me the size of the bird.", "gold": "tiny", "entity": "bird", "skills": ["size"], "distractors": ["wide", "white", "red"]},
                {"human": "Which color is the fox?", "gold": "white", "entity": "fox", "skills": ["color"], "distractors": ["red", "wide", "tiny"]},
                {"human": "How about the pig?", "gold": "red", "entity": "pig", "skills": ["how-about"], "distractors": ["white", "wide", "tiny"]},
            ],
        },
        {
            "id": "about_saystop_reuse",
            "n_turns": 4,
            "facts": "That cow is pink. That duck is green.",
            "turns": [
                {"human": "What do you know about the cow?", "gold": "pink", "entity": "cow", "skills": ["about"], "distractors": ["green"]},
                {"human": "Please say the color of the duck and stop.", "gold": "green", "entity": "duck", "skills": ["say-stop"], "distractors": ["pink"]},
                {"human": "Talk about the cow.", "gold": "pink", "entity": "cow", "skills": ["about", "come-back"], "distractors": ["green"]},
                {"human": "How about the duck?", "gold": "green", "entity": "duck", "skills": ["how-about", "come-back"], "distractors": ["pink"]},
            ],
        },
        {
            "id": "refuse_story_color",
            "n_turns": 4,
            "facts": "Long ago a hen went home. That hen was yellow. A dog came by. That dog was blue.",
            "turns": [
                {"human": "Stop the story. Which color is the hen?", "gold": "yellow", "entity": "hen", "skills": ["refuse-story", "color"], "distractors": ["blue"]},
                {"human": "Tell me the color of the dog.", "gold": "blue", "entity": "dog", "skills": ["color"], "distractors": ["yellow"]},
                {"human": "How about the hen?", "gold": "yellow", "entity": "hen", "skills": ["how-about", "come-back"], "distractors": ["blue"]},
                {"human": "Give the color of the dog then stop.", "gold": "blue", "entity": "dog", "skills": ["say-stop", "come-back"], "distractors": ["yellow"]},
            ],
        },
        {
            "id": "chat_varied_4",
            "n_turns": 4,
            "facts": "cat looks red. bear looks white.",
            "turns": [
                {"human": "which color is the cat?", "gold": "red", "entity": "cat", "skills": ["color"], "distractors": ["white"]},
                {"human": "the bear then?", "gold": "white", "entity": "bear", "skills": ["how-about"], "distractors": ["red"]},
                {"human": "tell me about the cat.", "gold": "red", "entity": "cat", "skills": ["about", "come-back"], "distractors": ["white"]},
                {"human": "how is the bear?", "gold": "white", "entity": "bear", "skills": ["how-about", "come-back"], "distractors": ["red"]},
            ],
        },
        {
            "id": "size_howabout_reuse",
            "n_turns": 4,
            "facts": "That pig is tiny in size. That duck is huge in size.",
            "turns": [
                {"human": "What is the size of the pig?", "gold": "tiny", "entity": "pig", "skills": ["size"], "distractors": ["huge"]},
                {"human": "How about the duck?", "gold": "huge", "entity": "duck", "skills": ["how-about", "size"], "distractors": ["tiny"]},
                {"human": "Tell me the size of the pig.", "gold": "tiny", "entity": "pig", "skills": ["size", "come-back"], "distractors": ["huge"]},
                {"human": "Which size is the duck?", "gold": "huge", "entity": "duck", "skills": ["size", "come-back"], "distractors": ["tiny"]},
            ],
        },
        {
            "id": "five_turn_color",
            "n_turns": 5,
            "facts": "fox looks blue. hen looks pink. cow looks white.",
            "turns": [
                {"human": "Which color is the fox?", "gold": "blue", "entity": "fox", "skills": ["color"], "distractors": ["pink", "white"]},
                {"human": "How about the hen?", "gold": "pink", "entity": "hen", "skills": ["how-about"], "distractors": ["blue", "white"]},
                {"human": "What do you know about the cow?", "gold": "white", "entity": "cow", "skills": ["about"], "distractors": ["blue", "pink"]},
                {"human": "The fox then?", "gold": "blue", "entity": "fox", "skills": ["how-about", "come-back"], "distractors": ["pink", "white"]},
                {"human": "Talk about the hen.", "gold": "pink", "entity": "hen", "skills": ["about", "come-back"], "distractors": ["blue", "white"]},
            ],
        },
        {
            "id": "refuse_then_about",
            "n_turns": 4,
            "facts": "In the yard a cat went out. The cat looks green. A bear walked along. The bear looks red.",
            "turns": [
                {"human": "Do not continue. What is the color of the cat?", "gold": "green", "entity": "cat", "skills": ["refuse-story", "color"], "distractors": ["red"]},
                {"human": "How about the bear?", "gold": "red", "entity": "bear", "skills": ["how-about"], "distractors": ["green"]},
                {"human": "What do you know about the cat?", "gold": "green", "entity": "cat", "skills": ["about", "come-back"], "distractors": ["red"]},
                {"human": "Please say the color of the bear and stop.", "gold": "red", "entity": "bear", "skills": ["say-stop", "come-back"], "distractors": ["green"]},
            ],
        },
        {
            "id": "five_turn_size_color",
            "n_turns": 5,
            "facts": "Remember: the hen is short in size. That fox is huge in size. duck looks yellow. cow looks blue.",
            "turns": [
                {"human": "Which size is the hen?", "gold": "short", "entity": "hen", "skills": ["size"], "distractors": ["huge", "yellow", "blue"]},
                {"human": "Tell me the size of the fox.", "gold": "huge", "entity": "fox", "skills": ["size"], "distractors": ["short", "yellow", "blue"]},
                {"human": "Which color is the duck?", "gold": "yellow", "entity": "duck", "skills": ["color"], "distractors": ["blue", "short", "huge"]},
                {"human": "How about the cow?", "gold": "blue", "entity": "cow", "skills": ["how-about"], "distractors": ["yellow", "short", "huge"]},
                {"human": "What is the size of the hen?", "gold": "short", "entity": "hen", "skills": ["size", "come-back"], "distractors": ["huge", "yellow", "blue"]},
            ],
        },
        {
            "id": "saystop_about_howabout",
            "n_turns": 4,
            "facts": "Remember: the bird is pink. That frog is white.",
            "turns": [
                {"human": "Give the color of the bird then stop.", "gold": "pink", "entity": "bird", "skills": ["say-stop"], "distractors": ["white"]},
                {"human": "What do you know about the frog?", "gold": "white", "entity": "frog", "skills": ["about"], "distractors": ["pink"]},
                {"human": "How about the bird?", "gold": "pink", "entity": "bird", "skills": ["how-about", "come-back"], "distractors": ["white"]},
                {"human": "Tell me the color of the frog.", "gold": "white", "entity": "frog", "skills": ["color", "come-back"], "distractors": ["pink"]},
            ],
        },
    ]


def build_sentence_decode_pack() -> list[dict]:
    """Held-out color prompts for no-train sentence operators. Not a train family."""
    return [
        {"id": "fox_white", "facts": "fox looks white. pig looks red.", "entity": "fox", "color": "white", "query": "Which color is the fox?"},
        {"id": "pig_red", "facts": "fox looks white. pig looks red.", "entity": "pig", "color": "red", "query": "Tell me the color of the pig."},
        {"id": "frog_blue", "facts": "Remember: the frog is blue. Remember: the bird is yellow.", "entity": "frog", "color": "blue", "query": "What is the color of the frog?"},
        {"id": "cow_pink", "facts": "That cow is pink. That duck is green.", "entity": "cow", "color": "pink", "query": "Which color is the cow?"},
        {"id": "hen_yellow", "facts": "That hen is yellow. That dog is blue.", "entity": "hen", "color": "yellow", "query": "Tell me the color of the hen."},
        {"id": "cat_green", "facts": "cat looks green. bear looks red.", "entity": "cat", "color": "green", "query": "What is the color of the cat?"},
        {"id": "bird_pink", "facts": "Remember: the bird is pink. That frog is white.", "entity": "bird", "color": "pink", "query": "Which color is the bird?"},
        {"id": "duck_yellow", "facts": "duck looks yellow. cow looks blue.", "entity": "duck", "color": "yellow", "query": "Tell me the color of the duck."},
    ]


def banks_from_language() -> Banks:
    from .data import read_u16

    return build_banks(read_u16(LANG_TRAIN))
