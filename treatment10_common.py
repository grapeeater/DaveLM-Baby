"""Shared structural helpers for the Treatment-10 strict counterfactual curriculum.

Standard-library helpers plus the already-frozen v0.9 generator primitives
(tokenizer, identity/filler pools, text builder). No model, no checkpoint, no
training, no behavioral/outcome logic lives here.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokenizers import Tokenizer

from treatment10_config import RAW_DOCUMENT_TOKEN_COUNT, TOKENIZER_PATH, TOKENIZER_SHA256

sys.path.insert(0, str(Path(r"C:\DaveLM-v0.9")))

import experiments.two_mapping_contextual_binding.run as original_run  # trusted frozen generator


class Treatment10Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Treatment10Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False)


def load_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    require(sha256_file(TOKENIZER_PATH) == TOKENIZER_SHA256,
            "Tokenizer SHA256 mismatch against frozen tokenizer hash.")
    return tokenizer


def load_identity_groups(tokenizer: Tokenizer):
    groups = original_run._identity_pool(tokenizer)
    identities = groups["identity"]
    filler = groups["filler"]
    require(len(identities) >= 64 and len(filler) >= 64, "Identity/filler pool under-sized.")
    return identities, filler


def verify_isolated_single_tokens(tokenizer: Tokenizer, items: List[Dict[str, Any]],
                                  label: str) -> Dict[str, int]:
    """Verify each item contributes exactly its own token when space-prefixed.

    Returns {value_string: token_id}. This is the structural guarantee that
    filler/identity words are single tokens, matching the approved generator
    semantics used by the frozen T5/T8 corpus.
    """
    out: Dict[str, int] = {}
    for item in items:
        token_id = int(item["token_id"])
        encoded = tokenizer.encode(" " + str(item["value"])).ids
        require(len(encoded) == 1 and int(encoded[0]) == token_id,
                f"{label} item {item['value']!r} is not an isolated single token.")
        out[str(item["value"])] = token_id
    return out


def draw_words(filler_words: List[str], count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    return [filler_words[rng.randrange(len(filler_words))] for _ in range(count)]


def build_member_doc(
    tokenizer: Tokenizer,
    mapping_lines: List[List[str]],
    query_key: str,
    answer_value: str,
    prefix_words: List[str],
    between_words: List[str],
    tail_words: List[str],
) -> List[int]:
    """Build one raw 192-token member document id sequence.

    mapping_lines: ordered mapping lines, each [source_word, value_word].
    Only query_key and answer_value differ between the two query members of an
    orientation; only the two mapping-value words differ between orientations.
    The shared prefix/between/tail filler word lists keep geometry matched.
    """
    prefix = " ".join(prefix_words)
    between = " ".join(between_words)
    lines = "\n".join(
        "Mapping: %s -> %s." % (src, val) for src, val in mapping_lines
    )
    prompt = "Notes: %s\n%s\n %s\nQuery: %s\nAnswer:" % (prefix, lines, between, query_key)
    initial_text = "%s %s\nEnd." % (prompt, answer_value)
    initial_ids = tokenizer.encode(initial_text).ids
    need = RAW_DOCUMENT_TOKEN_COUNT - len(initial_ids)
    require(need >= 0, f"Document initial text exceeds {RAW_DOCUMENT_TOKEN_COUNT} tokens.")
    require(len(tail_words) >= need, "Insufficient tail filler words for padding.")
    chosen_tail = tail_words[:need]
    final_text = initial_text + "".join(" " + word for word in chosen_tail)
    final_ids = tokenizer.encode(final_text).ids
    require(len(final_ids) == RAW_DOCUMENT_TOKEN_COUNT,
            "Padded document length != frozen raw token count.")
    require(final_ids[: len(initial_ids)] == initial_ids,
            "Padded document does not start with the initial prompt ids.")
    return final_ids
