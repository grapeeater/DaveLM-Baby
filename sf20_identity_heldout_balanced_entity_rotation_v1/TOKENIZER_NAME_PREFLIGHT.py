"""Read-only SF20 tokenizer eligibility check. No model or evaluation access."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
EXPECTED_TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"

# Fixed before inspection. This list is used only to demonstrate ordinary-name behavior;
# the exhaustive vocabulary scan below establishes the hard tokenizer limitation.
COMMON_NAME_REFERENCE = (
    "Adam Alice Anna Amy Ava Ben Bob Charlie Chloe Daniel David Ella Emma Emily Ethan Eva "
    "Frank George Grace Harry Henry Jack Jacob James Jane John Julia Kate Leo Lily Liam Lucy "
    "Luke Mary Max Mike Oliver Paul Peter Rose Ruby Sam Sarah Sophia Sophie Ted Tim Tom William Zoe"
).split()
ORIGINAL_NAMES = ["Alex", "Owen", "Mia", "Nora"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    actual_hash = sha256(TOKENIZER)
    assert actual_hash == EXPECTED_TOKENIZER_SHA256
    tokenizer = Tokenizer.from_file(str(TOKENIZER))

    capitalized_single_vocab_tokens = []
    for token_id in range(tokenizer.get_vocab_size()):
        decoded = tokenizer.decode([token_id])
        if re.fullmatch(r" ?[A-Z][a-z]{1,15}", decoded):
            capitalized_single_vocab_tokens.append({"token_id": token_id, "decoded": decoded})

    def inspect(name: str) -> dict:
        start = tokenizer.encode(name).ids
        leading = tokenizer.encode(" " + name).ids
        candidate = tokenizer.encode(" " + name + ".").ids
        return {
            "name": name,
            "sentence_start_ids": start,
            "leading_space_ids": leading,
            "candidate_with_period_ids": candidate,
            "sentence_start_single_token": len(start) == 1 and tokenizer.decode(start) == name,
            "leading_space_single_token": len(leading) == 1 and tokenizer.decode(leading) == " " + name,
            "eligible_both_boundaries": (
                len(start) == 1
                and tokenizer.decode(start) == name
                and len(leading) == 1
                and tokenizer.decode(leading) == " " + name
            ),
            "roundtrip_start": tokenizer.decode(start) == name,
            "roundtrip_leading": tokenizer.decode(leading) == " " + name,
        }

    reference = [inspect(name) for name in COMMON_NAME_REFERENCE]
    originals = [inspect(name) for name in ORIGINAL_NAMES]
    eligible_reference = [row["name"] for row in reference if row["eligible_both_boundaries"]]

    result = {
        "study": "SF20_IDENTITY_HELDOUT_BALANCED_ENTITY_ROTATION_V1",
        "status": "HARD_STOP_TOKENIZER_ELIGIBILITY",
        "tokenizer_path": str(TOKENIZER),
        "tokenizer_sha256": actual_hash,
        "vocabulary_size": tokenizer.get_vocab_size(),
        "eligibility_rule": {
            "sentence_start": "encode(name) is exactly one token and decodes exactly to name",
            "answer_boundary": "encode(' '+name) is exactly one token and decodes exactly to ' '+name",
            "required_count": 16,
        },
        "capitalized_alphabetic_single_vocab_tokens": capitalized_single_vocab_tokens,
        "capitalized_alphabetic_single_vocab_token_count": len(capitalized_single_vocab_tokens),
        "common_name_reference": reference,
        "eligible_common_names_at_both_boundaries": eligible_reference,
        "eligible_common_name_count": len(eligible_reference),
        "original_name_encodings": originals,
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "updates": 0,
        "evaluation_panels_scored": [],
        "historical_transfer_panel_content_used_for_name_selection": False,
        "final_or_sacred_accessed": False,
        "conclusion": (
            "The frozen tokenizer cannot supply the required 16 defensible person names that are "
            "single tokens at both sentence-start and answer boundaries. The scientific design cannot "
            "be implemented literally without authorizing multi-token names or another changed identity definition."
        ),
    }
    (ROOT / "TOKENIZER_ELIGIBILITY.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "status": result["status"],
        "eligible": len(eligible_reference),
        "required": 16,
        "capitalized_single_vocab_tokens": len(capitalized_single_vocab_tokens),
    }))


if __name__ == "__main__":
    main()
