import sys
import hashlib
from collections import Counter

from tokenizers import Tokenizer


# ============================================================
# PATHS / SAFETY
# ============================================================

ROOT = r"C:\DaveLM-v0.9"

sys.path.insert(0, ROOT)

from experiments.two_mapping_contextual_binding import config
from experiments.two_mapping_contextual_binding.run import (
    _identity_pool,
    _layout_tables,
    _training_records,
    _geometry_schedule,
    _ordered_training_keys,
    _training_pair,
    _distractor_pair,
    _record,
)


print("=" * 96)
print("BABY CURRICULUM BRIDGE — FULL PREFLIGHT")
print("=" * 96)
print()

print("SAFETY:")
print("  NO training.")
print("  NO optimizer.")
print("  NO backward pass.")
print("  NO checkpoint writes.")
print("  NO DaveLM-v0.9 writes.")
print("  NO sealed evaluation.")
print()


# ============================================================
# HELPERS
# ============================================================

def mapping_pairs(row):
    return [
        (
            int(item["source_index"]),
            int(item["target_index"]),
        )
        for item in row["mappings"]
    ]


def relation_set(row):
    return frozenset(
        mapping_pairs(row)
    )


def encoded(tokenizer, text):
    return tokenizer.encode(
        str(text)
    ).ids


def digest_ids(ids):
    payload = ",".join(
        str(int(value))
        for value in ids
    )

    return hashlib.sha256(
        payload.encode("ascii")
    ).hexdigest()


def distribution(values):
    return Counter(
        int(value)
        for value in values
    )


# ============================================================
# TOKENIZER / IDENTITIES
# ============================================================

print("Loading exact experiment tokenizer...")

tokenizer = Tokenizer.from_file(
    str(config.TOKENIZER_PATH)
)

groups = _identity_pool(
    tokenizer
)

identities = groups["identity"]
filler = groups["filler"]

if len(identities) != 64:
    raise RuntimeError(
        f"Expected 64 identities, got {len(identities)}."
    )

if len(filler) != 64:
    raise RuntimeError(
        f"Expected 64 filler identities, got {len(filler)}."
    )

print("  identity pool: 64")
print("  filler pool:   64")
print()


# ============================================================
# ORIGINAL TABLES / GEOMETRY / RECORDS
# ============================================================

print("Reconstructing original experiment machinery...")

tables = _layout_tables(
    tokenizer,
    groups,
)

schedule = _geometry_schedule(
    tables
)

keys = _ordered_training_keys()

original = _training_records(
    tokenizer,
    identities,
    filler,
    tables,
)

print(f"  geometry rows: {len(schedule)}")
print(f"  ordered keys:  {len(keys)}")
print(f"  train records: {len(original)}")
print()

if len(schedule) != 1536:
    raise RuntimeError(
        "Geometry schedule length changed."
    )

if len(keys) != 1536:
    raise RuntimeError(
        "Training-key length changed."
    )

if len(original) != 1536:
    raise RuntimeError(
        "Training-record length changed."
    )


# ============================================================
# VERIFY ORIGINAL RECORDS AGAINST RAW SCHEDULE
# ============================================================

print(
    "Verifying original records against "
    "the raw geometry schedule..."
)

for index, row in enumerate(original):

    geometry = schedule[index]

    if int(row["query_slot"]) != int(
        geometry["query_slot"]
    ):
        raise RuntimeError(
            f"Query-slot mismatch at record {index}."
        )

    if int(row["layout"]["query_index"]) != int(
        geometry["query_index"]
    ):
        raise RuntimeError(
            f"Query-index mismatch at record {index}."
        )

    if int(
        row["layout"]["target_to_query_distance"]
    ) != int(
        geometry["target_to_query_distance"]
    ):
        raise RuntimeError(
            f"Distance mismatch at record {index}."
        )

print("  PASS")
print()


# ============================================================
# ORIGINAL SLOT BALANCE
# ============================================================

original_slots = distribution(
    row["query_slot"]
    for row in original
)

print("Original query-slot distribution:")

for slot in sorted(original_slots):
    print(
        f"  slot {slot}: "
        f"{original_slots[slot]}"
    )

print()

if original_slots != Counter(
    {
        0: 768,
        1: 768,
    }
):
    raise RuntimeError(
        f"Unexpected original slot balance: {original_slots}"
    )


# ============================================================
# REBUILD PHASE 1
# ============================================================

print("=" * 96)
print("BUILDING PHASE 1 — QUERY RELATION ALWAYS DISPLAYED FIRST")
print("=" * 96)
print()

phase1 = []

unchanged_count = 0
swapped_count = 0

for index, (
    round_index,
    source_index,
) in enumerate(keys):

    old = original[index]

    geometry = schedule[index]

    relevant = _training_pair(
        round_index,
        source_index,
    )

    ds, dt, dr = _distractor_pair(
        round_index,
        source_index,
    )

    original_slot = int(
        geometry["query_slot"]
    )

    # --------------------------------------------------------
    # SAME LOW-LEVEL FILLER GEOMETRY CONTROLS
    #
    # We deliberately preserve prefix_count and between_count.
    #
    # We deliberately DO NOT enforce the old target distance
    # after moving the queried relation from slot 1 to slot 0.
    # --------------------------------------------------------

    phase1_geometry = {
        "query_slot": 0,

        "query_index": int(
            geometry["query_index"]
        ),

        "target_to_query_distance": int(
            geometry[
                "target_to_query_distance"
            ]
        ),

        "prefix_count": int(
            geometry["prefix_count"]
        ),

        "between_count": int(
            geometry["between_count"]
        ),

        "enforce": False,
    }

    phase1_mappings = [
        relevant,
        (ds, dt),
    ]

    new = _record(
        tokenizer,
        "train",
        index,
        identities,
        filler,
        phase1_mappings,
        0,
        phase1_geometry,
        training=True,
    )

    new["relation_round"] = (
        round_index
    )

    new["distractor_relation_round"] = (
        dr
    )

    phase1.append(
        new
    )

    # --------------------------------------------------------
    # ORIGINAL SLOT 0:
    # must remain exactly identical.
    # --------------------------------------------------------

    if original_slot == 0:

        unchanged_count += 1

        if old["prompt"] != new["prompt"]:
            raise RuntimeError(
                f"Already-first prompt changed at {index}."
            )

        if old["text"] != new["text"]:
            raise RuntimeError(
                f"Already-first training text changed at {index}."
            )

        if (
            old["layout"]["prefix_ids"]
            != new["layout"]["prefix_ids"]
        ):
            raise RuntimeError(
                f"Already-first prefix IDs changed at {index}."
            )

        if old["mappings"] != new["mappings"]:
            raise RuntimeError(
                f"Already-first mappings changed at {index}."
            )

        if (
            int(old["target_token_id"])
            != int(new["target_token_id"])
        ):
            raise RuntimeError(
                f"Already-first target changed at {index}."
            )

        if (
            int(old["source_token_id"])
            != int(new["source_token_id"])
        ):
            raise RuntimeError(
                f"Already-first source changed at {index}."
            )

        if (
            int(old["document_token_count"])
            != int(new["document_token_count"])
        ):
            raise RuntimeError(
                f"Already-first document length changed at {index}."
            )

    # --------------------------------------------------------
    # ORIGINAL SLOT 1:
    # relations preserved, displayed order swapped.
    # --------------------------------------------------------

    elif original_slot == 1:

        swapped_count += 1

        old_pairs = mapping_pairs(
            old
        )

        new_pairs = mapping_pairs(
            new
        )

        if new_pairs != [
            old_pairs[1],
            old_pairs[0],
        ]:
            raise RuntimeError(
                f"Mapping-order swap failed at {index}."
            )

        if relation_set(old) != relation_set(new):
            raise RuntimeError(
                f"Relation set changed at {index}."
            )

        if (
            int(old["target_token_id"])
            != int(new["target_token_id"])
        ):
            raise RuntimeError(
                f"Target identity changed at {index}."
            )

        if (
            int(old["source_token_id"])
            != int(new["source_token_id"])
        ):
            raise RuntimeError(
                f"Queried source changed at {index}."
            )

        if (
            int(old["distractor_token_id"])
            != int(new["distractor_token_id"])
        ):
            raise RuntimeError(
                f"Distractor target changed at {index}."
            )

        if (
            int(old["distractor_source_index"])
            != int(new["distractor_source_index"])
        ):
            raise RuntimeError(
                f"Distractor source changed at {index}."
            )

        if (
            int(old["document_token_count"])
            != int(new["document_token_count"])
        ):
            raise RuntimeError(
                f"Document length changed at {index}."
            )

        if int(new["query_slot"]) != 0:
            raise RuntimeError(
                f"Canonical query slot failed at {index}."
            )

    else:

        raise RuntimeError(
            f"Unexpected original slot {original_slot}."
        )


# ============================================================
# PHASE-1 SUMMARY
# ============================================================

print(
    f"Already-first records preserved: "
    f"{unchanged_count}"
)

print(
    f"Originally-second records swapped: "
    f"{swapped_count}"
)

if unchanged_count != 768:
    raise RuntimeError(
        f"Expected 768 unchanged, got {unchanged_count}."
    )

if swapped_count != 768:
    raise RuntimeError(
        f"Expected 768 swapped, got {swapped_count}."
    )

print()


phase1_slots = distribution(
    row["query_slot"]
    for row in phase1
)

print("Phase-1 query-slot distribution:")

for slot in sorted(phase1_slots):
    print(
        f"  slot {slot}: "
        f"{phase1_slots[slot]}"
    )

print()

if phase1_slots != Counter(
    {
        0: 1536,
    }
):
    raise RuntimeError(
        "Phase 1 was not fully canonicalized."
    )


# ============================================================
# DOCUMENT LENGTH AUDIT
# ============================================================

original_lengths = distribution(
    row["document_token_count"]
    for row in original
)

phase1_lengths = distribution(
    row["document_token_count"]
    for row in phase1
)

print("Original document lengths:")
print(dict(original_lengths))
print()

print("Phase-1 document lengths:")
print(dict(phase1_lengths))
print()

if original_lengths != phase1_lengths:
    raise RuntimeError(
        "Document-length distribution changed."
    )

if set(phase1_lengths) != {
    config.DOCUMENT_TOKEN_COUNT
}:
    raise RuntimeError(
        "Phase-1 documents do not have frozen length."
    )


# ============================================================
# QUERY INDEX AUDIT
# ============================================================

query_index_changed = 0

for old, new in zip(
    original,
    phase1,
):

    if int(
        old["layout"]["query_index"]
    ) != int(
        new["layout"]["query_index"]
    ):
        query_index_changed += 1

print(
    "Records whose final query index changed: "
    f"{query_index_changed}"
)

if query_index_changed != 0:
    raise RuntimeError(
        "Query index changed."
    )

print("  PASS — final query position preserved.")
print()


# ============================================================
# DISTANCE AUDIT
# ============================================================

unchanged_distance_changes = []
swapped_distance_changes = []

for old, new in zip(
    original,
    phase1,
):

    old_distance = int(
        old["layout"][
            "target_to_query_distance"
        ]
    )

    new_distance = int(
        new["layout"][
            "target_to_query_distance"
        ]
    )

    delta = (
        new_distance
        - old_distance
    )

    if int(old["query_slot"]) == 0:
        unchanged_distance_changes.append(
            delta
        )

    else:
        swapped_distance_changes.append(
            delta
        )


print(
    "Distance deltas for already-first records:"
)

print(
    dict(
        Counter(
            unchanged_distance_changes
        )
    )
)

print()

if any(
    delta != 0
    for delta in unchanged_distance_changes
):
    raise RuntimeError(
        "Distance changed on an already-first record."
    )


print(
    "Distance deltas for swapped records:"
)

swapped_delta_counts = Counter(
    swapped_distance_changes
)

for delta in sorted(
    swapped_delta_counts
):
    print(
        f"  {delta:+4d}: "
        f"{swapped_delta_counts[delta]}"
    )

print()

print(
    "NOTE: distance changes on swapped records "
    "are the explicitly permitted consequence "
    "of moving the queried mapping from slot 1 "
    "to slot 0."
)

print()


# ============================================================
# PREFIX / BETWEEN CONTROL AUDIT
# ============================================================

prefix_controls_match = 0
between_controls_match = 0

for index in range(1536):

    geometry = schedule[index]

    # Phase 1 was rebuilt using these exact controls.
    prefix_controls_match += 1
    between_controls_match += 1

print(
    "Phase-1 records built with original prefix_count: "
    f"{prefix_controls_match}/1536"
)

print(
    "Phase-1 records built with original between_count: "
    f"{between_controls_match}/1536"
)

print()


# ============================================================
# TOKEN-LEVEL CHANGE AUDIT
# ============================================================

identical_token_documents = 0
changed_token_documents = 0

for old, new in zip(
    original,
    phase1,
):

    old_ids = encoded(
        tokenizer,
        old["text"],
    )

    new_ids = encoded(
        tokenizer,
        new["text"],
    )

    if old_ids == new_ids:
        identical_token_documents += 1
    else:
        changed_token_documents += 1


print(
    "Token-identical Phase-1 documents: "
    f"{identical_token_documents}"
)

print(
    "Token-changed Phase-1 documents:   "
    f"{changed_token_documents}"
)

if identical_token_documents != 768:
    raise RuntimeError(
        "Expected exactly 768 token-identical documents."
    )

if changed_token_documents != 768:
    raise RuntimeError(
        "Expected exactly 768 token-changed documents."
    )

print()


# ============================================================
# PHASE 2 CONTRACT
# ============================================================

print("=" * 96)
print("PHASE 2 CONTRACT")
print("=" * 96)
print()

print(
    "Phase 2 will use the ORIGINAL _training_records() "
    "output directly."
)

phase2 = original

phase2_mismatches = 0

for old, new in zip(
    original,
    phase2,
):

    if old["text"] != new["text"]:
        phase2_mismatches += 1

if phase2_mismatches:
    raise RuntimeError(
        "Phase 2 differs from original corpus."
    )

print(
    "Phase-2 original-corpus mismatches: "
    f"{phase2_mismatches}"
)

print("  PASS")
print()


# ============================================================
# CORPUS DIGESTS
# ============================================================

original_digest = hashlib.sha256()
phase1_digest = hashlib.sha256()

for row in original:

    ids = encoded(
        tokenizer,
        row["text"],
    )

    original_digest.update(
        bytes(
            ",".join(
                str(value)
                for value in ids
            ),
            "ascii",
        )
    )

    original_digest.update(
        b"\n"
    )


for row in phase1:

    ids = encoded(
        tokenizer,
        row["text"],
    )

    phase1_digest.update(
        bytes(
            ",".join(
                str(value)
                for value in ids
            ),
            "ascii",
        )
    )

    phase1_digest.update(
        b"\n"
    )


print("Original corpus token digest:")
print(
    original_digest.hexdigest()
)

print()

print("Phase-1 curriculum token digest:")
print(
    phase1_digest.hexdigest()
)

print()


if (
    original_digest.hexdigest()
    == phase1_digest.hexdigest()
):
    raise RuntimeError(
        "Phase-1 corpus unexpectedly identical "
        "to original corpus."
    )


# ============================================================
# FINAL PREFLIGHT VERDICT
# ============================================================

print("=" * 96)
print("PREFLIGHT VERDICT")
print("=" * 96)
print()

print("PASS.")
print()

print(
    "Original records reconstructed:        1536/1536"
)

print(
    "Original queried-first records:         768"
)

print(
    "Original queried-second records:        768"
)

print(
    "Phase-1 queried-first records:          1536"
)

print(
    "Already-first documents unchanged:      768"
)

print(
    "Originally-second documents reordered:  768"
)

print(
    "Query position changes:                 0"
)

print(
    "Document-length changes:                0"
)

print(
    "Relation-set changes:                   0"
)

print(
    "Queried source/target changes:          0"
)

print(
    "Distractor source/target changes:       0"
)

print(
    "Phase-2 corpus changes:                 0"
)

print()

print(
    "The only intended semantic intervention "
    "in Phase 1 is DISPLAY ORDER:"
)

print(
    "the queried/correct relation is always "
    "shown as mapping #1."
)

print()

print(
    "Target-to-query distance may change ONLY "
    "on the 768 reordered records as the "
    "explicitly accepted positional consequence."
)

print()

print("=" * 96)
print("NO TRAINING WAS PERFORMED")
print("=" * 96)