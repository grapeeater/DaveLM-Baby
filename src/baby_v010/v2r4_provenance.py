from __future__ import annotations

"""Pinned identities for the v2R4 terminal isolation diagnostic.

These hashes are copied from the frozen v2R4 receipts. This module does not
rewrite historical artifacts; it only names them so new tools can refuse to
score the wrong files.

The v2R4 freeze hashes were computed on Windows CRLF bytes. A POSIX checkout
may store the same git blob with LF. Tools must accept either encoding of the
same text and must not rewrite the tracked files.
"""

import hashlib
from pathlib import Path

PROTOCOL = "BABY_V010_FOUNDATION_V2R4"
TERMINAL_SEED = 106001
TERMINAL_UPDATE = 16000
FROZEN_PANEL_SEED = 102000

FROZEN_PANELS_SHA256 = "5f0d1d2c59c5d9ac07cd93460a59f542d1c65b9a10f130e61a28388b52342bb3"
TERMINAL_METRICS_SHA256 = "b81cb5dad5084c30c95ca6b9b913e0c814125f6b511aec06cd75637aa345e74a"
TERMINAL_RUN_CONFIG_SHA256 = "fb8d172c3cd0e7d705558cb6ed012da4acb67ec9dbcef687ab37fe2abd2fb6b7"
TERMINAL_CHECKPOINT_SHA256 = "94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827"
U6000_CHECKPOINT_SHA256 = "75d2c761f34a5716d3f35c2118b5d3446f8b63decf2fe5781b45888c164037b3"

ALIAS_PANEL_NAMES = ("novel", "induction")
UNIQUE_SCORED_PANELS = (
    "primitive_induction",
    "primitive_keyed",
    "short_keyed",
    "same_surface_novel",
    "same_surface_induction",
    "heldout_surface",
    "unseen_length",
    "low_prior",
    "distractor",
    "broken_context",
    "broken_order",
)

EOS_TOKEN = 3
BOS_TOKEN = 2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identify_frozen_file(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    working = sha256_bytes(raw)
    lf_hash = sha256_bytes(lf)
    crlf_hash = sha256_bytes(crlf)
    if working == expected:
        mode = "exact"
    elif expected in {lf_hash, crlf_hash}:
        mode = "newline_normalized"
    else:
        mode = "mismatch"
    return {
        "path": str(path),
        "working_tree_sha256": working,
        "frozen_sha256": expected,
        "lf_sha256": lf_hash,
        "crlf_sha256": crlf_hash,
        "match": mode != "mismatch",
        "match_mode": mode,
    }


def require_frozen_file(path: Path, expected: str, label: str) -> dict:
    identity = identify_frozen_file(path, expected)
    if not identity["match"]:
        raise RuntimeError(
            f"{label} sha256 {identity['working_tree_sha256']} does not match frozen "
            f"{expected} even after newline normalization; refusing the wrong file"
        )
    return identity
