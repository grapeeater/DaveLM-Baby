r"""Outcome-blind structural census over the frozen Treatment 5 / Treatment 8 universe.

Treatment 5 and Treatment 8 share one frozen structural universe:

    treatment5_counterfactual_pairs_seed8382/treatment5_full_document_pair_pool.json
    treatment5_counterfactual_pairs_seed8382/treatment5_full_document_frozen_schedule.json

This script reads ONLY those two files and (optionally) the matching
treatment5_full_document_preflight_result.json for hash/count cross-checking.

BLINDNESS CONTRACT
------------------
- Standard library only. No torch, no model/Baby code, no training code,
  no evaluation code, no checkpoint loading, no project module imports.
- No behavioral/outcome data is inspected, inferred, or emitted. Outputs
  contain structural identities and counts only.

UNIT OF DISCOVERY
-----------------
The master structural registry is the unique frozen pair pool (1536 pairs ->
3072 twin documents). T5 and T8 are NOT counted as independent datasets because
they consume these exact artifacts. The schedule is used only to verify
exposure/coverage and to report exposure multiplicities; repeated schedule
exposures are NEVER independent structural observations and never inflate
unique comparison counts.

STAGE DEFINITIONS (literal)
---------------------------
    Stage 0 : same query_token_id (no other restriction).
    Stage 1 : Stage 0 + same unordered candidate set.
    Stage 2 : Stage 1 + different base_record_id.
    Stage 3 : Stage 2 + reversed target binding within that same candidate set
              (the same query identity is bound to the other candidate value).

Stage 0 deliberately does NOT require different pairs or different records.
Same-record comparisons, if any exist, attrit naturally at Stage 2. The
structural relationship between pair_id and base_record_id is validated and
reported; they are never silently assumed interchangeable.

REGISTRY CONTRACT
-----------------
A set of preregistered, outcome-blind structural comparison families is
reported (existence counts and structural characterization ONLY):
    Registry A (Tier A): same query + same unordered candidate set +
        different base record + reversed target binding. This is the only
        Tier-A binding-discrimination family and is identical to Stage 3.
    Registry B (Tier B): same query + same distractor identity + different
        target identity + different base record.
    Registry C (Tier B): same query + same target + same unordered candidate
        set + different base record, with one or more geometry/layout
        variables differing.
    Registry D (Tier B): same query + same target + different
        distractor/candidate context across different base records.
    Registry E (Tier B): same unordered candidate set + same target +
        different query identity across different base records.

No behavioral evaluation is performed on any registry.

INTERPRETATION RULE (emitted verbatim in the outputs)
-----------------------------------------------------
If Stage 3 = 0, the valid conclusion is only that the frozen T5/T8 pair pool
contains no strict same-query, same-candidate, cross-record reversed-binding
comparisons under the registered definition. This is not evidence that Baby
cannot bind contextually and is not evidence that T8 is a shortcut.
Tier-B groups may later test specific shortcut hypotheses under a separately
preregistered behavioral analysis. They must never be described as substitutes
for Stage 3 or as proof of binding.

No minimum-N, balance, ratio, significance, or adequacy thresholds are imposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")

DEFAULT_PAIR_POOL_PATH = (
    CADAVER_ROOT
    / "treatment5_counterfactual_pairs_seed8382"
    / "treatment5_full_document_pair_pool.json"
)

DEFAULT_SCHEDULE_PATH = (
    CADAVER_ROOT
    / "treatment5_counterfactual_pairs_seed8382"
    / "treatment5_full_document_frozen_schedule.json"
)

DEFAULT_PREFLIGHT_PATH = (
    CADAVER_ROOT
    / "treatment5_counterfactual_pairs_seed8382"
    / "treatment5_full_document_preflight_result.json"
)

DEFAULT_OUT_JSON = CADAVER_ROOT / "t5_t8_structural_census.json"
DEFAULT_OUT_MD = CADAVER_ROOT / "t5_t8_structural_census_report.md"

EXPECTED_PAIR_POOL_SHA256 = (
    "0b31e39361f7a45dc4182c49ab4b4967bffe539cd2ff12e7b196eabfb3dac307"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a4e64fd9d1b934aa440a5d036a3bec0ac7c9ac97d58c72fb45df6d8576bb7c12"
)

INTERPRETATION_RULE = (
    "If Stage 3 = 0, the valid conclusion is only that the frozen T5/T8 pair pool contains no "
    "strict same-query, same-candidate, cross-record reversed-binding comparisons under the "
    "registered definition. This is not evidence that Baby cannot bind contextually and is not "
    "evidence that T8 is a shortcut. Tier-B groups may later test specific shortcut hypotheses "
    "under a separately preregistered behavioral analysis. They must never be described as "
    "substitutes for Stage 3 or as proof of binding."
)

GEOMETRY_DIMENSIONS = [
    "query_slot",
    "native_query_slot",
    "mapping_order",
    "qdp",
    "query_clause_source_pos",
    "query_value_clause_pos",
    "distractor_source_clause_pos",
    "distractor_value_clause_pos",
    "q_to_relevant_source_dist",
    "q_to_relevant_value_dist",
    "q_to_distractor_source_dist",
    "q_to_distractor_value_dist",
    "answer_token_index",
    "answer_to_value_offset",
    "base_cell",
    "base_actual_cell",
    "relation_round",
]

COLLINEARITY_AXES = [
    "query_slot",
    "native_query_slot",
    "mapping_order",
    "qdp",
    "q_to_relevant_source_dist",
    "q_to_relevant_value_dist",
    "q_to_distractor_source_dist",
    "q_to_distractor_value_dist",
    "base_cell",
    "base_actual_cell",
    "relation_round",
]

ROLE_NAMES = [
    "query",
    "target",
    "distractor_value",
    "nonqueried_source",
    "candidate_set",
]


class CensusError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CensusError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _array_digest(ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(str(int(token)) for token in ids).encode("ascii"))
    return digest.hexdigest()


def _as_int(value: Any, label: str) -> int:
    return int(value)


def _ids(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    return [int(token) for token in value]


def _safe_minmax_mean(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


class PoolIndex:
    def __init__(self, pool: Dict[str, Any]) -> None:
        self.pool = pool
        self.docs: List[Dict[str, Any]] = []
        self.pairs: List[Dict[str, Any]] = []
        self.checks: List[Dict[str, str]] = []
        self.docs_by_pair_twin: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.pair_id_to_record: Dict[str, str] = {}
        self._index()

    def _pass(self, name: str, detail: str) -> None:
        self.checks.append({"name": name, "detail": detail})

    def _validate_reciprocity(self, pair: Dict[str, Any], pair_id: str) -> None:
        twin_a = pair["twin_a"]
        twin_b = pair["twin_b"]
        require(str(twin_a.get("twin", "")) == "A", f"{pair_id}: twin_a label != 'A'.")
        require(str(twin_b.get("twin", "")) == "B", f"{pair_id}: twin_b label != 'B'.")
        slot_a = _as_int(twin_a.get("query_slot", -1), f"{pair_id} slot_a")
        slot_b = _as_int(twin_b.get("query_slot", -1), f"{pair_id} slot_b")
        require(slot_a in (0, 1) and slot_b in (0, 1), f"{pair_id}: query slots not in {{0,1}}.")
        require(slot_a != slot_b, f"{pair_id}: twin query slots identical.")

        mapping = {
            0: pair["mapping_0"],
            1: pair["mapping_1"],
        }
        require(_as_int(mapping[0]["slot"], "slot0") == 0, f"{pair_id}: mapping_0.slot != 0.")
        require(_as_int(mapping[1]["slot"], "slot1") == 1, f"{pair_id}: mapping_1.slot != 1.")
        target_a = _as_int(twin_a["target_token_id"], f"{pair_id} A target")
        target_b = _as_int(twin_b["target_token_id"], f"{pair_id} B target")
        distractor_a = _as_int(twin_a["distractor_token_id"], f"{pair_id} A distractor")
        distractor_b = _as_int(twin_b["distractor_token_id"], f"{pair_id} B distractor")
        require(target_a == _as_int(mapping[slot_a]["target_token_id"], "A mapping target"),
                f"{pair_id}: twin A target != queried-slot mapping target.")
        require(target_b == _as_int(mapping[slot_b]["target_token_id"], "B mapping target"),
                f"{pair_id}: twin B target != queried-slot mapping target.")
        require(target_a == distractor_b, f"{pair_id}: A target != B distractor.")
        require(target_b == distractor_a, f"{pair_id}: B target != A distractor.")
        require(target_a != target_b and distractor_a == target_b and distractor_b == target_a,
                f"{pair_id}: reciprocal target/distractor roles inconsistent.")

        prefix_a = twin_a["prefix_token_ids"]
        prefix_b = twin_b["prefix_token_ids"]
        qdp = _as_int(pair["query_difference_position"], f"{pair_id} qdp")
        require(len(prefix_a) == len(prefix_b), f"{pair_id}: twin prefix lengths differ.")
        require(qdp < len(prefix_a), f"{pair_id}: qdp outside prefixes.")
        q_a = prefix_a[qdp]
        q_b = prefix_b[qdp]
        require(q_a != q_b, f"{pair_id}: twin A/B query token identities identical.")

        require(
            {target_a, distractor_a} == {target_b, distractor_b} == set(_ids(pair.get("candidate_pair", []))),
            f"{pair_id}: twin A/B candidate sets differ from pair candidate_pair.",
        )
        answer_a = _as_int(pair["answer_token_index"], "answer index")
        require(len(prefix_a) == answer_a and len(prefix_b) == answer_a,
                f"{pair_id}: twin prefix lengths != answer_token_index.")

    def _extract_twin_doc(self, pair: Dict[str, Any], twin_key: str) -> Dict[str, Any]:
        pair_id = str(pair["pair_id"])
        twin = pair[twin_key]
        twin_label = str(twin["twin"])
        require(twin_label in ("A", "B"), f"{pair_id}: twin label {twin_label!r}.")
        query_slot = _as_int(twin["query_slot"], f"{pair_id}/{twin_key} query_slot")
        require(query_slot in (0, 1), f"{pair_id}/{twin_key}: query_slot {query_slot}.")

        mapping_a = pair["mapping_0"]
        mapping_b = pair["mapping_1"]
        require(_as_int(mapping_a["slot"], "mapping_0.slot") == 0, f"{pair_id}: mapping_0.slot != 0.")
        require(_as_int(mapping_b["slot"], "mapping_1.slot") == 1, f"{pair_id}: mapping_1.slot != 1.")

        queried_mapping = mapping_a if query_slot == 0 else mapping_b
        distractor_mapping = mapping_b if query_slot == 0 else mapping_a

        target_id = _as_int(twin["target_token_id"], f"{pair_id}/{twin_key} target")
        distractor_id = _as_int(twin["distractor_token_id"], f"{pair_id}/{twin_key} distractor")
        require(
            target_id == _as_int(queried_mapping["target_token_id"], "queried mapping target"),
            f"{pair_id}/{twin_key}: twin target != queried mapping target.",
        )
        require(
            distractor_id == _as_int(distractor_mapping["target_token_id"], "distractor mapping target"),
            f"{pair_id}/{twin_key}: twin distractor != distractor mapping target.",
        )
        require(target_id != distractor_id, f"{pair_id}/{twin_key}: target equals distractor token.")

        prefix = twin["prefix_token_ids"]
        doc_ids = twin["full_document_token_ids"]
        require(isinstance(prefix, list) and all(isinstance(t, int) for t in prefix),
                f"{pair_id}/{twin_key}: prefix not an int list.")
        require(isinstance(doc_ids, list) and all(isinstance(t, int) for t in doc_ids),
                f"{pair_id}/{twin_key}: full document not an int list.")

        answer_token_index = _as_int(pair["answer_token_index"], f"{pair_id} answer_token_index")
        answer_causal_position = _as_int(pair["answer_causal_position"], f"{pair_id} answer_causal")
        qdp = _as_int(pair["query_difference_position"], f"{pair_id} qdp")

        require(answer_causal_position == answer_token_index - 1,
                f"{pair_id}: answer_causal != answer_token_index - 1.")
        require(len(prefix) == answer_token_index,
                f"{pair_id}/{twin_key}: prefix length != answer_token_index.")
        require(qdp < len(prefix), f"{pair_id}/{twin_key}: qdp outside prefix.")
        require(len(doc_ids) == _as_int(pair["model_visible_document_length"], "doc length"),
                f"{pair_id}/{twin_key}: full doc length mismatch.")
        require(doc_ids[: len(prefix)] == prefix,
                f"{pair_id}/{twin_key}: full document does not start with prefix.")

        q_token_id = prefix[qdp]

        require(doc_ids[answer_token_index] == target_id,
                f"{pair_id}/{twin_key}: answer token index is not the target token.")

        q_occ = [i for i, token in enumerate(prefix) if token == q_token_id]
        require(len(q_occ) == 2 and qdp in q_occ,
                f"{pair_id}/{twin_key}: query token occurrences {len(q_occ)}; expected 2 with qdp.")
        query_clause_source_pos = [i for i in q_occ if i != qdp][0]
        require(query_clause_source_pos < qdp,
                f"{pair_id}/{twin_key}: query clause not before query line.")

        target_occ_prefix = [i for i, token in enumerate(prefix) if token == target_id]
        require(len(target_occ_prefix) == 1,
                f"{pair_id}/{twin_key}: queried value token appears {len(target_occ_prefix)}x in prefix.")
        query_value_clause_pos = target_occ_prefix[0]
        require(query_clause_source_pos < query_value_clause_pos,
                f"{pair_id}/{twin_key}: source not before value in queried clause.")
        require(doc_ids.count(target_id) == 2,
                f"{pair_id}/{twin_key}: queried value token count in doc != 2.")

        distractor_occ_prefix = [i for i, token in enumerate(prefix) if token == distractor_id]
        require(len(distractor_occ_prefix) == 1,
                f"{pair_id}/{twin_key}: distractor value token appears {len(distractor_occ_prefix)}x in prefix.")
        distractor_value_clause_pos = distractor_occ_prefix[0]
        require(doc_ids.count(distractor_id) == 1,
                f"{pair_id}/{twin_key}: distractor value token count in doc != 1.")

        # --------------------------------------------------------------------
        # Distractor mapping source reconstruction.
        #
        # The distractor mapping of this twin is the OTHER slot's mapping.  Its
        # source identity word is exactly the query identity of the reciprocal
        # twin.  The reciprocal twin's frozen prefix at the (pair-shared) qdp
        # holds that query token, which appears in this twin's prefix exactly
        # once, inside its own mapping clause.  The distractor source token
        # identity and its absolute clause position are therefore anchored
        # directly from frozen structural tokens, with NO use of a
        # queried-clause delta to choose the position.  The shared
        # "src -> tgt" clause token layout (identity words are punctuation
        # stable single tokens by the approved generator semantics) is applied
        # afterwards only as a consistency check.  If the anchor token cannot
        # be located uniquely the census fails rather than guessing.
        # --------------------------------------------------------------------
        queried_word = str(queried_mapping["source"])
        distractor_word = str(distractor_mapping["source"])
        stored_query_word = str(twin.get("query_identity", ""))
        require(stored_query_word == queried_word,
                f"{pair_id}/{twin_key}: twin query identity != queried mapping source.")

        other_twin_key = "twin_b" if twin_key == "twin_a" else "twin_a"
        other_twin = pair[other_twin_key]
        other_prefix = other_twin["prefix_token_ids"]
        require(len(other_prefix) == len(prefix),
                f"{pair_id}/{twin_key}: reciprocal twin prefix length differs.")
        require(qdp < len(other_prefix), f"{pair_id}/{twin_key}: qdp outside reciprocal prefix.")
        require(all(token == other_prefix[i] for i, token in enumerate(prefix) if i != qdp),
                f"{pair_id}/{twin_key}: twin prefixes differ beyond the query token.")
        require(str(other_twin.get("query_identity", "")) == distractor_word,
                f"{pair_id}/{twin_key}: reciprocal query identity != distractor source word.")

        reciprocal_query_token_id = other_prefix[qdp]
        require(reciprocal_query_token_id != q_token_id,
                f"{pair_id}/{twin_key}: reciprocal query token equals own query token.")

        d_src_occ = [i for i, token in enumerate(prefix) if token == reciprocal_query_token_id]
        require(len(d_src_occ) == 1,
                f"{pair_id}/{twin_key}: reciprocal query token appears {len(d_src_occ)}x in this prefix; "
                "expected exactly once as the distractor mapping clause.")
        distractor_source_clause_pos = d_src_occ[0]
        distractor_source_token_id = reciprocal_query_token_id
        require(doc_ids.count(distractor_source_token_id) == 1,
                f"{pair_id}/{twin_key}: distractor source token not unique in document.")

        require(prefix[query_clause_source_pos] == q_token_id,
                f"{pair_id}/{twin_key}: queried clause source is not the query token.")
        require(prefix[query_value_clause_pos] == target_id,
                f"{pair_id}/{twin_key}: queried clause value is not the target token.")
        require(prefix[distractor_value_clause_pos] == distractor_id,
                f"{pair_id}/{twin_key}: distractor clause value is not the distractor token.")
        require(prefix[distractor_source_clause_pos] == distractor_source_token_id,
                f"{pair_id}/{twin_key}: distractor clause source token mismatch.")

        clause_delta = query_value_clause_pos - query_clause_source_pos
        require(clause_delta > 0, f"{pair_id}/{twin_key}: non-positive queried clause delta.")
        require(distractor_value_clause_pos - distractor_source_clause_pos == clause_delta,
                f"{pair_id}/{twin_key}: distractor clause delta != queried clause delta (consistency).")
        arrow_a = prefix[query_clause_source_pos + 1: query_value_clause_pos]
        arrow_b = prefix[distractor_source_clause_pos + 1: distractor_value_clause_pos]
        require(arrow_a == arrow_b,
                f"{pair_id}/{twin_key}: mapping clause arrow token patterns differ.")

        four_positions = {
            query_clause_source_pos,
            query_value_clause_pos,
            distractor_source_clause_pos,
            distractor_value_clause_pos,
        }
        require(len(four_positions) == 4, f"{pair_id}/{twin_key}: clause positions not distinct.")

        if query_slot == 0:
            require(query_clause_source_pos < distractor_source_clause_pos,
                    f"{pair_id}/{twin_key}: queried slot0 clause not before distractor clause.")
        else:
            require(distractor_source_clause_pos < query_clause_source_pos,
                    f"{pair_id}/{twin_key}: queried slot1 clause not after distractor clause.")
        require(qdp > query_value_clause_pos and qdp > distractor_value_clause_pos,
                f"{pair_id}/{twin_key}: query line not after mapping clauses.")

        candidate_list = [_as_int(v, "candidate member") for v in pair["candidate_pair"]]
        mapping_target_list = sorted(
            [
                _as_int(mapping_a["target_token_id"], "mapping_a target"),
                _as_int(mapping_b["target_token_id"], "mapping_b target"),
            ]
        )
        require(candidate_list == sorted([target_id, distractor_id]),
                f"{pair_id}: candidate_pair not equal to sorted twin target/distractor.")
        require(candidate_list == mapping_target_list,
                f"{pair_id}: candidate_pair not equal to sorted mapping targets.")

        return {
            "pair_id": pair_id,
            "twin": twin_label,
            "query_slot": query_slot,
            "query_word": queried_word,
            "query_token_id": q_token_id,
            "target_token_id": target_id,
            "distractor_token_id": distractor_id,
            "distractor_source_word": distractor_word,
            "distractor_source_token_id": distractor_source_token_id,
            "candidate_set": candidate_list,
            "qdp": qdp,
            "answer_token_index": answer_token_index,
            "answer_causal_position": answer_causal_position,
            "query_clause_source_pos": query_clause_source_pos,
            "query_value_clause_pos": query_value_clause_pos,
            "distractor_source_clause_pos": distractor_source_clause_pos,
            "distractor_value_clause_pos": distractor_value_clause_pos,
            "clause_delta": clause_delta,
            "prefix_len": len(prefix),
            "document_len": len(doc_ids),
            "document_digest": _array_digest(doc_ids),
            "full_document_token_ids": doc_ids,
            "base_record_id": str(pair["base_record_id"]),
            "base_record_index": _as_int(pair["base_record_index"], "base_record_index"),
            "native_query_slot": _as_int(pair["native_query_slot"], "native_query_slot"),
            "base_cell": str(pair.get("base_cell", "")),
            "base_actual_cell": str(pair.get("base_actual_cell", "")),
            "base_relation_round": _as_int(pair.get("base_relation_round", -1), "relation_round"),
        }

    def _index(self) -> None:
        require(self.pool.get("artifact_type") == "full_document_counterfactual_pair_pool",
                "Pair pool artifact_type mismatch.")
        pairs = self.pool.get("pairs")
        require(isinstance(pairs, list), "Pair pool 'pairs' must be a list.")
        declared = _as_int(self.pool.get("pair_count", -1), "pair_count")
        require(declared == len(pairs), "Pair pool pair_count != len(pairs).")
        self.pairs = pairs
        self._pass("pair_pool_declared_count", f"pair_count={declared}")
        for pair in pairs:
            require(isinstance(pair, dict), "Pair entry is not a dict.")
            pair_id = str(pair["pair_id"])
            self._validate_reciprocity(pair, pair_id)
            for twin_key in ("twin_a", "twin_b"):
                doc = self._extract_twin_doc(pair, twin_key)
                key = (doc["pair_id"], doc["twin"])
                require(key not in self.docs_by_pair_twin, f"Duplicate doc key {key}.")
                self.docs_by_pair_twin[key] = doc
                self.docs.append(doc)
        pool_pair_ids = [str(pair["pair_id"]) for pair in pairs]
        pool_record_ids = [str(pair["base_record_id"]) for pair in pairs]
        require(len(set(pool_pair_ids)) == len(pairs), "Duplicate pair_id in pair pool.")
        require(len(set(pool_record_ids)) == len(pairs), "Duplicate base_record_id in pair pool.")
        require(len(set(zip(pool_pair_ids, pool_record_ids))) == len(pairs),
                "pair_id/base_record_id mapping is not one-to-one.")
        self.pair_id_to_record = dict(zip(pool_pair_ids, pool_record_ids))
        self._pass("pair_record_relationship",
                   "pair_id and base_record_id are one-to-one across the pool (1,536 unique each).")
        self._pass("reciprocal_twin_pairs_validated",
                   "all pairs passed A/B reciprocity validation (labels, slots, identities, "
                   "target/distractor cross-roles, candidate sets, prefix/qdp geometry).")
        self._pass("unique_pair_records", str(len(self.pairs)))
        self._pass("unique_twin_documents", str(len(self.docs)))


class ScheduleIndex:
    def __init__(self, schedule: Dict[str, Any], docs_by_pair_twin: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
        self.schedule = schedule
        self.docs_by_pair_twin = docs_by_pair_twin
        self.checks: List[Dict[str, str]] = []
        self.pair_presentations = Counter()
        self.events_by_pair = Counter()
        self.event_count = 0
        self.slot_events = Counter()
        self._index()

    def _pass(self, name: str, detail: str) -> None:
        self.checks.append({"name": name, "detail": detail})

    def _index(self) -> None:
        schedule = self.schedule
        require(schedule.get("artifact_type") == "full_document_frozen_training_schedule",
                "Schedule artifact_type mismatch.")
        steps_data = schedule.get("steps_data")
        steps_declared = _as_int(schedule.get("steps", -1), "steps")
        require(isinstance(steps_data, list) and len(steps_data) == steps_declared,
                "steps_data length != steps.")

        previous_step: Optional[int] = None
        for step_index, step in enumerate(steps_data):
            step_number = _as_int(step.get("step", -1), "step")
            require(step_number >= 1, f"Step {step_index}: step numbers must begin at 1.")
            if step_index == 0:
                require(step_number == 1, "Schedule step numbering must begin at 1.")
            if previous_step is not None:
                require(step_number == previous_step + 1, "Step numbers not strictly consecutive.")
            previous_step = step_number

            examples = step.get("examples")
            pair_ids = step.get("pair_ids")
            require(isinstance(examples, list), f"Step {step_index}: examples missing.")
            require(isinstance(pair_ids, list), f"Step {step_index}: pair_ids missing.")
            require(len(examples) == _as_int(step.get("example_count", -1), "example_count"),
                    f"Step {step_index}: example_count mismatch.")
            require(len(pair_ids) == _as_int(step.get("pair_count", -1), "pair_count"),
                    f"Step {step_index}: pair_count mismatch.")
            require(len(examples) == 2 * len(pair_ids), f"Step {step_index}: examples != 2*pair_count.")
            require(len(set(pair_ids)) == len(pair_ids), f"Step {step_index}: pair_ids not unique.")
            step_pair_from_examples = set()
            step_twin_counts = Counter()
            for example in examples:
                example_pair = str(example["pair_id"])
                twin_label = str(example["twin"])
                self.event_count += 1
                self.events_by_pair[example_pair] += 1
                step_pair_from_examples.add(example_pair)
                self.slot_events[str(example["query_slot"])] += 1
                key = (example_pair, twin_label)
                doc = self.docs_by_pair_twin.get(key)
                require(doc is not None, f"Schedule event references unknown pool doc {key}.")
                require(_ids(example.get("full_document_token_ids")) == doc["full_document_token_ids"],
                        f"Schedule event document content mismatch {key}.")
                require(_as_int(example.get("target_token_id", -1), "ex target") == doc["target_token_id"],
                        f"Schedule event target mismatch {key}.")
                require(_as_int(example.get("distractor_token_id", -1), "ex distractor") == doc["distractor_token_id"],
                        f"Schedule event distractor mismatch {key}.")
                require(_as_int(example.get("query_slot", -1), "ex qslot") == doc["query_slot"],
                        f"Schedule event query_slot mismatch {key}.")
                require(_as_int(example.get("answer_token_index", -1), "ex apos") == doc["answer_token_index"],
                        f"Schedule event answer index mismatch {key}.")
                require(_as_int(example.get("answer_causal_position", -1), "ex acausal") == doc["answer_causal_position"],
                        f"Schedule event answer causal mismatch {key}.")
                require(_as_int(example.get("query_difference_position", -1), "ex qdp") == doc["qdp"],
                        f"Schedule event qdp mismatch {key}.")
                require(str(example.get("base_record_id", "")) == doc["base_record_id"],
                        f"Schedule event base_record_id mismatch {key}.")
                step_twin_counts[(example_pair, twin_label)] += 1
            require(set(pair_ids) == step_pair_from_examples,
                    f"Step {step_index}: pair_ids != example pair ids.")
            for example_pair in pair_ids:
                require(step_twin_counts.get((example_pair, "A")) == 1,
                        f"Step {step_index}: pair {example_pair} does not carry exactly one Twin A in this step.")
                require(step_twin_counts.get((example_pair, "B")) == 1,
                        f"Step {step_index}: pair {example_pair} does not carry exactly one Twin B in this step.")
                self.pair_presentations[example_pair] += 1

        require(previous_step == steps_declared,
                "Final schedule step number != number of steps.")

        self.mismatch_events = 0
        require(all(count % 2 == 0 for count in self.events_by_pair.values()),
                "Odd event count for a pair.")
        self._pass("schedule_step_numbering",
                   "step numbers begin at 1, are consecutive, and end at the declared step count.")
        self._pass("schedule_steps", str(len(steps_data)))
        self._pass("schedule_events", str(self.event_count))
        self._pass("schedule_pair_presentations", str(sum(self.pair_presentations.values())))
        self._pass("slot_0_events", str(self.slot_events.get("0", 0)))
        self._pass("slot_1_events", str(self.slot_events.get("1", 0)))


def doc_geometry_dimensions(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_slot": doc["query_slot"],
        "native_query_slot": doc["native_query_slot"],
        "mapping_order": doc["query_slot"],
        "qdp": doc["qdp"],
        "query_clause_source_pos": doc["query_clause_source_pos"],
        "query_value_clause_pos": doc["query_value_clause_pos"],
        "distractor_source_clause_pos": doc["distractor_source_clause_pos"],
        "distractor_value_clause_pos": doc["distractor_value_clause_pos"],
        "q_to_relevant_source_dist": doc["qdp"] - doc["query_clause_source_pos"],
        "q_to_relevant_value_dist": doc["qdp"] - doc["query_value_clause_pos"],
        "q_to_distractor_source_dist": doc["qdp"] - doc["distractor_source_clause_pos"],
        "q_to_distractor_value_dist": doc["qdp"] - doc["distractor_value_clause_pos"],
        "answer_token_index": doc["answer_token_index"],
        "answer_to_value_offset": doc["answer_token_index"] - doc["query_value_clause_pos"],
        "base_cell": doc["base_cell"],
        "base_actual_cell": doc["base_actual_cell"],
        "relation_round": doc["base_relation_round"],
    }


def pair_metrics(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    same_candidate = set(a["candidate_set"]) == set(b["candidate_set"])
    diff_record = a["base_record_id"] != b["base_record_id"]
    same_record = a["base_record_id"] == b["base_record_id"]
    reversed_binding = same_candidate and a["target_token_id"] != b["target_token_id"]

    ga = doc_geometry_dimensions(a)
    gb = doc_geometry_dimensions(b)
    dims = {name: ga[name] == gb[name] for name in GEOMETRY_DIMENSIONS}

    nonquery_same = (
        a["distractor_source_token_id"] == b["distractor_source_token_id"]
        and a["distractor_source_word"] == b["distractor_source_word"]
    )

    return {
        "stage0_same_query": True,
        "stage1_same_unordered_candidate_pair": same_candidate,
        "stage2_different_base_record": same_candidate and diff_record,
        "stage3_reversed_binding": same_candidate and diff_record and reversed_binding,
        "same_record": same_record,
        "different_record": diff_record,
        "reversed_binding": same_candidate and reversed_binding,
        "controls": {
            "same_unordered_candidate_pair": same_candidate,
            "different_base_record": diff_record,
            "same_base_record": same_record,
            "reversed_binding": same_candidate and reversed_binding,
            "same_query_absolute_position": a["qdp"] == b["qdp"],
            "same_query_slot": a["query_slot"] == b["query_slot"],
            "same_native_query_slot": a["native_query_slot"] == b["native_query_slot"],
            "same_mapping_order": a["query_slot"] == b["query_slot"],
            "same_nonqueried_source_identity": nonquery_same,
            "same_distractor_value": a["distractor_token_id"] == b["distractor_token_id"],
            "same_target_value": a["target_token_id"] == b["target_token_id"],
            "same_queried_source_pos": dims["query_clause_source_pos"],
            "same_queried_value_pos": dims["query_value_clause_pos"],
            "same_distractor_source_pos": dims["distractor_source_clause_pos"],
            "same_distractor_value_pos": dims["distractor_value_clause_pos"],
            "same_answer_position_geometry": (
                dims["answer_token_index"] and dims["answer_to_value_offset"]
            ),
            "same_base_cell": dims["base_cell"],
            "same_base_actual_cell": dims["base_actual_cell"],
            "same_relation_round": dims["relation_round"],
        },
        "dims": dims,
    }


def build_document_payload_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_id": doc["pair_id"],
        "twin": doc["twin"],
        "query_slot": doc["query_slot"],
        "query_word": doc["query_word"],
        "query_token_id": doc["query_token_id"],
        "target_token_id": doc["target_token_id"],
        "distractor_token_id": doc["distractor_token_id"],
        "distractor_source_word": doc["distractor_source_word"],
        "distractor_source_token_id": doc["distractor_source_token_id"],
        "candidate_pair_sorted": doc["candidate_set"],
        "query_difference_position": doc["qdp"],
        "answer_token_index": doc["answer_token_index"],
        "answer_causal_position": doc["answer_causal_position"],
        "query_clause_source_pos": doc["query_clause_source_pos"],
        "query_value_clause_pos": doc["query_value_clause_pos"],
        "distractor_source_clause_pos": doc["distractor_source_clause_pos"],
        "distractor_value_clause_pos": doc["distractor_value_clause_pos"],
        "clause_delta": doc["clause_delta"],
        "base_record_id": doc["base_record_id"],
        "base_record_index": doc["base_record_index"],
        "native_query_slot": doc["native_query_slot"],
        "base_cell": doc["base_cell"],
        "base_actual_cell": doc["base_actual_cell"],
        "base_relation_round": doc["base_relation_round"],
        "document_digest": doc["document_digest"],
    }


def build_same_query_rows(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        groups[doc["query_token_id"]].append(doc)
    rows = []
    for qid, group in sorted(groups.items()):
        group_sorted = sorted(group, key=lambda d: (d["pair_id"], d["twin"]))
        for a, b in combinations(group_sorted, 2):
            metrics = pair_metrics(a, b)
            rows.append(
                {
                    "query_token_id": a["query_token_id"],
                    "query_word": a["query_word"],
                    "pair_a": a["pair_id"],
                    "twin_a": a["twin"],
                    "pair_b": b["pair_id"],
                    "twin_b": b["twin"],
                    "base_record_a": a["base_record_id"],
                    "base_record_b": b["base_record_id"],
                    "target_token_a": a["target_token_id"],
                    "target_token_b": b["target_token_id"],
                    "distractor_token_a": a["distractor_token_id"],
                    "distractor_token_b": b["distractor_token_id"],
                    "same_record": metrics["same_record"],
                    **{k: metrics[k] for k in (
                        "stage0_same_query",
                        "stage1_same_unordered_candidate_pair",
                        "stage2_different_base_record",
                        "stage3_reversed_binding",
                        "reversed_binding",
                    )},
                    "controls": metrics["controls"],
                    "dims": metrics["dims"],
                }
            )
    rows.sort(
        key=lambda r: (
            r["query_token_id"], r["query_word"], r["pair_a"], r["twin_a"], r["pair_b"], r["twin_b"],
        )
    )
    return rows


def registry_subset(rows: List[Dict[str, Any]], predicate) -> List[Dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def build_registry_summary(rows: List[Dict[str, Any]], name: str, tier: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "name": name,
        "tier": tier,
        "comparison_count": len(rows),
    }
    if not rows:
        summary["note"] = "No eligible comparisons found under the registered definition."
        return summary
    control_names = [
        "same_unordered_candidate_pair",
        "different_base_record",
        "same_query_slot",
        "same_native_query_slot",
        "same_mapping_order",
        "same_query_absolute_position",
        "same_nonqueried_source_identity",
        "same_distractor_value",
        "same_target_value",
        "same_queried_source_pos",
        "same_queried_value_pos",
        "same_distractor_source_pos",
        "same_distractor_value_pos",
        "same_answer_position_geometry",
        "same_base_cell",
        "same_base_actual_cell",
        "same_relation_round",
    ]
    control_counts = {}
    for name_key in control_names:
        yes = sum(1 for row in rows if row["controls"][name_key])
        control_counts[name_key] = {"equal": yes, "not_equal": len(rows) - yes}
    summary["control_counts"] = control_counts
    dim_counts = {}
    for dim in GEOMETRY_DIMENSIONS:
        equal = sum(1 for row in rows if row["dims"][dim])
        dim_counts[dim] = {"equal": equal, "variable": len(rows) - equal}
    summary["geometry_dimension_counts"] = dim_counts
    return summary


def _fmt_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "-".join(str(int(x)) for x in key)
    return str(int(key))


def _sort_key(item: Tuple[Any, Any]) -> Any:
    return item[0]


def summarize_role_frequencies(
    docs: List[Dict[str, Any]],
    multiplicities: Dict[str, int],
) -> Dict[str, Any]:
    def value_key(role: str, doc: Dict[str, Any]) -> Any:
        if role == "query":
            return doc["query_token_id"]
        if role == "target":
            return doc["target_token_id"]
        if role == "distractor_value":
            return doc["distractor_token_id"]
        if role == "nonqueried_source":
            return doc["distractor_source_token_id"]
        if role == "candidate_set":
            return tuple(doc["candidate_set"])
        raise ValueError(role)

    out: Dict[str, Any] = {}
    for role in ROLE_NAMES:
        doc_counts: Counter = Counter()
        record_sets: Dict[Any, set] = defaultdict(set)
        exposures: Counter = Counter()
        for doc in docs:
            key = value_key(role, doc)
            doc_counts[key] += 1
            record_sets[key].add(doc["base_record_id"])
            exposures[key] += multiplicities.get(doc["pair_id"], 0)
        out[role] = {
            "distinct_values": len(doc_counts),
            "total_document_role_occurrences": int(sum(doc_counts.values())),
            "total_schedule_role_exposure": int(sum(exposures.values())),
            "value_documents": {
                _fmt_key(k): v for k, v in sorted(doc_counts.items(), key=_sort_key)
            },
            "value_base_records": {
                _fmt_key(k): len(record_sets[k]) for k in sorted(record_sets, key=_fmt_key)
            },
            "schedule_role_exposure": {
                _fmt_key(k): v for k, v in sorted(exposures.items(), key=_sort_key)
            },
        }
    return out


def collinearity_summary(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        groups[doc["query_token_id"]].append(doc)

    per_query_rows = []
    axis_aggregates: Dict[str, Dict[str, Any]] = {}
    for axis in COLLINEARITY_AXES:
        status_counts = Counter()
        distinct_counts = []
        for qid in sorted(groups):
            vals = set(doc_geometry_dimensions(doc)[axis] for doc in groups[qid])
            if len(vals) == 1:
                status_counts["invariant"] += 1
            elif len(vals) > 1:
                status_counts["variable"] += 1
            else:
                status_counts["unavailable"] += 1
            distinct_counts.append(len(vals))
        axis_aggregates[axis] = {
            "invariant_queries": status_counts.get("invariant", 0),
            "variable_queries": status_counts.get("variable", 0),
            "unavailable_queries": status_counts.get("unavailable", 0),
            "distinct_values_per_query": _safe_minmax_mean([float(x) for x in distinct_counts])
            if distinct_counts else {"n": 0},
        }

    for qid in sorted(groups):
        docs_q = groups[qid]
        row: Dict[str, Any] = {
            "query_token_id": qid,
            "query_word": docs_q[0]["query_word"],
            "document_count": len(docs_q),
            "base_record_count": len(set(doc["base_record_id"] for doc in docs_q)),
            "axes": {},
        }
        for axis in COLLINEARITY_AXES:
            vals = sorted(set(doc_geometry_dimensions(doc)[axis] for doc in docs_q))
            row["axes"][axis] = {
                "status": "invariant" if len(vals) == 1 else ("variable" if len(vals) > 1 else "unavailable"),
                "distinct_values": len(vals),
                "values": vals,
            }
        per_query_rows.append(row)
    return {
        "axis_summary": axis_aggregates,
        "per_query_identity": per_query_rows,
    }


def higher_order_summary(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    edges = [(doc["query_token_id"], doc["target_token_id"]) for doc in docs]
    edge_counts = Counter(edges)
    unique_edges = set(edges)
    nodes = set()
    for q, t in unique_edges:
        nodes.add(q)
        nodes.add(t)
    parent = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for q, t in unique_edges:
        union(q, t)
    comp_sizes = Counter(find(n) for n in nodes)

    two_cycles = 0
    cycle_pairs = []
    for u, v in combinations(sorted(nodes), 2):
        if (u, v) in unique_edges and (v, u) in unique_edges:
            two_cycles += 1
            cycle_pairs.append([u, v])

    return {
        "note": (
            "Descriptive higher-order structure over unique frozen documents only. Reciprocal-twin "
            "chains and cross-query cycles do NOT recover the causal leverage of a missing direct "
            "Stage-3 reversal unless they actually contain an eligible direct reversal."
        ),
        "unique_directed_query_to_target_edges": len(unique_edges),
        "edge_occurrence_histogram": {
            str(k): v for k, v in sorted(Counter(edge_counts.values()).items())
        },
        "distinct_identity_nodes_in_edges": len(nodes),
        "weakly_connected_components": len(comp_sizes),
        "weak_component_size_histogram": {
            str(k): v for k, v in sorted(Counter(comp_sizes.values()).items())
        },
        "mutual_2_cycles": two_cycles,
        "mutual_2_cycle_pairs": cycle_pairs,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Outcome-blind structural census - frozen Treatment 5 / Treatment 8 universe")
    lines.append("")
    lines.append("Structural universe shared verbatim by Treatment 5 and Treatment 8: "
                 "`treatment5_full_document_pair_pool.json` + "
                 "`treatment5_full_document_frozen_schedule.json`.")
    lines.append("")
    lines.append("**Blindness:** stdlib only; no model, checkpoint, metrics, results, audit, "
                 "positive-control, or sealed-evaluation file is read or loaded; no behavior is "
                 "inferred or reported. Outputs contain structural identities and counts only.")
    lines.append("")
    lines.append("**Unit of discovery:** the unique frozen pair pool (master registry). "
                 "T5 and T8 are not independent datasets. Repeated schedule exposures are never "
                 "independent structural observations.")
    lines.append("")

    lines.append("## Provenance / input files")
    lines.append("")
    lines.append("| path | role | sha256 |")
    lines.append("|---|---|---|")
    for entry in payload["provenance"]["input_files"]:
        lines.append(f"| `{entry['path']}` | {entry['role']} | `{entry['sha256']}` |")
    lines.append("")
    lines.append("Frozen hash verification: pair_pool match = "
                 f"{payload['provenance']['frozen_hash_verification']['pair_pool']['match']}, "
                 "schedule match = "
                 f"{payload['provenance']['frozen_hash_verification']['schedule']['match']}.")
    lines.append("")

    universe = payload["universe"]
    lines.append("## Structural universe / validation")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| unique pair records | {universe['pair_records']} |")
    lines.append(f"| unique base records | {universe['base_records']} |")
    lines.append(f"| pair_id <-> base_record_id relationship | {universe['pair_record_relationship']} |")
    lines.append(f"| unique twin documents | {universe['unique_twin_documents']} |")
    lines.append(f"| unique document digests | {universe['unique_document_digests']} |")
    lines.append(f"| duplicate document digests | {universe['duplicate_document_digests']} |")
    lines.append(f"| built-in reciprocal twin pairs | {universe['built_in_reciprocal_twin_pairs']} |")
    lines.append(f"| distinct query identity tokens | {universe['query_identity_count']} |")
    dq = universe["docs_per_query_identity"]
    lines.append(f"| documents per query identity (min/median/max) | {dq['min']}/{dq['median']}/{dq['max']} |")
    lines.append(f"| distinct unordered candidate sets | {universe['unordered_candidate_set_count']} |")
    lines.append(f"| candidate_pair stored order | {universe['candidate_pair_stored_order']} |")
    lines.append("")

    schedule_summary = payload["schedule"]
    lines.append("## Schedule exposure multiplicities (not independent observations)")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| schedule events | {schedule_summary['event_count']} |")
    lines.append(f"| steps | {schedule_summary['step_count']} |")
    lines.append(f"| step numbering contract | {schedule_summary['step_numbering_contract']} |")
    lines.append(f"| twin-pair presentations | {schedule_summary['pair_presentation_count']} |")
    pp = schedule_summary["pair_presentations_per_pair"]
    lines.append(f"| pair presentations per pair (min/median/max) | {pp['min']}/{pp['median']}/{pp['max']} |")
    epair = schedule_summary["events_per_pair"]
    lines.append(f"| events per pair (min/median/max) | {epair['min']}/{epair['median']}/{epair['max']} |")
    ep = schedule_summary["events_per_unique_document"]
    lines.append(f"| events per unique document (min/median/max) | {ep['min']}/{ep['median']}/{ep['max']} |")
    lines.append(f"| slot-0 / slot-1 events | {schedule_summary['slot_0_event_count']} / {schedule_summary['slot_1_event_count']} |")
    lines.append(f"| distinct pairs present in schedule | {schedule_summary['pairs_present_in_schedule']} |")
    lines.append(f"| content mismatches between schedule and pool | {schedule_summary['content_mismatch_events']} |")
    lines.append("")

    lines.append("## Stage 0-3 attrition (literal definitions)")
    lines.append("")
    lines.append("| stage | restriction | unique comparison pairs |")
    lines.append("|---|---|---|")
    for entry in payload["attrition"]["stages"]:
        lines.append(f"| {entry['stage']} | {entry['label']} | {entry['count']} |")
    lines.append("")
    att = payload["attrition"]
    lines.append(f"| same-record comparisons at Stage 0 (natural attrition) | {att['same_record_comparisons']} |")
    lines.append(f"| Stage 2 subset with identical (non-reversed) binding | {att['stage_2_same_binding_comparisons']} |")
    lines.append("")
    lines.append("Stage 0 does not require different pairs or records; same-record comparisons, if "
                 "any, attrit at Stage 2.")
    lines.append("")

    lines.append("## Registry A - strict reversal (Tier A, = Stage 3)")
    lines.append("")
    lines.append(f"Eligible comparison count: {payload['registries']['A']['comparison_count']}")
    lines.append("")
    lines.append("## Registries B-E (Tier B, structural only)")
    lines.append("")
    lines.append("| registry | tier | definition | eligible comparison pairs |")
    lines.append("|---|---|---|---|")
    for key in ("B", "C", "D", "E"):
        reg = payload["registries"][key]
        lines.append(f"| {key} | {reg['tier']} | {reg['definition']} | {reg['comparison_count']} |")
    lines.append("")
    lines.append("Tier-B registries never substitute for Stage 3 and never prove binding.")
    lines.append("")

    col = payload["collinearity"]
    lines.append("## Query identity <-> structure collinearity")
    lines.append("")
    lines.append("| axis | invariant queries | variable queries | unavailable |")
    lines.append("|---|---|---|---|")
    for axis, agg in col["axis_summary"].items():
        lines.append(f"| {axis} | {agg['invariant_queries']} | {agg['variable_queries']} | {agg['unavailable_queries']} |")
    lines.append("")

    freq = payload["frequencies"]
    lines.append("## Unique-document frequencies (each unique twin document counted once)")
    lines.append("")
    lines.append("| role | distinct values |")
    lines.append("|---|---|")
    for role, data in freq["unique_documents"].items():
        lines.append(f"| {role} | {data['distinct_values']} |")
    lines.append("")
    lines.append("## Schedule-weighted role exposure multiplicities")
    lines.append("")
    lines.append("| role | distinct values |")
    lines.append("|---|---|")
    for role, data in freq["schedule_exposure"].items():
        lines.append(f"| {role} | {data['distinct_values']} |")
    lines.append("")
    lines.append("Schedule-weighted numbers are exposure multiplicities, never combined with "
                 "unique-document counts into one effective sample size.")
    lines.append("")

    ho = payload["higher_order"]
    lines.append("## Higher-order structure (descriptive)")
    lines.append("")
    lines.append(f"- unique directed query->target edges (unique docs): {ho['unique_directed_query_to_target_edges']}")
    lines.append(f"- distinct identity nodes in edges: {ho['distinct_identity_nodes_in_edges']}")
    lines.append(f"- weakly connected components: {ho['weakly_connected_components']}")
    lines.append(f"- mutual 2-cycles: {ho['mutual_2_cycles']}")
    lines.append("")
    lines.append(ho["note"])
    lines.append("")

    lines.append("## Interpretation rule (verbatim)")
    lines.append("")
    lines.append(payload["interpretation_rule"])
    lines.append("")
    lines.append("No minimum-N, balance, ratio, significance, or adequacy thresholds were imposed.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Outcome-blind structural census T5/T8.")
    parser.add_argument("--pair-pool", type=Path, default=DEFAULT_PAIR_POOL_PATH)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--preflight", type=Path, default=None,
                        help="Optional preflight result for hash/count cross-check only.")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip the optional preflight cross-check.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    pair_pool_path = args.pair_pool
    schedule_path = args.schedule

    require(pair_pool_path.is_file(), f"Missing pair pool: {pair_pool_path}")
    require(schedule_path.is_file(), f"Missing schedule: {schedule_path}")

    pair_pool_sha = sha256_file(pair_pool_path)
    require(pair_pool_sha == EXPECTED_PAIR_POOL_SHA256,
            "Pair pool SHA256 mismatch against frozen authoritative hash; aborting.")
    schedule_sha = sha256_file(schedule_path)
    require(schedule_sha == EXPECTED_SCHEDULE_SHA256,
            "Schedule SHA256 mismatch against frozen authoritative hash; aborting.")

    pool = load_json(pair_pool_path)
    pool_index = PoolIndex(pool)

    schedule = load_json(schedule_path)
    schedule_index = ScheduleIndex(schedule, pool_index.docs_by_pair_twin)

    preflight_check: Optional[Dict[str, Any]] = None
    preflight_used = False
    preflight_sha: Optional[str] = None
    if not args.no_preflight:
        preflight_path = args.preflight or DEFAULT_PREFLIGHT_PATH
        if preflight_path.is_file():
            preflight = load_json(preflight_path)
            preflight_used = True
            preflight_sha = sha256_file(preflight_path)
            preflight_check = cross_check_preflight(preflight, pool_index, schedule_index,
                                                    pair_pool_sha, schedule_sha, preflight_sha)
        else:
            preflight_check = {"present": False,
                               "note": f"optional preflight not present at {preflight_path}"}
    else:
        preflight_check = {"present": False, "note": "preflight cross-check disabled by flag"}

    docs = pool_index.docs
    rows = build_same_query_rows(docs)

    stage_counts = {
        0: sum(1 for r in rows),
        1: sum(1 for r in rows if r["stage1_same_unordered_candidate_pair"]),
        2: sum(1 for r in rows if r["stage2_different_base_record"]),
        3: sum(1 for r in rows if r["stage3_reversed_binding"]),
    }
    same_record_comparisons = sum(1 for r in rows if r["same_record"])
    stage2_same_binding = sum(
        1 for r in rows
        if r["stage2_different_base_record"] and not r["controls"]["reversed_binding"]
    )

    def reg(predicate) -> List[Dict[str, Any]]:
        return registry_subset(rows, predicate)

    reg_a_rows = reg(lambda r: r["stage3_reversed_binding"])
    reg_b_rows = reg(lambda r: (
        r["controls"]["different_base_record"]
        and r["controls"]["same_distractor_value"]
        and not r["controls"]["same_target_value"]
    ))
    reg_c_rows = reg(lambda r: (
        r["stage2_different_base_record"]
        and not r["controls"]["reversed_binding"]
        and (not all(r["dims"][d] for d in GEOMETRY_DIMENSIONS))
    ))
    reg_d_rows = reg(lambda r: (
        r["controls"]["different_base_record"]
        and r["controls"]["same_target_value"]
        and (
            not r["controls"]["same_distractor_value"]
            or not r["controls"]["same_nonqueried_source_identity"]
        )
    ))

    reg_a = build_registry_summary(reg_a_rows, "Registry A - strict reversal", "Tier A")
    reg_b = build_registry_summary(reg_b_rows, "Registry B - shared-distractor target-change", "Tier B")
    reg_c = build_registry_summary(reg_c_rows, "Registry C - same binding, varied geometry", "Tier B")
    reg_d = build_registry_summary(reg_d_rows, "Registry D - same query/target, varied distractor", "Tier B")

    reg_d_value_change = sum(
        1 for r in reg_d_rows if not r["controls"]["same_distractor_value"]
    )
    reg_d_source_change = sum(
        1 for r in reg_d_rows if not r["controls"]["same_nonqueried_source_identity"]
    )

    # Registry E: same unordered candidate set + same target + different query + different record.
    groups_e: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        groups_e[(tuple(doc["candidate_set"]), doc["target_token_id"])].append(doc)
    reg_e_rows = []
    for key, group in sorted(groups_e.items(), key=lambda kv: kv[0]):
        group_sorted = sorted(group, key=lambda d: (d["pair_id"], d["twin"]))
        for a, b in combinations(group_sorted, 2):
            if a["base_record_id"] == b["base_record_id"]:
                continue
            if a["query_token_id"] == b["query_token_id"]:
                continue
            metrics = pair_metrics(a, b)
            reg_e_rows.append(
                {
                    "query_token_id_a": a["query_token_id"],
                    "query_word_a": a["query_word"],
                    "query_token_id_b": b["query_token_id"],
                    "query_word_b": b["query_word"],
                    "pair_a": a["pair_id"],
                    "twin_a": a["twin"],
                    "pair_b": b["pair_id"],
                    "twin_b": b["twin"],
                    "base_record_a": a["base_record_id"],
                    "base_record_b": b["base_record_id"],
                    "target_token_a": a["target_token_id"],
                    "target_token_b": b["target_token_id"],
                    "controls": metrics["controls"],
                    "dims": metrics["dims"],
                }
            )
    reg_e_rows.sort(
        key=lambda r: (
            r["query_token_id_a"], r["query_word_a"], r["query_token_id_b"], r["query_word_b"],
            r["pair_a"], r["pair_b"],
        )
    )
    reg_e_summary_rows = []
    for row in reg_e_rows:
        reg_e_summary_rows.append({**row, "controls": row["controls"], "dims": row["dims"]})
    reg_e = build_registry_summary(reg_e_summary_rows,
                                   "Registry E - same candidate/target, varied query", "Tier B")

    definitions = {
        "A": "same query identity + same unordered candidate set + different base record + reversed target",
        "B": "same query identity + same distractor value + different target identity + different base record",
        "C": "same query identity + same target + same unordered candidate set + different base record + "
             "at least one geometry/layout variable differing",
        "D": "same query identity + same target + different distractor/candidate context + different base record",
        "E": "same unordered candidate set + same target + different query identity + different base record",
    }
    registries = {
        "A": {**reg_a, "definition": definitions["A"]},
        "B": {**reg_b, "definition": definitions["B"]},
        "C": {**reg_c, "definition": definitions["C"]},
        "D": {
            **reg_d,
            "definition": definitions["D"],
            "distractor_value_change_comparisons": reg_d_value_change,
            "nonqueried_source_change_comparisons": reg_d_source_change,
        },
        "E": {**reg_e, "definition": definitions["E"]},
    }

    pair_presentation_map = {
        str(pair["pair_id"]): schedule_index.pair_presentations.get(str(pair["pair_id"]), 0)
        for pair in pool_index.pairs
    }
    require(
        sum(pair_presentation_map[doc["pair_id"]] for doc in docs) == schedule_index.event_count,
        "Per-document schedule multiplicity total != schedule event count.",
    )
    per_doc_multiplicity = {
        doc["pair_id"]: pair_presentation_map[doc["pair_id"]] for doc in docs
    }

    doc_digests = Counter(doc["document_digest"] for doc in docs)
    candidate_sets = Counter(tuple(doc["candidate_set"]) for doc in docs)
    cell_counts = Counter(doc["base_actual_cell"] for doc in docs)
    qdp_histogram = Counter(doc["qdp"] for doc in docs)

    docs_by_qid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        docs_by_qid[doc["query_token_id"]].append(doc)
    docs_per_query = sorted(len(v) for v in docs_by_qid.values())

    word_to_token: Dict[str, set] = defaultdict(set)
    for doc in docs:
        word_to_token[doc["query_word"]].add(doc["query_token_id"])
    word_to_token_id_conflicts = sum(1 for s in word_to_token.values() if len(s) != 1)

    collinearity = collinearity_summary(docs)

    unit_multiplicities = {pid: 1 for pid in set(doc["pair_id"] for doc in docs)}
    frequencies = {
        "unit": (
            "unique frozen twin documents are the unit for 'documents'; 'base_records' counts "
            "distinct base records; 'schedule_role_exposure' counts role occurrences across the "
            "frozen schedule and is labeled as exposure multiplicity, never as independent "
            "observations."
        ),
        "unique_documents": summarize_role_frequencies(docs, unit_multiplicities),
        "schedule_exposure": summarize_role_frequencies(docs, per_doc_multiplicity),
    }

    higher_order = higher_order_summary(docs)

    doc_freq = frequencies["unique_documents"]
    uni = {
        "pair_records": len(pool_index.pairs),
        "base_records": len(set(doc["base_record_id"] for doc in docs)),
        "pair_record_relationship": "one-to-one (validated)",
        "unique_twin_documents": len(docs),
        "unique_document_digests": len(doc_digests),
        "duplicate_document_digests": sum(v - 1 for v in doc_digests.values() if v > 1),
        "built_in_reciprocal_twin_pairs": len(pool_index.pairs),
        "query_identity_count": len(docs_by_qid),
        "docs_per_query_identity": {
            "min": min(docs_per_query),
            "max": max(docs_per_query),
            "mean": statistics.mean(docs_per_query),
            "median": statistics.median(docs_per_query),
            "histogram": {str(k): v for k, v in sorted(Counter(docs_per_query).items())},
        },
        "unordered_candidate_set_count": len(candidate_sets),
        "documents_per_unordered_candidate_set": {
            "min": min(doc_freq["candidate_set"]["value_documents"].values()),
            "max": max(doc_freq["candidate_set"]["value_documents"].values()),
            "mean": statistics.mean(list(doc_freq["candidate_set"]["value_documents"].values())),
            "value_documents": doc_freq["candidate_set"]["value_documents"],
            "base_records_per_set": doc_freq["candidate_set"]["value_base_records"],
        },
        "candidate_pair_stored_order": "sorted_ascending_canonical",
        "base_actual_cell_unique_count": len(cell_counts),
        "word_to_token_id_conflicts": word_to_token_id_conflicts,
    }

    schedule_exposures = sorted(schedule_index.pair_presentations.values())
    events_per_pair = sorted(schedule_index.events_by_pair.values())
    events_per_doc = [per_doc_multiplicity[doc["pair_id"]] for doc in docs]

    schedule_summary = {
        "event_count": schedule_index.event_count,
        "step_count": _as_int(schedule.get("steps", -1), "steps"),
        "step_numbering_contract": "1-based, consecutive, final equals step count",
        "pair_presentation_count": sum(schedule_index.pair_presentations.values()),
        "pair_presentations_per_pair": {
            "min": min(schedule_exposures),
            "max": max(schedule_exposures),
            "mean": statistics.mean(schedule_exposures),
            "median": statistics.median(schedule_exposures),
            "histogram": {str(k): v for k, v in sorted(Counter(schedule_exposures).items())},
        },
        "events_per_pair": {
            "min": min(events_per_pair),
            "max": max(events_per_pair),
            "mean": statistics.mean(events_per_pair),
            "median": statistics.median(events_per_pair),
            "histogram": {str(k): v for k, v in sorted(Counter(events_per_pair).items())},
        },
        "events_per_unique_document": {
            "min": min(events_per_doc),
            "max": max(events_per_doc),
            "mean": statistics.mean(events_per_doc),
            "median": statistics.median(events_per_doc),
        },
        "slot_0_event_count": schedule_index.slot_events.get("0", 0),
        "slot_1_event_count": schedule_index.slot_events.get("1", 0),
        "pairs_present_in_schedule": len(schedule_index.pair_presentations),
        "pairs_with_zero_exposure": len(pool_index.pairs) - len(schedule_index.pair_presentations),
        "content_mismatch_events": schedule_index.mismatch_events,
    }

    attrition = {
        "stages": [
            {"stage": 0, "label": "same query_token_id (no other restriction)", "count": stage_counts[0]},
            {"stage": 1, "label": "Stage 0 + same unordered candidate set", "count": stage_counts[1]},
            {"stage": 2, "label": "Stage 1 + different base_record_id", "count": stage_counts[2]},
            {"stage": 3, "label": "Stage 2 + reversed target binding within same candidate set",
             "count": stage_counts[3]},
        ],
        "same_record_comparisons": same_record_comparisons,
        "stage_2_same_binding_comparisons": stage2_same_binding,
        "note": (
            "Stage 0 requires only the same query token. Same-record comparisons, if any, attrit "
            "naturally at Stage 2. pair_id and base_record_id were validated one-to-one and the "
            "Stage-2 test uses base_record_id directly."
        ),
    }

    input_files = [
        {"path": str(pair_pool_path), "role": "pair pool (master structural registry)",
         "sha256": pair_pool_sha},
        {"path": str(schedule_path),
         "role": "frozen schedule (event multiplicity/exposure verification)",
         "sha256": schedule_sha},
    ]
    if preflight_used:
        input_files.append(
            {"path": str(args.preflight or DEFAULT_PREFLIGHT_PATH),
             "role": "optional preflight result (hash/count cross-check only; not part of the "
                     "authoritative structural universe)",
             "sha256": preflight_sha}
        )

    payload = {
        "census_kind": "outcome_blind_structural_census",
        "census_schema_version": 2,
        "structural_universe": (
            "treatment5_full_document_pair_pool.json + treatment5_full_document_frozen_schedule.json; "
            "shared verbatim by Treatment 5 and Treatment 8."
        ),
        "unit_of_discovery": (
            "unique frozen pair pool (1536 pairs / 3072 twin documents). "
            "T5 and T8 are not independent datasets. Schedule repetitions are exposure "
            "multiplicities and are never independent structural observations."
        ),
        "blindness_declaration": {
            "standard_library_only": True,
            "torch_imported": False,
            "checkpoint_loaded": False,
            "model_loaded": False,
            "training_metrics_read": False,
            "training_result_read": False,
            "final_retention_audit_read": False,
            "positive_control_read": False,
            "sealed_evaluation_read": False,
            "behavioral_outcomes_emitted": False,
        },
        "provenance": {
            "input_files": input_files,
            "preflight_cross_check_used": preflight_used,
            "frozen_hash_verification": {
                "pair_pool": {
                    "expected_sha256": EXPECTED_PAIR_POOL_SHA256,
                    "observed_sha256": pair_pool_sha,
                    "match": pair_pool_sha == EXPECTED_PAIR_POOL_SHA256,
                },
                "schedule": {
                    "expected_sha256": EXPECTED_SCHEDULE_SHA256,
                    "observed_sha256": schedule_sha,
                    "match": schedule_sha == EXPECTED_SCHEDULE_SHA256,
                },
            },
        },
        "validation": {
            "passed": True,
            "pair_pool_checks": pool_index.checks,
            "schedule_checks": schedule_index.checks,
        },
        "preflight_cross_check": preflight_check,
        "interpretation_rule": INTERPRETATION_RULE,
        "universe": uni,
        "schedule": schedule_summary,
        "attrition": attrition,
        "registries": registries,
        "collinearity": collinearity,
        "frequencies": frequencies,
        "higher_order": higher_order,
        "geometry": {
            "dimension_definitions": {
                "mapping_order": "queried mapping clause position (0 = first/slot0, 1 = second/slot1)",
                "qdp": "absolute token index of the query identity word in the document",
            },
            "geometry_dimension_names": GEOMETRY_DIMENSIONS,
            "query_difference_position_histogram": {
                str(k): v for k, v in sorted(qdp_histogram.items())
            },
            "base_actual_cell_histogram": dict(sorted(cell_counts.items())),
        },
        "documents": [build_document_payload_doc(doc) for doc in sorted(
            docs, key=lambda d: (d["pair_id"], d["twin"]))],
        "comparison_rows": [
            {
                "query_token_id": r["query_token_id"],
                "query_word": r["query_word"],
                "pair_a": r["pair_a"],
                "twin_a": r["twin_a"],
                "pair_b": r["pair_b"],
                "twin_b": r["twin_b"],
                "base_record_a": r["base_record_a"],
                "base_record_b": r["base_record_b"],
                "stage0_same_query": r["stage0_same_query"],
                "stage1_same_unordered_candidate_pair": r["stage1_same_unordered_candidate_pair"],
                "stage2_different_base_record": r["stage2_different_base_record"],
                "stage3_reversed_binding": r["stage3_reversed_binding"],
                "same_record": r["same_record"],
                "controls": r["controls"],
                "dims": r["dims"],
            }
            for r in rows
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=1, sort_keys=False), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    return 0


def cross_check_preflight(
    preflight: Dict[str, Any],
    pool_index: PoolIndex,
    schedule_index: ScheduleIndex,
    pair_pool_sha: str,
    schedule_sha: str,
    preflight_sha: str,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def compare(label: str, actual: Any, expected_value: Any) -> None:
        ok = actual == expected_value
        checks.append({"check": label, "expected": expected_value, "observed": actual, "match": ok})
        require(ok, f"Preflight cross-check failed: {label}.")

    pool_meta = preflight.get("pair_pool", {})
    schedule_meta = preflight.get("frozen_schedule", {})
    compare("preflight_file_sha256_present", preflight_sha != "", True)
    compare("pair_pool_sha256", str(pool_meta.get("sha256", "")), pair_pool_sha)
    compare("schedule_sha256", str(schedule_meta.get("sha256", "")), schedule_sha)
    compare("preflight_unique_counterfactual_pairs", int(preflight.get("unique_counterfactual_pairs", -1)),
            len(pool_index.pairs))
    compare("preflight_unique_complete_synthetic_documents",
            int(preflight.get("unique_complete_synthetic_documents", -1)), len(pool_index.docs))
    compare("preflight_steps", int(preflight.get("steps", -1)),
            _as_int(schedule_index.schedule.get("steps", -1), "steps"))
    compare("preflight_scheduled_examples", int(preflight.get("scheduled_examples", -1)),
            schedule_index.event_count)
    compare("preflight_pair_presentations", int(preflight.get("pair_presentations", -1)),
            sum(schedule_index.pair_presentations.values()))
    compare("preflight_slot_0_presentations", int(preflight.get("slot_0_presentations", -1)),
            schedule_index.slot_events.get("0", 0))
    compare("preflight_slot_1_presentations", int(preflight.get("slot_1_presentations", -1)),
            schedule_index.slot_events.get("1", 0))
    return {
        "present": True,
        "passed": True,
        "file_sha256": preflight_sha,
        "authoritative_universe_member": False,
        "checks": checks,
    }


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CensusError as exc:
        print(f"CENSUS ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
