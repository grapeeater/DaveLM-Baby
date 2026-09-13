# Outcome-blind structural census - frozen Treatment 5 / Treatment 8 universe

Structural universe shared verbatim by Treatment 5 and Treatment 8: `treatment5_full_document_pair_pool.json` + `treatment5_full_document_frozen_schedule.json`.

**Blindness:** stdlib only; no model, checkpoint, metrics, results, audit, positive-control, or sealed-evaluation file is read or loaded; no behavior is inferred or reported. Outputs contain structural identities and counts only.

**Unit of discovery:** the unique frozen pair pool (master registry). T5 and T8 are not independent datasets. Repeated schedule exposures are never independent structural observations.

## Provenance / input files

| path | role | sha256 |
|---|---|---|
| `C:\DaveLM-CADAVER\treatment5_counterfactual_pairs_seed8382\treatment5_full_document_pair_pool.json` | pair pool (master structural registry) | `0b31e39361f7a45dc4182c49ab4b4967bffe539cd2ff12e7b196eabfb3dac307` |
| `C:\DaveLM-CADAVER\treatment5_counterfactual_pairs_seed8382\treatment5_full_document_frozen_schedule.json` | frozen schedule (event multiplicity/exposure verification) | `a4e64fd9d1b934aa440a5d036a3bec0ac7c9ac97d58c72fb45df6d8576bb7c12` |
| `C:\DaveLM-CADAVER\treatment5_counterfactual_pairs_seed8382\treatment5_full_document_preflight_result.json` | optional preflight result (hash/count cross-check only; not part of the authoritative structural universe) | `e0aa2639c05da25df1a3b7ec5a596d4c9d3ec885bddc45012c67c33aafd85ae0` |

Frozen hash verification: pair_pool match = True, schedule match = True.

## Structural universe / validation

| quantity | value |
|---|---|
| unique pair records | 1536 |
| unique base records | 1536 |
| pair_id <-> base_record_id relationship | one-to-one (validated) |
| unique twin documents | 3072 |
| unique document digests | 3072 |
| duplicate document digests | 0 |
| built-in reciprocal twin pairs | 1536 |
| distinct query identity tokens | 64 |
| documents per query identity (min/median/max) | 48/48.0/48 |
| distinct unordered candidate sets | 1024 |
| candidate_pair stored order | sorted_ascending_canonical |

## Schedule exposure multiplicities (not independent observations)

| quantity | value |
|---|---|
| schedule events | 32000 |
| steps | 1000 |
| step numbering contract | 1-based, consecutive, final equals step count |
| twin-pair presentations | 16000 |
| pair presentations per pair (min/median/max) | 10/10.0/11 |
| events per pair (min/median/max) | 20/20.0/22 |
| events per unique document (min/median/max) | 10/10.0/11 |
| slot-0 / slot-1 events | 16000 / 16000 |
| distinct pairs present in schedule | 1536 |
| content mismatches between schedule and pool | 0 |

## Stage 0-3 attrition (literal definitions)

| stage | restriction | unique comparison pairs |
|---|---|---|
| 0 | same query_token_id (no other restriction) | 72192 |
| 1 | Stage 0 + same unordered candidate set | 128 |
| 2 | Stage 1 + different base_record_id | 128 |
| 3 | Stage 2 + reversed target binding within same candidate set | 0 |

| same-record comparisons at Stage 0 (natural attrition) | 0 |
| Stage 2 subset with identical (non-reversed) binding | 128 |

Stage 0 does not require different pairs or records; same-record comparisons, if any, attrit at Stage 2.

## Registry A - strict reversal (Tier A, = Stage 3)

Eligible comparison count: 0

## Registries B-E (Tier B, structural only)

| registry | tier | definition | eligible comparison pairs |
|---|---|---|---|
| B | Tier B | same query identity + same distractor value + different target identity + different base record | 512 |
| C | Tier B | same query identity + same target + same unordered candidate set + different base record + at least one geometry/layout variable differing | 128 |
| D | Tier B | same query identity + same target + different distractor/candidate context + different base record | 1536 |
| E | Tier B | same unordered candidate set + same target + different query identity + different base record | 1024 |

Tier-B registries never substitute for Stage 3 and never prove binding.

## Query identity <-> structure collinearity

| axis | invariant queries | variable queries | unavailable |
|---|---|---|---|
| query_slot | 0 | 64 | 0 |
| native_query_slot | 0 | 64 | 0 |
| mapping_order | 0 | 64 | 0 |
| qdp | 0 | 64 | 0 |
| q_to_relevant_source_dist | 0 | 64 | 0 |
| q_to_relevant_value_dist | 0 | 64 | 0 |
| q_to_distractor_source_dist | 0 | 64 | 0 |
| q_to_distractor_value_dist | 0 | 64 | 0 |
| base_cell | 0 | 64 | 0 |
| base_actual_cell | 0 | 64 | 0 |
| relation_round | 0 | 64 | 0 |

## Unique-document frequencies (each unique twin document counted once)

| role | distinct values |
|---|---|
| query | 64 |
| target | 64 |
| distractor_value | 64 |
| nonqueried_source | 64 |
| candidate_set | 1024 |

## Schedule-weighted role exposure multiplicities

| role | distinct values |
|---|---|
| query | 64 |
| target | 64 |
| distractor_value | 64 |
| nonqueried_source | 64 |
| candidate_set | 1024 |

Schedule-weighted numbers are exposure multiplicities, never combined with unique-document counts into one effective sample size.

## Higher-order structure (descriptive)

- unique directed query->target edges (unique docs): 1536
- distinct identity nodes in edges: 64
- weakly connected components: 1
- mutual 2-cycles: 0

Descriptive higher-order structure over unique frozen documents only. Reciprocal-twin chains and cross-query cycles do NOT recover the causal leverage of a missing direct Stage-3 reversal unless they actually contain an eligible direct reversal.

## Interpretation rule (verbatim)

If Stage 3 = 0, the valid conclusion is only that the frozen T5/T8 pair pool contains no strict same-query, same-candidate, cross-record reversed-binding comparisons under the registered definition. This is not evidence that Baby cannot bind contextually and is not evidence that T8 is a shortcut. Tier-B groups may later test specific shortcut hypotheses under a separately preregistered behavioral analysis. They must never be described as substitutes for Stage 3 or as proof of binding.

No minimum-N, balance, ratio, significance, or adequacy thresholds were imposed.
