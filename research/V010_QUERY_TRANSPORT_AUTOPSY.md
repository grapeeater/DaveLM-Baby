# Query-transport autopsy: the selection framework was measuring the wrong thing

Status: **read-only recomputation + new read-only causal probe**. No optimizer
step was taken. Frozen panels, S1 receipts, and S2 receipts were not rewritten.
Gate L/C/R were not changed. Protected TEST was not opened.

Probe module: [`src/baby_v010/query_locality.py`](../src/baby_v010/query_locality.py)
Runner: [`scripts/probe_query_locality.py`](../scripts/probe_query_locality.py)
Tests: [`tests/test_query_locality.py`](../tests/test_query_locality.py)
Machine-readable: `runs/query_locality/QUERY_LOCALITY.json`

Parent SHA-256 verified `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
S2 `DIAGNOSTIC.json` verified against `runs/selection_s2/MANIFEST.json`.
Device `cuda` / AMD Radeon RX 9060 XT, `torch 2.9.1+rocmsdk20260116`.

## Headline

Baby v0.10 **already has** a working query→key→value bind-and-copy circuit. Its
effective reach is **about one token**. Queried selection fails whenever the query
key sits two or more tokens before the generation position, because the query
key's *identity is never transported* to the position where the answer is
emitted — not because binding, candidate representation, span copy, or
first-token logit strength are missing.

"First-token selection failure", "query blindness", and "the query signal merely
needs to be stronger" are all **falsified** as descriptions of the defect.

## The instrument was aggregating over a hard structural boundary

`data_v2._keyed` renders six templates. Two of them place the query key exactly
one token before the generation position **by construction**; one places it
before the body; three place it a variable filler-length away:

| variant | tail of `input` | query→generation gap |
|---|---|---|
| `keyed_0` | `… body … filler_b, m0, query_key, m1` | **always 1** |
| `keyed_2` | `… body … filler_b, query_key, m5` | **always 1** |
| `keyed_1` | `… body, m3, query_key, *filler_b` | `len(filler_b)` |
| `keyed_4` | `body, m1, *filler_a, m3, query_key, *filler_b` | `len(filler_b)` |
| `keyed_3` | `body, *filler_a, query_key, m0, *filler_b` | `1 + len(filler_b)` |
| `keyed_5` | `*filler_a, query_key, m2, *body, m4, *filler_b` | always large |

Recomputed from the frozen S2 diagnostic (432 rows, 144 all-K bodies), scoring
the queried head's argmax among the K candidate heads:

| variant | gap | n | parent | chance | excess |
|---|---|---:|---:|---:|---:|
| `keyed_0` | 1 | 88 | 0.511 | 0.318 | **+0.19** |
| `keyed_2` | 1 | 62 | 0.387 | 0.323 | +0.06 |
| `keyed_4` | ≤1 | 8 | 0.625 | 0.375 | +0.25 |
| `keyed_4` | >1 | 82 | 0.366 | 0.354 | +0.01 |
| `keyed_1` | ≥2 | 61 | 0.361 | 0.344 | +0.02 |
| `keyed_3` | ≥2 | 70 | 0.343 | 0.329 | +0.01 |
| `keyed_5` | ≥20 | 61 | 0.328 | 0.328 | **+0.00** |

Long-gap strata are at **exact** chance, which is the arithmetic signature of a
query-invariant favourite: one query per body is right, every time. Pooled over
the three variants that actually span a gap range (`keyed_1/3/4`, so wrapper
identity is held roughly fixed):

| gap | n | parent | chance | excess |
|---|---:|---:|---:|---:|
| 0–3 | 29 | 0.448 | 0.345 | +0.10 |
| 4–12 | 38 | 0.368 | 0.316 | +0.05 |
| 13–30 | 54 | 0.352 | 0.352 | +0.00 |
| >30 | 100 | 0.350 | 0.350 | +0.00 |

Because gap and variant are nearly collinear in the frozen panels, this alone is
suggestive, not conclusive. The probe below breaks the collinearity.

## The logit decomposition: bias is 12× the signal

For each body, with all K queries scored, write the first-answer logit of
candidate `c` under query `i` as a query-invariant salience `a_c` plus a
query-dependent deviation. Recomputed from the frozen evals:

| checkpoint | salience spread (median) | query-dependent spread (median) | raw argmax | bias-removed argmax |
|---|---:|---:|---:|---:|
| parent U16000 | **2.776** | 0.237 | 0.394 | 0.495 |
| S2 control +400 | 2.843 | 0.252 | 0.382 | 0.526 |
| S2 treatment +400 | 2.660 | **0.341** | 0.438 | 0.542 |

Two things follow.

1. S2's contrast `effect_ij = (L_i[i] − L_i[j]) − (L_j[i] − L_j[j])` is
   **algebraically invariant** to `a_c`. S2 could only ever move the 0.24-nat
   term and was structurally blind to the 2.78-nat term it had to beat. It did
   exactly that: query-dependent spread 0.237 → 0.341, salience spread unchanged.
2. Even an oracle that removed `a_c` perfectly would reach only ~0.50. So the
   query-dependent term is not a suppressed-but-sufficient signal either. Both
   halves of the previous framework are wrong.

For K=2 the algebra is exact: both queries are right only when the
query-dependent effect exceeds twice the salience bias. Parent
`body_all_correct` = 5/144; S2 raised the effect from 0.78 to 1.55 against a
required ≈2.8 and got 12/144. That is the whole S2 result.

## The causal probe: transport, demonstrated

Every arm keeps the body, separators, markers, query key, and candidate
inventory byte-identical, and only **appends** material after the original input,
so no arm can delete or reorder the inventory being selected from. Scored on the
same 432 frozen rows, no training.

Parent U16000, accuracy of candidate argmax (chance ≈ 0.336):

| arm | what it does | accuracy |
|---|---|---:|
| `as_is` | unchanged | 0.394 |
| `append_query` | append the queried key (gap→0) | **0.607** |
| `append_other_key` | append a *different* pair's key; scored as "did she pick *that* pair" | **0.556** |
| `append_unused_key` | append a key-bank token absent from the item | 0.336 |
| `append_filler_key_8` | 8 extra key-bank nuisance tokens | 0.352 |
| `append_filler_span_8` | 8 extra span-bank nuisance tokens | 0.361 |
| `append_filler_key_32` | 32 extra key-bank nuisance tokens | 0.340 |

Dose-response on transport distance, from a known-good starting point
(`append_query_then_filler_n` = append the queried key, then push it away by `n`
nuisance tokens):

| distance from generation position | parent | S1 treat | S2 control | S2 treat |
|---|---:|---:|---:|---:|
| 0 | **0.607** | 0.569 | 0.611 | **0.690** |
| 1 | 0.433 | 0.398 | 0.461 | 0.507 |
| 2 | 0.340 | 0.338 | 0.329 | 0.359 |
| 4 | 0.340 | 0.322 | 0.357 | 0.363 |
| 8 | 0.368 | 0.329 | 0.338 | 0.370 |

The reach is one token wide and the cliff is total. Independently, the as-is
gap=1 stratum scores 0.468 and `append_query_then_filler_1` scores 0.433 — two
unrelated constructions agreeing on the same range limit.

Stratified by the row's *original* gap, parent:

| base gap | n | `as_is` excess | `append_query` excess | `append_other_key` excess | `append_unused_key` excess |
|---|---:|---:|---:|---:|---:|
| 0–1 | 158 | +0.146 | +0.310 | +0.285 | +0.000 |
| 2–3 | 21 | +0.048 | +0.286 | +0.238 | −0.143 |
| 4–12 | 38 | +0.053 | +0.263 | +0.026 | +0.000 |
| 13–30 | 56 | −0.000 | +0.214 | +0.143 | −0.036 |
| **31+** | **159** | **+0.000** | **+0.258** | **+0.233** | +0.038 |

And by pair count, parent:

| arm | K=2 | K=3 | K=4 |
|---|---:|---:|---:|
| `as_is` | 0.552 | 0.354 | 0.344 |
| `append_query` | **0.760** | **0.576** | **0.552** |
| chance | 0.500 | 0.333 | 0.250 |

## What this falsifies

- **"Queried selection / first-token binding has failed."** Falsified. On the
  rows that sit at exact chance, making the query adjacent lifts accuracy by
  +0.26 with no weight change.
- **"Query-swap does not follow; follow is at chance for every K."** Falsified as
  a statement about the capability. The frozen query-swap panel swapped the query
  *in place*, i.e. mostly at a gap the circuit cannot reach. Swapped adjacently
  (`append_other_key`), she follows the new key at 0.556 overall and +0.233
  excess on the 31+ stratum.
- **"Persistent multi-candidate K≥3 failure."** Falsified as intrinsic. K=4 goes
  from 0.344 (chance 0.250) to 0.552 under adjacency.
- **"The query signal merely needs to be stronger."** Falsified twice: by S2's
  own algebra (its functional cannot see the dominant term) and by the probe (the
  limit is a range cliff, not an amplitude shortfall).
- **"S2 was a regression."** Wrong instrument. S2 measurably *strengthened* the
  adjacent circuit — `append_query` 0.607 → 0.690, `append_other_key` 0.556 →
  0.644, K=4 adjacency 0.552 → 0.609, as-is gap≤1 0.468 → 0.620 — and was scored
  by a metric in which ~63% of rows are structurally unreachable. Its two
  retention losses are real and remain real.
- **"Held-out separator OOD and copy weakness are the story."** Unchanged but now
  secondary; they are Gate C problems on top of a missing transport pathway.
- **"Key-bank filler jams the key-match channel."** Falsified:
  `append_filler_span_8` ≈ `append_filler_key_8`. It is distance, not bank.
- **"Appending anything at the end perturbs the choice."** Falsified by
  `append_unused_key`, flat at chance in every stratum.

## What survives

- Copy-a-value-span-from-context is real and strong (inventory copy 93/96).
- Teacher-forced continuation / gold rest-lock is real and strong (~0.97).
- Value-absent original-span copy is 0; she copies from context.
- Held-out exact is killed in part by OOD separators `{82–85}`.
- The S1 and S2 retention losses were genuine.
- Architecture change is still not justified — the circuit exists, it is short.

## Corrected reading of a prior positional claim

`isolation_transforms.parse_records` reads `item["source"]`, which `data_v2`
populates in **pre-shuffle record order**, not render order. So
`candidate_heads` and `query_index` index the original records, not body slots.
Any analysis that treated a candidate index as a body position is measuring
nothing. The gap analysis here uses `query_position` directly and is unaffected.

## Why the salience bias exists, and why no logit loss removes it

Roughly two thirds of keyed training rows place the query outside the circuit's
reach. On those rows no parameter setting can lower the loss by attending to the
query, and the loss-minimising behaviour is exactly what is observed: spread mass
over the inventory value heads and commit to a per-body favourite. The 2.78-nat
salience bias is not a bug or a shortcut artefact — it is the Bayes-rational
response to a training distribution that is, as posed, unsolvable for most of its
keyed mass. Adding first-token CE (S1) pushes that prior *up*, which is why S1
made collapse worse and `query_logit_effect` fall while moving induction. Adding
a bias-invariant contrast (S2) cannot touch it at all.

This is a **curriculum / information-flow** defect expressed as an objective
failure. It is not repairable by re-weighting logits at the answer position.

## Highest-information next action

Extend the transport range by curriculum, not by amplitude. Start keyed selection
where the circuit already earns reward (gap 0–1) and grow the query→generation
distance, with a matched control that is simply more training at the natural gap
distribution — which is what v2R4 already did for 10,000 updates without effect.
Preregistered as `design/V010_SELECTION_REPAIR_T1_GAP.md`.

Falsifier stated in advance: if the curriculum moves gap≤1 but leaves gap≥13 at
chance, then the one-token reach is architectural or representational rather than
curricular, and the next claim becomes an architecture claim.

## Local-only artefacts

`runs/query_locality/ROWS_*.json` (per-row dumps, ~2 MB each) stay local.
Hashes: [`V010_QUERY_TRANSPORT_SHA256SUMS.txt`](V010_QUERY_TRANSPORT_SHA256SUMS.txt).
Protected material opened: false.
