# Stack2 R3: internalize the traffic light

Status: **EXPERIMENTAL HYBRID, NOT PROMOTED**.
U16000 remains AUTHORITATIVE.
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. R2 is not replaced. Nothing is promoted.

Weights parent (verified loaded SHA, not merely CLI):
`runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt`
SHA `0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd`.

Learned module (frozen backbone; head only):
`runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt`
SHA `8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9`.

Lineage:
s4m → s5m → s5b3 → R2 (WhoProp + RelAssist, no weight change)
→ R3b PropMatchHead on frozen s5b3 (no backbone change)

R3 is not a claim that the Python router vanished from the whole stack.
It is a claim that **inverse who_2e retrieval** no longer needs the Python
property-token cue scanner.

Receipts: `runs/actual_baby/stack2/{CAMPAIGN.json,LEDGER.md,r3/}`
and `research/V010_STACK2_R3_SHA256SUMS.txt`.

## What the three layers actually do

Label every number. R2 showed batched `eval_panels` can bypass last-token
routing. R3 keeps that warning.

### A. What the s5b3 weights can do alone

No runtime module. Ordinary last-token decode.

| panel | who_2e | last≠gold |
|---|---:|---:|
| official | 0.531 | 0.471 |
| seed 324777 | 0.562 | 0.500 |
| seed 324888 | 0.469 | 0.667 |

The bind is still in the hidden states (property cosine 0.9375). Readout
still defaults to recency. D3 weights-only **181/215**, first_correct 215/215.

Batched `eval_panels` (teacher-forced; may bypass routing): color 0.969,
size_stop 1.000, mixed_2e first_top1 0.969, fact_combine 0.656.

### B. What the learned PropMatchHead does

Tiny head on frozen s5b3. Ordinary `model.forward` last-token path.

Mechanism (no gold query position, no gold entity, no Python cue scanner):

1. Learned cue locator over the last K=6 suffix hidden states.
2. Frozen cosine of that cue hidden against earlier hiddens (the R2 0.9375 geometry).
3. Learned entityness hop onto the nearest prior entity hidden.
4. Boost that token's spaced first-piece logit, only while the sequence still ends on the question.
5. Cue-confidence gate 0.7 so chat is not always-on copied.

Unbatched native-head `who_2e` (the evaluator that actually uses the head):

| panel | native | last≠gold | pred_is_last |
|---|---:|---:|---:|
| official | **0.969** | **0.941** | 0.500 |
| seed 324777 | **0.906** | **0.917** | 0.594 |
| seed 324888 | 0.750 | 0.667 | 0.625 |

Gate `who_2e >= 0.85` on at least two held-out/order-balanced panels: **PASS**.
Prefer >= 0.90: **PASS** on official and seed 324777.

Diagnostics on official: cue_locate 0.969, property_match 0.9375,
entity_hop_gold 0.9375, entity_hop_last 0.4375 (not a recency pointer).

WHO sentences, native head, official 16-item bare pack: **0.562**
(color 0.625 / size 0.500). Below the 0.80 sentence target. Predicate 1.000.
The one-word retrieval is internalized; the sentence readout is not.

Usable-chat with the head installed: usable4 **0.844**, stop **1.000**,
reuse **0.857**. Matches the R2 / s5b3 chat hold.

### C. What still requires external runtime assistance

**RelAssist** is still required for has/beside expression.

| family | native head only | native + RelAssist | R2 ceiling (WhoProp+RelAssist) |
|---|---:|---:|---:|
| has bare | 0.417 | **0.833** | 0.917 |
| has who / what | 0.833 / 0.000 | **0.833 / 0.833** | 1.000 / 0.833 |
| beside bare | 0.833 | **1.000** | 0.917 |
| beside where / who | 0.833 / 0.833 | **1.000 / 1.000** | 0.833 / 1.000 |
| who_sent bare | 0.562 | 0.500 | 1.000 |

WhoPropRouter remains the who_2e ceiling (official 1.000 / seed324777 0.969).
R3 does not delete it. Default stack2 chat now uses PropMatchHead + RelAssist
(`--runtime r2` restores WhoProp).

## Lesson 1 — internalize the router

R2 recommended: a tiny trained query→fact head that reproduces WhoProp scores
without a Python cue scanner. That is what r3b is.

### Killed arms

| attempt | lesson |
|---|---|
| Last-token cosine of h[-1] | Argmax is always "other". The cue is the last **property** hidden, not the last sequence token. |
| r3a suffix-attn entity pointer | Copied recency (pointer_gold 0.50 / pointer_last 0.53). Same failure as r2a. Killed. |
| r3c local hop window | WHO sentences rose (0.56→0.75 with piece map) but who_2e seed2 0.844 missed 0.85, usable 0.59. Extra lock: who_2e 0.906/0.875 with WHO 0.625 and usable 0.56. Not the safest architecture. |
| Always-on hop / ungated copy | usable4 0.53. Chat is not a retrieval exam. Gate or skip. |

Distillation teacher was WhoProp `route_entity_pos` on **train-generated**
who_bind items only. Held-out panels stayed held out. Property IDs are used
in the teacher and in diagnostics, not in PropMatch inference.

### Survivor: r3b entityness

50 steps, seed 329011, backbone frozen. Parent SHA verified on load.

This is a small reduction vs R2's external 1.000/0.969, not a collapse.
Baby owns the who_2e traffic light.

Seed 324888 at 0.750 is the weak third panel. Do not average it away.
The two-panel gate is official + 324777.

## Lesson 2 — clean relational expression

Tokenizer evidence, not a tokenizer redesign:

| word | spaced pieces |
|---|---|
| dog | ` dog` |
| hen | ` he` + `n` |
| duck | ` d` + `u` + `ck` |
| bear | ` be` + `ar` |
| cat | ` c` + `at` |
| frog | ` f` + `ro` + `g` |

`hen` first-piece is the same token as English ` he`. Boosting it without
finishing `n` yields `he is beside the duck.`

Fix: `entity_piece_seqs` / `next_entity_finish_id` complete a known entity
spelling after the first piece, only on the **answer suffix**. First-token
copy is restricted to the question turn (R2's always-on-copy lesson).
`bind_piece_map` maps bare first-pieces onto the spaced first-token used in
answers.

This is readout canonicalization, not a Python property scanner.

Documented malformations after the fix (hybrid PropMatch+RelAssist):

| was (R2) | now |
|---|---|
| `he is beside the duck.` | official beside pack **1.000**; hen completes as ` hen` |
| `hen has the dog has the white object.` | still malformed; one probe decoded ` hen has the redog.` |

Subject first-piece vs whole-word is materially reduced. has_what object
finish is not solved. Do not treat the literal strings as a unit-test suite.

WHO sentences 0.50 → 0.562. Still below 0.80. Remaining misses are hop-None
(recency) and wrong-entity hops (`who_hen_tiny` → dog), not `he` truncation
on the hits (`who_hen_red` → ` hen looks red.`).

## Lesson 3 — paraphrase expansion

**Deferred.** Lessons 1–2 consumed the work-shift budget. HAS/BESIDE
paraphrases (`Which animal has…`, `next to`, …) are tonight's work, not a
fake new relation.

## Retention

| metric | R3 | note |
|---|---:|---|
| who_2e native official / 324777 | 0.969 / 0.906 | Lesson 1 survivor |
| who_sent native bare | 0.562 | below 0.80; labeled |
| has / beside hybrid | 0.833 / 1.000 | RelAssist still on |
| color (batched eval_panels) | 0.969 | hold |
| size_stop | 1.000 | hold |
| mixed_2e first_top1 | 0.969 | hold |
| fact_combine | 0.656 | hold |
| usable4 / stop / reuse | 0.844 / 1.000 / 0.857 | hold vs R2 |
| D3 long-gap | 181/215 | first_correct 215; logged |
| induction | 0.500 | logged |

No catastrophic forgetting. Modest WHO-sentence gap vs R2's external 1.000
is the cost of not wrapping WhoProp and calling it internalized.

## Blockers left for tonight

1. WHO sentences still need a native hop that hits sentence-initial names
   without wrecking chat. r3c local hop is the near-miss, not the ship.
2. has_what object/color finish (`redog`, duplicated has).
3. Seed 324888 who_2e 0.750 under the two-panel gate.
4. Lesson 3 paraphrases untouched.

Do not brute-force another recency pointer. Do not throw generic sentence CE.
Do not open TEST / FINAL / SACRED.

## Lineage / SHA

| id | path | SHA256 | role |
|---|---|---|---|
| U16000 | structured_v2r4 …/checkpoint_16000.pt | 94b3a9daf0…c917827 | AUTHORITATIVE |
| s5b3 | …/s5b3_compose_lock_326191/checkpoint_00025.pt | 0d3e694750…cf2747cd | frozen R3 parent |
| r3b head | …/r3/r3b_329011/prop_match_head_00050.pt | 8ea3dbfd48…7d62d9 | Lesson 1 survivor |
| r3a | r3a_329001 who_fact_head | killed | recency pointer |
| r3c | r3c_329021 / lock | killed | hop wrecks chat |

Loaded parent on every canary/verify/recheck matched `0d3e6947…cf2747cd`.
