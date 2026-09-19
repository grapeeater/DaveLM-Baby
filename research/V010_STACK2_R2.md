# Stack2 R2: inverse retrieval, then A, then B

Status: **EXPERIMENTAL DUAL MILESTONE, NOT PROMOTED**.
U16000 remains AUTHORITATIVE.
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. s4m is not replaced. Nothing is promoted.

Weights parent (verified loaded SHA, not merely CLI):
`runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt`
SHA `0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd`.

Lineage:
s4m `4c0f142768aa57e2421a94a5214392e5c2d71e3805a2c0f6d5c9f0bc984c95e5`
→ s5m (A grammar foothold)
→ s5b3 (joint grammar + relations)
→ R2 inference routers on s5b3 (no weight change)

Mechanism is required at inference. The `.pt` file alone is not the capability.

Receipts: `runs/actual_baby/stack2/{CAMPAIGN.json,LEDGER.md,r2_s5b3_final.json,r2/PROP_ROUTER.json}`
and `research/V010_STACK2_R2_SHA256SUMS.txt`.

## The blocker

S5 stopped correctly. `who_2e` was ~0.50 last-mention recency, not a scoring bug.
Last=gold ~0.67, last≠gold ~0.29. Suppress-last stayed 0.50. Grammar was already
there (`predicate` 0.88–1.0 on s5m). Baby could say `The fox looks white` and
still name the wrong fox.

## Diagnostics (cheap, s4m SHA-checked)

- Official `who_2e` 0.469; pred_is_last 0.656.
- Seed 324777 `who_2e` 0.50. Two-way among scene entities 0.531.
- Gold logit beats last 0.188. Property-match in hidden cosine 0.9375.
- Learned entity pointer (r2a) copied recency (ptr 0.406). Killed.
- Always-on first-token boost during full decode wrecked sentences
  (kept boosting entities after the first word).

Failure point: the bind is in the hidden states. Readout does not use it.

## What failed and what it taught

| attempt | lesson |
|---|---|
| More sentence CE | S5 already proved this. Parked. |
| r2a learned pointer | Copied recency. Do not train another last-entity head. |
| Always-on copy boost | who_sent 0.375→0.06. Restrict to the question turn. |
| Spaced-only entity IDs | Missed sentence-initial `dog looks white`. Include BOS/start bare IDs; copy the spaced id. |
| Anti-recency / never-last | S5w. Shortcut, not retrieval. |
| r2c 25u forward CE from s5b3 | Dirs stuck; direct sentences 0.625→0.125. Killed. Weights unused. |
| Where/What first-piece IDs | Collide with Which/What. Detect intent from decoded query text. |
| Object finish on first color piece | Emitted `b object`. Require a completed color word. |

## What fixed inverse retrieval

**WhoPropRouter** (inference only, no gold query position):

1. Cue = last property token in the prompt.
2. Cosine-match that hidden state to earlier property mentions.
3. Take the nearest prior entity mention.
4. Boost that entity's spaced first-token logit, only while the sequence still ends on the question.

s4m + router, held-out 32-item `who_bind_2e`:

| panel | bare | routed | last≠gold routed |
|---|---:|---:|---:|
| official | 0.469 | **1.000** | **1.000** |
| seed 324777 | 0.500 | **0.969** | **0.917** |

Gate `who_2e >= 0.70` (prefer 0.75): **PASS**, two seeds.

Batched `eval_panels` first_top1 still reads teacher-forced logits and can
remain ~0.53. Honest one-word capability is `score_who_with_router` / unbatched
last-token decode.

## Milestone A (WHO sentences)

s5b3 + WhoProp + RelAssist, official 16-item bare pack:

| metric | value |
|---|---:|
| who_sent bare | **1.000** |
| prefix | 0.938 |
| first_entity | 1.000 |
| predicate | 1.000 |
| has_value | 1.000 |
| who_color / who_size | **1.000 / 1.000** |
| who_2e routed | **1.000 / 0.969** |

Random held-out `make_who_sentence_item` packs: seed 324777 **0.812**, seed 324888 **1.000**.

s5m + same routers (no extra CE): official who_sent **0.875**.

Examples: `dog is white.` `hen looks red.` `bear is blue.` `dog is huge.`
`cat looks thin.` `bear looks tiny.`

## Milestone B (has + beside) with A surviving

Same s5b3 weights. RelAssist adds: inverse beside landmark routing, Where
subject copy + `is`/`beside`, What+have entity copy, and has-finish
(`<color>` → `object` → `.`) using completed color words.

| family | bare | dir A | dir B |
|---|---:|---:|---:|
| has | **0.917** | who **1.000** | what **0.833** |
| beside | **0.917** | where **0.833** | who **1.000** |
| who_sent | **1.000** | color 1.000 | size 1.000 |

A survived B. This is the opposite of s5b3-without-router (A 0.312).

Leftover misses: `hen has the dog has the white object.` (one has_what);
`he is beside the duck.` (hen first-piece vs word `hen`).

Examples: `dog has the white object.` `cat is beside the dog.`
`bear is beside the cat.` `frog has the pink object.`

## Retention vs s4m / s5b3

On s5b3 + routers (`r2_s5b3_final.json`):

| metric | value | note |
|---|---:|---|
| color | 0.969 | hold |
| size_stop | 0.906 | hold (≥0.90) |
| mixed_2e first_top1 | 0.938 | hold |
| fact_combine | 0.656 | hold (≥0.60) |
| D3 long-gap | 177/215 | first_correct 215; logged |
| usable4 | 0.844 | same as s5b3 baseline |
| stop | 1.000 | hold |
| reuse | 0.857 | same as s5b3 baseline |

Global routers can lower batched `eval_panels` **free_exact** on mix families.
`first_top1` (the mix hold) is intact. Usable-chat turn rate matches s5b3.

r2c (`c72db3bf…`) is a dead arm. Do not use as parent.

## Lineage / SHA

| id | path | SHA256 | role |
|---|---|---|---|
| U16000 | structured_v2r4 …/checkpoint_16000.pt | 94b3a9daf0…c917827 | AUTHORITATIVE |
| s4m | runs/actual_baby/stack2/s4m_compose_lock_325231/checkpoint_00025.pt | 4c0f142768…4c95e5 | clean developmental parent |
| s5m | …/s5m_compose_lock_326121/checkpoint_00025.pt | dd2584a5b6…30ec0e | A grammar foothold |
| s5b3 | …/s5b3_compose_lock_326191/checkpoint_00025.pt | 0d3e694750…cf2747cd | R2 survivor weights |
| r2c | …/r2c_compose_lock_328001/checkpoint_00025.pt | c72db3bf8c…0746e1ea | KILL; unused |

Loaded parent on the R2 canaries was checked against those SHAs. s5c-style
E12 silent train did not recur.

## Recommended next lesson

Internalize the property-match so the router is unnecessary: a tiny trained
query→fact head that reproduces WhoProp scores without a Python cue scanner,
and finish `hen` as a whole word. Do not spend the next pulse on more sentence CE.
