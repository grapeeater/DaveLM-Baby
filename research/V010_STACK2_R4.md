# Stack2 R4: native WHO sentences, native HAS, held-out paraphrases

Status: **EXPERIMENTAL HYBRID, NOT PROMOTED**.
U16000 remains AUTHORITATIVE.
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. R3 is the parent survivor, not replaced as
the who_2e claim. Nothing is promoted over U16000.

Weights parent (verified loaded SHA):
`runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt`
SHA `0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd`.

Learned module (unchanged r3b weights):
`runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt`
SHA `8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9`.

Lineage:
s5b3 (frozen) → R3b PropMatchHead (frozen) → R4 native WHO/HAS finish
and paraphrase aliases. No new trained head. No backbone CE.

Receipts: `runs/actual_baby/stack2/{CAMPAIGN.json,LEDGER.md,r4/}`
and `research/V010_STACK2_R4_SHA256SUMS.txt`.

## A / B / C

### A. Frozen s5b3 backbone alone

No runtime module. Ordinary last-token decode.

who_2e official **0.531** (R3 measurement, unchanged weights).
The bind is still in the hidden states. Readout still defaults to recency.

Batched `eval_panels` (teacher-forced; may bypass last-token routing):
color 0.969, size_stop 1.000, mixed_2e first_top1 0.969, fact_combine 0.656.

D3 weights-only **181/215**, first_correct 215/215, induction 0.500.

### B. Baby + learned PropMatchHead + native finish

Same r3b head. Ordinary `model.forward` last-token path.

R3 already internalized inverse who_2e retrieval (no WhoProp cue scanner).
R4 adds Baby-owned **sentence finish** after that retrieved first piece:

1. WHO: completed entity → ` is` → subject-bound color/size → `.`
2. HAS: completed entity → ` has` (full ` h`+`as`, not first-piece ` h`) →
   ` the` → subject-bound color → ` object` → `.`
3. Aliases ` have` / ` belong` route to HAS finish; ` next` / ` beside`
   do not steal the WHO template.

No gold answers or gold query positions at inference.

| panel | R3 B | R4 B |
|---|---:|---:|
| who_2e official / 324777 | 0.969 / 0.906 | **0.969 / 0.906** |
| who_2e last≠gold official | 0.941 | **0.941** |
| who_sent official bare | 0.562 | **0.812** |
| who_sent seeds 324777 / 324888 | (R3 0.56 pack) | **0.875 / 0.812** |
| has native who / what | 0.833 / 0.000 | **1.000 / 1.000** |
| WHO paraphrases held-out | — | **1.000** |
| HAS paraphrases held-out | — | **1.000** |
| BESIDE paraphrases native | — | 0.833 |
| paraphrase overall native | — | **0.950** |
| usable4 / stop / reuse | 0.844 / 1.000 / 0.857 | **0.844 / 1.000 / 0.857** |

Official who_sent color **1.000**, size **0.625**. Remaining misses are
wrong-pair hops (`who_hen_tiny` → `dog is huge`), not one-word stops.
Prefer 0.90 was not reached without the killed window-argmax hop.

### C. What still requires external runtime assistance

**RelAssist** is still required for reliable BESIDE (canonical hybrid 1.000;
native 0.833). One held-out paraphrase still copies the query entity:
`Which animal is next to the dog?` → `The dog is beside the dog.`

**WhoPropRouter** remains the who_2e ceiling (official 1.000 / seed324777
0.969). R4 does not delete it. Default chat is PropMatchHead + RelAssist
(`--runtime r2` restores WhoProp). RelAssist is no longer required to
*choose* HAS answers.

Do not call RelAssist or WhoProp "native."

## Lesson 1 — full WHO sentences

Diagnosis: when the hop lands on the correct first-piece, frozen LM often
continues. Misses were (1) hop-None on sentence-initial bare names and
(2) retrieved-then-period-stop (`bear.`).

Survivor: entityness last-≥0.5, else argmax in the hop window only;
canonicalize any piece of a known entity to the spaced first-token;
WHO finish after the copied entity.

Window-argmax as the *primary* hop was killed (who_2e seed324777 0.844).

## Lesson 2 — clean native HAS

Failures were not one cause:

| symptom | cause |
|---|---|
| `hen has the redog.` | trailing first-piece `d` finished `dog` |
| has_what 0.000 | no object-side finish; RelAssist scanned query text |
| size WHO collapse | treating ` h` (356) as HAS; it is also in ` huge` |
| empty answer span | `query_boundary` treated ` wh` of `white` as `who` |
| has_who stole last fact | query-entity over the full prefix, not the question span |

Survivor: whole-suffix-only entity finish; `has_seq` `[356, 309]`; last `?`
as the stable query bound; question-span query entity for has_what;
native HAS finish without RelAssist.

## Lesson 3 — natural paraphrases

Train stems in `selection_stack2_r4.TRAIN_PARAPHRASE_STEMS` are **not
scored**. Held-out packs use disjoint wrappers (`Which animal`, `Which one`,
`What object does … have`, `belongs to`, `next to`).

Zeroshot before WHO finish: WHO 0.625 (first_entity 1.0, then period-stop),
HAS 1.000, BESIDE native 0.833, overall 0.80.

After WHO finish: WHO **1.000**, HAS **1.000**, BESIDE native 0.833,
overall native **0.950**, hybrid **1.000**. No family is dead.

## Killed arms

| attempt | lesson |
|---|---|
| Window-argmax hop as primary | WHO official 0.875 but who_2e 324777 0.844. Reverted. |
| `has_id` = first piece of ` has` | Collides with ` huge`. Use full `has_seq`. |
| `query_boundary` on question IDs | `white` first piece ` wh` == `who`. Use last `?`. |
| Trailing first-piece finish | `d` in `red` completed `dog` → `redog`. |
| Query-entity over full prefix | has_who copied the last fact entity. |
| Generic sentence CE | Not justified. Retrieval already existed. |

## Retention

| metric | R4 | note |
|---|---:|---|
| who_2e native official / 324777 | 0.969 / 0.906 | held |
| who_sent official / seeds | 0.812 / 0.875 / 0.812 | Lesson 1 |
| has native | 1.000 | Lesson 2; RelAssist off |
| beside native / hybrid | 0.833 / 1.000 | RelAssist still on |
| color / size_stop | 0.969 / 1.000 | hold |
| mixed_2e / fact_combine | 0.969 / 0.656 | hold |
| usable4 / stop / reuse | 0.844 / 1.000 / 0.857 | hold |
| D3 long-gap | 181/215 | first_correct 215; logged |
| induction | 0.500 | logged |

## Next developmental lesson

Internalize BESIDE the same way HAS was internalized (partner retrieval +
native finish), without RelAssist choosing the neighbor. Do not open
world-knowledge / coding / tools. Do not promote.

## Lineage / SHA

| id | path | SHA256 | role |
|---|---|---|---|
| U16000 | structured_v2r4 …/checkpoint_16000.pt | 94b3a9daf0…c917827 | AUTHORITATIVE |
| s5b3 | …/s5b3_compose_lock_326191/checkpoint_00025.pt | 0d3e694750…cf2747cd | frozen backbone |
| r3b head | …/r3/r3b_329011/prop_match_head_00050.pt | 8ea3dbfd48…7d62d9 | frozen learned module |
| R4 runtime | src/baby_v010/selection_stack2_r{2,3,4}.py | see SHA256SUMS | experimental finish |

Loaded parent on every canary/retain matched `0d3e6947…cf2747cd`.
