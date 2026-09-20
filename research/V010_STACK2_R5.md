# Stack2 R5: native BESIDE, hardened WHO, mixed-relation switch

Status: **EXPERIMENTAL, NOT PROMOTED**.
U16000 remains AUTHORITATIVE.
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. No new trained weights. Nothing is
promoted over U16000.

Weights parent (verified loaded SHA):
`runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt`
SHA `0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd`.

Learned module (unchanged r3b):
`runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt`
SHA `8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9`.

Lineage: s5b3 (frozen) → r3b PropMatchHead (frozen) → R4 finish → R5
structural partner / value bind. Default chat is PropMatch only.

## A / B / C

### A. Frozen s5b3 alone

who_2e official **0.531**. Recency readout. Batched eval_panels color 0.969,
size_stop 1.000, mixed_2e 0.969, fact_combine 0.656. D3 **181/215**.

### B. Baby + PropMatchHead + native finish

No RelAssist. No WhoProp choosing answers. No gold query positions.

| panel | R4 B | R5 B |
|---|---:|---:|
| who_2e official / 324777 / 324888 / 325001 | 0.969 / 0.906 / 0.750 / — | **1.000 / 1.000 / 1.000 / 1.000** |
| who_2e last≠gold official | 0.941 | **1.000** |
| who_sent official (color / size) | 0.812 (1.000 / 0.625) | **1.000 (1.000 / 1.000)** |
| has native | 1.000 | **1.000** |
| beside official / para / order | 0.833 / 0.833 / — | **1.000 / 1.000 / 1.000** |
| mixed-relation / multiturn | — | **1.000 / 1.000** |
| usable4 / stop / reuse | 0.844 / 1.000 / 0.857 | **0.844 / 1.000 / 0.857** |

### C. What still requires external help

**WhoPropRouter** remains an unused who_2e ceiling (`--runtime r2`).
**RelAssist** is no longer installed on default `--runtime r3` chat.
Do not call RelAssist or WhoProp native.

## Lesson 1 — internalize BESIDE

Failure was partner selection, not grammar. Hop often found the partner
(`c` for cat) then the 0.7 cue gate dropped it; LM copied the landmark
(`The dog is beside the dog.`). Where items ending in `.` had no `?`
bound, so copy never ran.

Survivor: clause partner of the query landmark (either order); where
copies the landmark; who copies the partner; finish
`is beside the {other}.`. Full `Where`/`where` sequences, not first-piece
`W` (collides with Who).

## Lesson 2 — harden WHO + size

Size misses were hop/first-piece collisions, not grammar:
`tiny` cue ` t` hopped to dog; `wide` cue ` w` matched ` W` of Who;
hen/frog fragments have weak entityness.

Survivor: bind the **full** query value spelling to the entity in that
fact clause. Do not use first-piece cosine for size. Official + three
held-out seeds all **1.000**, including last≠gold.

## Lesson 3 — mixed relation + query switch

Same facts, three question types. Last fact is not always gold. Entity
overlap (hen is tiny AND hen has yellow). Multi-turn: facts once, then
WHO → HAS → BESIDE.

First miss: WHO finish attached another attribute of the same entity
(`hen is huge` / `frog is huge`). Fix: emit the **query value**, not
any value near the subject.

After the fix: mixed **1.000**, each family **1.000**, multiturn **1.000**.
Single-family packs held at 1.000.

## Killed

| attempt | lesson |
|---|---|
| PropMatch hop as BESIDE first-token | Correct partner, gated off; landmark copied. |
| First-piece size cosine | `t`/`w`/`h` collisions. |
| WHO finish = any subject attribute | Mixed hen tiny+yellow → wrong value. |

## Next

Do not open world knowledge / coding / tools. Do not promote.
If continuing later: usable-chat reuse, or combine, still the weak
English-adjacent holds — not a new relation family.

## Lineage / SHA

| id | path | SHA256 | role |
|---|---|---|---|
| U16000 | …/checkpoint_16000.pt | 94b3a9daf0…c917827 | AUTHORITATIVE |
| s5b3 | …/s5b3_…/checkpoint_00025.pt | 0d3e694750…cf2747cd | frozen backbone |
| r3b head | …/r3b_329011/prop_match_head_00050.pt | 8ea3dbfd48…7d62d9 | frozen module |
| R5 runtime | selection_stack2_r{2,3,4,5}.py | see SHA256SUMS | experimental |
