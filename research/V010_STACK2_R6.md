# Stack2 R6: PRE-v1.0 FINAL-EXAM CANDIDATE

Status: **PRE-v1.0 FINAL-EXAM CANDIDATE. NOT GRADUATED. NOT v1.0.**
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

Tokenizer: `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json`
SHA `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.

Lineage: s5b3 (frozen) → r3b PropMatchHead (frozen) → R4 finish → R5
BESIDE/WHO bind → R6 cross-attribute + dialogue bind. Default chat is
`--runtime r3` (PropMatch only).

## A / B / C

### A. Frozen s5b3 alone

Official who_2e **~0.531** (R5). Recency readout. Batched eval_panels
color 0.969, size_stop 1.000, mixed_2e 0.969, fact_combine **0.656**,
story_combine **0.438**. D3 **181/215**. Allowed to remain weaker than B.

### B. Baby + PropMatchHead + native finish

No RelAssist. No WhoProp choosing answers. No gold query positions.

| panel | R5 B | R6 B |
|---|---:|---:|
| fact_combine decode (e13 + 2 seeds + 3e) | hijacked ~0.06 | **1.000** |
| fact_combine eval free_exact | — | **1.000** |
| fact_combine eval first_top1 (batched) | 0.656 A | 0.688 (noisy; see below) |
| story_combine decode (e13 + seed) | ~0.08 | **1.000** |
| story_combine eval free_exact | — | **1.000** |
| story_combine eval first_top1 (batched) | 0.438 A | 0.438 (noisy; see below) |
| about_report (both attrs) | one-fact | **1.000** |
| usable4 / stop / reuse | 0.844 / 1.000 / 0.857 | **1.000 / 1.000 / 1.000** |
| reuse pack come-back | — | **1.000** |
| integration (fresh, 9 families) | — | **1.000** |
| who_2e / who_sent / HAS / BESIDE | 1.000 | **1.000** |
| paraphrases / mixed / multiturn | 1.000 | **1.000** |
| color / size_stop / mixed_2e | held | **0.969 / 1.000 / 0.969** |

### C. What still requires external help

**WhoPropRouter** remains an unused who_2e ceiling (`--runtime r2`).
**RelAssist** is not installed on default `--runtime r3` chat.
Do not call RelAssist or WhoProp native.

## Old vs behavioral combine metrics

Official `eval_panels` `first_top1` is batched teacher-forced first-token.
It does not reliably apply last-token PropMatch routing. R6 **does not
replace** those numbers.

Behavioral metrics (ordinary decode, same items):

- `fact_combine_decode` / eval `free_exact` = **1.000**
- `story_combine_decode` / eval `free_exact` = **1.000**
- `about_report` = both requested attributes, not last sentence copy

## Lesson 1 — fact combination

R5 WHO finish answered combine queries with the cue (`bear is blue` for
`Which size is the blue one?`). Survivor: classify color-ask+size-cue /
size-ask+color-cue, bind the named attribute of that entity, emit the
**other** attribute with full spelling. About-report uses `X is C and S.`
Held-out seeds 611001/611777/611888/612001/e13/3e all **1.000**.

## Lesson 2 — story combination

Same cross-attribute job inside story frames (`a tiny hen sat. That hen
was yellow.`). Diagnosed independently: size is often adjective-before-entity.
Subject lookup uses entity after the value when none precedes it. Official
story_combine decode **1.000**; progression canaries (2 facts → distractor →
separated → multi-entity) **1.000**.

## Lesson 3 — conversational reuse

Failures were dialogue mechanics, not missing facts: `Baby:` after `?`
looked like an answer; last `?` from an earlier turn stole the bound from
a later period-ended question; `pig`/`pink` shared a first piece so finish
had to complete the full spelling. Survivors: treat role tokens as empty
answer; take the later of last `?` and last period-ended query; direct
color/size of a named bridge entity (including bird/fox/pig). usable
reuse **0.857→1.000**, usable4 **0.844→1.000**, period-stop **1.000**.

## Lesson 4 — integration

Fresh pack, not trained. Color, size, WHO, WHO-sent, HAS, BESIDE,
paraphrase, combine, story-combine, about, mixed switch: **1.000**
overall and per family.

## Killed

| attempt | lesson |
|---|---|
| WHO finish on combine queries | Cue property / entity sentence, not the other attr. |
| One-token value plan | `short`→`sh.` Multi-piece values need full spelling. |
| Two-sentence about plan | `greedy_decode` stops at first period. Use `and`. |
| Ignore `Baby:` role prefix | Finishes never matched; chat stayed backbone-only. |
| Last `?` only as bound | Earlier question stole later period-ended turns. |

## PRE-v1.0 freeze

Reproduce with:

- backbone `checkpoint_00025.pt` SHA above
- r3b head SHA above
- tokenizer SHA above
- `--runtime r3` (`PropMatchRuntime` in `selection_stack2_r3.py`)
- RelAssist / WhoProp off

NOT v1.0. NOT graduated. Owner constructs a closed final exam separately.
Do not train against that exam.

## Lineage / SHA

See `research/V010_STACK2_R6_SHA256SUMS.txt`.
