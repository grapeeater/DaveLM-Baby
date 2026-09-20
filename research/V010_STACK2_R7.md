# Stack2 R7: generalized inverse entity binding

Status: **PRE-v1.0 CANDIDATE. NOT GRADUATED. NOT v1.0.**
U16000 remains AUTHORITATIVE.
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. No new trained weights. Nothing is
promoted over U16000. Form A was not used for treatment.

Weights parent (verified loaded SHA):
`runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt`
SHA `0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd`.

Learned module (unchanged r3b; not retrained):
`runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt`
SHA `8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9`.

Tokenizer: `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json`
SHA `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.

Git HEAD (no commit created this lesson):
`8430e293abf0dd203448eac0463113eec5016bfb`.

Default chat is `--runtime r3`. RelAssist OFF. WhoProp OFF.
PropMatchHead is **not** the inverse-entity decoder.

## Job

One bind: given a queried property, value, or stored relation, identify the
bound **entity**. Finish wording is a separate canned plan. Inverse WHO,
inverse HAS, and BESIDE partner selection share that bind.

## What changed

Runtime (`src/baby_v010/selection_stack2_r3.py`):

- Inverse bind uses the combination/story subject lookup: full bridge
  `ENTITIES`, full spellings, bidirectional inside the clause
  (entity-before-value and value-before-entity).
- `_subject_of_value` now wraps `_subject_word_of_value` (R6 Lesson 2).
- HAS-value still names the query entity and emits its object.
- Inverse HAS (`Which one has the {color} object?`) binds value→entity
  instead of copying a query entity or hopping.
- BESIDE selection uses relation pairs already in memory. A WHO-ask that
  names a landmark, even without `beside`/`next`/`where` in the query,
  selects the partner. Finish follows the relation speech-act. `near` was
  **not** added to the stem list.
- First-token PropMatch hop is no longer the inverse-entity fallback.
- Answers for WHO/HAS/BESIDE follow word-level plans so pig/fox/bird/cow
  are not decoded from colliding first-pieces.

Scoring (`selection_stack2_r6.py`): inverse WHO must name the **entity**.
Right property + wrong entity fails.

New pack/scorer: `src/baby_v010/selection_stack2_r7.py`.

## What did not change

- Backbone s5b3 (not retrained).
- PropMatchHead weights (not retrained).
- Combination / story / about / direct / dialogue-bound plans (reused).
- RelAssist / WhoProp remain off on default `--runtime r3`.
- No Form A items, no Form B, TEST/FINAL/SACRED not opened.

## Canaries (independent of Form A and of the frozen pack)

| probe | n | result |
|---|---:|---|
| L1 mechanism (WHO left/adj/mid, HAS both ways, BESIDE stem/no-stem/where, direct, combine, about) | 12 | **1.000** |
| L2 all 10 bridge entities × left / adjective-before / which + distractor | 30 | **1.000** |
| L2 extra HAS/BESIDE/story/who_sent/mixed | 8 | **1.000** |
| L2 reuse (12 turns, come-back 1.000) | 12 | **1.000** |
| Official who_2e heldout (last≠gold **1.000**) | 32 | **1.000** |
| WHO-sentence / HAS / BESIDE cheap decode | — | **1.000** each; HAS who+what; BESIDE who+where |

Unit tests: `tests/test_selection_stack2_r3.py` **15 passed**, including
pig/fox/cow bind, value-before-entity, HAS-value vs HAS-entity, and
`Who is near the duck?` classified as WHO then finished as BESIDE.

## Frozen integration pack

Generated and hashed **before** eval. Not edited after seeing results.

- `runs/actual_baby/stack2/r7/INTEGRATION_PACK.json` n=**40**
  SHA `b1fda072e1d6c0a7ab31fda46eddce921c4344e9cd758a2c012f5529fbf2dd93`
- `runs/actual_baby/stack2/r7/REUSE_PACK.json` 4 scripts / 12 turns
  SHA `696a2563b28bdf1ef65f380ae9c72dd75462f1ceda746e6b24fa9cae5c2446a4`

Families: who, who_sent, has_value, has_entity, beside (who/where/next/near),
color, size, direct, combine, story_combine, about, mixed scenes.
Scoring is entity-first: first bridge entity in the answer must be gold
for inverse WHO/HAS/BESIDE (landmark-copy fails). Combine/direct grade the
attribute. HAS requires a `has … object` sentence.

Result: integration **40/40 = 1.000** per family. Reuse **1.000**, come-back
**1.000**, period-stop **1.000**.

One pack at 1.000 is not graduation.

## Retention (must not break)

| panel | R6 B | R7 B |
|---|---:|---:|
| fact_combine decode | 1.000 | **1.000** |
| story_combine decode | 1.000 | **1.000** |
| about_report | 1.000 | **1.000** |
| usable4 / stop / reuse | 1.000 | **1.000** |
| R6 integration (who now entity-graded) | 1.000 | **1.000** |
| official who_2e / who_sent / HAS / BESIDE | 1.000 | **1.000** |
| color / size_stop / mixed_2e first_top1 | 0.969 / 1.000 / 0.969 | **0.969 / 1.000 / 0.969** |
| D3 long-gap | 181/215 | **181/215** |

Batched `eval_panels` first_top1 for fact_combine **0.688** and
story_combine **0.438** is the same noisy teacher-forced metric R6 refused
to replace. Behavioral decode / free_exact stay **1.000**.

## Killed / not used

| attempt | lesson |
|---|---|
| PropMatch hop as inverse decoder | OOD off 2e WHO-bind; first-piece collisions (pig/pink, tiny→dog). |
| WHO_ENTITIES-only + leftward `_subject_of_value` | Drops pig/fox/bird/cow; fails value-before-entity. |
| HAS copies query entity else hop | Inverse HAS never binds the possessor. |
| Growing query stems (`near`, `owner`, …) | Selection must use stored relation/value structure. |
| Retrain PropMatch / s5b3 | Not needed once the native bind is generalized. |

## PRE-v1.0 freeze

Reproduce with:

- backbone `checkpoint_00025.pt` SHA above
- r3b head SHA above (weights unchanged)
- tokenizer SHA above
- `--runtime r3` (`PropMatchRuntime` in `selection_stack2_r3.py`, lexicon via
  `attach_prop_match_lexicon`)
- RelAssist / WhoProp off

Source SHA:

- `selection_stack2_r3.py` `e7bd0b3006c22289182a4f5b9093b3c84606040f09ebfd595b747023aa852b33`
- `selection_stack2_r7.py` `7520622c24f54c7d3aac7ed790403cff2e84b153a4593a149278ce45f3f7f1c0`
- `selection_stack2_r6.py` `ab89e6d9610f627d8e8f2c0ebde4214044c117922cff444a3458565d9d94bee6`
- `selection_stack2_r2.py` `4ab1a1912224e82c7ed529233844f1f5e8f39d49b4620a584842cf4062f4d9de`
- `stack2_chat.py` `91627007aaeeb2646d33c097d93f6156560673376e31dc684448c87ca12c944f` (unchanged)

NOT v1.0. NOT graduated. Owner constructs a closed final exam separately.
Do not train against that exam. Do not inspect Form B.

## Remaining risks

- Relation sites in **memory** are still `beside` / `next`. Unseen fact-side
  relation verbs will not build a pair.
- Interrogative detection still needs Who/Which/What (bare or spaced) so a
  named-entity follow-up like `the dog then?` stays about-report.
- Frozen pack omitted frog (L2 all-entity canary included it at 1.000).
- A 40-item pack at 1.000 does not estimate a closed exam.
- Hop weights still exist in the head file but are unused for inverse bind.

## Lineage / SHA

See `research/V010_STACK2_R7_SHA256SUMS.txt` and
`runs/actual_baby/stack2/r7/R7_FREEZE.json`.
