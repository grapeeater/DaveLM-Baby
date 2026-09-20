# Stack2 s4m: direct-fact short English from s3s

Status: **EXPERIMENTAL SURVIVOR**. Not authoritative. Not promoted.
U16000 remains AUTHORITATIVE:
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. s3s is not replaced.

Parent (unpromoted composition-first survivor):
`runs/actual_baby/stack2/s3s_compose_lock_324121/checkpoint_00025.pt`
SHA `8101c421bedf0514bd4ad7c403557f4e72cc2cfc0626090d779eafb2343b382d`.

Survivor:
`runs/actual_baby/stack2/s4m_compose_lock_325231/checkpoint_00025.pt`
SHA `4c0f142768aa57e2421a94a5214392e5c2d71e3805a2c0f6d5c9f0bc984c95e5`.

Protocol: `src/baby_v010/selection_stack2.py` `--mode sentence` plus evidence
follow-ups in `src/baby_v010/selection_stack2_s4.py`.
Receipts: `runs/actual_baby/stack2/{CAMPAIGN.json,LEDGER.md,S4.json,s3s_sentence_zeroshot.json,s4m_verify.json,s4m_verify_d3.json}`.

This is a **direct-fact sentence** milestone. Baby can now emit a short
grammatical answer to an ordinary bare color question. Two-entity / WHO
sentences are the next lesson, not claimed here.

## Why this is genuine

s3s already retrieves and composes. On the official sentence decode pack she
answered one word (`bare sentence_ok` 0.0, `one_word_color` 1.0). Prefix
instructions did not help (`prefix` 0.0).

s4m answers some of those same **unscaffolded** held-out questions as
short sentences that contain the right entity and color:

| id | question (abbrev.) | s4m bare decode | ok |
|---|---|---|---|
| fox_white | Which color is the fox? | The fox looks white. | yes |
| pig_red | Tell me the color of the pig. | pig is red. | yes |
| cat_green | What is the color of the cat? | The cat is green. | yes |
| frog_blue | What is the color of the frog? | blue. | one-word |
| hen_yellow | Tell me the color of the hen. | yellow. | one-word |
| bird_pink | Which color is the bird? | pink. | one-word |
| cow_pink | Which color is the cow? | The duck looks green. | wrong entity |
| duck_yellow | Tell me the color of the duck. | blue. | wrong one-word |

`bare` 0.375 vs `prefix` 0.125. The instruction "Please answer in a sentence"
does **not** inflate the score. Train and hold answer templates are disjoint
(`The/e is/looks` vs `That/This/e looks`). Exact hold-template CE stays 0.0;
success is generated meaning+syntax, not worksheet copy.

Three surface families appear (`The e looks v.`, `e is v.`, `The e is v.`).
This is not one memorized wrapper around a color word.

## Metrics vs s3s

| metric | s3s | s4m | note |
|---|---:|---:|---|
| bare sentence_ok | 0.000 | **0.375** | primary |
| prefix sentence_ok | 0.000 | 0.125 | not scaffold |
| bare one_word_color | 1.000 | 0.375 | style split, not wipe |
| sent_1e / sent_2e exact | 0 / 0 | 0 / 0 | hold templates unused |
| who_2e / who_3e | 0.500 / 0.469 | **0.500 / 0.500** | hold |
| mixed_3e | 0.938 | 0.938 | hold |
| mixed_2e | 0.969 | 0.812 | −0.156, still high |
| fact_combine | 0.656 | **0.688** | +0.031 |
| story_combine | 0.594 | 0.500 | −0.094 |
| color first_top1 | 0.969 | 0.750 | style mix: some QA now starts with The/entity |
| size-stop / story / dialogue | 1.0 / 1.0 / 1.0 | 1.0 / 0.906 / 1.0 | hold |
| usable-chat overall | 0.905 | 0.857 | −0.048, inside 0.05 of s3s |
| usable-chat 4-turn | 0.938 | 0.906 | −0.031 |
| period-stop / fact-reuse | 1.0 / 1.0 | **1.0 / 1.0** | hold |
| D3 long-gap | 190/215 | 185/215 | −5, above 180 |
| D3 first_correct | 215/215 | **215/215** | hold |
| primitive induction | 0.484 | **0.547** | +0.063 |

Color first_top1 is not English death. Held-out QA decodes still name the
right color, often as a sentence (`The duck looks red.`, `hen is white.`)
and often as one word (`green.`). Size-stop stays 1.0. Usable-chat stop/reuse
stay 1.0. Collapse would be color < 0.70 plus dead chat; that did not happen.

## Lineage (cheap canaries)

Do not repeat s3b–s3e: heavy sentence CE inside the mix bucket made a
sentence metric move by overwriting one-word English. Light dose as a
**fourth bucket**, mix-protected.

1. **s4a** 8% 1e, 25 updates from s3s: English/mix hold, `bare` 0. Too few
   sentence examples. who 0.50→0.375.
2. **s4c** 15% 1e, 25 updates: still `bare` 0; combine 0.656→0.500. Kill.
   Stealing mix budget wobbles combine without teaching expression.
3. **s4f** 8% at 0.25 LR: hold, `bare` 0. Lower LR is not the missing piece.
4. **s4d** 8% instructed 1e: `prefix` still 0. An instruction is not a
   shortcut. Combine 0.562. Kill.
5. **s4j** 8% × 75 updates: `bare` 0 the whole way; train-surface first
   token never moves. 8% is invisible to generation. who soft at u75.
6. **s4p** mix-protected **25%** 1e pulse, 25 updates: English perfect,
   combine hold, `bare` still 0. Train-surface entity-first ranks 6–8.
   Gradients are real (`task=sentence`, loss ~1.61). Need more pulse, not
   more mix theft.
7. **s4px** continue pulse +50: u25 still `bare` 0, English hold. **u50
   `bare` 0.500** with genuine decodes (`The fox looks white.`) **and**
   color first_top1 0.625 collapse. Pulse teaches expression and, if left
   on, overwrites one-word first tokens.
8. **s4l** lock from that u50: language/structured/compose_lock + 8%
   sentence. Color 0.844, usable recovering, who 0.406, `bare` 0.250
   (2/8 still real sentences). D3 192/215.
9. **s4m** from s4l, 12% sentence: `bare` 0.375, prefix 0.125, who
   0.500/0.500, combine 0.688, usable4 0.906, stop/reuse 1.0, D3 185/215
   first_correct 215. This is the survivor.

## What failed and what it taught

- Light 8–15% for 25–75 updates does not start sentence generation. The
  one-word first-token prior is too strong.
- 15% that steals mix hurts combine before any sentence appears.
- Instruct prefixes do not create bare sentences.
- A mix-protected 25% pulse is the smallest mechanism that actually fits
  sentence CE without immediately killing English.
- Leaving that pulse on past the onset (s4px u50) recreates s3b: sentences
  appear and one-word first_top1 collapses together.
- Lock + light rehearsal separates the skills: one-word QA and short
  sentences can coexist. That coexistence is the milestone, not 100%
  sentence answers.

## Not claimed

- Two-entity compositional sentences (`compose_sent_heldout` exact 0;
  2e decodes stay one-word).
- WHO questions answered as sentences (who stays one-word entity, correctly).
- Hold-template exact match.
- Promotion over U16000 or s3s.

## Next Baby School lesson

**Compositional / WHO short sentences.** Direct-fact "What color is Alice?"
→ "Alice is red." is real. "Who has the large object?" → "Alice has the
large object." and 2e color sentences under distractors are not.
