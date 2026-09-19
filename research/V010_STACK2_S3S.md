# Stack2 s3s: composition-first who-bind from s2m

Status: **EXPERIMENTAL SURVIVOR**. Not authoritative. Not promoted.
U16000 remains AUTHORITATIVE:
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. s2m is not replaced as a promotion
candidate; s3s is the composition child, still experimental.

Parent (unpromoted mix/combine/D3 recover):
`runs/actual_baby/stack2/s2m_protect40_combine_323091/checkpoint_00025.pt`
SHA `bd10f0ea8239fe09bd5bec933611ba92b27c20402b24474ad287331153240ff9`.

Survivor:
`runs/actual_baby/stack2/s3s_compose_lock_324121/checkpoint_00025.pt`
SHA `8101c421bedf0514bd4ad7c403557f4e72cc2cfc0626090d779eafb2343b382d`.

Protocol: `src/baby_v010/selection_stack2.py` `--mode compose` plus evidence
follow-ups in `src/baby_v010/selection_stack2_s3.py`.
Receipts: `runs/actual_baby/stack2/{CAMPAIGN.json,LEDGER.md,S3.json,s3s_verify.json,s3s_verify_d3.json}`.

This is a **composition-first** milestone. Unprompted short-sentence generation
is the next lesson, not claimed here.

## Why this is the composition milestone

s2m already binds color/size for 2- and 3-entity scenes (`mixed_2e` 0.875,
`mixed_3e` 0.875–0.937) and answers in one word. It cannot invert the bind:
`who_2e` / `who_3e` zeroshot are 0.0 on held-out “who is the {color/size}
one?” templates. That is entity-from-property under distractors, not another
D3 campaign.

s3s teaches that inversion on held-out wording and entity assignments while
keeping English, stop/reuse, mixed bind, and D3 first-token 215/215.

| metric | s2m | s3s | vs s2m |
|---|---:|---:|---|
| who_2e held-out n=32 balanced | 0.000 | **0.500** | +0.500 |
| who_3e held-out n=32 balanced | 0.000 | **0.469** | +0.469 |
| mixed_3e | 0.875 | **0.938** | +0.063 |
| mixed_2e | 0.875 | **0.969** | +0.094 |
| fact_combine | 0.719 | 0.656 | −0.063, s2a floor |
| story_combine | 0.562 | **0.594** | +0.031 |
| color / size-stop / story | 0.969 / 1.0 / 1.0 | 0.969 / 1.0 / 1.0 | hold |
| usable-chat 4-turn | 0.969 | 0.938 | −0.031, inside 0.05 bar |
| usable-chat overall | 0.929 | 0.905 | −0.024, inside 0.05 bar |
| period-stop / fact-reuse | 1.0 / 1.0 | 1.0 / 1.0 | hold |
| D3 long-gap free_exact | 197/215 | 190/215 | −7, above 180 floor |
| D3 first_correct | 215/215 | 215/215 | hold |
| primitive induction | 0.453 | **0.484** | +0.031 |
| bare sentence_ok | 0.000 | 0.000 | unsolved |
| prefix sentence_ok | 0.000 | 0.000 | not scaffolded |
| bare one_word_color | 1.000 | 1.000 | still one-word Baby |

## Not prompt-fed, not a worksheet

- Train and hold **query templates are disjoint**. Gold is the entity name,
  never `query_position`.
- Official who panels are **entity-balanced n=32**. A lucky duck-heavy n=16
  draw is not the score.
- Extra held-out seeds (same n=32 balancer):

| seed | who_2e | who_3e |
|---:|---:|---:|
| 324001 (official) | 0.500 | 0.469 |
| 324777 | 0.344 | 0.375 |
| 325001 | 0.531 | 0.250 |

who_2e stays clearly above s2m’s 0.0 on every seed. who_3e is noisier but
not zero. Errors are mostly wrong-entity, not color leaks; Baby learned the
who-answer format and is partway through the bind.

- Sentence operators: prefix does **not** inflate `sentence_ok`. Baby still
  emits one-word colors on the bare sentence pack. This milestone does not
  claim sentence generation.

## Lineage (cheap canaries)

Pulse then lock, not one long run.

1. **s3a** from s2m, heavy who-bind mix, 25 updates: who_2e 0→~0.44 on the
   original n=16 panel, but `fact_combine` 0.719→0.656. Who-bind is learnable
   and taxes combine.
2. **s3b/s3c/s3d/s3e** sentence families, 25 updates: English collapse.
   Phrase training can fake `bare sentence_ok` 0.875 by destroying one-word
   color (0.969→0.250). Instruction prefixes move prefix more than bare
   (scaffold). Killed.
3. **s3i** more 2e-combine rehearsal with who: still taxes combine; who
   weaker than s3a. Combine drop is who-interference, not under-rehearsal.
4. **s3k** pure `protect40_combine` lock from s3a: who vanishes (0.44→0.06).
   Who is not sticky without rehearsal.
5. **s3n** half-LR lock from s3a with 18% who rehearsal: mix/English hold,
   who survives. Balanced n=32: who_2e 0.406 / who_3e 0.50. Full D3 159/215
   (first still 215). Too much span tax.
6. **s3r** lock-story from s3a: combine back to 0.719, D3 183/215, who
   weaker (0.375 / 0.406). D3-healthy local optimum, not enough who.
7. **s3s** from s3n: steal mix→language 35% + structured remainder 30%, keep
   `compose_lock` who rehearsal, half LR, 25 updates.

s3s is the survivor: who-bind is real, D3 190, mixed up, combine at the s2a
floor rather than s2m’s extra 0.063.

## Recipe

From s3n (`s3n_compose_lock_324081/checkpoint_00025.pt`), 25 updates, seed
324121, `lr_scale` 0.5:

- 35% language CE
- 30% remainder-span structured keyed/induction CE
- 35% `compose_lock` mix (≈18% of mix is who-bind; rest is 2e mix/combine,
  story, color, size)

s3n itself was 25 half-LR updates of `compose_lock` from the s3a who-pulse.

## Tradeoffs (accepted)

- **fact_combine** 0.719→0.656: returns to the s2a combine survivor, not to
  mix-cold. Mixed_2e improved. Hard floor is s2a combine, not s2m’s extra.
- **usable-chat** 0.929/0.969 → 0.905/0.938. Stop and reuse stay 1.0.
- **D3** 197→190/215. First-token still 215/215. Not a D3-max campaign.
- **3e fact-combine** stays weak (~0.22). Not this lesson.
- **Sentences** stay unsolved. Heavy sentence CE (s3b/s3e) is the wrong
  next step; it overwrites one-word English.

## Next Baby School lesson

Genuine short natural-English answers from this composition parent, with a
**light** sentence dose that cannot be a prefix scaffold and cannot be
allowed to destroy color/mix. Do not restart a D3-max campaign. Do not
promote s3s.
