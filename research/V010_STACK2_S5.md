# Stack2 s5: WHO-sentences + richer relations from s4m

Status: **SUPERSEDED AS THE LIVE A+B LINE BY R2**. S5 remains the retrieval-blocker
receipt. R2 (`research/V010_STACK2_R2.md`) is the experimental dual-milestone
continuation. Still not authoritative. Still not promoted.
U16000 remains AUTHORITATIVE:
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. s4m is not replaced.

Parent (unpromoted sentence survivor):
`runs/actual_baby/stack2/s4m_compose_lock_325231/checkpoint_00025.pt`
SHA `4c0f142768aa57e2421a94a5214392e5c2d71e3805a2c0f6d5c9f0bc984c95e5`.

Best A-only checkpoint (s5m):
`runs/actual_baby/stack2/s5m_compose_lock_326121/checkpoint_00025.pt`
SHA `dd2584a5b6252adae70dc1729666b937c41dd0686af347d0b2ee2a0bbe30ec0e`.

Best joint checkpoint (s5b3):
`runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt`
SHA `0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd`.

Receipts: `runs/actual_baby/stack2/{CAMPAIGN.json,LEDGER.md,S5.json,s4m_s5_zeroshot.json,s5b3_verify.json,s5b3_verify_d3.json}`.

Neither graduation bar hit 0.75 on held-out bare packs at the same time.
This is a diagnosed retrieval ceiling plus skill interference, not a quiet
bar-lowering.

## Preregistered A gate vs what the stack can do

The written A target was bare compositional/WHO sentence_ok ≥ 0.75.

s4m / s5m one-word WHO (`who_2e`) is **0.50**. WHO-sentence scoring requires
that same inverse retrieval **plus** independent expression. A model that
finds the right entity half the time cannot systematically clear 0.75
sentence_ok on a 16-item 2e pack.

Revised valid A gate, stated before treating 0.375 as enough: **who_sent_bare
≥ who_2e (about 0.50)** with both color/size families, no prefix scaffold,
who_bind not destroyed, direct-fact sentences and mix alive. s5m reached
**0.375**, not 0.50. A is **not declared**.

## What failed

- **s5a** 25% pulse on the same Who-is queries as one-word WHO: syntax appeared
  (`predicate` 0.75) and `who_2e` fell 0.50→0.16. Two answers, one question.
- **s5c** `--mode train` used E12, not s4m. Invalid arm. Train mode now uses
  `parent_checkpoint`.
- **s5d** animal queries from s4m: `who_sent` 0.25 but color collapsed to 0.47.
- **s5e–s5f** lock restored English; `who_sent` plateaued. Decodes were
  `The animal is white.` — dummy-noun wrapper, not WHO.
- **s5g–s5k** looks-queries + entity-first CE: syntax real, many
  **wrong-fact** sentences (`frog is pink` when asked green). Last-fact /
  distractor verbalization. `who_sent` stuck at 0.31.
- **s5n / s5p** lock or medium dose from s5m **faded** who_sent back to 0.12–0.25.
- **s5bx** raised has to 0.667 both directions and **cut A** to 0.19.
- **s5bl / s5b4 / s5b5** oscillate: raising has or beside steals WHO-sentences
  and vice versa. 50u at 15% did not stabilize a joint 0.75.

## What worked

**s5m** (from s4m): 1e then 2e WHO-sentences, **asked-attribute facts only**,
entity-first gold (`dog is white.` not `The dog is white.`), looks-queries
disjoint from Who-is.

| metric | s4m | s5m |
|---|---:|---:|
| who_sent bare | 0.000 | **0.375** |
| who_sent prefix | 0.062 | 0.375 |
| who_color / who_size | 0 / 0 | **0.375 / 0.375** |
| who_2e / who_3e | 0.500 / 0.500 | **0.500 / 0.375** |
| direct bare | 0.375 | 0.250 |
| mixed_2e | 0.812 | **0.875** |
| combine | 0.688 | **0.750** |
| color | 0.750 | **1.000** |

Held-out bare WHO-sentences that are real (entity + value + is/looks):

- `bear is blue.`
- `duck is green.`
- `frog looks pink.`
- `dog is huge.`
- `cat looks thin.`
- `bear looks tiny.`

**s5b / s5bx / s5b3** taught possession and beside with existing tokens
(` object`, ` beside`). Bare, no answer-prefix dependence (has prefix 0.42
< bare 0.67 on s5bx; beside prefix 0.33 < bare 0.75 on s5b3).

s5b3 held-out examples:

- `dog has the white object.`
- `The cat has the green object.`
- `The frog has the pink object.`
- `The dog is beside the hen.`
- `The duck is beside the frog.`
- `bear is beside the cat.`

Forward and inverse both move: s5bx has_who=has_what=0.667; s5b3
beside_where=0.667 / beside_who=0.833.

## s5b3 verify (joint candidate, not a pass)

| metric | s4m | s5b3 verify |
|---|---:|---:|
| who_sent bare | 0.000 | 0.312 |
| has bare | 0.000 | 0.583 |
| beside bare | 0.250* | **0.750** |
| who_2e | 0.500 | 0.406 |
| direct bare | 0.375 | **0.375** |
| color | 0.750 | **0.969** |
| mixed_2e | 0.812 | **0.969** |
| combine | 0.688 | 0.656 |
| usable / usable4 | 0.857 / 0.906 | 0.857 / 0.844 |
| stop | 1.0 | **1.0** |
| reuse | 1.0 | 0.889 / 0.857 |
| D3 long-gap | 185/215 | 181/215 |
| D3 first_correct | 215/215 | **215/215** |
| induction | 0.547 | 0.500 |

\*s4m beside 0.25 was mostly fact-copy / coincidence, not a trained family.

## Retrieval follow-up (after the first A/B sweep)

s4m/s5m who_2e ≈ 0.50 is **last-entity recency**, not a scoring bug.

| ckpt | acc | P(pred=last) | acc \| last=gold | acc \| last≠gold |
|---|---:|---:|---:|---:|
| s4m | 0.47 | 0.66 | 0.67 | 0.29 |
| s5m | 0.50 | 0.59 | 0.60 | 0.41 |

Asked-attr 2-fact WHO (A's actual scene) is **not** easier than 4-fact who_bind
(s5m asked-attr 0.50). The evaluator is not the ceiling. Wrong-entity fluent
sentences correctly fail.

Tried next, all 25-update cheap canaries:

1. **s5w** anti-recency + margin from s4m: **KILL**. who_2e 0.50→0.31.
   Forcing non-last gold taught never-last.
2. **s5x** uniform 4-fact + light margin 0.3: last-not 0.29→0.41, who_2e 0.531.
   story_combine 0.50→0.375. Real direction, not a pass.
3. **s5xx** continue s5x: who_2e stuck 0.531; story_combine recovered 0.594.
4. **s5z** from s5m, A-shaped asked-attr sentences + margin: who_sent **stuck 0.375**.
5. **s5pp** same-scene pair questions (both entities, matching the eval pack):
   who_sent **0.375→0.250**. Kill.

Pulse → lock, anti-recency, first-token hinge, asked-attr matching, and
paired-scene CE all fail to put inverse 2e selection honestly above ~0.53
or WHO-sentences above 0.375. Further 25u CE variants would repeat this.

## Blocker

This is an owner-level retrieval/architecture limit, not a routine failed arm.

- Inverse 2e WHO saturates at chance-among-the-two (~0.50) with last-mention bias.
- Sentence **syntax** is already present (`predicate` 0.88–1.0).
- Milestone A at 0.75 requires retrieval the stack does not have.
- Milestone B families can hit 0.67–0.75 **separately** but steal A when pulsed.

Do not quietly lower A to 0.375. A revised gate of who_sent ≥ who_2e (~0.50)
was stated before treating 0.375 as enough; **s5m did not hit it**.

U16000 / TEST / FINAL / SACRED untouched. Nothing promoted. No commit.

## Next lesson

Do **not** keep pulsing sentence CE. Owner choices:

1. A pointer / copy / contrastive-retrieval mechanism that can beat last-mention.
2. A longer dedicated 2e selection campaign with a new signal, not more CE.
3. Only after who_2e is honestly above ~0.70, return to WHO-sentences then B.
