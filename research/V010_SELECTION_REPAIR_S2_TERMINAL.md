# Selection repair S2: matched +400 terminal

Status: **REGRESSION — hypothesis only half-survived**

Isolated diagnostic fork of v2R4 U16000. Not a replacement of v2R5/v2R6.
Not foundation graduation. No merge to `main`. Protected TEST was not opened.
Frozen Gate L/C/R were not changed. S1 receipts were not rewritten.

Protocol: [`design/V010_SELECTION_REPAIR_S2.md`](../design/V010_SELECTION_REPAIR_S2.md)
Independent S1 autopsy: [`research/V010_S1_INDEPENDENT_AUTOPSY.md`](V010_S1_INDEPENDENT_AUTOPSY.md)

## Identity

| item | identity |
|---|---|
| Protocol freeze commit | `d2dacd7484c4d28d03d5333e989b344c9a39fff8` |
| Parent checkpoint | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` |
| Parent SHA-256 | `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| Manifest SHA-256 | `006b3889bead1d6f45aa9e66a6ad3f090fa84b91f267a6e1ed522381af7d9e01` |
| Device | CUDA / AMD Radeon RX 9060 XT, `torch 2.9.1+rocmsdk20260116` |
| Preflight | `runs/selection_s2/PREFLIGHT.json` `status=PASS` |
| Padding max logit Δ | `0.0` |
| Contrast | `softplus(2.0 - effect_ij)` weight 1.0 on all-K query groups |
| Control | same frozen S2 schedule, full-answer CE only |
| Seed | 120001 (replicate 120002 not launched) |

Control and treatment `eval_0000.json` are byte-identical (`ace67e60…3264`).

## What this tested

S1 first-token vocabulary CE did not amplify the rank-2 residue and stole from induction. S2 trained the statistic S1 only measured: cross-query relative ranking of in-context value heads, packed as complete K-query groups, versus a matched all-K full-answer control.

## Results (independently read from eval JSON)

| metric | parent | control +400 | treatment +400 |
|---|---:|---:|---:|
| body-macro | 0.4115 | 0.4109 | 0.4479 |
| query_logit_effect | 0.784 | 0.875 | **1.551 (+0.767)** |
| same-first-token rate | 0.833 | 0.799 | 0.715 |
| queried inventory rank-1 | 0.394 | 0.382 | 0.438 |
| mean vocab margin | −0.770 | −0.757 | −0.554 |
| rest-lock | 0.977 | 0.975 | 0.970 |
| K=2 / 3 / 4 | 0.542 / 0.354 / 0.339 | 0.573 / 0.368 / 0.292 | 0.542 / 0.438 / 0.365 |
| all-queries-correct | 5/144 | 8/144 | 12/144 |
| query-swap follow / stuck-old / other / off | 36 / 33 / 23 / 4 | 32 / 36 / 25 / 3 | 40 / 24 / 27 / 5 |
| query-swap 2-pair follow | 16/31 | 17/31 | 21/31 |
| new-gold rest-lock | 0.979 | 0.969 | 0.979 |
| primitive induction first top-1 | 0.2969 | 0.2969 | 0.3125 |
| language DEV CE | 1.2430 | 1.2428 | 1.2443 |
| value-absent exact | 0 | 0 | 0 |
| primitive keyed first top-1 | 1.000 | 1.000 | **0.9375** |
| short keyed free exact | 0.8438 | 0.7969 | **0.7812** |

Paired-body bootstrap 10,000 seed 120300, treatment−control body-macro 95% CI **[0.0029, 0.0718]**; lower bound > 0. Point gain vs parent +0.036, vs control +0.037.

+200 mechanism triad vs parent (futility needs all three misses): effect +0.128 (bar 0.15), collapse drop 0.056 (bar 0.05), rank-1 +0.007 (bar 0.05). Futility **false**; run completed +400. Treatment receipt `terminal`.

## Adjudication

`REGRESSION`. Futility false. Two of ten success criteria true (`query_logit_effect` gain; bootstrap vs control). All others false.

Regression flags (loss > 0.05 vs parent on frozen intact panels):

- `primitive_keyed_first_top1` 1.000 → 0.9375
- `short_keyed_free_exact` 0.8438 → 0.7812

Induction did **not** regress (the S1 failure mode). Language CE +0.001. Gold rest-lock held. Value-absent stayed 0. Held-out exact stayed 0.

## Did the hypothesis survive?

Partially, as a loss-functional claim. Not as a selection remedy.

The extra term moved the thing it was pointed at: `query_logit_effect` +0.767 vs control's +0.091. Collapse fell 0.833→0.715 (control 0.799). Query-swap stuck-old 33→24. Isolation 2-pair follow 16/31→21/31. Mean margin less negative (−0.77→−0.55). That is not S1's "extra CE, no selection movement, induction dies" signature.

It did **not** convert that logit flip into greedy bind-and-copy:

- Diagnostic K=2 body-macro glued at 0.542 (parent *and* treatment).
- Queried inventory rank-1 only +0.044 (bar 0.15).
- Same-first-token still 0.715 (bar ≤0.50).
- Query-swap follow 36→40 (bar +0.15).
- 3-pair and 4-pair follow remain at chance.
- Body-macro 0.45, far from 0.70.

Mean margin is still **negative**. The query-differential grew from 0.78 to 1.55 and still loses to a query-invariant offset of similar size. Nearby "raise `m` / train longer" would chase that remainder with the same recipe. Protocol forbids that as the next move.

Control (all-K packing + full-answer CE) did not produce the effect jump. The packing change is not the explanation for +0.767.

Retention: no 1-pair train stream, and primitive 1-pair first-token dropped 4/64. Short-keyed exact also crossed the 0.05 line. Contrast is not free.

## Ruled out / not ruled out

Ruled out, this parent, this budget:

- Paired query-contrast at `m=2` on all-K twins as a **sufficient +400 remedy** for greedy query-conditioned selection vs matched full-answer CE.
- "The contrast gradient cannot reach the query-differential at all."
- "All-K packing alone explains the effect jump."
- "This extra first-position term must smash induction" (S1-specific, not reproduced).

Not ruled out:

- Remainder-masked continuation CE, so teacher-forced copy cannot keep locking the query-invariant favorite (predicted S2 miss mode).
- A retention 1-pair mix, because S2 nicked primitive keyed.
- A loss on the *greedy* competitor rather than paired logit effect.
- Architecture / sidecar (still not justified: the functional moved).

## Baby's resulting state

Authoritative Baby remains **v2R4 U16000 parent**
`checkpoint_16000.pt` SHA `94b3a9da…17827`.
S2 U16200/U16400 checkpoints are local diagnostic artifacts. Do not promote.
Do not open TEST. Do not move Gate L/C/R.

## Highest-information next action

Do **not** raise S2's margin, λ, or duration. Do not launch 120002.

Justified later protocol, not launched here: keep paired contrast, **mask the first answer token out of token-mean CE** (remainder-only copy), and add a small primitive-keyed retention stream so 1-pair copy cannot be the tax. Matched control required. Futility on whether greedy inventory rank-1 / K=2 collapse actually move, not on `query_logit_effect` (already shown trainable).

## Local-only artifacts

Checkpoints stay local. Hashes:
[`research/V010_SELECTION_REPAIR_S2_SHA256SUMS.txt`](V010_SELECTION_REPAIR_S2_SHA256SUMS.txt).
Adjudication: `runs/selection_s2/ADJUDICATION_120001_400.json`.
Protected material opened: false.
