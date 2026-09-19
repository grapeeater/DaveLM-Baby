# Selection repair S2: paired query-contrast vs matched all-K control

Status: **FROZEN before generate / train / score.** Isolated diagnostic fork of v2R4 U16000. Not a replacement of v2R5/v2R6. Not foundation graduation. No merge to `main`. Protected TEST/FINAL/SACRED stay closed. Frozen Gate L/C/R stay unchanged.

Parent branch: `codex/selection-repair-s1` at `bea0963789eb2c0317c65078cfd0501075b8a787`.
This file does not rewrite S1 receipts, v2R4 terminal metrics, or isolation panels.

Name: `BABY_V010_SELECTION_REPAIR_S2`

## Exact hypothesis

S1's extra first-token **vocabulary CE** on counterfactual twins is the wrong functional of the existing query residue. Baby already changes inventory logits when the query changes, but greedy first-token selection remains a **query-invariant body attractor**. Independent gold NLL can be reduced by reinforcing that attractor for whichever query already matches it, while the losing query's CE is absorbed as noise and induction pays the gradient tax.

S2 therefore optimizes the **cross-query relative ranking** that S1 only *measured*: for the same body, swapping the query must flip which in-context value-first-token is preferred. That operator cannot be satisfied by copying one favorite span for every query.

If this is the missing bind-and-copy selection pressure, then at the first informative checkpoint the treatment — and not the matched all-K full-answer control — will raise `query_logit_effect`, drop same-first-token collapse, and move queried inventory rank-1. Query-swap follow should follow those logit changes. If those mechanism metrics do not move, the hypothesis is wrong at this budget and the run stops.

## Evidence supporting it (independently recomputed from disk)

Independent autopsy: `scripts/autopsy_selection_s1_independent.py` → `research/V010_S1_INDEPENDENT_AUTOPSY.md`. Parent SHA-256 `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` matches. S1 SHA256SUMS 21/21 match.

| observation | parent | S1 control +400 | S1 treatment +400 |
|---|---:|---:|---:|
| body-macro | 0.4022 | 0.4074 (+0.0052) | 0.4097 (+0.0075) |
| bootstrap treatment−control 95% CI | — | — | [−0.0191, +0.0243] |
| query_logit_effect | 0.7806 | 0.7674 | **0.7244 (fell)** |
| mean vocab margin | −0.7921 | −0.7758 | **−0.8168 (fell)** |
| queried inventory rank-1 | 0.377 | 0.382 | 0.387 (+0.009) |
| queried inventory rank-2 | 0.338 | 0.333 | **0.338 (unmoved)** |
| same emitted first token / 144 bodies | 118 | 115 | **123 (worse)** |
| K=2 body-macro | 0.53125 | 0.53125 | **0.53125 (stuck)** |
| query-swap novel follow | 36/96 | 28/96 | 35/96 |
| miss queried vocab rank-1 | 0/60 | 1/68 | 2/61 |
| primitive induction first top-1 | 0.2969 | 0.2656 | **0.1719 (−0.125)** |
| language DEV CE | 1.2430 | 1.2491 | 1.2490 (+0.006) |
| gold rest-lock | 0.9745 | 0.9769 | 0.9653 |
| value-absent exact | 0 | 0 | 0 |

S1 K=2 packing already used **both** queries of each 2-pair body. Collapse on K=2 stayed 45/48. First-slot copy on the S1 diagnostic is only 0.30, and only 39/118 collapsed bodies emit the first rendered head. The attractor is a body-specific favorite, not a pure first-slot machine.

S1's extra term was not too weak to move the network: induction first top-1 dropped 0.125 on treatment vs 0.031 on control in 400 updates. The extra first-token CE was live. It spent itself on the wrong circuit.

## Alternatives already ruled out

- Query-blind copy (isolation query-swap and S1 `query_logit_effect` ≈ 0.78).
- Payload continuation / free-generation of a selected span (gold rest-lock ≥ 0.96; inventory copy high).
- "Insufficient explicit first-token CE on this S1 data/order/loss as a +400 remedy" (S1 terminal; confirmed).
- Nearby λ or duration variants of **that** first-token vocabulary CE (low information; S1 extra term already moved induction).
- "Just amplify the rank-2 residue with gold NLL": rank-2 did not become rank-1; the residue was not amplified.
- Matched counterfactual **data** alone as a +400 remedy (control also failed every success bar).
- Suffix-loss-only and frozen-lower-layers (prior v0.10 history; language held, capability did not appear).
- Architecture / binding sidecar / tokenizer: not justified. S1 tested the wrong objective, not capacity. Sidecar stays disabled.

Not ruled out, and **not** this experiment: remainder-only masking of continuation CE; inventory-restricted softmax that is ~equivalent to full-vocab CE given 95% inventory copy; attention-head localization; sidecar.

Read-only attention dump was considered and **not** run as a gate on this launch. Rank-2→rank-1 already falsified S1's amplification story from eval JSON. Layer localization would not change the paired-contrast functional.

## Exact intervention

Fresh hashed data, **not** a mutation of frozen S1 schedules.

Training items: full-difficulty keyed bodies with distinct value first tokens, K ∈ {2,3,4} only (no 1-pair). Render every query with body/filler/wrappers **unchanged** (true query-swap twins). Pack **all K queries** of a body into the same batch.

Structured batch (16): 12 keyed + 4 induction. Keyed packing is homogeneous-K:

| K | bodies | keyed rows |
|---:|---:|---:|
| 2 | 6 | 12 |
| 3 | 4 | 12 |
| 4 | 3 | 12 |

Each structured step draws one K uniformly from {2,3,4}. Language updates remain 20%. Low-prior probability 20% on keyed generation. Train-surface markers/separators only.

**Control:** inherited token-mean full-answer CE on the frozen S2 schedule (keyed + induction). No extra term.

**Treatment:** the same full-answer CE **plus** `1.0 * mean softplus(m - effect_ij)` over all unordered query pairs `(i,j)` in each body group, with `m = 2.0`.

For candidate heads `h` and first-answer logits `z_q`:

```text
effect_ij = (z_i[h_i] - z_i[h_j]) - (z_j[h_i] - z_j[h_j])
```

This is the S1 `query_logit_effect` statistic used as a **loss**, not as a post-hoc score. It is not first-token vocabulary CE and not an inventory-only softmax on gold.

Induction CE and language CE are unchanged. The extra term is keyed body-groups only.

## Why this targets the diagnosed mechanism

Collapse (same first token for every query on a body) yields `effect_ij ≈ 0` when logits themselves are query-invariant, and cannot drive `effect_ij` to the margin `m=2` without a query-conditioned flip. Independent gold CE *can* be improved by backing the current favorite. S1 did exactly that (collapse 118→123, effect 0.78→0.72). Paired contrast punishes that solution.

All-K packing makes a favorite wrong on K−1 queries every update. That is a data change vs S1's always-2-of-K, so a matched control on the **same** S2 schedule is required. For K=2, all-K equals S1's packing; K=2 movement isolates the new loss from the packing change.

## What remains unchanged

- Architecture, tokenizer, dropout, binding sidecar **disabled**
- Parent checkpoint and SHA above
- AdamW reset; layerwise LR 1.5e-5 / 3.75e-5; betas (0.9, 0.999); eps 1e-8; decay 0.05; clip 2.0
- Batch 16, context 256, no checkpoint selection
- Frozen v2 panels, isolation panels, Gate L/C/R
- No TEST/FINAL/SACRED, no merge to `main`, no v2R5 interference

## Data

- Diagnostic panel seed **120200**: 48 bodies × K=2,3,4, all queries, full difficulty, distinct heads. Independent of S1's 144 bodies.
- Schedule seed **120100** for train seed 120001; **120101** for optional replicate 120002 (replicate is forbidden unless this protocol's success bundle passes).
- Exclude exact inputs and all candidate value spans from frozen v2 panels, isolation-eval sources already denied via those panels, **and** the frozen S1 diagnostic (`runs/selection_s1/DIAGNOSTIC.json`).
- A body and all its counterfactual queries belong to one partition.
- Write/hash `DIAGNOSTIC.json`, `SCHEDULE_120001.json`, protocol, code. Do not edit S1 artifacts.

## Optimizer / masking / curriculum

See unchanged optimizer. Token-mean CE mask is the full answer span (value + train sep + EOS), identical to S1 control. Treatment does **not** mask away the first token from that CE; the only added term is paired contrast. Remainder-only CE is a later fork if this run shows copy-CE fighting the contrast.

Curriculum: full difficulty, K≥2, all-K groups, 20% language, 25% of structured rows induction.

## Seeds

Train/eval seed **120001**. Optional replicate **120002** only after success. Bootstrap **120300**, 10,000 paired-body replicates. Diagnostic **120200**. Data rng **120100**.

## Checkpoints / eval

Updates 0 / 200 / 400. Maximum **400**. No 800, no silent extension. Evaluate diagnostic + language DEV CE + frozen panels + isolation panels at 0, 200, and 400. Checkpoints named `checkpoint_{16000+step}.pt`.

Run **control first**, then treatment. Treatment futility at +200 may read control `eval_0200.json`.

## Success gates (S2 mechanism win, not Gate C)

Hard stop > regression > success > partial > failure.

No regression, then **all** of:

1. New diagnostic body-macro accuracy ≥ 0.70
2. Body-macro gain ≥ 0.20 vs parent and ≥ 0.15 vs matched control
3. Each K above chance + 0.15
4. Isolation query-swap same-surface novel first-token accuracy gain ≥ 0.15 vs parent
5. Same-surface novel free-exact gain ≥ 0.15 vs parent
6. `query_logit_effect` gain ≥ 0.50 vs parent
7. Same-emitted-first-token rate ≤ 0.50 (must break the ~0.82 collapse)
8. Queried inventory rank-1 gain ≥ 0.15 vs parent
9. Paired-body bootstrap 10,000 seed 120300: 95% lower bound of treatment−control body-macro > 0

This is **not** Gate C and not foundation graduation. Held-out exact may remain separator-OOD. Report value-span metrics separately. Do not move Gate L/C/R.

## Failure gates

Otherwise failure (if not hard stop / regression / partial / futility).

Partial: no regression, and (body-macro gain ≥ 0.10 parent and ≥ 0.05 control) **or** (query_logit_effect gain ≥ 0.25 and same-first-token drop ≥ 0.15), but not full success.

## Early futility (U+200)

Stop treatment (do not continue to 400) if **all** of the following hold vs parent:

- `query_logit_effect` gain < 0.15
- same-first-token rate drop < 0.05
- queried inventory rank-1 gain < 0.05

Complete the +200 measurements, freeze, autopsy. Do not raise `m` or λ. Do not switch to first-token CE.

## Negative controls

- Value-absent exact must stay ≤ 0.05 (prefer 0)
- Query-swap: report follow / stuck-old / other / off, not only first_top1
- Body-reorder query-first vs query-last (first-slot prior must not become the whole recipe)
- Broken-context / broken-order free exact ≤ 0.05

## Retention

- Language DEV CE: hard stop if +0.20; regression if +0.10
- Primitive induction first_top1 or free_exact loss > 0.05 vs parent: regression (S1 treatment failed this)
- Primitive keyed first_top1 / free_exact loss > 0.05: regression
- New gold-remainder lock loss > 0.05: regression
- Any other original intact frozen panel first_top1 or free_exact loss > 0.05: regression

## Expected outcomes

**If the hypothesis is right:** by +200, treatment `query_logit_effect` is up, collapse is down, queried inventory rank-1 is up, control is near parent on those three. By +400, query-swap follow is above chance with rest-lock held, K=2 is no longer glued at 0.531, induction stays inside 0.05, value-absent stays 0.

**If the hypothesis is wrong:** mechanism triad does not move by +200 (futility); or treatment≈control (all-K packing, not contrast, was the intervention); or effect rises but greedy/follow stay collapsed (offset larger than this margin at this LR — a *new* protocol, not a λ bump); or induction regresses again (extra first-position keyed losses steal induction — also a new protocol, not a silent mix patch).

## Authoritative Baby

Until a later treatment independently passes its frozen gates, authoritative Baby remains v2R4 U16000 `checkpoint_16000.pt` SHA `94b3a9da…17827`. S2 checkpoints are diagnostic artifacts only unless this protocol's success bundle passes **and** is independently re-scored.
