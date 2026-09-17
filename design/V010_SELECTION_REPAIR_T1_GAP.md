# Selection repair T1: query-transport gap curriculum

Protocol id: `V010_SELECTION_REPAIR_T1_GAP`
Status: **preregistration — frozen before data generation and before any optimizer update**

This is not S1 with a bigger λ and not S2 with a bigger margin or duration. It
adds **no loss term at all**. The only thing that differs between the two arms is
the distribution of the query→generation *distance* in the keyed training rows.

Prior evidence: [`research/V010_QUERY_TRANSPORT_AUTOPSY.md`](../research/V010_QUERY_TRANSPORT_AUTOPSY.md)

## 1. Hypothesis

**H1.** Baby's query→key→value bind-and-copy circuit works but has an effective
reach of about one token from the generation position. Queried selection at
gap ≥ 2 fails because the query key's *identity is not transported* to the
position where the answer is emitted. A curriculum that starts keyed selection at
gap 0–1, where the circuit already earns reward, and then progressively increases
the gap, will **extend the transport range**, raising as-is candidate selection on
the untouched broad-gap frozen diagnostic above chance.

## 2. Causal mechanism targeted

Formation and extension of a query-identity transport pathway (retrieve the query
key from an earlier position into the residual stream at the generation
position), scaffolded from an already-working short-range match-and-copy circuit
rather than searched for from scratch as a conjunctive circuit.

Secondary predicted consequence: the query-invariant candidate-salience prior
(median spread 2.78 nats at the parent) shrinks, because it is currently the
loss-minimising response to keyed rows that are unreachable as posed. This
prediction is **secondary** and its failure does not by itself falsify H1.

## 3. Why the existing evidence licenses this and not an architecture change

Measured on the frozen S2 diagnostic with the query-locality probe, the one-token
cliff reproduces on **eight** checkpoints spanning four curricula (v2R4, v2R5
seeds 107001/107002, v2R6 seeds 108001/108002), five seeds, and three objectives
(plain CE, S1 first-token CE, S2 paired contrast). `append_query` lifts selection
far above `as_is` on every one of them; `append_query_then_filler_2` is at chance
on every one of them. No curriculum in the repository has ever scaffolded the
gap: every one presents far queries from the first structured update. So
"curriculum never taught transport" is consistent with all eight observations,
and it is strictly cheaper to test than an architecture claim. Objective and data
isolations come first, per the standing constraint.

## 4. Identity

| item | value |
|---|---|
| parent checkpoint | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` |
| parent SHA-256 | `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen eval diagnostic | `runs/selection_s2/DIAGNOSTIC.json` (hash checked against `runs/selection_s2/MANIFEST.json`) |
| frozen panels | `data/generated/foundation_v2/panels.json` at `FROZEN_PANELS_SHA256` |
| isolation panels | `runs/v2r4_isolation_panels/ISOLATION_PANELS.json` |
| train seed | `130001` (replicate `130002` generated but **not launched** unless T1 succeeds) |
| data seed | `130100` |
| device | `cuda` / AMD Radeon RX 9060 XT |

S1 and S2 data are read-only inputs and are not mutated. Both arms deny any
input sequence or candidate span appearing in the frozen panels, the S1
diagnostic, or the S2 diagnostic.

## 5. Dataset construction

Keyed rows come from the **unmodified** `data_v2.make_item(kind="keyed",
difficulty="full")`. The gap is then set by relocating *nuisance filler tokens
only*: a chosen number of filler tokens is moved from after the query key into
the template's own `filler_a` slot. The token multiset, the body, the rendered
pair order, the markers, the separator, the query key, and the target are all
unchanged. This is exactly the operation the generator already performs when it
draws a different `left` split, so every produced row lies inside the original
generator's support.

Only variants `keyed_1`, `keyed_3`, `keyed_4` are used for the keyed stream,
because those are the templates whose query→generation distance is a free
parameter. `keyed_0` and `keyed_2` are structurally pinned at gap 1 and
`keyed_5` is structurally pinned long, so none of them can carry a gap
manipulation. Insertion points, per template:

| variant | template | filler_a slot used |
|---|---|---|
| `keyed_1` | `[BOS, m, *filler_a, *body, m, q, *filler_b]` | body start |
| `keyed_3` | `[BOS, m, *body, *filler_a, q, m, *filler_b]` | query position |
| `keyed_4` | `[BOS, *body, m, *filler_a, m, q, *filler_b]` | query position − 1 |

Evaluation is on the untouched frozen diagnostic, which contains **all six**
variants, so `keyed_0/2/5` are a held-out-template generalization check.

### Batch composition (identical in both arms)

Every structured update is 16 rows: **10** full keyed rows, **3** primitive keyed
rows (1-pair retention, the stream S2 lacked and whose absence S2 paid for), and
**3** full induction rows (retention). Language-retention batches fire with
probability 0.20 per update, as in v2R4/S1/S2.

### Gap schedule — TREATMENT

Structured updates only, over a maximum of 800:

| updates | allowed keyed gap |
|---|---|
| 1–150 | 0–1 |
| 151–300 | 0–3 |
| 301–450 | 0–8 |
| 451–600 | 0–20 |
| 601–800 | natural (no relocation) |

### Gap schedule — MATCHED CONTROL

Natural gap for all 800 updates, i.e. exactly what v2R4 already did for 10,000
updates. The control is generated to match the treatment's
`(variant, pair_count, value_length)` histogram row for row, so the **only**
difference between the arms is the query→generation distance.

Same parent, same seed, same optimizer (`train_v2r4.capability_optimizer`), same
learning rates, same gradient clip, same language-retention probability and RNG
seeds, same batch composition, same budget, same evaluation, same loss:
`F.cross_entropy` over the answer-span mask, and nothing else.

## 6. Metrics

Primary instrument is **candidate-restricted argmax** on the frozen diagnostic:
did the queried value head win among the K in-context candidate heads. This
separates selection from answer-onset detection; full-vocab body-macro is
reported but is not the primary endpoint, because it conflates the two.

Reported, always stratified and never collapsed into one number:

- as-is candidate argmax by gap bucket `{0–1, 2–3, 4–12, 13–30, 31+}`
- as-is candidate argmax by `pair_count` `{2,3,4}` and by `value_length`
- as-is candidate argmax by `variant` (including untrained `keyed_0/2/5`)
- body-macro (full-vocab rank-1), `body_all_correct`, `mean_margin`
- `query_logit_effect`, `same_first_token_rate` (reported; **not** endpoints)
- query-invariant salience spread and query-dependent spread
- gold `rest_lock`, language DEV CE
- frozen panels: `primitive_induction`, `primitive_keyed`, `short_keyed`,
  `same_surface_novel`, `heldout_surface`, `broken_context`, `broken_order`
- post-hoc query-locality probe on the terminal checkpoints of both arms

## 7. Expected directional changes if H1 is true

- gap 13–30 and 31+ as-is excess over chance **rises**.
- `append_query` minus `as_is` on long-gap rows **shrinks** (the crutch stops
  being needed).
- `append_query_then_filler_2/4` **rises** above chance (range widened).
- K=3 and K=4 as-is excess **rises**.
- Salience spread shrinks or query-dependent spread grows, or both.
- `keyed_0/2/5` also rise, if the pathway generalizes rather than being
  template-memorized.

## 8. Primary endpoint and decision rules — fixed before seeing results

Primary endpoint `P` = as-is candidate argmax accuracy on frozen diagnostic rows
with gap ≥ 13. Parent: **74/215 = 0.3442**, chance **0.3391**, excess **+0.0051**.

| verdict | rule at update 800 |
|---|---|
| **SUCCESS** | treatment excess ≥ **+0.10** and (treatment − control) ≥ **+0.07** with paired bootstrap 95% CI lower bound > 0, and no retention regression |
| **MECHANISM SUPPORTED, DOSE INSUFFICIENT** | treatment excess ≥ +0.05 and (treatment − control) ≥ +0.04 with CI lower bound > 0 |
| **FALSIFIED** | treatment excess < +0.02 while gap 0–1 excess rose ≥ +0.10 |
| **NULL** | treatment excess < +0.02 and gap 0–1 excess did not rise |
| **REGRESSION** | any retention gate below, regardless of `P` |

Bootstrap: 10,000 resamples of diagnostic **bodies** (not rows), seed `130300`.

### Futility

At update 400, stop if treatment long-gap excess gain over its own update-0
baseline is **< +0.02**. Futility keys on the mechanism metric that matters
(long-gap greedy candidate selection), never on `query_logit_effect`, which is
already known to be trainable and insufficient.

### Hard stops

Language DEV CE > parent + 0.20; non-finite loss or gradient; free disk < 10 GiB;
wall clock > 2 h per arm.

### Retention gates — regression if any drops more than 0.05 versus parent

| metric | parent |
|---|---:|
| language DEV CE | 1.2430 (regression flag at +0.05) |
| `primitive_induction` first top-1 | 0.2969 |
| `primitive_keyed` first top-1 | 1.0000 |
| `short_keyed` free exact | 0.8438 |
| gold `rest_lock` | 0.9769 |

### Negative controls — must hold

- `value_absent` exact copy of the original span stays **0**.
- `broken_context` free exact ≤ 0.05.
- `broken_order` top-1 ≤ 0.20 of intact accuracy.
- `append_unused_key` excess over chance ≤ +0.05 (post-hoc probe). If this rises,
  the model has learned "attend to the last token" rather than key matching, and
  the result is a shortcut, not the capability.
- `append_other_key` follow must not collapse; the adjacent circuit must survive.

### What would falsify H1

Treatment lifts gap 0–1 but leaves gap ≥ 13 at chance. That would mean the
one-token reach is architectural or representational rather than curricular,
after nine checkpoints, and the next claim becomes an architecture claim with
evidence earned rather than assumed.

### Shortcut audits required before any success is reported

- Selection must rise on `keyed_0/2/5`, which are never trained here, or the
  result is template-specific.
- Selection must rise at gap values never trained (> 20 in blocks 1–4 terms).
- `append_unused_key` and `broken_*` controls above.
- No training input sequence or candidate span may appear in any eval panel.

## 9. Budget and replication

Maximum 800 updates per arm, two arms, one seed. No λ sweep. No second treatment
without new evidence or a revised mechanism. Replication on seed `130002` is
required before any promotion claim and is **not** launched by this protocol.
Promotion of a new authoritative checkpoint, opening protected TEST, tokenizer
changes, and merges to `main` are all out of scope here.

## 10. Frozen

These rules are not to be changed after results are seen. Gate L/C/R in
`design/V010_CAPABILITY_GATES.md` are not touched by this protocol.
