# Independent S1 autopsy (not Astra's terminal write-up)

Status: **read-only recomputation**. Frozen S1 receipts were not rewritten. Protected TEST was not opened. Gate L/C/R were not changed.

Script: `scripts/autopsy_selection_s1_independent.py`  
Parent SHA-256: `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` (match)  
S1 SHA256SUMS: 21/21 files match, including control/treatment evals, receipts, and adjudication.

## Headline verification

Astra's terminal numbers reproduce from `eval_0000.json` / `eval_0400.json`:

- Parent body-macro 0.4022; control +0.0052; treatment +0.0075.
- Bootstrap treatment−control 95% CI [−0.0191, +0.0243].
- Language DEV CE +0.0060. Gold rest-lock 0.9745 → 0.9653. Value-absent exact 0.
- Primitive induction first top-1 0.2969 → 0.1719 (treatment). Control only 0.2969 → 0.2656.
- Adjudication JSON: `REGRESSION`, `futility=true`, all eight original success criteria false.

## What S1 actually did to the hypothesized signal

The "weak query residue, amplify with first-token CE" story does **not** survive the row-level ranks.

| mechanism | parent | control +400 | treatment +400 |
|---|---:|---:|---:|
| queried inventory rank-1 | 0.377 | 0.382 | 0.387 |
| queried inventory rank-2 | 0.338 | 0.333 | 0.338 |
| query_logit_effect | 0.781 | 0.767 | 0.724 |
| mean vocab margin | −0.792 | −0.776 | −0.817 |
| same first token / 144 bodies | 118 | 115 | 123 |
| all-queries-correct | 5 | — | 6 |
| K=2 body-macro | 0.53125 | 0.53125 | 0.53125 |
| query-swap novel follow / stuck-old / other / off | 36 / 33 / 23 / 4 | 28 / 39 / 25 / 4 | 35 / 30 / 26 / 5 |
| miss queried vocab rank-1 | 0/60 | 1/68 | 2/61 |

First-token CE did **not** buy a stronger selection signal. Rank-2 stayed rank-2. The query-conditioned logit effect fell. Collapse got worse. K=2 is glued to the same 0.53125 in all three columns.

Query-swap follow is still chance. New-gold rest-lock stays high (parent 0.979, treatment 0.958). Copy is not the failure.

First-slot is not the diagnostic collapse recipe: parent first-slot copy 128/432 (0.296); only 39/118 collapsed bodies emit the first rendered value head. Treatment slightly *increased* first-slot copy (148/432) and collapse.

S1 packing already used two distinct queries per body. For K=2 that is all-K. 45/48 two-pair bodies still emitted one first token. "Same body, two queries + gold CE" is not an untested idea; it is the failed recipe.

The extra term was strong enough to move *something*: induction first top-1 dropped 0.125 on treatment vs 0.031 on control. Budget and loss weight were sufficient to disturb a circuit. They disturbed induction, not bind-and-select.

## Diagnosis revision

Keep: Baby copies in-context value spans; greedy selection is not query-bound; a rank-2 queried residue exists; architecture is not the next claim.

Revise: the residue is **not** sitting in a basin that gold first-token NLL will climb. Gold NLL on twins is compatible with a query-invariant favorite (the query that already matches the favorite gets cheap CE; the other query does not create a bind operator). Nearby λ/duration variants of S1 are low information.

S1 **rules out** first-token vocabulary CE on this counterfactual packing as a +400 selection remedy, and rules out "the extra CE never reached the network."

S1 does **not** rule out a loss whose minimum *requires* a query flip (paired contrast on `query_logit_effect`), nor all-K packing for K=3,4 (S1 only used 2-of-K there), nor later remainder-mask / induction-priority forks.

## Highest-information next action

Not attention localization (would not change the objective once rank-2 is known not to have moved). Not S1 with bigger λ.

Next: **S2 paired query-contrast vs a matched all-K full-answer control**, frozen in `design/V010_SELECTION_REPAIR_S2.md`.
