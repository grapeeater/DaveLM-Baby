# SF2 residual-item autopsy v1 — why the one Alex-correct TRAINING item failed

**Nature:** READ-ONLY post-SF2 autopsy. No training, no checkpoint modification, no SF3 design/implementation,
no locked transfer/copy/competing-name panels, no FINAL/sacred access. SF2's frozen classification
`SF2_ACQUISITION_FAIL` is preserved and not reinterpreted.

**Authorized sources:** the 16 SF1 TRAIN records; the frozen SF1/SF2 `SCHEDULE.json`; raw acquisition outputs at
updates 0/100/200; and the existing checkpoints Pilot1-parent (= SF2 u0 weights), SF1 u100, SF2 u100, SF2 u200.

---

## 1. Identity of the residual item and its matched counterpart

- **Failed Alex-correct item:** `TRAIN:g0:carried:wooden boat:a0`
  - prompt: `"Alex carried the wooden boat.\nThe person who carried the wooden boat was"`
  - candidates `[" Alex.", " Owen."]`, correct = Alex (index 0); object `wooden boat`; predicate `carried`.
- **Matched Owen counterfactual:** `TRAIN:g0:carried:wooden boat:a1`
  - identical except subject `Owen`; correct = Owen.
- **Other Alex-correct items (passed):** `TRAIN:g0:found:small drum:a0` (margin +0.049),
  `TRAIN:g0:found:wooden boat:a0` (+0.184), `TRAIN:g0:carried:small drum:a0` (+0.178).
- **All four Owen-correct items:** `g0:found:small drum:a1` (-0.623), `g0:found:wooden boat:a1` (-0.402),
  `g0:carried:small drum:a1` (-0.428), `g0:carried:wooden boat:a1` (-0.613) — ll4 Alex-minus-Owen margins at u200.

## 2. Item-level margin/logit/probability trajectory

Full per-item, per-checkpoint table: `ITEM_TRAJECTORIES.jsonl` (all 16 items x 4 checkpoints). Alex/Owen key
values (ll4 Alex-minus-Owen sequence margin / first-token Alex-vs-Owen logit margin / Alex & Owen first-token probs):

| item | checkpoint | ll4 margin | first margin | P(Alex) | P(Owen) | exact |
|---|---|---|---|---|---|---|
| carried:wooden boat:a0 (FAILED) | Pilot1 | +0.617 | +0.730 | 3.0e-5 | 1.5e-5 | no |
| " | SF1 u100 | +0.431 | +0.441 | 0.344 | 0.221 | yes |
| " | SF2 u100 | +0.228 | +0.271 | 0.260 | 0.198 | yes |
| " | SF2 u200 | **-0.030** | **-0.045** | **0.421** | **0.440** | **no** |
| carried:wooden boat:a1 (matched Owen) | Pilot1 | +0.269 | +0.701 | - | - | no |
| " | SF1 u100 | +0.180 | +0.190 | 0.313 | 0.259 | no |
| " | SF2 u100 | +0.029 | +0.079 | 0.243 | 0.224 | no |
| " | SF2 u200 | -0.613 | -0.616 | 0.303 | 0.561 | yes |

## 3. Is the miss near-boundary or materially wrong?

**Near-boundary.** At u200 the failed item's first-token distribution is essentially a tie (Alex 0.421 vs Owen
0.440) and its sequence margin is -0.030 nats. This is an order of magnitude smaller than every Owen item's
margin magnitude (0.40-0.62) and the Mia/Nora margins (1.8-2.8). The item is the minimum-margin Alex item and was
also the minimum-margin Alex item at u100 (+0.228), i.e., it is the weakest member of the group, not a random or
materially wrong prediction.

## 4. Is the failed item structurally unusual?

**Only weakly.** The cell (object=wooden boat, predicate=carried) is not unique within g0; both lexical elements
appear elsewhere. Its subject-conditioning strength (Owen-minus-Alex margin -0.58) is large but comparable to
found:small drum (-0.67). The second-weakest Alex item (found:small drum, +0.049) uses a different object/predicate
yet nearly failed. So item-level geometry orders which item is nearest the boundary but does not uniquely explain
the failure.

## 5. Does training-order/recency evidence support Gemini's hypothesis?

**No - contradicted structurally.** The frozen schedule places every one of the 16 records exactly twice in every
one of the 180 English updates (verified: 180 presentations per item; every English batch contains all 16 twice;
all items present in the last 10 English updates). Item-level stochastic recency imbalance is therefore
impossible. Within-batch position of the failed item varies across the last 10 updates (12 distinct positions over
20 slots), so no fixed positional bias either. Gemini's stochastic-recent-imprint hypothesis is not supported by
the actual schedule.

## 6. Is the error upstream, output-path, or interactive?

**Upstream-dominant with a small output-head interaction.** A 2x2x2 component swap (Pilot1 P vs SF2 u200 S) on the
failed item gives: PPP +0.617, SPP (S upstream, P norm/head) +0.108, SSP (S upstream, S norm, P head) +0.102, SPS
(S upstream, P norm, S head) -0.038, SSS -0.030. Upstream replacement alone accounts for the large movement
(+0.62 -> +0.11); the SF2 head contributes the final ~-0.13 to -0.15 that pushes the near-tie across to Owen. Norm
is immaterial. This is the same pattern the SF1 component-swap forensic reported (upstream-dominant with material
head interaction), now at the level of the residual AO error.

## 7. Ranked causal explanations (full table in HYPOTHESIS_RANKING.json)

1. Global Alex/Owen boundary imbalance (Owen-side offset) - **SUPPORT** (all four cell midpoints negative at u200;
   group sweep of -0.2 to -0.5 over u100->u200).
2. Near-boundary miss - **SUPPORT** (genuine near-tie; settling vs offset not resolvable at this checkpoint
   resolution).
3. Output-head contribution - **WEAK SUPPORT** (marginal tipping factor, not origin).
4. Item-specific geometry - **WEAK SUPPORT** (orders which item is weakest; not unique).
5. Constant-LR instability - **UNRESOLVED** (no high-resolution checkpoints; losses smooth).
6. Symmetric-pair-batching (exposure) - **WEAKENED** (pairs already appear together every batch).
7. Stochastic training-order / recency - **CONTRADICTED** (schedule structure forbids item-level recency).
8. Synthesis: KL removed the Alex prior; factual CE then drove a monotonic boundary sweep past center to a mild
   Owen-lean; the weakest Alex item crossed - **SUPPORT** (see REPORT §"mechanism").

## 8. Mechanism supported by the evidence (narrative)

The SF1 Alex default (all eight AO items Alex-leaning at u100 in both SF1 and SF2) was produced under CE-only with
no retention constraint. Under SF2's KL retention, ordinary-context pollution and language damage were suppressed
and the *unconditional* name prior was removed, which forced the factual CE to build true subject-conditioned
selection. Between SF2 u100 and u200 the model's Alex/Owen decision boundary swept monotonically toward Owen
(mean AO margin: Alex items +0.403->+0.095, Owen items +0.224->-0.516), converting 0/4 Owen-correct at u100 into
4/4 at u200 while compressing all Alex-correct margins toward zero. The sweep overshot the balanced point by a
systematic ~0.1-0.3 nats (all four cell midpoints negative), leaving the four Alex items at margins +0.05..+0.18
(and one at -0.03). The single residual failure is the consistently-weakest Alex item sitting ~0.03 nats on the
Owen side, tipped across by a small (~0.15) output-head contribution. Net: the KL treatment did not fail; the 
residual is a near-threshold group-boundary offset - a one-item, ~0.03-nat endpoint miss under a constant-LR,
fixed-budget run.

## 9. Treatment-family implications (rank only, no design)

1. **Learning-rate schedule / annealing** - most mechanistically aligned (constant-LR run swept monotonically past
   the balanced point; annealing is the canonical way to stop at the balanced crossing). WEAK but best-supported.
2. **Parameter/readout constraint (head)** - second (head supplies the last ~0.15 tip on the weakest item).
3. **Pairwise-margin / symmetric-loss objective** - third; note prior pairwise/discrimination losses failed in the
   harder two-fact binding task, so this is unvalidated here.
4. **Additional fixed-budget training** - unresolved/risky both directions (the single remaining error is now the
   only CE error and would be pushed back, but the boundary is still sweeping monotonically toward Owen).
5. **Data-topology correction** - WEAK (no unique structural anomaly identified).
6. **Symmetric pair batching** - contradicted as an exposure fix (already satisfied).
7. Defensible alternative: **NO TREATMENT YET / more read-only evidence** (a fine-grained trajectory is impossible
   with existing checkpoints; only u0/u100/u200 exist). If SF3 is pursued, LR-annealing is the best-supported
   family; a decisive falsifier is a future annealed run that still ends with a negative Alex margin on this item.

## 10. Artifacts

Directory `C:\DaveLM-CADAVER\sf2_residual_item_autopsy_v1\`:
REPORT.md, ITEM_TRAJECTORIES.jsonl, MATCHED_PAIR_ANALYSIS.json, TRAINING_ORDER_AUDIT.json,
BOUNDARY_TRAJECTORY.json, OUTPUT_PATH_ANALYSIS.json, HYPOTHESIS_RANKING.json, PROVENANCE.json, SHA256SUMS.txt.
Hashes and inspected-artifact list in PROVENANCE.json / SHA256SUMS.txt.
