# Treatment-Design Audit v1 — SF1 ordinary-language factual-learning failure

**Scope:** Read-only. No optimizer, autograd, weight mutation, checkpoint, dataset, report, gate, or artifact was
touched. No locked/withheld panel, FINAL, or sacred material was accessed. This audit proposes a treatment; it
does not execute one.

**Authoritative roots verified during this audit:**
- Active research/evidence tree: `C:\DaveLM-CADAVER\` (root). Its `DaveLM-v0.9\` subdirectory is a **stale
  snapshot** (see INSPECTED_PATHS.md and the prior Nemotron audit); the current dev codebase lives at
  `C:\DaveLM-v0.9\` (top-level sibling). All experiment artifacts cited here are under the CADAVER root.
- Parent checkpoint lineage: T13 graduation checkpoint -> Pilot1
  (`language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt`,
  SHA-256 `2281d20e...`) -> SF1 update-100
  (`single_fact_acquisition_sf1_seed87011_run\checkpoint_100.pt`, SHA-256 `550ce430...`).

---

## 1. What the mission asked me to verify (endpoints)

All listed endpoints were checked against artifacts and are **confirmed**, with the stated status caveats:

| Mission endpoint | Verification | Status |
|---|---|---|
| SF1 stopped at update 100 for severe language regression, binding intact | `STATUS.json`: `binding_pass:true, language_cost_pass:false, status:STOP_REGRESSION, update:100`. Aligned CE 3.391->4.653 (guard = parent+0.25 = 3.641). Both binding pools 80/80 answer, 80/80 BD, 0 collapse at update 100 (`update100_checks.json`). | **Confirmed** |
| Position forensic classification | `sf1_token_position_patching_forensic_v1\REPORT.md` headline: `ANSWER_GENERATION_STATE_DOMINANT_WITH_OUTPUT_PATH_INTERACTION; LANGUAGE_DAMAGE_DISTRIBUTED_ACROSS_PREDICTIVE_POSITIONS`. | **Confirmed** |
| Full answer-generation trajectory required, not just first-token state | Block-7-MLP table: final-prompt-predictor-only patch -> 12/16 first-token but 9/16 sequence, 0/16 exact+EOS, 1/8 pairs; full answer-generation path -> 12/16, 12/16, 12/16, 4/8. | **Confirmed** |
| Current predictive state sufficient for broad name-prior inflation, amplified by output path | D3 section: predictive-state-only patch with Pilot1 head = 0.0364 (all-position value); SF1 head amplifies to 0.0678 (native SF1). | **Confirmed** |
| Language damage distributed across ordinary predictive positions | CE section: replacing "preceding valid/scored positions" is what moves CE (3.29 -> 4.37); final-record-predictor-only replacement leaves CE at 4.72. | **Confirmed** |
| No clean useful-only token route | "Established: ... cannot be cleanly separated by the tested token groups"; "Supported: ... partial separation by breadth but no demonstrated useful-only route." | **Confirmed (as established/supported mix; no useful-only route demonstrated)** |
| Alex/Owen vs Mia/Nora behave differently | Native SF1: Mia/Nora 4/4 complete reversal pairs, Alex/Owen 0/4 (readout D4, upstream-localization, position forensics all agree). | **Confirmed** |
| Candidate-pool audit weakened a simple frequency explanation | `sf1_candidate_pool_hypothesis_audit_v1\REPORT.md`: `ALREADY_WEAKENED`. Equal exposure (1,440/name), larger inherited Mia/Nora prior overcome while smaller Alex/Owen not. Survives only as "low conditional target-name diversity / fixed name-object topology may encourage an upstream shortcut." | **Confirmed** |
| Patching localizes causal sufficiency, not training locus | All forensics: "patches do not identify where training changed the mechanism"; "inference-time patching localizes causal sufficiency, NOT where learning occurred." | **Confirmed** |

---

## 2. Reconstruction of the causal story (independent, from code + evidence)

### 2.1 What SF1 actually trained (verified in code)

Parent **Pilot1** = T13 orthogonal-shared-unbounded localizer checkpoint after 1,000 updates (900 language with
blocks 0-3 + T13 modules frozen, 100 binding full-scope; AdamW 3e-4) — per `TRAINING_SPEC.json`. It has
near-chance behavior on the 16 cloze records (9/16 "restricted-correct" margins, 0/16 full-vocab top-1, 0/16
exact+EOS) and a large *inherited* non-uniform name prior (Pilot1 corpus counts Alex 85, Owen 7, Mia 1,190,
Nora 0; unrelated-context mean probabilities Mia ~25x Nora, Alex ~4.6x Owen).

**SF1** (`single_fact_acquisition_sf1_seed87011` bundle): 16 single-fact **identity-copy cloze** records:
`BOS + "<Subject> <predicate> the <object>.\nThe person who <predicate> the <object> was" + " <Name>." + EOS`,
where the correct answer is literally the subject of sentence 1 (so Alex-correct and Owen-correct records differ
only in the first ~3 subject tokens). English scope = every `base_model` parameter except `base_model.blocks[0..3]`
(so embeddings, positions, blocks 4-7, final norm, output head train; T13 localizer/retrieval frozen) —
`PINNED_MASKING.py set_scope`. Objective = mean full-vocabulary CE over the **five response/EOS positions only**
(`prepare_example`/`pad_batch`); correct-answer exposure exactly balanced (each name supervised 1,440 times);
no sentence rehearsal; no contrastive/discrimination term; AdamW lr 5e-5; 200 updates = 20 cycles of 9 factual
+ 1 binding; English batch = the 16 records x2.

### 2.2 What happened (verified numbers)

- Per-position correct-target NLL at update 100: **[1.056, 0.0051, 0.0067, 0.0047, 0.0036]** for the four
  response tokens + EOS (`REPORT.md` saved-output diagnosis). Positions 2-4 and EOS are essentially learned; the
  **entire residual difficulty is the first answer token**.
- Endpoint: 12/16 correct/exact; failures are exactly the **4 Owen-correct** records (all output " Alex.").
  Mia/Nora 8/8 with all 4 reversal pairs correct; Alex/Owen 4/8 (all Alex). Binding preserved; language CE
  exploded (3.391 -> 4.653) across *distributed* predictive positions; D3 unrelated-context four-name mass rose
  0.0009 -> 0.068; run stopped by the language cost guard at update 100.
- Owen evidence genuinely strengthened on Owen-correct records (Owen first-token logit +12.15, probability
  +0.27) yet Alex still won all four (readout D4: `STAGE_B_SIGNAL_INCREASED_BUT_SUPPRESSED`).

### 2.3 Mechanism attribution (from the four completed forensics)

| Endpoint | Established attribution |
|---|---|
| Factual selection gain (9->12/16, Mia/Nora reversals) | **Upstream** (embeddings/blocks 4-7), head/norm-independent; requires the block-7 MLP residual in cumulative splice direction; requires the *full answer-generation trajectory* (not only the first-token predictor) for exact+EOS. |
| Unrelated-context name prior (D3) | Late/cumulative, **current-predictive-state dominant** at block 7, **amplified by the SF1 output head** (0.0364 -> 0.0678). Distributed upstream x head interaction. |
| Language regression | Distributed across ordinary predictive positions; largest increments late (block-7 MLP); amplified by head; **not** concentrated at final-record positions. |
| Alex-default / Owen suppression | Non-monotonic identity flips across blocks 4-7 (block-4 MLP favors Owen, block-5 MLP returns to Alex); residual Alex preference **not localized** by component swap; unconditional name boost and conditional Owen evidence are entangled in the same late route. |

**No clean separation** of "useful factual route" from "harmful prior route" exists at any tested token group or
depth. Inference-time patching establishes *causal sufficiency of states*, not *where learning occurred*.

### 2.4 Independent synthesis (why SF1 failed)

[A = established; B = suggested by evidence; C = unknown; see EVIDENCE_LEDGER.md for the full split.]

My reconstruction, kept distinct from what is proven:

1. SF1's objective (CE only on 5 response/EOS positions, over 16 records whose correct answers are drawn from a
   closed set of four names) offers the optimizer **two routes** to reduce mean loss: (a) a *conditional* route —
   gate the first answer token on the subject tokens (needed for Owen and Nora); (b) a *cheap unconditional*
   route — shift the late predictive-state/head mapping so that the four trained names carry elevated logits
   generally. Route (b) raises P(correct) on the items whose correct name is already the model's preferred one
   and, as a shared readout change, needs no per-context discrimination. It does not by itself lower loss on the
   hard items, but it lets the easy items saturate while the hard items stall — consistent with mean training
   loss collapsing to ~0.28 while position-1 NLL on the Owen items stays ~4.2.
2. Route (b) is exactly the observable D3 pollution: the same predictive-state/output change that inflates names
   at factual answer positions also inflates them at **every** ordinary predictive position (hence distributed
   language damage and the D3 mass rise). The head's amplification of the S-upstream state (0.036 -> 0.068) and
   its mild role in language regression match a shared readout-path mechanism.
3. The Alex-over-Owen outcome at Owen records is the *unconditional/default* component out-competing a real but
   weaker *conditional* Owen component (Owen logit +12 but Alex still top-1). Because the pollution mechanism is
   global (position-independent) and the useful mechanism is conditional (subject-gated), the two are *in
   principle separable by training constraint* even though they are entangled in shared parameters.
4. Therefore the demonstrated harms (pollution, distributed language damage, and plausibly the Owen default) all
   trace to **unconstrained drift of the ordinary-context predictive distribution**, which SF1 never penalized:
   no language rehearsal was introduced, and nothing constrained blocks 4-7/head from shifting ordinary
   next-token behavior.

This synthesis (points 1-4) is **B (suggested)**, not A. What is A is narrower: SF1's change was global rather
than conditional (D3 is context-independent), the damage is position-distributed, and the head amplifies it.
The treatment below is chosen to test exactly the point-4 hypothesis while changing as little else as possible.

---

## 3. Does the evidence support training?

**Yes — one bounded treatment, TRAIN_ONE_BOUNDED_TREATMENT.**

Justification:
- The failure is characterized, reproducible, and stopped by a pre-registered cost guard *before* its own
  endpoint; it is not an unexplained plateau or a mechanical defect (all mechanical checks passed).
- The two harms whose mechanism is *established* (D3 name-prior pollution; distributed ordinary-language damage)
  are caused by drift of the ordinary-context predictive distribution during English factual updates. SF1 had
  **no** mechanism opposing that drift. A single additive retention constraint is directly mechanism-matched.
- The natural read-only follow-ups are exhausted or low-information: the last three forensics each ended
  "no treatment choice follows from this diagnostic," and the candidate-pool audit's gating read-only diagnostic
  (upstream localization) is now complete and returned `DISTRIBUTED_OR_INTERACTIVE` (no narrow-boundary
  signature), which under that audit's own decision rule does **not** raise the deferred diversity-training study
  to top priority.
- A training run of the proposed size is cheap (SF1-scale: 200 updates on 16 records) and is *falsifiable* with
  pre-registered signatures; even a null result cleanly distinguishes "pollution can be suppressed without losing
  acquisition" (treatment works) from "pollution and acquisition share an inseparable mechanism" (treatment fails
  acquisition or fails to suppress pollution) — a genuine architectural conclusion no read-only patch can draw.

**What is NOT supported (and therefore not proposed):** architecture changes (any), tied embeddings, frozen
transfer-panel access, removal/weakening of any gate, unlikelihood/discrimination/contrastive losses (every
earlier pairwise/discrimination objective at the binding task failed, and SF1 deliberately excluded them),
candidate-pool/wider-name diversity as the *primary* variable (already `ALREADY_WEAKENED` as a frequency account;
its surviving form is a data-topology hypothesis that the completed localization did not elevate), and simple
LR/update-budget extension (no evidence Owen was improving; the cost guard exists precisely to prevent this).

---

## 4. The proposed treatment (summary; full protocol in PROPOSED_TREATMENT.md)

**Add a single forward-KL-to-frozen-parent retention term on ordinary (non-factual) next-token positions during
the English factual updates of an otherwise-identical SF1 rerun.** One variable changed; everything else
(parent, 16 records, schedule, scope, optimizer, seed, monitoring, gates) identical to SF1.

Rationale in one sentence: the evidence shows SF1's harms are a *position-unconditional* drift of the
ordinary predictive distribution; penalizing exactly that quantity (KL of the model's output distribution
against the frozen Pilot1 parent on a disjoint frozen sample of ordinary English positions, per English update)
removes the free lunch that made the unconditional route cheap, without touching factual data, scope, cadence,
or any gate, and without adding any new labels or data modality.

This addresses the five scientific criteria as follows:
1. **Conditional acquisition / complete answer+EOS** — targeted indirectly but mechanistically: by removing the
   unconditional boost, the model can only reduce factual CE via subject-conditional selection; the conditional
   Owen evidence already exists (logit +12) and must merely outrank Alex. Full answer+EOS is already 12/16 under
   SF1; the retained trajectory supervision is unchanged.
2. **Reversal not global preference** — the Alex default is treated as the unconditional component; suppressing
   it is the hypothesis under test. A remaining concentration of errors on one name is a pre-registered
   falsifier.
3. **Name-prior pollution** — directly penalized on ordinary positions (the D3 mechanism).
4. **Language preservation** — the distributed ordinary-position drift is directly penalized; the aligned-CE cost
   guard is retained unchanged.
5. **Binding** — binding updates and their full-scope rehearsal are unchanged; the KL applies only to English
   updates and only to ordinary positions, and SF1 already preserved binding through update 100.

**Predicted signatures** and **falsifiers** are pre-registered in PROPOSED_TREATMENT.md §10.

**Biggest remaining uncertainty:** whether constraining ordinary-position behavior (shared blocks 4-7 + head)
also suppresses the *conditional* factual gain — i.e., whether the entanglement is learnable-apart or structural.
This is precisely the question the run answers; it cannot be answered by further read-only patching.

---

## Recommendation class

TRAIN_ONE_BOUNDED_TREATMENT
