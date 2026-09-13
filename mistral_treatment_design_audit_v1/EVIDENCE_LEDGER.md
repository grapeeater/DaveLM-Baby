# Evidence Ledger

Discipline used throughout: **A** = established by artifact; **B** = suggested/consistent with evidence but not
proven; **C** = unknown/not established; **D** = what the proposed treatment specifically tests.
"Established" here means reproduced across at least one authoritative artifact with hashes, and consistent with
the replication tolerances declared in the relevant protocols.

All artifact paths are under `C:\DaveLM-CADAVER\` unless noted. `C:\DaveLM-v0.9\` (top-level) is the current dev
codebase; `C:\DaveLM-CADAVER\DaveLM-v0.9\` is a stale snapshot and was not used as authority.

---

## A. ESTABLISHED

### A1. Training/objective facts (verified in code + frozen JSON)
- SF1 parent = Pilot1 checkpoint, SHA-256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`
  (`PROTOCOL.json`; `sf1_readout_selection_forensic_v1\REPORT.md`).
- Pilot1 = T13 orthogonal-shared-unbounded parent after 1,000 updates (900 language blocks0-3-protected, 100
  binding full-scope), AdamW 3e-4 (`language_pilot_1_early_block_protection_seed8380\pilot_run\TRAINING_SPEC.json`,
  `run.py`).
- SF1 training set = 16 identity-copy cloze records, answer exposure exactly balanced, each record 360
  presentations, each name 1,440 supervised answers (`single_fact_acquisition_sf1_seed87011\TRAIN.json`,
  `SCHEDULE.json`; `sf1_candidate_pool_hypothesis_audit_v1\EVIDENCE.md`).
- SF1 objective = mean full-vococab CE over the five response/EOS tokens only; English scope = every
  `base_model` parameter except `base_model.blocks[0..3]`; T13 localizer/retrieval frozen on English; binding
  full scope; fresh AdamW lr 5e-5, wd 0.05, clip 2.0; 200 updates = 20x(9 factual + 1 binding)
  (`sources\PINNED_MASKING.py`, `PROTOCOL.json`, `OPTIMIZER_METADATA.json`).
- SF1 update-100 checkpoint SHA-256 `550ce4306b450c2bdb64e57314e45ec7db5bb2ee68fc0978ed9d3617f4fc1e7e`
  (`single_fact_acquisition_sf1_seed87011_run\checkpoint_100.sha256`).

### A2. SF1 outcome facts
- Stopped at update 100 by the prospective language cost guard (aligned CE 3.3907 -> 4.6529 > parent+0.25);
  binding fully preserved (both pools 80/80 answer, 80/80 BD, 0 collapse)
  (`single_fact_acquisition_sf1_seed87011_run\STATUS.json`, `update100_checks.json`, `REPORT.md`).
- At update 100: 12/16 correct, 12/16 exact answer+EOS, 2/4 families, 4/8 reversals; failures all Owen-correct;
  Alex/Owen 4/8 (all " Alex."), Mia/Nora 8/8
  (`update100_acquisition_RESULT.json`, `REPORT.md`).
- Per-position correct-target NLL at update 100 ~ [1.056, 0.005, 0.007, 0.005, 0.004]; residual difficulty is
  the first answer token (`REPORT.md` saved-output diagnosis).
- On Owen-correct records Owen logit rose +12.15 / probability +0.27 yet Alex remained top-1
  (`sf1_readout_selection_forensic_v1\REPORT.md` D4; `D4_SUMMARY.json`).
- Pilot1 baseline on the 16 records: 9/16 restricted-correct margins, 0/16 full-vocab top-1, 0/16 exact
  (`update0_acquisition_RESULT.json`; readout D4).

### A3. Forensic attribution facts
- Factual gain is upstream (embeddings/blocks 4-7) and head/norm-independent: all S-upstream conditions 12/16,
  all P-upstream 9/16 regardless of norm/head
  (`sf1_component_swap_forensic_v1\REPORT.md`, `FACTUAL_SUMMARY.json`).
- D3 four-name mass: parent 0.000907 -> SF1 0.067833; S-upstream + P-head = 0.036394 (head amplifies ~1.9x);
  head alone on P-upstream = 0.001786. Factorial: upstream main 0.0507, head main 0.0159, interaction 0.0151
  (`sf1_component_swap_forensic_v1\REPORT.md`, `FACTORIAL_EFFECTS.json`).
- Language regression upstream-dominant with material head interaction: SPP CE 4.2769 (70% of regression), SSS
  4.6529 (`sf1_component_swap_forensic_v1\REPORT.md`).
- Final norm immaterial on tested endpoints (`sf1_component_swap_forensic_v1\REPORT.md`).
- Useful factual signal and the D3 prior both require the late path; largest increments at block-7 MLP; Mia/Nora
  full reversal requires block-7 MLP residual; Alex/Owen flips non-monotonically (block-4 MLP favors Owen,
  block-5 MLP returns toward Alex)
  (`sf1_upstream_localization_forensic_v1\REPORT.md`, `SUMMARY.json`).
- First-token factual choice and D3 inflation are carried by the current/final predictive state; full
  answer+EOS and Mia/Nora reversals additionally require the whole answer-generation trajectory; language damage
  is distributed across ordinary predictive positions; no clean useful-only token route
  (`sf1_token_position_patching_forensic_v1\REPORT.md`, `SUMMARY.json`, `POSITION_PATCH_RESULTS.jsonl`).
- Candidate-pool/frequency account weakened: exposure balanced; inherited Mia/Nora prior (Mia ~25x Nora on
  unrelated contexts) larger than Alex/Owen (~4.6x), yet the former pair succeeded and the latter failed
  (`sf1_candidate_pool_hypothesis_audit_v1\REPORT.md`, `EVIDENCE.md`).
- Broader historical context: the earlier four-name/all-six-pair factual-supervision treatment failed its harder
  two-fact acquisition gate (97/192, 22/96 reversals, 0/24 families) — supporting context, not a matched
  comparison (`sf1_candidate_pool_hypothesis_audit_v1\EVIDENCE.md`, citing the factual-treatment artifacts).
- Binding regression machinery (pool structure, gates) is preserved and unchanged across Pilot1/HR/SF1
  (`single_fact_acquisition_sf1_seed87011\data\binding_*.json`; `sources\hr3_block3_runtime.py` `binding_gate`).

### A4. Integrity facts
- SF1 monitor/`D3` sample = first-128 rows of `data\ENGLISH_DEV.jsonl` (SHA-256 `2053a803...`), 15,866 rows
  total; D3 = deterministic 256 positions, selection receipt `3ae0f6a7...`
  (`sf1_readout_selection_forensic_v1\D3_SELECTION_RECEIPT.json`, `PROVENANCE.json`).
- Pilot1 language corpus hashes: train `450f78bd...`, dev `deff4fc7...`
  (`PILOT1_MATERIAL_MANIFEST.json`).
- Replication tolerances satisfied across forensics (max abs logit error 0.0 at 1e-6; cross-run aggregate
  1e-5) (`BASELINE_REPRODUCTION.json` in component-swap and upstream-localization forensics).

## B. SUGGESTED (consistent, not proven)

- The SF1 change is best described as a **global/unconditional** predictive-distribution shift (context-
  independent name-logit inflation) plus a **partial conditional** subject->name component, rather than as a
  purely relational failure. Basis: D3 is context-independent; readout controls moved nonuniformly; Owen logit
  rose on Owen records (+12) while Alex still won; component-swap places pollution in an upstream x head
  (readout-path) interaction.
- The **Alex default** at Owen records is a competition between an unconditional/default component and a real but
  weaker conditional Owen component (`STAGE_B_SIGNAL_INCREASED_BUT_SUPPRESSED`), rather than an absence of
  learned Owen evidence.
- SF1's five-position mean CE **underweights** the position-1 selection error and hid the stalled Owen items
  (mean train loss ~0.28 while position-1 NLL on those items ~4.2). Whether the mean rule *caused* the stall is
  not established (`single_fact_acquisition_sf1_seed87011_run\REPORT.md` states the same caution).
- The mid-stack (block 4-5) non-monotonic identity flips indicate intermediate layers *do* represent the
  subject identity but later layers re-impose the default — consistent with, but not proof of, an
  output-route/default that dominates a present contextual signal.
- A retention/regularization term on ordinary-context output distributions is the natural countermeasure to a
  position-unconditional drift that no current SF1 component penalizes.

## C. UNKNOWN / NOT ESTABLISHED

- Where learning actually occurred during training (patching establishes only causal sufficiency of states).
- Any held-out generalization from the 16 training items (all transfer panels HELDOUT16/ALTERNATE48/COPY8/
  COMPETING64 remain locked and unscored; readiness/FINAL/sacred untouched).
- Whether SF1 should be called "factual understanding": **no** — at most 12/16 on its own training records with
  an identity-copy interface; not a defensible capability claim.
- The exact cause of the Alex/Owen vs Mia/Nora asymmetry (token geometry, object-family confound, inherited
  prior, shared-suffix structure, or a genuine default mechanism). Existing SF1 data cannot separate these.
- Whether constraining ordinary-context behavior (the proposed KL) will suppress the conditional factual gain
  along with the pollution — i.e., whether the entanglement is separable under a training constraint.
- Whether a simple LR/update-budget extension or added rehearsal would have changed the SF1 outcome (the run
  stopped early by design; unchanged continuation is untested and would require weakening the cost guard, which
  is not proposed).

## D. WHAT THE PROPOSED TREATMENT SPECIFICALLY TESTS

Single manipulated variable: **an additive forward-KL-to-frozen-parent retention term applied at ordinary
(non-factual) next-token positions during SF1-identical English factual updates.**

- D1 (pollution): does penalizing ordinary-context drift reduce D3 four-name mass toward parent without
  sacrificing acquisition? (Target <= 0.01 at update 100; parent = 0.0009, SF1 = 0.068.)
- D2 (language): does it hold aligned CE within the unchanged cost guard (<= parent+0.25) at updates 100/200?
- D3 (Owen/reversal): with the unconditional boost suppressed, does the existing conditional Owen evidence win
  (Owen-correct margins >= 0, reversal pairs 8/8), or does a fixed-default concentration persist (falsifier for
  the "unconditional component causes the default" hypothesis)?
- D4 (acquisition suppression): does the KL leave the blocks 4-7 pathway free to learn the conditional rule at
  factual answer positions (acquisition >= SF1's 12/16 by update 100), or does it choke factual learning
  (acquisition <= 9/16 at 100 -> early stop, KL too strong/ill-targeted)?
- D5 (binding): unchanged binding updates keep both pools at gate (>=76/80, collapse 0) at 100/200.

Falsification logic and the single fallback are specified in PROPOSED_TREATMENT.md.
