# Proposed Treatment — `sf2_kl_parent_retention_v1` (PROPOSAL ONLY; NOT EXECUTED)

This document is a complete, implementation-ready protocol for one bounded prospective training treatment. It is
a **proposal**: nothing here has been built, frozen, or run. A separate engineer/scientist operating under an
authorized experimental mandate must first materialize, freeze (checksums), pre-register, and only then execute
this protocol.

**Design rule:** change exactly **one** causal variable relative to SF1 (`single_fact_acquisition_sf1_seed87011`)
— add a retention term to the English-factual objective. Every other element is copied verbatim from SF1 so the
result is interpretable as the effect of that single variable.

---

## 1. Treatment identity

- **Working id (prospective):** `sf2_kl_parent_retention_v1`
- **One-line description:** SF1 rerun whose English-factual loss is `L = CE_factual + 1.0 * KL`, where KL is the
  forward KL of the training model's output distribution against the **frozen Pilot1 parent** output
  distribution, averaged over a fixed, frozen, disjoint sample of **ordinary (non-factual) next-token
  positions** drawn from the Pilot1 language-train corpus, applied only on the 180 English (factual) updates.
- **Scientific question it answers:** Does constraining ordinary-context predictive drift (the demonstrated D3
  pollution and distributed language-damage mechanism) force subject-conditional acquisition and thereby
  (a) reduce name-prior pollution, (b) preserve ordinary language within the unchanged cost guard, (c) let the
  already-present conditional Owen evidence win, and (d) leave binding untouched — without suppressing the
  factual gain itself?

## 2. Parent checkpoint (exact)

- Same as SF1: `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt`
- SHA-256: `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`
- Tokenizer: `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json`, SHA-256
  `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`
- Runtime: torch `2.12.0+rocm7.14.0`, deterministic settings identical to SF1/HR3 runtime
  (`configure_runtime` in the pinned runtime: TF32 off, deterministic algorithms on, seed 87011).

## 3. Frozen / trainable parameter policy (identical to SF1)

Use the pinned scope verbatim (bundle `sources\PINNED_MASKING.py`, `set_scope(model, binding)`):
- **English (factual) updates:** train every `base_model` parameter whose name does NOT begin with
  `base_model.blocks.{0,1,2,3}.` — i.e. token embeddings, position embeddings, blocks 4-7, final norm, and the
  untied output head. Blocks 0-3 and the T13 special modules (`model.wq, wk, wv, wo`, `model.localizer`) are
  frozen (`requires_grad=False`).
- **Binding updates:** full model trainable (all 334 parameter-set entries), identical to SF1.
- Frozen-parameter and optimizer-state inactivity checks on every English update, exactly as SF1
  (`PROTOCOL.json` scope/persistence clauses).
- **No output-head freeze** in this treatment (that is the fallback, §11, not the primary). The head must stay
  trainable so the only difference from SF1 is the KL term.

## 4. Dataset construction / balancing / counterfactual structure (identical to SF1)

- Factual records: the exact 16 SF1 `TRAIN.json` records (four Alex/Owen + four Mia/Nora per predicate/object
  cell structure; two assignment reversals per predicate/object; identity-copy cloze; candidates not in input).
- Schedule: SF1 `SCHEDULE.json`; 200 updates = 20 cycles of 9 English (factual) + 1 binding; English batch = the
  16 records x2 (32 rows), each record twice per English update; no resampling.
- **No** distractor candidate is added to inputs or labels (unchanged from SF1).

## 5. Answer/EOS supervision (identical to SF1)

`prepare_example`: `BOS(2) + prompt_token_ids + candidate_token_ids + EOS(3)`; `x = seq[:-1]`;
`y = -100` on context positions then `candidate_token_ids + [EOS]`. Mean full-vocabulary cross-entropy over the
five response/EOS positions per row. No contrastive term, no prior subtraction, no length normalization.

## 6. The KL retention term (THE ONE NEW VARIABLE)

**KL sample pool (must be frozen before training, disjoint from every monitor/gate):**
- Source rows: `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\language_train.jsonl`
  (SHA-256 `450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c`), the Pilot1 language-training
  corpus. This file is entirely disjoint from the SF1 language monitor source
  (`single_fact_acquisition_sf1_seed87011\data\ENGLISH_DEV.jsonl`, SHA-256 `2053a803...`) whose first 128 rows
  define the aligned-CE monitor and the D3 positions. This keeps both the aligned-CE gate and the D3 diagnostic
  pristine (they are never touched by the KL loss).
- Selection rule: deterministic, **no RNG** (mirror the D3 selection rule style). From the first N rows of
  `language_train.jsonl`, select row indices and within-row next-token positions by a fixed arithmetic rule
  (e.g., row `r = (k*37) % 500`, position `p = base_offset_k`, skipping rows/positions that would exceed the
  context window 256 or whose aligned row is byte-identical to any of the first-128 `ENGLISH_DEV` monitor rows).
  Materialize ~2,560 ordinary positions total. Hash the selection manifest (rule + resolved
  row/token-id/position list) and the resolved input tensors **before** loading any checkpoint; record the SHA-256
  in the pre-registration and FREEZE_RECEIPT.
- Per English update: consume the next **160 KL positions** (round-robin, deterministic), i.e., the same number
  of supervised positions as the factual CE (32 rows x 5), so the two terms are on a comparable per-position
  scale. Rows are batched (<=32 rows) through the ordinary path (`model.base_model`) exactly as the aligned
  language loss is computed.
- **KL definition (direction matters):** forward KL `D_KL(P_model || P_parent)` averaged over the 160 positions,
  where both distributions are the full 1024-class softmax of the logits at that position. `P_parent` is computed
  with the **frozen Pilot1 parent** under `torch.no_grad()` (on-the-fly, deterministic; no cached logits file is
  required, but if cached, the cache must be checksummed at creation). Forward KL is chosen deliberately: it
  penalizes the model placing probability mass where the parent places little (exactly the observed name-token
  inflation on ordinary contexts), while tolerating the model being more diffuse than the parent.
  - Gradient flows only into the training model. Parent forward is detached.
  - KL is computed only at **ordinary positions inside ordinary sentences** — never at factual answer positions,
  never on the monitor rows, never on binding documents.
- **Weight:** a single pre-registered scalar, **`lambda_kl = 1.0`**, justified as follows: with position counts
  matched (160 KL positions vs 160 factual CE positions) both terms are mean nats-per-position, so
  `lambda_kl = 1.0` prices one nat of ordinary-context divergence equal to one nat of factual CE error. No sweep.
- **When applied:** every English (factual) update 1..180. Never on binding updates. Binding updates remain pure
  SF1 binding rehearsal.

## 7. Language rehearsal / replay

- **None added** (same as SF1). The KL term is the only language-retention mechanism in this treatment; adding
  sentence CE rehearsal would be a second variable and is deliberately excluded. Rationale: KL-to-parent is a
  softer, better-calibrated retention signal than CE-on-language and is the minimal test of the hypothesis.

## 8. Binding rehearsal cadence (identical to SF1)

- 20 binding updates (one per cycle, after each block of 9 factual updates), binding pool unchanged (80 quartets
  / 320 documents, `binding_rehearsal.json`), each quartet scheduled twice, objective = exact pinned answer CE +
  `LAM` * permutation-invariant localization loss (`PINNED_PILOT1_BINDING_IMPLEMENTATION.py`), full scope. Same
  frozen binding schedule seeds as SF1.

## 9. Optimizer / LR / update ceiling (identical to SF1)

- Fresh AdamW: lr 5e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0.05, amsgrad=False, decoupled WD; no
  scheduler; clip gradient norm 2.0 over gradient-bearing parameters. Seeds: Python/torch/CPU/GPU 87011,
  `PYTHONHASHSEED=87011`; English-order seed 87012 and binding-order seed 87013 as in SF1
  (`PROTOCOL.json` schedule). Update ceiling = 200. Deterministic float32, no autocast/TF32.

## 10. Prospective safety stops, gates, monitoring, and cadence (unchanged SF1 + one additive guard)

- **Language cost guard (unchanged):** at updates 100 and 200, aligned sentence CE on the first-128 rows of
  `data\ENGLISH_DEV.jsonl` must be <= (update-0 value + 0.25 nats). Stop if exceeded.
- **Binding gates (unchanged):** at updates 100 and 200, each pool independently: answer >= 76/80, BD >= 76/80,
  collapse == 0. Stop on failure.
- **Additive acquisition guard (new, does not weaken any existing gate):** at update 100, if acquisition
  correctness < 10/16 AND exact answer+EOS < 10/16 (i.e., no better than the parent's update-0 9/16), stop early:
  this is the pre-registered signature that the KL is choking factual learning.
- **Monitoring (unchanged):** acquisition panel (16/16 correctness, 16/16 exact+EOS, 8/8 reversals, 4/4
  families), aligned CE, and both binding pools at updates 0/100/200; additionally record the **pristine D3**
  four-name mass and mean four-name logit delta at 0/100/200 as a diagnostic (never a gate). Checkpoint every 25
  updates (rolling) + permanent at 100 and 200, with atomic fsync persistence exactly as SF1.
- **What remains locked until prerequisites pass:** SF1 transfer panels `HELDOUT16`, `ALTERNATE48`, `COPY8`,
  `COMPETING64` stay frozen and unscored; they may be opened **only if** the endpoint acquisition gate AND
  language AND binding checks pass at update 200 (identical rule to SF1). Human-readiness `FINAL` and sacred
  material are never accessed. No gate is weakened or moved.
- Required baseline reproduction (update 0, before any step): acquisition 9/16, aligned CE 3.3907 +/- declared
  tolerance, D3 mass 0.000907 +/- tolerance, both binding pools 80/80. If reproduction fails, stop (parent or
  harness mismatch).

## 11. Fallback (used ONLY if the primary design is judged infeasible)

The primary design is feasible only if (i) an admissible disjoint ordinary-corpus KL pool exists (yes:
`language_train.jsonl`, allowed/non-sacred, disjoint from the monitor) and (ii) computing on-the-fly frozen
parent logits does not violate any deterministic-run constraint (it does not). If a reviewer nevertheless judges
the primary infeasible, the single fallback is:

- **Freeze the output head (and final norm) during English factual updates only** (scope change; all else
  identical to SF1), based on the component-swap evidence that the S head amplifies the S-upstream state
  (D3 0.036 -> 0.068) and the factual gain survives upstream-only change (SPP = 12/16 exact). Predicted effect
  (from the swap table, not from training): retain ~12/16 factual, D3 ~0.036 (halved but still elevated),
  aligned CE ~4.28 (improved but still over the 3.64 guard). This fallback is expected to be **insufficient** on
  language retention precisely because most regression is upstream — which is *why* it is not the primary.

## 12. Exact hypothesis and predicted signatures

**Hypothesis H1 (target of the run):** SF1's name-prior pollution and distributed language damage are driven by
an *unconstrained, position-unconditional* drift of the ordinary predictive distribution during English factual
updates, and this drift is *learnably separable* from the subject-conditional factual mechanism. Adding a forward
KL-to-parent on ordinary positions removes the benefit of the unconditional route and forces conditional
acquisition.

**Signatures that would SUPPORT H1 (pre-registered):**
- S1 (pollution): D3 four-name mass at update 100 <= 0.01 (parent 0.0009; SF1 0.0678), mean four-name logit
  delta <= +1.0.
- S2 (language): aligned CE at updates 100 and 200 <= 3.6407 (parent+0.25) and monotonically far below SF1's
  4.65 at update 100.
- S3 (acquisition retained): >= 12/16 correctness and exact at update 100, and 16/16 + 16/16 exact + 8/8
  reversals + 4/4 families at update 200.
- S4 (reversal not global preference): at update 100 at least 1/4 Owen-correct records correct (SF1 had 0/4) and
  the mean Owen-correct sequence margin is >= 0 by update 200; wrong-answer mass is not concentrated on a single
  name.
- S5 (binding): both pools >= 76/80 answer and BD, 0 collapse at 100/200.

**Signatures that would WEAKEN/FALSIFY H1:**
- F1: acquisition at update 100 <= 9/16 and no better at 200 (KL chokes conditional learning) -> the ordinary/
  factual entanglement is not learnably separable under this constraint; KL too blunt for shared blocks 4-7/head.
- F2: aligned CE still exceeds the guard at 100 despite KL (KL positions don't bind the actual damage) -> damage
  mechanism is not (only) ordinary-position drift.
- F3: D3 mass stays high (>=0.03) while language passes -> pollution and language damage are partly independent;
  the KL is targeting the wrong subspace.
- F4: acquisition reaches 16/16 but Owen-correct remains 0/4 with all wrong answers on Alex -> the Alex default
  is *not* the unconditional component but a conditional/default property of the g0 family that ordinary-context
  KL cannot reach; escalation would need a factual-position-level intervention instead.

**Interpretation constraints (mandatory):** training-record results alone do NOT establish held-out factual
generalization or factual understanding; do not open transfer panels unless the §10 rule passes; do not claim the
KL "localizes" any training-time mechanism; report raw family/reversal/subgroup tables, not aggregates only.

## 13. Cost estimate

SF1-scale: 200 updates on 16 records, batch 32, context <= 256, model 10.6M params, one extra frozen-parent
forward over ~160 ordinary positions per English update (~180 extra small forwards total). Expected runtime well
under the SF1 run's cost; no architectural or data-scale change.
