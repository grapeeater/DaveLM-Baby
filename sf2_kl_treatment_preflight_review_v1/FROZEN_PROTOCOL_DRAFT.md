# FROZEN PROTOCOL DRAFT — `sf2_kl_parent_retention_v1`

**Status: DRAFT for freeze review, incorporating all corrections (M1-M10).** Nothing here has been executed.
A separate authorized engineer must materialize, checksum-freeze, pre-register, and only then run this protocol.
No implementation choice is left open in this document; any item that could not be resolved from authoritative
evidence without changing the scientific treatment would be marked **[BLOCKING]** — none is.

**Scientific content (unchanged from the approved treatment):** SF1 rerun whose only change is an additive
forward-KL retention term, applied during the 180 English factual updates, keeping the model's output
distribution at ordinary (non-factual) TinyStories next-token positions close to the frozen Pilot1 parent.

---

## 1. Identity and scope

- Working id: `sf2_kl_parent_retention_v1`
- Parent run harness: `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\ENGINE.py` (authoritative SF1
  trainer), copied and edited in **one place only**: the English-update loss branch (see §9). All other branches
  (schedule, scope, optimizer, commit, eval, guards, transfer gating, finalize) are byte-identical.

## 2. Checkpoints / tokenizer / runtime (exact)

- Parent (= SF2 init = KL teacher weights): `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt`;
  SHA-256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`.
- Tokenizer: `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json`; SHA-256
  `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.
- Runtime: torch `2.12.0+rocm7.14.0`, Python `3.12.14`, `tokenizers 0.23.1`, CUDA; `configure_runtime(87011)`
  (TF32 off, deterministic algorithms on), `PYTHONHASHSEED=87011`; float32 deterministic; no autocast.

## 3. Teacher (M1)

- Teacher = new `DaveLMV082` instance via `build_model("untied")` (v0_8_2), loaded **only** from the parent
  checkpoint's `base_model.*` keys (`raw["model_state_dict"]` filtered to keys starting with `base_model.`).
- `.eval()`; `for p in teacher.parameters(): p.requires_grad_(False)`; all teacher forwards under
  `torch.no_grad()`. Teacher is never an optimizer parameter and shares no `Parameter` object or storage with the
  student. Verify `id()` and `.untyped_storage().data_ptr()` disjointness for `token_embedding.weight` and
  `language_head.weight`.
- Student = the normal SF1 load path (`rt.load_model(parent, device, bundle)` -> `Treatment13Model` with
  `OrthoLocalizer`), as in `ENGINE.py`.
- Consequence: at update 0 student == teacher in the ordinary path, so KL == 0 exactly (floating point identical
  weights and identical eval forward) -> assert `KL == 0` within 0 tolerance at baseline.

## 4. SF1 elements held byte-identical (no change)

- 16 factual TRAIN records from `single_fact_acquisition_sf1_seed87011\TRAIN.json` (SHA-256
  `08318d68e0d1b005ae340448af9dabb933013af08046a9cf8fd3ab64153ed9dd`), consumed via the frozen schedule
  (`SCHEDULE.json`, SHA-256 `91dad3510e7997b0d55ab53e6189db9337857d582c9399a720723412f70a7ab6`).
- Supervision: `prepare_example`/`pad_batch` from `PINNED_MASKING.py`; CE `F.cross_entropy(logits.reshape(-1,1024),
  y.reshape(-1), ignore_index=-100)` over the five response/EOS positions; mean over the 160 supervised positions.
- Scope: `set_scope(m, binding)` from `PINNED_MASKING.py` (English: train `base_model.*` except
  `base_model.blocks.{0..3}`; T13 `wq/wk/wv/wo/localizer` frozen; binding: full). Scope audit and per-update
  inactive-parameter + optimizer-state equality checks unchanged.
- Cadence: 200 updates = 20 x (9 English + 1 binding). Optimizer: fresh `AdamW(m.parameters(), lr=5e-5, betas=(.9,
  .999), eps=1e-8, weight_decay=.05, amsgrad=False, foreach=False, fused=False)`; clip 2.0; one step per update.
- Rolling restart after every update; permanent checkpoints and full evaluations at 0/100/200 only.
- Binding pools pilot0/pilot1 (`data\binding_dev_pilot0.json`, `data\binding_dev_pilot1.json`), rehearsal pool
  `data\binding_rehearsal.json`; binding objective = pinned answer CE + LAM x localization loss
  (`PINNED_PILOT1_BINDING_IMPLEMENTATION.py`); gates: per pool answer >= 76/80 AND BD >= 76/80 AND collapse == 0
  at 100 and 200.
- Language monitor: aligned CE over the first-128 rows of `data\ENGLISH_DEV.jsonl` (SHA-256
  `2053a8036d7d0f02ba56ab28f739df95a1743760d0f8c980e112d6667b38104e`); stop if `> update0 + 0.25` nats at 100 or 200.
- Transfer panels HELDOUT16/ALTERNATE48/COPY8/COMPETING64 remain frozen and unscored; opened only if endpoint
  acquisition AND language AND binding pass at 200. Readiness FINAL/sacred never accessed.

## 5. Baseline reproduction (M9) — required before the first English step

At update 0, eval/no_grad, assert within declared tolerance:
- Acquisition: 9/16 correct, 0/16 exact (SF1 update-0 values).
- Aligned CE 3.3907 (tolerance 1e-4 relative), D3 four-name mass 0.000907 (tolerance 1e-4 absolute), both binding
  pools gate PASS, KL == 0.
Stop if any check fails (parent/harness/teacher mismatch).

## 6. Retention pool — exact deterministic construction (M2)

- Source: `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\language_train.jsonl`
  (SHA-256 `450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c`), each line `{id,text,token_ids}`
  with `token_ids` authoritative (pinned tokenizer).
- Retain rows in file order with `2 <= len(token_ids) <= 254`.
- Disjointness preflight (assert zero rejections; if any, STOP — contamination): no retained row's `token_ids`
  array equals the `token_ids` of any of the first-128 rows of `ENGLISH_DEV.jsonl`, nor any of the 16 SF1 TRAIN
  `prompt_token_ids` arrays.
- Aligned form per row i: `z_i = [2] + token_ids_i + [3]`, length `L_i = len(token_ids_i) + 2`. Valid labeled
  columns are `j = 0 .. L_i-2` (column j predicts `z_i[j+1]`).
- Pool `P` = row-major ordered list of all `(i, j)` with `j in [0, L_i-2]`, over retained rows in file order.
- Per English update `u in 1..180`: `start = ((u-1)*160) mod |P|`; selected = next 160 entries of `P` (wrap at
  the end of `P`). The same 160 indices repeat only after `|P|/160` updates (never within 180 if |P| >= 28,800;
  assert |P| >= 28,800, else enlarge by retaining more rows).
- Freeze artifacts (SHA-256, before any checkpoint load): `KL_POOL.json` (the materialized `(i,j)` list), a
  `KL_POOL_MANIFEST.json` (source file hash, retention filter counts, |P|, rotation rule), and the per-update
  selection table (or the closed-form rule above). Record hashes in the FREEZE receipt.

## 7. KL term — exact definition (M3, M4)

Per English update u, after `set_scope(m, False)` and before the factual CE:
1. `m.eval()` (dropout off). Select the 160 `(i,j)` entries. Group by row i; for each row run
   `z_s_row = student.base_model(x_row)` with grad enabled and `z_t_row = teacher.base_model(x_row)` under
   `torch.no_grad()`, where `x_row` is the aligned input for that row (`z[:-1]`, padded as in `aligned_tensors`).
2. For every selected column j gather student logits `ls = z_s_row[b, j]` and teacher logits `lt = z_t_row[b, j]`.
3. In float64: `ps = softmax(ls.double())`;
   `kl_pos = sum_v ps[v] * (log_softmax(ls.double())[v] - log_softmax(lt.double())[v])`.
   (Full 1024-class support; temperature T = 1; logsumexp-stable via `log_softmax`.)
4. `KL = mean over the 160 selected positions`, cast to float32 scalar.
5. `m.train()`. Compute the factual CE exactly as SF1 (train mode, dropout 0.05 active).
6. `loss = CE_factual + lambda_kl * KL` with `lambda_kl = 1.0` fixed (see §8). Single `backward()`, clip 2.0,
   one `step()`. Run the frozen-parameter / optimizer-state equality audits exactly as SF1 (teacher params are
   not in the graph's trainable set; blocks 0-3 and T13 modules receive no grad).
Ordering note: the eval KL forward consumes no dropout RNG, so the train-mode factual dropout stream per update is
unchanged relative to SF1's draws.

## 8. lambda_kl (M5)

- Single fixed value `lambda_kl = 1.0`. No sweep, no schedule, no adaptive adjustment, no post-hoc change.
- Pre-registered rationale (replaces the audit's equal-counts claim):
  1. Gradient-scale: `dKL/dz_s = ps - pt`, O(1) per position, same order as the CE per-position gradient; with
     both terms as per-position means over 160 positions, lambda=1 prices the per-position gradient fields
     equally.
  2. Guard/floor: `lambda* = factual-CE floor (~0.2876, SF1 last-10 English mean at update 100) / language-guard
     allowance (0.25) ~= 1.15`; 1.0 is within ~13%.
- Pre-registered readings if a stop trips: language guard trips at 100 -> "lambda=1 too weak" (result, no
  adjustment); acquisition guard trips at 100 -> "lambda=1 too strong / constraint chokes conditional learning"
  (result, no adjustment).

## 9. Harness change (M10)

Single edit to the English branch of `ENGINE.py`:
`loss = CE_factual`  ->  `loss = CE_factual + kl_loss(selected_positions)` per §7.
Add the D3 diagnostic call at updates 0/100/200 (eval/no_grad) using the authoritative
`C:\DaveLM-CADAVER\sf1_readout_selection_forensic_v1\D3_SELECTION.json` manifest and the frozen `sf1_forensic.py`
measurement definition (report four-name mass, per-name logit/rank/prob, entropy, EOS, true-next).
Freeze + SHA-256: modified `ENGINE.py`, teacher-construction snippet, `KL_POOL.json`,
`KL_POOL_MANIFEST.json`, and the D3 measurement code, all before any checkpoint load. Keep all existing
integrity checks (`scope_audit`, `verify`, restart round-trip, alignment prediction equivalence) unchanged.

## 10. Stops / gates — exact Booleans (M6)

- Endpoint gate at 200 (unchanged, all-AND): `correct == 16 AND exact == 16 AND reversals == 8 AND families == 4`.
- Language stop at 100 and 200 (unchanged): stop if `aligned_CE > update0_aligned_CE + 0.25`.
- Binding stop at 100 and 200 (unchanged): stop if any pool fails `answer >= 76 AND BD >= 76 AND collapse == 0`.
- **New additive acquisition stop at 100 (AND):** stop if `correct < 10 AND exact < 10`. Code literally:
  `if (summary["correct"] < 10) and (summary["exact"] < 10): stop_acquisition_guard = True`. Intent: stop only if
  the run is no better than the parent's update-0 9/16 on correctness AND also failing complete decoding.
- Monitoring is eval/no_grad and consumes no RNG; it cannot alter training except through the frozen stops above.

## 11. Fallback (M7) — explicitly out of scope for this run

The head-freeze fallback is a **separate future experiment** requiring its own freeze and pre-registration. It is
not implemented, not reachable by any flag/branch in this run, and must not execute automatically. If this
protocol is judged infeasible for a non-scientific reason, stop and report; do not silently substitute the
fallback.

## 12. Monitoring, persistence, and evaluation cadence (M8)

- Rolling atomic restart after every update (model/optimizer/RNG/provenance), permanent checkpoints and full
  evaluation at 100 and 200, D3 diagnostics at 0/100/200. No 25-step evaluations. All persistence via the pinned
  atomic writers; corrupt provenance stops.

## 13. Predicted signatures (support / falsify)

Support H1 (all pre-registered):
- S1 D3 four-name mass <= 0.01 at 100 (parent 0.0009; SF1 0.0678); mean four-name logit delta <= +1.0.
- S2 aligned CE <= 3.6407 at 100 and 200 (guard passes), far below SF1's 4.65 at 100.
- S3 acquisition >= 12/16 correct and exact at 100; 16/16 correct/exact + 8/8 reversals + 4/4 families at 200.
- S4 >= 1/4 Owen-correct correct at 100 and Owen-correct mean sequence margin >= 0 by 200; wrong answers not
  concentrated on a single name.
- S5 binding both pools >= 76/80 answer and BD, collapse 0, at 100/200.

Falsify / weaken H1:
- F1 acquisition stop trips at 100 (<=9/16 AND <=9/16 exact) -> constraint too strong; entanglement not learnably
  separable at lambda=1.
- F2 language stop trips at 100 despite KL -> KL positions do not bind the damage; drift is not (only)
  ordinary-position drift.
- F3 D3 mass stays >= 0.03 while language passes -> pollution and language damage partly independent.
- F4 acquisition reaches 16/16 but Owen-correct stays 0/4 with all errors on Alex -> Alex default is not the
  unconditional component reachable by ordinary-context KL.

## 14. Interpretation constraints (unchanged from SF1 discipline)

Training-record results do not establish held-out generalization or factual understanding (identity-copy remains a
possible shortcut). Transfer panels open only under §10. No claim that KL or any patch localizes where learning
occurred. Report raw family/reversal/subgroup/D3/position tables, not aggregates only. SF1 comparison valid only at
update 0 and update 100 (SF1 has no update-200).

## 15. **[BLOCKING] items**

None. Every choice above is resolved from authoritative artifacts without changing the scientific treatment.
