# SF2 KL-to-parent retention treatment — FINAL adversarial preflight review v1

**Nature of this document:** read-only preflight. No training, optimizer, autograd, weight mutation, locked-panel
scoring, FINAL/sacred access, gate change, or new mechanistic experiment was performed. This review decides only
whether the proposed SF2 treatment is ready to be frozen and implemented as written, or what must be corrected
first.

**Note on the referenced DeepSeek V4 Flash review:** an exhaustive search of `C:\DaveLM-CADAVER\` (recursive
directory and file-name search), the `C:\` version-sibling tree, user profile/Documents/Temp/Desktop, and a
content search of root-level markdown found **no DeepSeek V4 Flash independent-review artifact anywhere in the
repo** (the only "deepseek" hit is an unrelated AutoGPT frontend image; the only "review" hit is a readiness
rubric unrelated to SF2). Per the mission instruction ("do not rely on assumptions or reconstruct missing
conclusions"), this review does **not** reconstruct that review. It proceeds on the authoritative SF1/Pilot1
artifacts and the Mistral/Codestral treatment-design audit (`mistral_treatment_design_audit_v1\`), which are
present and were read in full.

---

## Headline

**Classification: APPROVE_WITH_MECHANICAL_CORRECTIONS**

The treatment's *science* — a single manipulated variable (add forward-KL-to-frozen-Pilot1 retention on ordinary
TinyStories positions during otherwise-identical SF1 English factual updates) — is coherent, mechanism-matched
to the established D3/language-damage evidence, falsifiable, and does not weaken any gate. It is **not** ready
to freeze as literally written in `PROPOSED_TREATMENT.md`, because several implementation choices are
underspecified or would introduce ambiguity. All identified problems are **non-hypothesis-changing mechanical
corrections**; none requires a scientific change. The corrections are enumerated in `REQUIRED_CORRECTIONS.md`
and are already resolved in `FROZEN_PROTOCOL_DRAFT.md`.

---

## Item-by-item review

### 1. Exact Pilot1 parent checkpoint / tokenizer and provenance — VERIFIED

- Parent: `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt`,
  SHA-256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb` (SF1 `PROTOCOL.json` `parent` /
  `parent_sha256`; reverified at runtime by `ENGINE.py` line 21 and the readout forensic REPORT).
- Tokenizer: `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json`, SHA-256
  `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.
- Teacher identity: the KL teacher is the **same parent checkpoint object** as the SF2 starting weights, so KL == 0
  at update 0. The audit's `PROPOSED_TREATMENT.md` did not say how to instantiate the teacher as a distinct,
  frozen module. **Correction M1** (below) pins this: teacher is a *separate* `DaveLMV082` instance
  (`build_model("untied")`), loaded from the parent checkpoint's `base_model.*` keys, `.eval()`,
  `requires_grad_(False)`, never in the optimizer, forward only under `torch.no_grad()`; no shared `Parameter`
  object or storage with the student.

### 2. Exact SF1 baseline mechanics that must remain unchanged — VERIFIED, with two spec corrections

Confirmed against the authoritative frozen trainer `single_fact_acquisition_sf1_seed87011\ENGINE.py`,
`PINNED_MASKING.py`, `PROTOCOL.json`, `SCHEDULE.json`:

- Factual TRAIN records: 16 (`TRAIN.json`, hash `08318d68e0d1b005...`), consumed from the frozen file; batch = all
  16 twice (32 rows) per English update, order from `SCHEDULE.json`.
- Answer+EOS supervision: `prepare_example`/`pad_batch` (BOS+prompt+4 candidate tokens+EOS; CE with
  `ignore_index=-100` over the five response/EOS positions only).
- Parameter protection: `set_scope` — English trains every `base_model.*` parameter except
  `base_model.blocks.{0..3}`; T13 `wq/wk/wv/wo/localizer` frozen on English; binding full scope. Per-update
  inactive-parameter and optimizer-state equality audits are enforced.
- Cadence: 200 updates = 20 x (9 English + 1 binding); rolling restart every update; permanent checkpoints and
  evaluations at 0/100/200 only.
- Optimizer/LR: fresh `AdamW` over `m.parameters()` lr 5e-5, betas (0.9, 0.999), eps 1e-8, wd 0.05, amsgrad
  False, foreach False, fused False; clip 2.0; deterministic float32 (TF32 off), torch `2.12.0+rocm7.14.0`,
  seed 87011, `PYTHONHASHSEED=87011`.
- Binding gates: each of pilot0/pilot1 pools independently answer >= 76/80, BD >= 76/80, collapse == 0, at 100
  and 200; stop on failure.
- Language regression stop: aligned CE on first-128 rows of `data\ENGLISH_DEV.jsonl` <= update-0 value + 0.25
  nats at updates 100 and 200; stop if exceeded.
- Locked panels: HELDOUT16/ALTERNATE48/COPY8/COMPETING64 frozen; opened **only** if endpoint acquisition AND
  language AND binding all pass at 200. Readiness FINAL/sacred never accessed.
- **Key mechanical facts the audit left implicit:** (a) the factual forward is in **train mode** (`m.train()`
  each update) and the model has **dropout 0.05** on embedding/attention/residual (`v0_8\config.py` MODEL_CONFIG),
  so the factual loss is stochastic; (b) all monitoring is **eval + no_grad** (aligned CE, binding, panel, greedy
  decode), so the monitor and the training objective live in different modes. Corrections M3/M4 below make the KL
  term's mode explicit and make the harness reuse `ENGINE.py` with a single edit so none of the above drifts.

### 3. Mathematical audit of the proposed KL — MOSTLY SOUND, must be pinned exactly

- **Direction:** forward KL `D_KL(P_student || P_teacher)` is the correct choice to suppress the demonstrated
  failure mode (student mass on tokens the teacher deems near-impossible, i.e., the four-name inflation on
  ordinary contexts). Preserve.
- **Support:** full 1024-class softmax/log-softmax over logits of the untied language head; `p_p(v) > 0` for all v
  with finite logits, so forward KL is finite by construction. Preserve.
- **Frozen/detached teacher:** must be a separate module (M1), eval/no_grad. On-the-fly recomputation each
  English update is deterministic because teacher weights are frozen and inputs fixed.
- **Temperature:** none was specified; pin **T = 1** (log-softmax of raw logits). No temperature is a hidden
  variable once stated.
- **Numerical stability:** compute via `log_softmax` (logsumexp-stable) and accumulate the reduction in float64
  after casting both logits tensors to float64; cast the final scalar back to float32 to join the CE term. This
  avoids cancellation in `p*(lp_s - lp_t)` over 1024 classes.
- **Reduction/averaging:** mean over the KL positions actually used in that English update (divide by the number
  of selected positions, 160), to match the CE term which is `F.cross_entropy(..., ignore_index=-100)` default
  mean over its 160 supervised positions. Both terms are then mean nats-per-position.
- **Token positions receiving KL:** ordinary next-token positions inside ordinary TinyStories rows of the KL pool
  only — never factual answer positions, never monitor rows, never binding documents, never D3 positions.
- **Combination with factual loss:** `L_total = CE_factual + lambda_kl * KL`, single graph, one `backward()`,
  then the unchanged clip(2.0) and one `step()`. Gradients flow to exactly the same student parameters the CE
  term reaches (blocks 4-7, embeddings, positions, final norm, head); frozen params and teacher receive none.
- **Gradients reach only SF2 student params:** guaranteed by teacher no_grad/requires_grad(False) and by reusing
  the SF1 `assert all(x.grad is None ...)` checks. Sound.

### 4. Critical audit of lambda_kl = 1.0 — VALUE RETAINED, RATIONALE MUST BE REPLACED (Correction M5)

The audit's stated rationale ("equal position counts 160:160 therefore lambda=1.0") is **not by itself
defensible**: CE and forward-KL are different functionals with different per-position scales, so equal counts do
not make equal weights principled.

Two independent *prospective* anchors nevertheless support **lambda_kl = 1.0** (no sweep; single fixed value):
1. **Gradient-scale anchor.** `∂KL/∂z_s = p_s − p_t`, bounded in magnitude by 2 per logit and O(1) per position,
   the same order as the CE per-position gradient `(p_s − 1_{correct})`. With matched position counts and per-
   position means, `lambda = 1` prices the two per-position gradient fields equally. This is the natural
   dimensionless scale.
2. **Guard/floor anchor.** The only existing numeric quantities that couple the two terms are SF1's measured
   factual CE floor (~0.2876 mean-per-position over the last 10 English updates at update 100) and the language
   guard allowance (0.25 nats). Pricing a full guard-scale ordinary drift (0.25 nats) equal to the factual floor
   gives `lambda* = 0.2876/0.25 ≈ 1.15`, i.e., `lambda = 1.0` is within ~13% of the anchored value.

Decision: **keep `lambda_kl = 1.0`**, pre-register both anchors, and explicitly forbid any within-run or
post-hoc adjustment. If the language guard still trips at update 100, the reading is "KL at lambda=1 is too weak"
(a result, not a license to raise lambda in the same run); if the acquisition guard trips, the reading is
"lambda=1 too strong / constraint chokes conditional learning." Both outcomes are interpretable and falsify or
support H1 in pre-registered directions.

### 5. Retention pool — must be replaced by ONE exact deterministic construction (Correction M2)

The audit's pool rule was vague ("first N rows", "(k*37)%500", "base_offset_k"). `FROZEN_PROTOCOL_DRAFT.md` §6
replaces it with an exact, arithmetic, no-RNG construction: rows = retained lines of
`language_pilot_1_early_block_protection_seed8380\language_train.jsonl`
(hash `450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c`) with `2 <= len(token_ids) <= 254`, in
file order, with a preflight assertion that no retained row's token_ids equals any of the first-128 monitor rows'
token_ids or any SF1 TRAIN prompt array (expected: zero rejections). The pool `P` is the row-major list of
`(row, col)` next-token positions (col 0 .. L_i-2, i.e., all labeled columns of the aligned `BOS+ids+EOS`
sequence). Per English update `u`, KL positions = the next 160 entries of `P` by fixed rotation
`start = ((u-1)*160) mod |P|`. Pool is **ordinary TinyStories language only**, is disjoint from D3 positions and
the language-monitor rows (different file: train vs dev) and from the 16 factual records, and is **materialized
and SHA-256-hashed before any checkpoint is loaded**. This resolves the ambiguity completely.

### 6. Stop/gate Boolean audit (AND vs OR) — CORRECT SEMANTICS, MUST BE WRITTEN EXPLICITLY (Correction M6)

- SF1 endpoint gate `acquire()` is all-AND (16/16 correct AND 16/16 exact AND 8/8 reversals AND 4/4 families);
  unchanged.
- Language stop: stop if `(language_loss > update0_language_loss + 0.25)` at update 100 or 200 — unchanged.
- Binding stop: stop if any pool fails `(answer>=76 AND BD>=76 AND collapse==0)` — unchanged.
- **New additive acquisition guard** (the audit's only new stop): the stated intent is "stop only if the KL has
  fully blocked acquisition (no better than the parent's update-0 9/16)." The correct Boolean is **AND**, not OR:
  `if (correct < 10) AND (exact < 10): STOP`. With OR, a healthy run that has restored correctness (12/16) but
  not yet exact decoding (0/16) at the mid-point would be killed; with AND it continues. Document that `exact`
  is a strict subset signal, so AND stops only when the run is simultaneously at/below parent on correctness and
  failing complete decoding. Code must literally encode `(correct < 10) and (exact < 10)`.
- No other stop/gate changes; the new guard is purely additive and cannot weaken any existing gate.

### 7. Monitoring cannot alter training adaptively — VERIFIED, with one constraint (Correction M4)

All SF1 monitoring is eval + no_grad and consumes no RNG (deterministic aligned CE, argmax greedy decode,
deterministic binding eval). SF2 adds only: (a) frozen D3 diagnostics (four-name mass, per-name logits/ranks,
entropy, EOS, true-next) at updates 0/100/200, reusing the authoritative `D3_SELECTION.json` manifest and the
frozen `sf1_forensic.py` measurement definition; (b) the KL-term forward each English update, which must be run
in **eval mode with grad enabled** (Correction M4). Running the student KL forward in eval mode (a) matches the
eval-mode monitor that defines the guard and the D3 metric, and (b) consumes zero dropout RNG, so it does not
perturb the train-mode factual dropout stream relative to SF1's per-update draws. No monitoring output feeds any
adaptive mechanism; the only training-visible effect of monitoring is the prospectively frozen stop Booleans.

### 8. Fallback must NOT auto-execute — FENCED (Correction M7)

The audit's §11 fallback (freeze the output head during English updates) is described as "used ONLY if the
primary is judged infeasible," which is a *reviewer judgment before freeze*, not an in-run trigger — but it must
be fenced explicitly so no code path can select it. `FROZEN_PROTOCOL_DRAFT.md` §11 declares the fallback a
**separate future experiment** requiring its own freeze/pre-registration; the SF2 run contains no branch,
flag, or dynamic switch to it, and it is not implemented in the harness.

### 9. Evidential limits — PRESERVED

Restated and binding in `FROZEN_PROTOCOL_DRAFT.md`: results are SF1-TRAIN-family behavior only; no held-out or
generalization claim follows from the 16 training records; the transfer panels remain locked under the SF1 rule;
the activation/position-patching forensics established inference-time causal sufficiency of states, **not** the
location of learning, and SF2 does not claim otherwise; SF2's success on training items would not by itself
constitute "factual understanding" (identity-copy remains a possible shortcut), consistent with SF1's own
interpretation clause.

### 10. Hidden second variable / contamination / invalid baseline / gate weakening scan — RESULTS

- **Hidden second variable:** none found *provided* the harness is `ENGINE.py` with a single edit in the English
  branch and M1-M7 are applied. Two latent risks were identified and removed by correction: (i) the audit left
  KL student mode unspecified (train vs eval) — if run in train mode the KL would consume dropout RNG and shift
  the factual dropout stream *and* regularize a different (stochastic) distribution than the eval-mode monitor;
  resolved by M4 (eval mode). (ii) the audit added "checkpoint/eval every 25 updates," which is *not* SF1's
  cadence — removed (M8); cadence stays exactly 0/100/200 + rolling restart.
- **Contamination path:** KL pool is from the Pilot1 *train* corpus; monitor and D3 are from the SF1 *dev*
  extraction (`ENGLISH_DEV.jsonl`). Files are disjoint; a preflight exact-token-array disjointness check is added
  (M2). Using the parent's own training corpus as the retention target is *by design* (we retain the parent's
  behavior on its training distribution; the monitor then tests whether that retention transfers to held-out DEV
  text). This is not label leakage or evaluation contamination because the KL pool is never scored as a
  generalization measure.
- **Invalid baseline comparison:** SF2's valid comparators are SF1 update-0 (reproduce: acquisition 9/16,
  aligned CE 3.3907, D3 four-name mass 0.000907, binding gates PASS — required before the first English step) and
  SF1 update-100 (12/16, CE 4.6529, D3 0.0678). SF1 has **no** update-200 measurements (it stopped at 100), so
  no SF2-vs-SF1 claim is possible at 200; SF2's update-200 endpoint is its own pre-registered gate. Documented to
  prevent an invalid comparison.
- **Gate weakening:** none. All SF1 gates, guard, monitor, and transfer-opening rule are unchanged; the only
  added stop is stricter (early stop on fully-blocked acquisition).
- **Optimizer/graph subtlety:** KL adds a second forward and an additive loss before the single backward; the
  clip is applied to the summed gradient, exactly as SF1's structure intended; frozen-parameter and optimizer-
  state equality audits still hold because teacher is separate and frozen params still receive no grad.

---

## Classification

**APPROVE_WITH_MECHANICAL_CORRECTIONS**

Plain-English reason: the treatment is scientifically sound and its single-variable logic is preserved, but it is
not yet freezable as written because the KL pool, KL mode/math, lambda rationale, new acquisition-guard Boolean,
and fallback fencing are under-specified or ambiguous in `PROPOSED_TREATMENT.md`. All of these are mechanical,
non-hypothesis-changing corrections already resolved in `FROZEN_PROTOCOL_DRAFT.md`; none requires a scientific
revision, a new experiment, a different treatment, or access to locked material.
