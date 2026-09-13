# SF2 update-100 replay-mismatch autopsy — why the historical SF2 state cannot be reproduced exactly

**Scope:** strictly forensic, read-only. NO SF4 design, NO treatment, NO annealing, NO transfer panels, NO
FINAL/sacred access, NO checkpoint mutation. The SF3 classification
`SF3_HARD_STOP_UPDATE100_REPLAY_MISMATCH` and the frozen SF2 classification `SF2_ACQUISITION_FAIL` are preserved
and not reinterpreted. The only new computation performed was the mission-authorized diagnostic: two replays of
the frozen SF2 engine (first 200 updates each, using the u100 checkpoints) plus read-only tensor/metric
comparisons.

## Headline

**The historical SF2 update-100 parameter state cannot be reproduced exactly because the execution environment's
per-step training numerics changed between the historical SF2 run and every later run on this machine. The
current environment is deterministic against itself, but it is not numerically identical to the historical
environment. SF3's hard stop was correct and unavoidable; SF3's own replay was bit-exact with the frozen SF2
engine's replay under the current environment.**

## Evidence summary

| Comparison (u100 checkpoints, 414 tensors) | Bitwise equal | Mismatched tensors | Max abs delta | RMS delta |
|---|---|---|---|---|
| Replay-A vs Replay-B (same frozen SF2 engine, current env) | **YES** | 0 | 0.0 | 0.0 |
| Replay-A vs SF3 replay (current env) | **YES** | 0 | 0.0 | 0.0 |
| Historical SF2 vs Replay-A | no | 334 | 9.047e-4 | 4.19e-6 |
| Historical SF2 vs Replay-B | no | 334 | 9.047e-4 | 4.19e-6 |
| Historical SF2 vs SF3 replay | no | 334 | 9.047e-4 | 4.19e-6 |

- All four runs produced byte-identical update-0 artifacts (acquisition, D3, checks) — pre-training forwards are
  bitwise reproducible.
- The first logged metric divergence between historical and current replays is **update 2, KL field, delta
  1.397e-9**. Update 1 is byte-identical across all fields (including grad_norm).
- The 334 divergent tensors are exactly the trainable parameters; the 80 bit-exact tensors are the non-parameter
  per-head causal mask buffers (verified equal to the parent in both runs).
- Divergence magnitude is largest in the English-updated groups (token embedding 9.0e-4, position embedding
  3.6e-4, blocks4 8.5e-4, blocks5 6.1e-4, blocks6 3.8e-4, blocks7 1.9e-4, T13 special 2.1e-4, language head
  2.7e-5, final norm 2.4e-6) and smaller in binding-only-updated groups (blocks0-3 ≈ 4.7-8.6e-5). Both update
  classes diverge; nothing is isolated to one module.
- Behavior is unaffected at the level measured: replay acquisition counts match historical SF2 u100 exactly
  (11/16 correct, 10/16 exact, 3/8 reversals, 1/4 families).

## Answers in order

### 1. Does historical SF2 checkpoint_100 preserve everything needed for exact continuation?
**No.** `checkpoint_100.pt` contains only `model_state_dict`, `update`, `provenance`. It has **no optimizer
state, no Adam moments, no RNG state, and no scheduler state** (no scheduler exists in this design). The rolling
`restart.pt` is overwritten every update and only the **latest** state survives (completed=200 for SF2;
completed=100 for SF3's own run). Therefore exact *continuation* from historical u100 was impossible from
surviving artifacts; the SF3 protocol's design (replay updates 1-100 from the untouched parent, then compare)
was the correct available approach — but see answer 7.

### 2. Historical vs replay mismatch profile
See table above and `TENSOR_DELTA_ANALYSIS.json`. 334/414 tensors differ (all trainable parameters); max abs
9.047e-4 in `token_embedding`; mean abs 1.08e-6; RMS 4.19e-6. The mismatch concentrates by *magnitude* in
English-updated parameters (embeddings, blocks 4-7), but no parameter group is exempt — blocks 0-3 and T13
modules (updated only by the 20 binding updates) also diverge, at roughly one-tenth the magnitude. The 80 exact
tensors are mask buffers, not parameters.

### 3. Self-reproducibility test (Replay-A vs Replay-B)
**Bitwise identical**: all 414 tensors, max abs 0.0, zero differing metric records across all 200 updates.
Additionally Replay-A equals the SF3 replay's u100 checkpoint bitwise. The current runtime is deterministic
against itself, and the SF3 controller's updates 1-100 are numerically identical to the frozen SF2 engine's.

### 4. Since A == B exactly but both differ from historical SF2:
Classification: the historical trajectory is not recoverable by current replay
(see CLASSIFICATION.json). Concrete differences between historical and current execution searched for and
excluded: code (frozen SF2_ENGINE.py hash verified unchanged), schedule/data/KL-pool/tokenizer/parent (all
hashes verified identical), interpreter and library versions (identical venv, Python 3.12.14, torch
2.12.0+rocm7.14.0, tokenizers 0.23.1), deterministic flags (TF32 off, deterministic algorithms on), seeds
(87011) and PYTHONHASHSEED (87011), GPU (AMD Radeon RX 9060 XT), and source hashes of the copied SF1 bundle used
by both runs. **No recorded difference exists**; the divergence must come from an unrecorded, persistent,
environment-level numeric state change (e.g., ROCm kernel selection/autotuning cache or driver state) between the
historical run (~01:07 local) and all later runs (~02:24+ local). A filesystem scan for driver/kernel-cache
artifacts modified in that window found nothing, so the specific low-level cause is **unidentified** — but the
environment-level origin is established by elimination: identical inputs, identical code, identical recorded
runtime, mutually-identical replays, one unique historical sample.

### 5. A != B? Not observed.
Replay-A and Replay-B are bitwise identical; there is no A/B divergence to localize. The earliest *historical-vs-
current* divergence is update 2 (KL 1.397e-9), with update-1 and update-0 artifacts bit-identical — consistent
with a per-step numeric difference entering at the first optimizer step, not with any RNG/mask/data difference.

### 6. Not dismissed as floating-point noise.
Quantified: 334/414 tensors differ; max 9.047e-4; RMS 4.19e-6; per-component maxima in
TENSOR_DELTA_ANALYSIS.json. These are small in absolute terms but real, systematic (every trained parameter
diverges), and incompatible with the exact-equality gate. Behavior-matched aggregates (11/16 etc.) do not
overturn the gate; the gate's purpose was exactly to prevent mixing a new treatment with a non-reproduced
trajectory.

### 7. Exact replay of historical SF2 is:
**B — impossible from surviving historical state under the current environment**, with one unresolved
sub-element (the specific changed kernel/driver state is unidentified; see RUNTIME_PROVENANCE_COMPARISON.json).
It is not A: no mechanical correction is identifiable from surviving artifacts (there is no recorded difference
to correct, and historical u100 optimizer/RNG state was never preserved). It is not merely C: the current
environment has now been demonstrated deterministic against itself twice, and three independent current replays
(A, B, SF3) all converge to the same non-historical trajectory.

### 8. Consequence for any future run
Do not resume SF3 or train any treatment here. The scientifically clean prospective family is: **fresh
matched-control branches from a newly frozen common trajectory** — freeze a single fresh replay once, verify
self-consistency (two replays bitwise identical in the current session environment), and branch control vs
treatment from that fresh common state so both arms share the identical environment numerics. An
"exact-tensor-match against a historical checkpoint" gate is not achievable on this hardware/stack and should not
be reused as an SF3/SF4 prerequisite. (Family-level recommendation only; no design, no implementation.)

## What this means for the annealing hypothesis
**Nothing directly — it remains UNTESTED** (zero annealed updates ran). The hard stop preserved that status
correctly. Comparative-model background evidence (Granite's power schedule; Qwen3's staged mixing) is unchanged
and remains weak background only.

## SF3's frozen classification
`SF3_HARD_STOP_UPDATE100_REPLAY_MISMATCH` is **correct and upheld**: at update 100, exact tensor reproduction
failed (334/414), the gate fired, zero annealed updates ran. This autopsy additionally establishes that SF3's
replay was a faithful current-environment replay of the frozen SF2 controller — the mismatch was environmental,
not an SF3 implementation error.

## Artifacts
All files in `C:\DaveLM-CADAVER\sf2_update100_replay_mismatch_autopsy_v1\` (see PROVENANCE.json and
SHA256SUMS.txt). Key measured artifacts: historical/SF3/A/B u100 checkpoint SHA-256 in CLASSIFICATION.json.
