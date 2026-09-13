# DECISION LEDGER — overnight reproducibility resolution

## Path-by-path disposition

### PATH 1 — Bitwise determinism: FAILED / INFEASIBLE (under mission fences)
- Evidence: all frozen inputs byte-identical; update-0 and update-1 outputs and update-1 gradients/clip-norm
  bits identical across runs; first observable divergence at update 2 (KL, 1.4e-9) or update 12 (SF4 a/b,
  3e-8); cross-run u100 tensor deltas stereotyped (334/414, max 9e-4, RMS 3-4e-6, all trainable params);
  behavioral endpoints identical (11/16@100, 15/16@200, CE within 4e-6, D3 within 7e-7, zero margin-sign
  flips). Divergence is stochastic per-process (matching bursts and divergent pairs both observed; hist==p1a
  and A==B==SF3==SF4a both observed). Runtime levers tested (default; TORCH_BLAS_PREFER_HIPBLASLT=0) do not
  remove the stochasticity. Root cause is at the GPU numeric-kernel level (ROCm/torch), first observable in the
  fp64 KL reduction; not fixable from project code without changing a frozen-pinned numeric implementation
  (forbidden) and not controllable at runtime.
- Acceptance criterion not met (cannot robustly make replicas bitwise-identical through the required prefix).
- See path1_evidence/PATH1_FINDINGS.md.

### PATH 2 — Prospective numerical-equivalence standard: FAILED / NOT JUSTIFIABLE
- Exactness unavailable; empirical tolerances would be reverse-engineered from observed SF3/SF4-era mismatch
  magnitudes (forbidden); parameter-space distance cannot be bounded below the stochastic spread; behavioral
  tolerance would not certify parameter-level comparability for a small LR manipulation.
- Superseded: the matched-prefix design (PATH 3) removes the need for any equivalence standard by giving both
  arms an exact shared in-process prefix.
- See path1_evidence/PATH2_FINDINGS.md.

### PATH 3 — Replicated matched-prefix control-vs-treatment design: PASSED (design frozen)
- A defensible preregistered replicated design was frozen (SF5_MATCHED_PREFIX_ANNEAL_V1): 3 seeds (87011,
  87012, 87013), single-process fork at update 100, control (constant 5e-5) vs treatment (late-English LR
  linear-to-zero), unchanged data/KL/scope/optimizer/gates/locked-panel rules, pre-registered support/weak/harm
  decision rule and endpoint reporting.
- NOT EXECUTED, per mission instruction.
- See path3_design/.

### PATH 4 — Retire LR question: NOT REACHED (Path 3 passed).

## Outcome
Exactly one path passes (PATH 3, as a frozen design). Historical gates/classifications unchanged. No scientific
artifact modified. No treatment executed. No locked panels / FINAL / sacred touched.

## Key measured numbers
- Mode-noise floor at endpoints: per-item margin deltas <=2.05e-3 (u100), <=6.97e-3 (u200); zero sign flips;
  same-mode pairs bitwise identical (0.0).
- Historical SF2 prefix IS bitwise reproducible by luck (hist==p1a exactly), but not controllably.
- SF3/SF4 remain hard-stopped forever; LR-annealing hypothesis remains untested.
