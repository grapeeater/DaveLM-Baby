# Overnight reproducibility resolution — FINAL REPORT

## 1. Executive conclusion

The SF3/SF4 reproducibility roadblock is resolved with **exactly one scientifically defensible path forward**:
a **replicated matched-prefix (single-process fork) design** that compares a constant-LR control against the
frozen late-English-LR-annealing treatment, with both arms forked from one shared in-process prefix. This design
was frozen (SF5_MATCHED_PREFIX_ANNEAL_V1) and was NOT executed. It removes the need for bitwise cross-process
identity (PATH 1: failed) and for any numerical-equivalence standard (PATH 2: not justifiable). It does not
weaken, reinterpret, or revive any SF2/SF3/SF4 gate or classification.

## 2. What was actually causing/limiting reproducibility

Run-to-run execution on this ROCm/torch stack (torch 2.12.0+rocm7.14.0, AMD RX 9060 XT) is only *stochastically*
bitwise-identical. Identical code, data, seeds, and runtime sometimes produce exactly identical trajectories
(observed for A, B, SF3-replay, SF4a over full 100-update prefixes; and for hist with p1a) and sometimes diverge
(first observable difference at update 2 or update 12, at ~1e-9 in the KL term), after which the difference
amplifies to ~9e-4 max in trainable weights by update 100 while every measured behavioral endpoint remains
effectively identical (acquisition counts, language CE within 4e-6, D3 within 7e-7, per-item margins within
~2e-3 at u100 / ~7e-3 at u200, zero margin-sign flips). Update-1 gradients and clip-norms are bitwise identical
across modes, localizing the first divergence to a rare GPU numeric-kernel event (first observable in the fp64
KL reduction). No runtime/environment lever (default; TORCH_BLAS_PREFER_HIPBLASLT=0) eliminated the
stochasticity. The historical SF2 checkpoint is reproducible only by luck (hist==p1a bitwise), not on demand.
Historical SF2 also never preserved update-100 optimizer/RNG state, so exact continuation was always impossible.

## 3. Evidence from each attempted path

- PATH 1: probe replays (engine-level), cluster matrix over 8 u100 checkpoints, gradient/clip-norm bit
  fingerprinting, env-var experiments, endpoint noise-floor quantification. Files in path1_evidence/.
- PATH 2: analysis only (no new runs needed) - no justifiable prospective tolerance; superseded by PATH 3.
  Files in path1_evidence/PATH2_FINDINGS.md.
- PATH 3: preregistered frozen design (path3_design/PATH3_PREREGISTRATION.json, PATH3_DESIGN.md), grounded in
  the measured noise floor (mode_noise_floor.json). Not executed.

## 4. Why rejected paths were rejected

- PATH 1 rejected because bitwise identity cannot be guaranteed (stochastic divergence), and the only candidate
  fixes (deterministic KL reduction rewrite; kernel/driver control) either change a frozen-pinned numeric
  implementation (forbidden) or are unavailable.
- PATH 2 rejected because (a) exactness is unavailable, (b) empirical tolerances would be reverse-engineered
  from the observed mismatch magnitudes (forbidden), (c) parameter-space distance cannot be bounded below the
  observed stochastic spread, and (d) it is unnecessary once arms share an exact in-process prefix.

## 5. Winning path

PATH 3, frozen as a design only. It is scientifically defensible because: control and treatment arms are
bitwise-identical through update 100 by construction (shared in-process prefix); post-fork stochastic divergence
is identically distributed across arms and is bounded at the endpoint level (<=6.97e-3 nats per item, zero sign
flips) - 10-100x smaller than the boundary movements the annealing hypothesis addresses; three seeds provide an
independent-noise replication dimension; all data, objectives, gates, and locked-panel rules are reused
unchanged. LR annealing remains UNTESTED until this design is executed under its own freeze.

## 6. Exact scientific claims allowed / not allowed

Allowed:
- SF2, SF3, SF4 classifications and gates are unchanged and remain authoritative.
- The LR-annealing hypothesis remains untested.
- Reproducibility claims stated in path1_evidence (mode structure, noise floor, stochasticity) apply to this
  machine/stack only.
- The frozen PATH 3 design may be executed later by an authorized engineer under its own preflight, per the
  preregistration.

Not allowed:
- Any claim that SF3 or SF4 "passed" or was "close".
- Any claim that annealing helps, hurts, or leaves the residual unchanged.
- Any held-out/generalization claim; any population p-value claim.
- Any inference that the mode-noise floor will hold on another machine/driver/torch build.

## 7. Recommended next experiment/design (DESIGN ONLY, not executed)

Execute (by a separate authorized engineer, after building and freezing the SF5 controller) the frozen
SF5_MATCHED_PREFIX_ANNEAL_V1 design in path3_design/. One manipulated variable: English LR schedule after global
update 100. Seeds 87011/87012/87013. Per seed: shared 100-update prefix, in-process fork into control (constant
5e-5) and treatment (5e-5*(90-j)/89), endpoint gates identical to SF2/SF3/SF4 definitions. Pre-registered
decision rule: support / weak-no-support / harm as written in the preregistration. No locked-panel access unless
an arm passes every endpoint gate at update 200.

## 8. Complete provenance/hashes

See PROVENANCE.json and SHA256SUMS.txt in this directory. All authoritative inputs (SF2/SF3/SF4 packages,
frozen engines, checkpoints, schedules, KL pools) were hash-verified before use; nothing was modified.

## 9. Explicit confirmation

FINAL/sacred material: NOT accessed. Locked transfer/copy/held-out/competing-name panels: NOT opened or scored.
No historical gate or classification weakened or rewritten. No scientific artifact modified. No treatment
executed. No SF5 controller written or run. Diagnostic replays and probes were confined to this mission's new
directory (overnight_reproducibility_resolution_v1).
