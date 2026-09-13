# Recommendations

**Scope reminder:** these are recommendations to consider/investigate. Nothing here has been implemented,
executed, or scheduled. No Baby code, checkpoint, dataset, or frozen gate was touched in producing this
document.

## Ranked findings (see REPORT.md Part 3 for full 11-point analysis of each)

| # | Finding | Rank |
|---|---|---|
| 1 | Preserve prior/unrelated-context behavior when adding new capability (distillation/replay-to-parent regularization), inspired by Nemotron Elastic's calibration-preserving distillation | **A — high value** |
| 2 | Increase within-capability candidate/surface-form diversity (more actor names, more phrasings) to dilute suspected frequency-prior shortcuts, inspired by Nemotron's deliberately diverse multi-teacher synthetic data construction | **B — diagnose first** |
| 3 | Reserve inert vocabulary slots for future extensibility | C — interesting/premature |
| 4 | Adopt hybrid Mamba-2/Attention SSM backbone, GQA, long context | D — do not copy (irrelevant at scale) |
| 5 | Revisit tied embeddings | D — do not copy (already tried 8 ways at Baby's scale; Nemotron's own choice, at 4B scale, is also untied — this corroborates stopping, not restarting) |

No finding reached rank A on architectural grounds alone; the one A-ranked finding is a **training-objective/
curriculum** idea, not an architecture change, which is consistent with the audit's conclusion that Baby's
current bottleneck does not look like an architecture-capacity problem.

## Recommended order of investigation

1. **First, run Finding 2's diagnostic** (candidate-pool diversity) if only because it is cheaper, requires no
   new loss term, and directly tests a specific, falsifiable hypothesis about *why* the Alex-default bias
   exists (a base-rate/frequency artifact of a 2-name pool) versus the alternative (a deeper, non-frequency
   mechanism) that the existing component-swap forensic could not resolve ("MIXED_OR_AMBIGUOUS... not
   localized").
2. **Then, if the frequency-artifact hypothesis is falsified** (bias persists undiminished with a wider,
   balanced candidate pool), proceed to Finding 1's diagnostic (KL-to-parent regularization on unrelated
   contexts), since a non-frequency-artifact explanation is more consistent with a genuine distributional-drift
   problem that a distillation-style term is designed to address.
3. **Do not pursue Findings 3–5** without new evidence specifically implicating vocabulary extensibility,
   architecture capacity, or embedding tying in a demonstrated Baby failure — the current evidence base does
   not support them.

## Proposed smallest next diagnostic for the strongest finding (Finding 2) — PROPOSED ONLY, NOT EXECUTED

**Working title (proposed, not registered):** `sf1_candidate_pool_diversity_diagnostic_v1`

**Design sketch (for a future researcher to formally pre-register, build, and freeze — this audit does not do
so):**
- Start from the exact same parent checkpoint, protected-block mask, causal answer+EOS CE objective, update
  budget (200 updates, regression-guard active at update 100), and evaluation battery structure as
  `single_fact_acquisition_sf1_seed87011`.
- The **only** manipulated variable: expand each 2-name role-pair to a 4-name (or 6-name) role-group, holding
  total number of training items, update budget, and learning-rate schedule fixed (this may require reducing
  per-name exposure count to keep total items constant — that trade-off should itself be pre-registered and
  reported, not silently absorbed).
- Reuse SF1's own acquisition gate definition unmodified (16/16 correct AND 16/16 exact+EOS AND all-reversals
  AND all-complete-families, scaled to the new item count) so results are comparable on the same pass/fail
  standard.
- Add the same forensic instrumentation already built for SF1 (readout-selection forensic's D3 unrelated-context
  prior-shift measurement, and the component-swap forensic's factorial localization) so that, whatever the
  acquisition-gate outcome, the *mechanism* is measured with the same tools already validated on the original
  SF1 run — enabling a direct before/after comparison of whether prior pollution (D3) and the fixed-default
  bias both shrink, both persist, or diverge.
- Pre-register the specific falsification criterion in advance (as in point 10 of Finding 2 in REPORT.md): if
  the fixed-default pathology persists at comparable severity with a wider, more balanced name pool, treat the
  frequency-artifact hypothesis as falsified and do not pursue further pool-diversity variants; escalate instead
  toward Finding 1's distributional-regularization diagnostic.

**Why this is the smallest clean next step:** it changes exactly one variable (candidate-pool composition),
reuses every other piece of SF1's already-validated infrastructure (parent checkpoint, masking scheme,
objective, gate definition, forensic tooling) unchanged, and produces a directly falsifiable answer to the one
open question the two most recent forensics (readout-selection and component-swap) could not resolve on their
own: is the Alex-default bias a shallow frequency artifact of the experimental design, or a deeper property of
what blocks 4–7 have learned to represent?

## What this audit explicitly does NOT recommend

- Do not re-architect Baby around Mamba/SSM, GQA, or long-context mechanisms.
- Do not resume tied-embedding experimentation.
- Do not adopt chat-template/reasoning-toggle scaffolding before naturalistic transfer is established.
- Do not treat any finding in this document as validated — every finding above is, at best, a *candidate*
  worth a small bounded diagnostic; none should be treated as a decision to change Baby's production training
  recipe without first running (and pre-registering) the diagnostic and inspecting its result.

## Final note

Per the mission's scope, this audit's authority ends here. No diagnostic, treatment, or code change described
above has been built, run, or scheduled. A future researcher (human or agent) operating under a properly
pre-registered, separately-authorized experimental mandate would be responsible for deciding whether and how to
actually execute any of the above.
