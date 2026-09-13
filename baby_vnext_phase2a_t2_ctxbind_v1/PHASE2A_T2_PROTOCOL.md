# Baby vNext Phase 2A Treatment 2 — Natural-Language Contextual Binding (decision-focused)

Status: **prospectively frozen** (BABY_VNEXT_PHASE2A_T2_CTXBIND_V1)

## Parent
Phase1G U6000 `best.pt`, SHA-256 `c5406f80…`, model-state digest `0101cec9…` (not v1 U100).

## Sole treatment
QA objective replaced by a **length-normalized hardest-distractor pairwise margin hinge**:
- candidate score `s = mean per-token log-likelihood of canonical ' Name.'` (trailing period included, EOS excluded)
- QA loss = `max(0, M − (s_correct − max_distractor s))`, `M = 1.0 nat`
- no prompt CE, no continuation CE, no EOS supervision, no KL, no localization term.
Everything else identical to Phase 2A v1: 9:1 QA:language cadence, AdamW 5e-5, wd 0.05, clip 2.0,
all base parameters trainable, binding/localizer frozen byte-identical.

## Panels (fresh; family-disjoint from v1 inspectable panels)
- T2-EVAL-DEV: 640 core (80/level A–H) + 120 shortcut-audit + 120 novel-name (2-candidate).
- T2-EVAL-TEST: 320 core + 60 audit + 60 novel-name; **sealed, hashed, never opened until a valid
  terminal commit; scored exactly once**.
- Language retention: frozen Phase1G DEV windows + train-fit + generation prompts.

## Gates
U100/U250 early-stop (language, AB/ABCDE no-progress, audit no-progress, V1_FAILURE_MODE),
final success (AB ≥0.80 & margin ≥0.50; ABCDE ≥0.65; every level ≥0.50; audit ≥0.55 & margin
≥0.10; novel ≥0.65; reversal ≥0.70; language ≤1.30; generation bounds; V1 diagnostic negative;
terminal TEST core ≥0.60/audit ≥0.50/novel ≥0.60).

Seeds: primary 610005; panel 61000211; schedule 61000212; rehearsal 61000213; train family
61000214; reserved 610006. Replication not authorized.
