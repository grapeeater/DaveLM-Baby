# Comparative model anatomy post-SF2 — Granite 3.1 8B / Qwen3 8B / LFM2-24B-A2B

This is a READ-ONLY comparative-anatomy study to extract general engineering principles for Baby/DaveLM. It is
NOT a request to copy, transplant, distill from, or imitate these models. Baby's architecture, frozen evidence,
governance, and identity remain authoritative. No training, no SF3 implementation, no checkpoint modification, no
locked-panel/FINAL/sacred access.

Baby's problem being explained: a 10.6M-parameter model must learn and MAINTAIN a conditional factual decision
boundary without (a) collapsing to an unconditional name prior, (b) damaging ordinary language, or (c) sweeping
the learned boundary past the balanced solution. SF2 evidence: unconditional Alex prior removed; Owen 0/4 -> 4/4;
Alex 4/4 -> 3/4; all four AO cell midpoints Owen-shifted; residual = one near-tie (-0.03 nats), upstream-dominant
with a small output-head interaction.

## Scorecard (per model)

### Granite 3.1 8B
- MOST RELEVANT LESSON: documented muP + power-LR pretraining recipe and a two-stage data mixture; head
  "calibration" is via constant logit scale which is boundary-neutral.
- WHY IT MAY MATTER TO BABY: independent evidence that LR decay at the end of training is a deliberate production
  choice and that cosine is only one decay shape; caution that fixed logit scaling cannot move a decision
  boundary.
- EVIDENCE QUALITY: architecture [LOCAL][DOC] high; recipe [DOC] medium (exact schedule numbers UNKNOWN).
- BIGGEST ANALOGY LIMITATION: 12T-token pretraining regime vs 200-update fine-tune.
- ONE IDEA WORTH CARRYING FORWARD: treat "end-of-stage LR decay" as a schedule family (power/linear/anneal), with
  cosine as one member.
- ONE IDEA NOT WORTH COPYING: logit/embedding scaling as a fix for the boundary offset (boundary-neutral).

### Qwen3 8B
- MOST RELEVANT LESSON: staged pretraining (S1/S2/S3) plus a dedicated "thinking-mode fusion" SFT stage that
  mixes new-capability and old-capability data to retain both; per-head QK-norm as an explicit stabilizer.
- WHY IT MAY MATTER TO BABY: strongest large-model precedent for retention-by-data-mixing in a dedicated stage
  (relevant to the language-vs-factual rehearsal family) and for explicit per-head scale control.
- EVIDENCE QUALITY: architecture [LOCAL][DOC] high; stage structure [DOC] high; exact optimizer/LR UNKNOWN.
- BIGGEST ANALOGY LIMITATION: fusion SFT is post-training at 1e13-token scale, not a micro-fine-tune; QK-norm
  addresses attention stability, which SF2 did not implicate.
- ONE IDEA WORTH CARRYING FORWARD: model any future language rehearsal on a dedicated fixed-fraction "fusion"
  stage rather than ad-hoc interleaving.
- ONE IDEA NOT WORTH COPYING: QK-norm or tie/untie changes as a response to the SF2 boundary offset.

### LFM2-24B-A2B
- MOST RELEVANT LESSON: teacher-distribution matching is treated as a designed objective - "tempered, decoupled
  Top-K knowledge distillation ... that avoids support mismatch" - plus curriculum difficulty-ordered data.
- WHY IT MAY MATTER TO BABY: directly corroborates Baby's design choice to apply the parent-KL only on ordinary
  (non-factual) positions (avoiding a factual support mismatch) and flags temperature/support/top-k as real
  design levers for any future retention term.
- EVIDENCE QUALITY: architecture [LOCAL][DOC] high; KD/curriculum [DOC] medium (abstract-level); optimizer/LR
  UNKNOWN.
- BIGGEST ANALOGY LIMITATION: cross-model distillation during large pretraining vs same-model self-retention
  during a fine-tune.
- ONE IDEA WORTH CARRYING FORWARD: a "support check" for any retention term: does the teacher distribution
  support the tokens/positions the student must learn? If not, temper/decouple.
- ONE IDEA NOT WORTH COPYING: the hybrid conv/attention MoE backbone or preference-optimization post-training.

## Rankings requested by the mission

1. Which reference model taught us the most about Baby's CURRENT problem? **LFM2** - its explicitly-designed
   teacher-distribution objective (tempered, decoupled, support-mismatch-avoiding) is the closest documented
   relative of Baby's parent-KL and gives the most direct design vocabulary for Baby's retention mechanism.
   Granite is second (LR-schedule + scale-discipline lesson); Qwen3 third for Baby's specific residual (its
   fusion-stage is a data-mixing lesson more than a boundary lesson).
2. Which individual mechanism/principle deserves the most attention? **Retention-by-a-designed
   teacher-distribution term with explicit support/weighting handling** (LFM2's KD design; corroborated by Qwen3
   small-from-flagship and by Baby's own SF2 result). The single most transferable *new* engineering caution is
   LFM2's "support mismatch": match where the teacher assigns mass to where the student is allowed to change.
3. Which apparent similarity is most likely to be misleading? **Distillation/KL as evidence that Baby should keep
   KL or change its strength.** Cross-model KD (Qwen3, LFM2) is for *building a new small model from a large
   teacher*; Baby's KL is *same-model retention while fine-tuning*. The apparent similarity (both use a teacher
   distribution) is real but the purpose and regime differ, so comparative anatomy is NEUTRAL on continuing or
   scheduling SF2's KL.

## Recommendation to the Baby team

A. Did comparative anatomy materially change which SF3 family looks most promising? **No.** It modestly
   re-weights the candidates (slightly strengthens replay/rehearsal and LR-decay; slightly weakens "more of the
   same CE") but no reference model's evidence is decisive for a 200-update, 16-record factual fine-tune, and the
   SF2 autopsy's own boundary-offset finding remains the controlling evidence.

B. Does it strengthen or weaken the autopsy's tentative preference for LR annealing? **Slightly strengthens it as
   a family** (Granite's documented power schedule shows production labs deliberately decay LR late), but it does
   NOT specifically endorse cosine annealing, and it provides no direct evidence that annealing cures a
   near-boundary miss. Treat "LR decay (power/linear/anneal)" as a family, not cosine specifically.

C. Did it uncover a better one-variable treatment candidate? **Not a better one, but two sharper framings:**
   (1) LFM2's "support-mismatch" lens suggests that IF a future run changes the KL or adds rehearsal, the 
   retention term should be designed with explicit attention to where the teacher supports the tokens being
   learned; (2) Qwen3's fusion-stage framing suggests structuring any language rehearsal as a dedicated,
   fixed-fraction stage rather than ad-hoc interleaving. Both are design framings within existing candidate
   families, not new one-variable winners.

D. Did it reveal any reason NOT to run another treatment yet? **No new reason.** The reference evidence does not
   contradict running a bounded SF3; it also does not elevate any specific treatment above Baby's own evidence.
   The existing reason to be cautious remains checkpoint sparsity in SF2's trajectory, which is Baby-internal,
   not comparative.

E. What should the independent Sol High reviewer be told before designing the prospective preflight?
   - Comparative anatomy (Granite 3.1 8B, Qwen3 8B, LFM2-24B-A2B) was literature/design evidence ONLY and must
     not override Baby's direct SF1/SF2 evidence.
   - SF2's classification remains SF2_ACQUISITION_FAIL; the residual is a genuine near-tie (-0.03 nats) on
     TRAIN:g0:carried:wooden boat:a0, caused by a group-level Owen-side boundary offset (~0.1-0.3 nats across all
     four AO cells) with a small output-head contribution; upstream is the dominant mover.
   - Comparative evidence leans: slightly toward LR-decay-as-a-family (not cosine specifically) and toward
     replay/mixed-rehearsal as a dedicated stage; slightly against additional CE on identical data; neutral on
     head constraints, margin objectives, and continuing/changing the KL.
   - Do not use reference models' tied/untied embedding choices or QK-norm/scale mechanisms as evidence for a
     Baby head fix (mixed or boundary-neutral).
   - Any retention/rehearsal design should pass the "support-mismatch" check (where does the teacher/parent
     assign mass vs where must the student change?).
   - The Sol reviewer should rely on Baby's own artifacts (SF2 run, residual-item autopsy) for the treatment
     decision and treat this comparative report as background only.

## Conservative research recommendation (bottom line)

Keep the SF3 candidate family list as-is from the SF2 autopsy (LR-decay family first among equals), with two
comparative refinements: treat LR decay as a family (power/linear/anneal), and, if any rehearsal/KL variant is
contemplated, structure it as a dedicated fixed-mix stage with an explicit support check on the retention term.
No reference model's evidence justifies removing or reordering Baby's own candidates on its own.
