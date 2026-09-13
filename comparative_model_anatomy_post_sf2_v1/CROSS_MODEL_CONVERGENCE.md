# Cross-model convergence — recurring principles with evidence

For each principle: models using it; exact evidence; documented/intended purpose; possible Baby analogy; and
differences that weaken the analogy. Evidence classes: [LOCAL]/[DOC]/[INF].

## Principle 1 — Staged / multi-phase training with per-stage data-mixture changes and a dedicated late stage

- Models: Granite 3.0/3.1, Qwen3, LFM2 (all three).
- Evidence:
  - Granite: two-stage pretraining data mixture (MiniCPM/JetMoE style) balancing high+medium quality data, then
    base -> instruct via SFT+RL+merging. [DOC]
  - Qwen3: S1 (>30T, 4K ctx) -> S2 (+5T, raised STEM/code/reasoning share) -> S3 (long-context, 32K); then 4
    post-training stages including a dedicated "thinking-mode fusion" SFT stage. [DOC]
  - LFM2: curriculum learning with difficulty-ordered data; three-stage post-training (SFT, length-normalized
    preference optimization, merging). [DOC]
- Documented purpose: capability/quality staging; add a new capability (reasoning, long-context, instruction)
  with data chosen for that stage rather than one monolithic mixture.
- Baby analogy: Baby already runs capability stages (v0.9 copy gate, Pilot1 language, SF1/SF2 factual) — the
  reference models reinforce the principle "when you add a new objective, give it a *stage* with its own mixture
  and its own end-of-stage behavior", rather than a single blended objective forever.
- Weakening differences: reference stages are 1e12–1e13-token pretraining/post-training phases on new data; Baby
  stages are ≤200-update fine-tunes on the same 16 records. The principle transfers as *structure*, not as scale.

## Principle 2 — Knowledge distillation from a teacher for compact models (2 of 3 + external precedent)

- Models: Qwen3, LFM2 (documented); (external: NVIDIA Nemotron Elastic, prior audit — KD-based compression).
- Evidence:
  - Qwen3: abstract — smaller models built "by leveraging the knowledge from the flagship models … reduce the
    computational resources required to build smaller-scale models". [DOC]
  - LFM2: "tempered, decoupled Top-K knowledge distillation objective that avoids support mismatch" in the
    training pipeline. [DOC]
- Documented purpose: transfer capability from a large teacher to a small student efficiently; LFM2's design
  note ("avoid support mismatch") is an explicit acknowledgment that naive KD can fail when teacher and student
  support differ.
- Baby analogy: SF2's KL-to-frozen-Pilot1 is Baby's own teacher-retention mechanism. Cross-model evidence
  validates the *family* (constraining a student toward a teacher distribution) and, via LFM2, warns that the
  *support/temperature/weighting* of such objectives is consequential.
- Weakening differences: reference KD is cross-model (large -> small) during pretraining of a *new* model; Baby's
  KL is same-model (earlier snapshot -> current self) during a small fine-tune. Different purpose
  (distillation-to-learn vs retention-while-learning). No reference model uses "KL against your own frozen
  earlier checkpoint on an unrelated-context sample" — that specific design has no direct analog.

## Principle 3 — LR decay/schedule treated as a first-class, documented design element (weak convergence)

- Models: Granite documents a specific schedule; Qwen3 and LFM2 exact schedules are UNKNOWN in the sources read.
- Evidence: Granite uses "maximal update parameterization and a power scheduler" (Shen et al. 2024c). [DOC]
- Purpose: stable, transferable LR control; power decay is a smooth late-training decay (not cosine).
- Baby analogy: directly relevant to the "LR annealing" candidate — it shows a production lab deliberately ends
  pretraining with a decay schedule, and that cosine is not the only accepted decay shape.
- Weakening differences: only one of three models documents this; and it is a pretraining device, not evidence
  that annealing fixes a near-boundary 200-update miss.

## Principle 4 — Explicit scale/normalization control beyond a plain final norm (3 of 3, but heterogeneous)

- Models: Granite (embedding_scale 12, residual_scale 0.22, logit_scale 16, attention scale; [LOCAL]), Qwen3
  (per-head QK-norm; [LOCAL]), LFM2 (token_embd_norm + per-head QK-norm; [LOCAL]).
- Evidence: all three ship GGUF metadata/tensors showing scale or norm mechanisms in addition to standard
  per-block RMSNorm. [LOCAL] Each is creator-documented to differing degrees (Qwen3/LFM2 QK-norm documented;
  Granite constants consistent with muP [INF]).
- Purpose (heterogeneous): gradient/activation stability (muP scales), attention-logit stability (QK-norm),
  embedding-scale control.
- Baby analogy: Baby's architecture uses a plain pre-norm ExplicitLayerNorm with no auxiliary scale control.
  The reference models show that independent labs converge on *explicit* control of activation/logit scale.
  Note however that a constant scale is monotone and cannot move a decision boundary (see Granite note), so this
  theme does not itself explain or fix the −0.03-nat miss.
- Weakening differences: three different mechanisms serving three different purposes; no shared "one mechanism"
  to copy; and none is shown to affect a two-class boundary offset.

## Principle 5 — Tied vs untied embeddings: NO convergence (mixed)

- Granite 8B: tied (shared). [DOC][LOCAL] Qwen3 8B: untied (tied at ≤4B). [DOC][LOCAL] LFM2: no separate output
  tensor observed (tied-like in conversion). [LOCAL][INF]
- Purpose: an engineering trade-off that flips with scale/implementation inside Qwen itself.
- Baby analogy: Baby is untied (matching Qwen3-8B and Nemotron). No reference principle favors changing Baby's
  head; treating tie/untie as a boundary-fix would be unsupported.

## Principle 6 — Retention-by-data-mixing (Qwen3 fusion SFT) as the closest "keep old while adding new" design

- Qwen3: thinking-mode fusion stage = SFT on a mix of new (long-CoT) and old (instruction) data. [DOC]
- Purpose: fuse a new mode into a model without losing the existing mode.
- Baby analogy: supports the "replay/mixed ordinary-language rehearsal" family and a "dedicated late mixing
  stage" structure; but note Baby's own Pilot0/1 evidence already shows naive mixing is delicate.
- Weakening differences: post-training SFT scale and content are far from Baby's setting; no distributional
  regularizer involved.

## Summary of genuinely convergent principles

1. Add new capabilities in dedicated stages with stage-appropriate data mixtures (all three).
2. Teacher/distillation-based distribution matching is a valid and used tool for compact models; its
   support/weighting design matters (Qwen3, LFM2).
3. Late-training LR decay is a deliberate production choice, and cosine is only one decay shape (Granite).
4. Explicit activation/logit scale and per-head normalization controls are common, heterogeneous stabilizers
   (all three) — a theme, not a recipe.
5. Tied/untied embeddings show no cross-model consensus — not a lever indicated by comparison.
