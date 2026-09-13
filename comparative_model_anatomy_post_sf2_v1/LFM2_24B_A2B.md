# Liquid LFM2-24B-A2B — comparative anatomy note

Evidence labels: **[LOCAL]** = observed locally (LM Studio hub + GGUF). **[DOC]** = creator-documented/public.
**[INF]** = inference. UNKNOWN = not found in the sources read.

## 1. Local forensic inventory

Sources: LM Studio hub `C:\Users\jdman\.lmstudio\hub\models\liquid\lfm2-24b-a2b\` and GGUF
`LFM2-24B-A2B-Q4_K_M.gguf` (metadata + 428-tensor inventory parsed read-only).

- Architecture: `lfm2moe`, hybrid. **[LOCAL]**
- 40 blocks. Structure (per tensor inventory): blk.0/1 = dense FFN (gate/up/down, 11776) + shortconv
  (conv/in_proj/out_proj), 8 tensors (leading dense blocks, `leading_dense_block_count=2`); blocks 2,6,10,…,38
  (every 4th, 10 blocks) contain full GQA attention (q/k/v + per-head `attn_q_norm`/`attn_k_norm`) + MoE;
  remaining 28 blocks are shortconv + MoE (no full attention). Total = 2 dense + 10 attention + 28 conv-MoE =
  40. **[LOCAL]**
- Hidden 2048; 32 heads; KV heads per block array = 8 only on the attention blocks, 0 elsewhere. **[LOCAL]**
- MoE: 64 experts, 4 active (`expert_used_count=4`), expert FFN 1536; router `ffn_gate_inp`; experts stored as
  3-D tensors `ffn_up_exps`/`ffn_down_exps`; extra `exp_probs_b.bias` per MoE block (function not explained by
  local material). **[LOCAL]**
- Embeddings: `token_embd.weight` AND `token_embd_norm.weight` (input embedding normalization). No separate
  top-level `output.weight`/`output_norm.weight` observed — consistent with a tied LM head in this conversion.
  **[LOCAL][INF]**
- Context: GGUF metadata `context_length=128000`; model card says 32,768 (discrepancy recorded, not resolved
  here). Vocab 65536. **[LOCAL]**
- Quantization: Q4_K_M. Mixed BF16/FP8 documented at training. **[DOC]**

## 2. Training-recipe research

- 24B-A2B: ~17T training tokens (card table); family 350M–8.3B trained 10–12T. **[DOC]**
- Architecture found by "hardware-in-the-loop architecture search" under edge latency/memory constraints;
  "gated short convolutions with a small number of grouped query attention blocks". **[DOC]**
- LFM2 (family) training pipeline documented in the technical report: a **"tempered, decoupled Top-K knowledge
  distillation objective that avoids support mismatch"**; **curriculum learning with difficulty-ordered data**;
  and a **three-stage post-training recipe (SFT, length-normalized preference optimization, model merging)**.
  **[DOC]** (Report text is for the ≤8.3B family; the 24B card points to the LFM2 technical report for details.)
- Optimizer/peak-LR/warmup/decay specifics: UNKNOWN from sources read.

## 3. Baby-relevance

- **Tempered, decoupled top-K KD "that avoids support mismatch"** is the single most Baby-relevant mechanism
  found across the three models. It is a direct, documented statement that teacher-distribution matching must be
  designed around the mismatch between what the teacher supports and what the student must learn. Baby's SF2 KL
  already sidesteps a factual support mismatch by applying the parent-KL **only on ordinary (non-factual)
  positions**, leaving factual answer positions free to learn names the parent disfavors. LFM2's design
  corroborates that such support handling matters and suggests that any future *extension* of the KL to other
  position classes should consider temperature and top-k decoupling. DIRECTLY RELEVANT (to the design-space of
  the KL term), INTERESTING-but-not-required for the current SF2 design.
- Curriculum with difficulty-ordered data: a general staged/ordering principle; Baby has no evidence that item
  difficulty ordering caused the residual (all 16 items co-exposed equally), so this is INTERESTING BUT CURRENTLY
  IRRELEVANT to the specific residual, though relevant to a data-topology family in general.
- QK-norm and `token_embd_norm` (embedding normalization): further evidence of "explicit scale/norm control"
  across independently built models (with Granite's embedding/logit/residual scales and Qwen3's QK-norm). Not a
  directional signal for Baby's output boundary. PLAUSIBLY RELEVANT as a general "normalization/scale discipline"
  theme; NOT APPLICABLE as a specific fix.
- Length-normalized preference optimization and model merging: alignment-stage techniques; NOT APPLICABLE to a
  10.6M single-fact acquisition problem (would be a different objective family; no Baby evidence).

## Scorecard

MODEL: Liquid LFM2-24B-A2B
MOST RELEVANT LESSON: teacher-distribution matching is treated as a first-class, *designed* objective —
tempered and decoupled top-K — precisely to avoid support mismatch; plus curriculum data ordering.
WHY IT MAY MATTER TO BABY: SF2's parent-KL is Baby's own instance of teacher-distribution retention; LFM2 shows
that the community has found the *support/weighting/temperature* of such objectives to be consequential, which
validates thinking carefully about where the KL is applied (Baby already restricts it to ordinary positions) and
how strong/tempered it should be.
EVIDENCE QUALITY: architecture [LOCAL] high + [DOC] high (card table; report abstract); KD/curriculum [DOC]
medium (abstract-level for the report; exact losses/values UNKNOWN); optimizer/LR UNKNOWN.
BIGGEST ANALOGY LIMITATION: LFM2's KD is cross-model distillation during large-scale pretraining of a hybrid
MoE; Baby's KL is same-model self-retention during a 200-update factual fine-tune — different regime and
different purpose (distillation-to-learn vs retention-while-learning).
ONE IDEA WORTH CARRYING FORWARD: an explicit design check for any future Baby retention term: does the teacher
distribution "support" the tokens/positions the student is being asked to learn? If not, temper or decouple
(top-K) the loss on those positions — a named mechanism to consider, not a copy.
ONE IDEA NOT WORTH COPYING: the hybrid conv/attention MoE backbone or the preference-optimization post-training
(as architecture/alignment for a different problem; no Baby evidence connects them to the residual).
