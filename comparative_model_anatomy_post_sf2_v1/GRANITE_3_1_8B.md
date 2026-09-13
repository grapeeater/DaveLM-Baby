# Granite 3.1 8B — comparative anatomy note

Evidence labels: **[LOCAL]** = observed on this machine (LM Studio metadata/GGUF). **[DOC]** = creator-documented /
public primary source. **[INF]** = inference. Values marked UNKNOWN were not found in the sources consulted and
are not filled with "typical practice".

## 1. Local forensic inventory

Sources: LM Studio hub `C:\Users\jdman\.lmstudio\hub\models\ibm\granite-3.1-8b\` (manifest.json, model.yaml,
README.md) and GGUF `granite-3.1-8b-instruct-Q4_K_M.gguf` (metadata + 362-tensor inventory parsed read-only).

- Architecture: `granite` (dense decoder-only). **[LOCAL]**
- Blocks: 40; hidden 4096; 32 attention heads / 8 KV heads (GQA); head/key/val dim 128; RoPE base 1e7; FFN
  12800 (gate/up/down, SwiGLU); RMSNorm eps 1e-5. **[LOCAL]**
- Context: 131072. Vocab 49155 (GPT-2 byte-BPE, "refact" pretokenizer). **[LOCAL]**
- Quantization: Q4_K_M (per-tensor mix: Q4_K 240, Q6_K 41, F32 81). **[LOCAL]**
- Embeddings: NO separate `output.weight` tensor present; `token_embd.weight` only at top level — consistent with
  shared input/output embeddings. **[LOCAL][INF]** Model card states architecture includes "shared input/output
  embeddings". **[DOC]**
- No bias tensors observed on q/k/v/o or FFN (attention/output biases absent). **[LOCAL][INF]**
- Granite-specific scaling metadata: `embedding_scale=12.0`, `residual_scale=0.22`, `logit_scale=16.0`,
  `attention.scale=0.0078125`. **[LOCAL]**

## 2. Training-recipe research

- Pretraining: 8B trained on ~12T tokens of curated language+code. **[DOC]**
- Data mixture: explicit two-stage mixture strategy "used in MiniCPM and JetMoE" balancing high- and medium-
  quality data. **[DOC]**
- LR/hyperparameter recipe: IBM's own method (Shen et al. 2024c) = **maximal update parametrization (muP)** +
  **power LR scheduler**. **[DOC]** (Exact peak LR / warmup length / final-LR value: UNKNOWN from sources read.)
- Post-training (instruct): SFT (open permissive + internal synthetic long-context data + small human-curated),
  RL alignment (PPO; best-of-N; BRAIn), and model merging (Granite 3.0); Granite 3.1 card: SFT + RL + model
  merging. **[DOC]**
- Explicit replay/retention of base capabilities during instruct: NOT documented as a named mechanism; retention
  is implied by data mixing and post-training curricula. **[UNKNOWN / not claimed]**
- Distillation: not explicitly claimed for Granite 3.0/3.1 in the sources read. **[UNKNOWN]**

## 3. Baby-relevance

- `logit_scale=16.0` and `embedding_scale=12.0` are best understood as muP-style gradient/activation scale
  controls **[INF]**. A **constant** multiplicative logit scale is monotone: it sharpens probabilities but does
  NOT change the sign of any logit margin, so it cannot correct a boundary offset such as Baby's −0.03-nat
  Alex/Owen error. Any claim that "scaling the head fixes the boundary" would be unsound. PLAUSIBLY RELEVANT only
  as evidence that scale discipline matters for stable training, not for decision calibration.
- Power LR scheduler (not cosine): supports the general "late-training LR decay" family; is a concrete
  documented alternative to cosine annealing. DIRECTLY RELEVANT to the annealing question (see CROSS_MODEL).
- Two-stage data mixture: a curriculum-style data-quality rebalancing between stages — a general pattern for
  "more training should come with a changed mixture", relevant to the replay/rehearsal family. PLAUSIBLY RELEVANT.
- Shared input/output embeddings: Granite ties at 8B (Qwen3-8B does not, LFM2 appears tied). Because reference
  models disagree, embedding tying does NOT give Baby a directional signal. NOT APPLICABLE as evidence for a head
  change; noted only for completeness.
- QK-norm / attention stabilizers: not present in Granite (uses plain RMSNorm + scaled residuals). Not a signal.

## Scorecard

MODEL: IBM Granite 3.1 8B
MOST RELEVANT LESSON: a documented **muP + power-LR** recipe and an explicit two-stage pretraining data mixture;
head "calibration" is done via constant logit scale (boundary-neutral), not via head architecture.
WHY IT MAY MATTER TO BABY: independently confirms that (a) LR decay at the end of training is a first-class,
non-cosine "power" design choice; (b) staged data-mixture changes accompany continued training; (c) fixed logit
scaling cannot move a decision boundary, which cautions against expecting head scaling to fix Baby's −0.03 miss.
EVIDENCE QUALITY: architecture [LOCAL][DOC] high; training recipe [DOC] medium (README-level; exact numbers not
published in the sources read).
BIGGEST ANALOGY LIMITATION: muP/power-schedule and 12T-token staged pretraining operate at a scale and in a
from-scratch regime wholly unlike a 10.6M-parameter 200-update factual fine-tune; the power schedule is a
pretraining device, not an evidence that annealing fixes a near-boundary miss.
ONE IDEA WORTH CARRYING FORWARD: treat "LR decay at the end of a fixed stage" as a schedule family worth
pre-registering (power/linear/anneal), and note cosine is only one member.
ONE IDEA NOT WORTH COPYING: any Granite "scale/logit-multiplier" head trick as a fix for Baby's boundary offset
(it is boundary-neutral).
