# Qwen3-1.7B-Base reference study v1

Status: **COMPLETE, read-only specimen study.**  
Date: 2026-09-12.  
Specimen path: `C:\AI\QwenLab\base_model`.  
Artifacts: `qwen_readonly_anatomy_v1.json`, `qwen_vs_baby_name_tokenization_v1.json`, `qwen_readonly_anatomy_v1.py`.  
The Qwen model was **not** modified. No weights, tokenizer, layers, or outputs were copied into Baby. No distillation. No teacher use. No Baby training during this phase.

Question answered: what general architectural / representational / optimization / tokenizer / capacity-allocation lessons can inspire **original** Baby research — not how to turn Baby into Qwen.

---

## 1. Exact model identity

Identified from local artifacts, not folder naming:

| Evidence | Value |
|---|---|
| `config.json` `architectures` | `Qwen3ForCausalLM` |
| `config.json` `model_type` | `qwen3` |
| README title | **Qwen3-1.7B-Base** |
| License | Apache-2.0 |
| `torch_dtype` | bfloat16 |
| Weight file | single `model.safetensors` (3,441,185,608 bytes) |
| Counted parameters | **1,720,574,976** (1.721B) |
| Non-embedding parameters | **1,409,410,048** (1.409B) |
| README claim | 1.7B total / 1.4B non-embedding — matches the count |
| Training stage (card) | Pretraining (Base, not instruct-finetuned) |
| Existing lab eval | `evals/baseline_untouched.json` labels checkpoint `Qwen3-1.7B-Base` |

Tokenizer class in `tokenizer_config.json` is `Qwen2Tokenizer` (Qwen3 reuses that tokenizer family). Vocab in config: **151,936**. Fast tokenizer reports 151,669 registered entries plus 26 added specials; unused config slots exist. BOS/EOS/PAD share `<|endoftext|>` (id 151643). `add_bos_token` is false.

This is a **dense** Qwen3 Base model, not an MoE.

---

## 2. Architecture summary

Observed from `config.json` + safetensors tensor names/shapes:

| Property | Value |
|---|---|
| Layers | 28 |
| Hidden width | 2048 |
| Attention heads (Q) | 16 |
| KV heads | 8 (GQA) |
| Head dimension | 128 |
| MLP intermediate | 6144 (3× hidden) |
| Activation | SiLU on a gated MLP (`gate_proj`, `up_proj`, `down_proj`) |
| Normalization | RMSNorm (`rms_norm_eps` 1e-6); **QK-Norm** (`q_norm`, `k_norm` of size 128) |
| Positions | RoPE, `rope_theta` 1e6, `rope_scaling` null |
| Context (config) | `max_position_embeddings` 32,768 |
| Tokenizer max length field | 131,072 (tokenizer metadata; model card says 32,768) |
| Sliding window | present in config but `use_sliding_window` false, `sliding_window` null |
| Embeddings | **tied** (`tie_word_embeddings` true); no separate `lm_head` tensor |
| Attention bias | false |
| Residual layout | decoder-only pre-norm (input LN → attn → residual; post-attn LN → MLP → residual) |
| Unusual / noteworthy | QK-Norm on every layer; GQA 16/8; SwiGLU; huge vocab with tied head; late-layer RMSNorm scales grow by ~200× |

Card also claims three-stage pretraining and scaling-law hyperparameter tuning. Those are **documentation claims**, not facts recoverable from weights.

---

## 3. Parameter-budget breakdown

Counted from streamed BF16 tensors (read-only):

| Group | Params | % of 1.721B | RMS |
|---|---:|---:|---:|
| MLP down | 352,321,536 | 20.48 | 0.0344 |
| MLP gate | 352,321,536 | 20.48 | 0.0376 |
| MLP up | 352,321,536 | 20.48 | 0.0352 |
| Token embedding (tied head) | 311,164,928 | 18.08 | 0.0334 |
| Attn Q | 117,440,512 | 6.83 | 0.0360 |
| Attn O | 117,440,512 | 6.83 | 0.0320 |
| Attn K | 58,720,256 | 3.41 | 0.0321 |
| Attn V | 58,720,256 | 3.41 | 0.0348 |
| Pre-attn RMSNorm | 57,344 | 0.003 | 8.61 |
| Pre-MLP RMSNorm | 57,344 | 0.003 | 1.49 |
| Q-Norm / K-Norm | 3,584 + 3,584 | ~0 | 1.78 / 3.01 |
| Final RMSNorm | 2,048 | ~0 | 2.27 |

**Priorities implied by the budget (fact):** ~61.4% of parameters are MLP (gated expansion). ~20.5% are attention. ~18.1% are the tied embedding/readout. Normalization is cheap in parameters and expensive in scale.

**GQA saving (fact):** K and V are half of Q/O (8 vs 16 heads). That is a KV-cache / attention-capacity trade, not a free lunch for a 61.5M model.

**Tied head (fact):** one 311M table serves both input identity and output softmax. Baby spends ~1.3M on **untied** embed+head at vocab 1024.

---

## 4. Attention / MLP / depth observations

**Facts**

- Every layer has the same tensor shapes. There is no MoE router and no sliding-window split in the weights.
- Per-layer attention RMS stays in a narrow band (~0.033–0.042). MLP RMS is similarly narrow (~0.033–0.038) with a mild late-layer rise.
- **Pre-attn RMSNorm scale grows with depth:** ~0.10 at L0 to ~24 at L26, then 17.9 at L27. Pre-MLP scale grows more slowly (0.25 → 4.65).
- K-Norm RMS is large at L0 (7.18) and then settles ~2–4. Q-Norm stays ~1.5–2.1.
- QK-Norm tensors exist on all 28 layers (shape `[128]`). This matches the model card’s “qk layernorm for all models.”
- Attention is GQA: `k_proj`/`v_proj` are `[1024, 2048]`, `q_proj`/`o_proj` are `[2048, 2048]`.

**Descriptive pattern**

Late layers apply much larger residual-stream gain control before attention than early layers. Weight RMS does **not** explode; the **norm scales** do. That is consistent with a residual stream whose magnitude grows with depth and is re-stabilized per layer.

**Plausible inference (not proven)**

Depth is used as a specialization axis via *gain and routing of an already-rich stream*, not via different module shapes. QK-Norm is a cheap stabilizer so Q/K can keep distinct scales without breaking attention. Gated MLP gives a per-token “which features to write” control that a ReLU-MLP lacks.

**Unsupported**

Tensor RMS alone does not identify which layers do language vs relations vs retrieval. Do not claim Qwen “solved” Baby’s T3–T15 coexistence because late RMSNorm scales are large.

---

## 5. Tokenizer / entity-representation observations

**Facts — Qwen tokenizer structure**

- Byte-level BPE (`merges.txt` 1.67MB, `vocab.json` 2.78MB, `tokenizer.json` 7.0MB).
- Config vocab 151,936; specials include chat (`im_start`/`im_end`), tool, FIM, vision pads, and `<think>`/`</think>` even on this Base checkpoint.
- Leading space is its own piece: `" Alice"` is one token, distinct from `"Alice"`.
- Period is one token. Common clauses tokenize as whole words (` likes`, ` cats`, ` book`).

**Facts — name probes (do not copy the vocabulary)**

- 32 generic English/world names: Qwen single-token rate **28%**, mean **1.78** tokens.
- Frequent names (`Alice`, `Bob`, `Charlie`, `Elizabeth`) are often one token; less common names split (`D`+`iana`, `S`+`of`+`ia`).
- T3 **train** names only (16 unique; TEST never loaded): Qwen single-token **5/16 (31%)**, mean **1.69**.

**Facts — Baby tokenizer on the same T3 train names**

- Vocab **1024**, learned absolute context 256.
- Single-token rate **0/16**. Mean **2.81** tokens. Histogram: 2 tokens ×4, 3 ×11, 4 ×1 (`Marie` → `M/ar/i/e`).
- `Bea` → `B e a` (3). `York` → `Y ork` (2). Clause `Who has the red book?` is 6 Qwen tokens vs **11** Baby tokens.

**Question: does Baby’s tiny vocab create unnecessary friction for names/entities/fact clauses?**

Yes, **as a representational cost**, not as a proven cause of the reversal miss. Baby’s pointer keys are name-through-period spans. Those spans are 2–4 character-ish pieces plus a period, not a single entity atom. That forces the late base to compose an entity from fragments before it can bind a relation. Qwen often (not always) gets a name atom for free.

**Do not change Baby’s tokenizer because Qwen’s is larger.** Baby’s frozen token identities, evaluators, and T3–T15 time series depend on the 1024 tokenizer. A tokenizer change would confound capacity, pointer keys, and language CE. The principle worth testing later is **entity-span composition**, not “adopt 150k BPE.”

---

## 6. Qwen vs Baby (only after Qwen analysis)

Baby source of truth: `baby_vnext_60m_design_v1` (`BABY_VNEXT_CONFIG.json`, `PARAMETER_COUNTS.json`, `BABY_VNEXT_ARCHITECTURE_SPEC.md`). Read-only.

| Axis | Qwen3-1.7B-Base | Baby vNext ~61.5M |
|---|---|---|
| Params | 1.721B (1.409B non-embed) | 61.52M (60.54M base + 0.98M binding) |
| Depth | 28 | 12 |
| Width | 2048 | 640 |
| Heads / KV | 16 Q / 8 KV | 10 / 10 (full MHA) |
| Head dim | 128 | 64 |
| MLP | gated SwiGLU 6144 (~3×) | ReLU 2560 (4×) |
| Norm | RMSNorm + QK-Norm | Explicit LayerNorm (mean+var, affine) |
| Positions | RoPE θ=1e6, 32k | Learned absolute, 256 |
| Vocab | 151,936, tied head | 1,024, **untied** head |
| Embed/head cost | 18% of the model | ~2.1% of the model (655k+656k) |
| Extra structure | none in weights | 984,321-param binding sidecar (frozen in T6–T15) |
| Dropout | 0 (config) | 0.05 embed/attn/residual |

**Jobs a 1.7B transformer can spread across distinct capacity that Baby forces into shared small regions**

1. **Language modeling vs entity binding.** Qwen spends 311M parameters on a tied table that already atomizes many names, plus 28 layers and 61% MLP to write residual updates. Baby spends ~1.3M on tokens+head and must compose names in 12 ReLU blocks of 4.9M each. T3–T15 already showed the binding sidecar does **not** carry relational gain; late base does. That late base is also the language network.
2. **Depth specialization.** Qwen has 28 residual writes and exploding late pre-attn scales. Baby has 12 writes. T14X localized reversal gain to blocks 4–7 and language cost to 8–11 — Baby is already using depth as a specialization axis, with only four blocks on each side of that cut.
3. **Attention vs MLP labor.** Qwen’s MLP is 3× its attention budget. Baby’s MLP is also larger than attention per block (3.28M vs 1.64M) but is an ungated ReLU. Both are MLP-heavy; Qwen adds a gate.
4. **Readout isolation.** Qwen’s tied head is a huge shared identity/output table. Baby’s pointer is **parameter-free** (cosine over `forward_hidden`, i.e. after **all 12 blocks + final_norm**). Pointer and language_head share the same late state; `language_head` has no pointer gradient (T14X).

**Qwen designs likely inappropriate to copy at 61.5M**

- 150k vocab / 311M embed (would dominate Baby’s budget).
- GQA (KV savings matter at long context / batch; Baby context is 256).
- 32k RoPE (no Baby evidence that 256 is the bottleneck).
- 28-layer depth without width (Baby already rejected deeper/narrower 16×560).
- Chat/tool/think/vision specials on a research LM that does not use them.

---

## 7. General principles worth retaining (class A)

1. **Most capacity in the residual writers (MLP), not in a sidecar.** Qwen and Baby both put the majority of non-embed params in blocks. Baby experiments already agree: frozen binding never produced loc/routing; relational gains were late-base.
2. **Depth is a specialization axis.** Different layers can carry different jobs even with identical shapes. T14X is Baby’s own evidence; Qwen’s growing late-norm scales are consistent with the same general idea.
3. **A module can be useful during training and optional (or harmful) at the trained endpoint.** This is a general optimization fact. T15 vs T14X is the Baby instance.
4. **Entity granularity is a first-class representational choice.** Tokenizer atoms vs composed spans change how much residual compute must be spent on “who is this.”
5. **Cheap per-head / per-stream gain control (QK-Norm, RMSNorm scales) is how large models keep competing writes stable.** The *principle* is stabilize-then-write, not “install RMSNorm.”

---

## 8. Scale-dependent findings (class B)

- GQA, 32k context, 1e6 RoPE θ, 151k BPE, 1.4B non-embed, three-stage 36T pretrain (card).
- Late RMSNorm scales of 10–24: a 12-layer 640-wide LN model will not reproduce this residual-growth regime by copying the numbers.
- Tied 311M head: at vocab 1024, tying vs untying is a ~0.6M decision already frozen; Qwen’s reason for tying is memory at 150k, not Baby’s reason.

---

## 9. Qwen-specific ideas not worth copying (class C)

- Qwen3 special-token inventory (vision, tool, `<think>`) on a Base checkpoint.
- Exact SwiGLU width 6144, head_dim 128, GQA 16/8.
- Reproducing Qwen’s architecture to “get Qwen behavior at 61.5M.”
- Using Qwen generation (`baseline_untouched.json`) as training targets. Those outputs are a Base-model continuation prior (repetition, prompt-echo, weak instruction following) and are not a teacher signal.

---

## 10. Original Baby-native inspirations (class D)

**D1 — Training scaffold ≠ inference circuit**  
Qwen observation: identical late layers with very different gain scales; card claims staged pretraining.  
Principle: later capacity can stabilize or route a stream while earlier capacity learns a feature; the late weights need not remain the ones you want at evaluation.  
Baby bottleneck: T14X revert of 8–11 to Phase1G *improved* language and reversals after joint 4–11 training; T15 freeze of 8–11 from init *prevented* reversals while holding language.  
Baby-native hypothesis: blocks 8–11 are a **training-time scaffold** for the 4–7 reversal pathway, not the inference circuit. Next test is not “copy Qwen staged pretrain,” it is a Baby two-phase or stop-grad protocol justified by T14X+T15.

**D2 — Entity-span composition, not a bigger vocab**  
Qwen observation: many names are 1 token; T3 train names are 0/16 single-token on Baby.  
Principle: entity atoms reduce the composition burden on residual blocks.  
Baby bottleneck: pointer keys are multi-token name-through-period spans; reversals stall ~43–49.  
Baby-native hypothesis: reversal difficulty may include **span-composition** cost. Do **not** retokenize now. A later diagnostic could compare pointer geometry on short vs long name spans inside the existing tokenizer.

**D3 — Gated residual writes as a future architecture candidate**  
Qwen observation: 61% of params are gated MLP.  
Principle: a gate lets a layer write a subset of features, which can reduce overwrite of a preserved skill.  
Baby bottleneck: language vs relation overwrite in shared late MLP (T14X: reverting MLP 4–11 collapsed reversals and *helped* CE).  
Baby-native hypothesis: an original Baby gate or write-control on blocks 4–7 (not a Qwen SwiGLU transplant) might let relational features write without as much language overwrite. **After v1.0 or after the scaffold question is settled** — this is an architecture change and would confound the current freeze/scope series.

---

## 11. Implications for Baby’s language-vs-relational coexistence

T3–T15 already mapped the conflict:

- Full-base (T3) breaks language.
- Sidecar (T4/T5/T8) is dead.
- Late-base fact-clause (T9–T14) can form representations but not at 2/3 without language death, except T14’s 1/3 hit.
- T14X: gain in 4–7 (esp. block 4); cost in 8–11.
- T15: freeze 8–11 from Phase1G holds language 3/3 (CE ~1.24, gap 0.36) and **fails representation 0/3** (rev 38/41/43).

Qwen does **not** override that. It only adds: a mature model *expects* late layers to apply different gain than early layers, and it spends most of its budget on gated residual writes plus a tokenizer that often atomizes entities. The coexistence problem at 61.5M is still “shared late residual + pointer-after-all-layers + multi-token names,” not “missing GQA.”

The immediately actionable implication is **D1**: do not freeze the language-costly late blocks from initialization; they appear necessary as a **moving** training path even if their trained endpoint is optional.

---

## 12. Future ideas that may matter after v1.0

- Original (not copied) gated MLP or write-control on selected Baby blocks.
- RMSNorm or QK-Norm as a **separate** architecture experiment after the current recipe is closed.
- RoPE vs learned positions (Baby spec already deferred this).
- Tokenizer / entity-lexicon redesign with a new frozen eval suite (would reset T3–T15 comparability).
- Staged objectives (language-first, then relation, then long-context) as Baby-native schedules — analogy to the card’s three-stage claim only.

---

## 13. Three highest-value ideas for current Baby development

1. **Treat T15 as a closed negative on “freeze 8–11 from Phase1G.”** Next information is how 4–7 differ when 8–11 are plastic vs frozen (read-only T14 vs T15 vs Phase1G drift), then a prospectively frozen **scaffold** treatment if that diagnostic agrees. Do not relaunch T15. Do not parent T14/T15/730002.
2. **Keep the pointer definition and 1024 tokenizer stable** while the scaffold question is open. A tokenizer change now would erase the T9–T15 LR/scope series.
3. **Do not spend the next run on Qwen-shaped architecture** (SwiGLU, GQA, RoPE, 150k vocab). Those are recorded as post-v1.0 candidates. Baby evidence already says the live bottleneck is late-base coexistence under the current architecture.

---

## 14. Confidence, limitations, and unresolved questions

**High confidence**

- Identity, shapes, parameter counts, tied embeddings, GQA, QK-Norm, SwiGLU, RMSNorm scale-vs-depth trend, tokenizer name-split contrast on T3 train names.

**Medium confidence**

- Interpretation of growing late RMSNorm scales as depth specialization / residual-growth control.
- Claim that name fragmentation contributes to Baby reversals (plausible, untested).

**Low / not claimed**

- Qwen training recipe, data, or “why they chose X” beyond the public card.
- Any statement that a Qwen mechanism would pass Baby’s 2/3 gates.

**Limitations**

- Weight statistics are not circuits. No activation patching, no Qwen probing on Baby tasks.
- Baseline generations in `baseline_untouched.json` are Base-model continuations, not instruction quality, and were not used as supervision.
- Baby comparison used published Baby specs and T3 **train** name lists only.

**Unresolved (for Baby, not for more Qwen sightseeing)**

- Do T15 blocks 4–7 fail to drift like T14’s block 4, or do they drift to a different, reversal-weak solution?
- Is 8–11 plasticity required as a **gradient path** (they stay frozen but grads flow through them in T15 already) or as **weight motion** (they must update)?
- Can a prospective two-phase “train 4–11, then lock or revert 8–11” replicate T14X without parenting T14?

Non-copying boundary held: no Qwen weights, code, tokenizer, or outputs enter Baby from this study.

---

## Receipts

- Anatomy script: `C:\AI\QwenLab\evals\qwen_readonly_anatomy_v1.py`
- Anatomy dump: `C:\AI\QwenLab\evals\qwen_readonly_anatomy_v1.json`
- Name-tokenization compare: `C:\AI\QwenLab\evals\qwen_vs_baby_name_tokenization_v1.json`
- This report: `C:\AI\QwenLab\evals\QWEN3_1P7B_BASE_REFERENCE_STUDY_V1.md`
- Qwen `config.json` identity: `Qwen3ForCausalLM` / `qwen3` / vocab 151936 / 28×2048 / GQA 16/8 / head_dim 128 / intermediate 6144 / tied embeddings / bf16
