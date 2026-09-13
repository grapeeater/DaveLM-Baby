# Comparative Engineering Audit: NVIDIA Nemotron 3 Nano 4B vs. DaveLM "Baby" v0.9

**Status:** Research only. No Baby code, checkpoint, dataset, or frozen gate was modified. Nothing was trained.
This document reports findings and recommendations only; it does not implement anything.

**Labeling convention (Nemotron claims only):**
- **[UPSTREAM]** = AUTHORITATIVE UPSTREAM FACT, sourced directly from NVIDIA-published artifacts (model card,
  config.json, arXiv abstracts).
- **[LOCAL]** = OBSERVED LOCALLY, from this machine's LM Studio metadata or the GGUF header/metadata I parsed.
- **[INFERENCE]** = a reasoned inference, not directly documented by NVIDIA. Marked explicitly every time.

Baby claims are all OBSERVED LOCALLY (Baby is not a public model); file citations are given instead.

---

## PART 1 — Nemotron 3 Nano 4B: what it is and how it is built

### 1.1 Identity and lineage

NVIDIA-Nemotron-3-Nano-4B-BF16 is a 3.97B-parameter **[UPSTREAM]** small language model, released 3/16/2026
**[UPSTREAM]**, "trained from scratch by NVIDIA" and then **compressed from NVIDIA-Nemotron-Nano-9B-v2 using
the Nemotron Elastic framework" [UPSTREAM]** (model card). It is a hybrid Mamba2/Transformer model whose
ultimate architecture family descends from the Nemotron-H hybrid Mamba-Transformer line **[UPSTREAM,
arXiv:2504.03624]**. Data freshness (pretraining cutoff) is September 2024 **[UPSTREAM]**; model training dates
are Dec 2025–Jan 2026 **[UPSTREAM]** (i.e., the 9B-v2 parent was pretrained on pre-Sep-2024 data, then the 4B
was elastically compressed from it more recently).

### 1.2 Architecture (config.json + GGUF header cross-checked)

| Field | Value | Source |
|---|---|---|
| Architecture class | `NemotronHForCausalLM` / `nemotron_h` | **[UPSTREAM]** config.json |
| Total parameters | 3.97 × 10^9 | **[UPSTREAM]** model card |
| Hidden size | 3136 | **[UPSTREAM]** config.json; **[LOCAL]** GGUF `nemotron_h.embedding_length=3136` (cross-confirmed) |
| Layers | 42 | **[UPSTREAM]** `num_hidden_layers=42`; **[LOCAL]** GGUF `block_count=42` |
| Attention heads (global) | 40, head_dim 128, 8 KV heads (GQA) | **[UPSTREAM]** config.json |
| Attention layers actually present | **only 4 of 42 layers** — "primarily Mamba-2 and MLP layers combined with just four Attention layers" | **[UPSTREAM]** model card; **[LOCAL]** GGUF per-layer `attention.head_count_kv` array is 0 for most of the 42 layers and 8 only at a handful of positions, consistent with 4 attention layers |
| Mamba-2/SSM params | `mamba_head_dim=80`, `mamba_num_heads=96`, `ssm_state_size=128`, `n_groups=8`, `conv_kernel=4`, `expand=2` | **[UPSTREAM]** config.json; **[LOCAL]** GGUF confirms `ssm.state_size=128`, `ssm.conv_kernel=4`, `ssm.group_count=8` |
| Per-layer type pattern | `hybrid_override_pattern`: interleaved Mamba2 ("M"), attention ("*"), and MLP-only ("-") layers across the 42-layer stack | **[UPSTREAM]** config.json field `hybrid_override_pattern` |
| MLP activation | `relu2` (squared ReLU), `intermediate_size=12544` (4× hidden) | **[UPSTREAM]** config.json |
| Norm | `rms_norm_eps`/`layer_norm_epsilon` both `1e-05`; **[INFERENCE]**: Nemotron-H-family models are documented elsewhere (arXiv:2504.03624) as using RMSNorm-style blocks; this was not independently re-verified against modeling code in this pass, so treat the exact norm type (RMSNorm vs. LayerNorm) as **[INFERENCE]**, not confirmed here | config.json |
| Position encoding | Learned/rotary hybrid; `max_position_embeddings=262144`; `rope.dimension_count=78` on attention layers only | **[UPSTREAM]**/ **[LOCAL]** |
| Embeddings | **`tie_word_embeddings: false`** — separate input embedding and output (`lm_head`) matrices | **[UPSTREAM]** config.json; **[LOCAL]** GGUF shows distinct `token_embd.weight` and `output.weight` tensors, both `[3136 × 131072]`, confirming untied at the weight level |
| Vocabulary | 131,072 tokens, BPE (`gpt2`-style byte-BPE, `pixtral` pretokenizer regex), 269,443 merge rules | **[UPSTREAM]** config.json `vocab_size`; **[LOCAL]** GGUF `tokenizer.ggml.tokens`/`merges` |
| Reserved special tokens | tokens 18 through well past 249 are literal placeholders `<SPECIAL_18>` … `<SPECIAL_2xx>` reserved but not yet semantically assigned, alongside functional tokens for chat (`<\|im_start\|>`/`<\|im_end\|>`), reasoning (`<think>`/`</think>`), and tool use (`<tool_call>`/`[TOOL_CALLS]` etc.) | **[LOCAL]** GGUF + **[UPSTREAM]** tokenizer_config.json |
| Context length | up to 262,144 tokens | **[UPSTREAM]** model card / config.json |

**[LOCAL]** GGUF tensor-type histogram (263 tensors total) shows a genuinely mixed-precision quantization
(the file is a Q4_K_M community re-pack): the embedding table and `ffn_up` tensors are kept in a higher-precision
type (GGML type 6) while `ffn_down`/norm/SSM tensors use a 4-bit K-quant type — i.e., quantization granularity is
per-tensor-role, not uniform. This is a third-party (`lmstudio-community`) packaging choice, not necessarily
representative of NVIDIA's own BF16/FP8 release; flagged as **[LOCAL]**, not upstream.

### 1.3 Training/compression method **[UPSTREAM, from abstracts]**

- **Nemotron Elastic** (arXiv:2511.16664): "a framework for building reasoning-oriented LLMs, including hybrid
  Mamba-Attention architectures, that embed multiple nested submodels within a single parent model... Each of
  these submodels shares weights with the parent model and can be extracted zero-shot during deployment
  without additional training." Enabled by "an end-to-end trained router, tightly coupled to a two-stage
  training curriculum," "group-aware SSM elastification," "heterogeneous MLP elastification," "normalized
  MSE-based layer importance for improved depth selection," and "knowledge distillation enabling simultaneous
  multi-budget optimization." Applied to Nemotron Nano V2 12B, producing nested 9B/6B submodels using only
  110B training tokens (>360× cheaper than training separate models from scratch). This is the mechanism by
  which the 4B model here was derived from the 9B-v2 parent (per the model card).
- **Nemotron-H** (arXiv:2504.03624): the base architectural family — hybrid Mamba/Transformer replacing most
  self-attention with Mamba layers for constant per-token compute/memory, claimed "up to 3× faster at
  inference" vs. same-accuracy Transformers; introduces "MiniPuzzle" pruning+distillation compression and an
  FP8 training recipe.
- **[INFERENCE]**: The exact loss functions, distillation temperature, teacher-selection policy, and precise
  layer-importance thresholds used for the specific 9B→4B compression step were not independently verified
  beyond the abstract in this pass; treat those specifics as unconfirmed pending a full-text read of
  arXiv:2511.16664.
- Post-training data **[UPSTREAM]**: >10 trillion pretraining tokens; post-training corpus explicitly
  multilingual, multi-domain (code/legal/math/science/finance/QA/alignment), built with dozens of distinct
  teacher models (DeepSeek-R1/V3, Qwen2.5/3 family, Mixtral, phi-4, gpt-oss-120b, etc.) generating diverse
  synthetic reformulations of the same underlying seed datasets (e.g. six+ independently-generated "Synthetic
  Art of Problem Solving" variants from different teacher models over the same AoPS/AMC seed corpus).

---

## PART 2 — Baby (DaveLM): architecture, history, and current status

*(Condensed from two independent full-context research passes over the codebase; all facts below are file-cited
in `INSPECTED_PATHS.md`; page/line citations for every individual claim were recorded by the two research
sub-agents and are available on request — this section is the synthesized narrative.)*

### 2.1 Repository identity

The **currently maintained** DaveLM v0.9 codebase is `C:\DaveLM-v0.9` (confirmed newer than the stale copy at
`C:\DaveLM-CADAVER\DaveLM-v0.9`, which was snapshotted no later than 8/31/2026 and never updated again). The
active **research/forensic work** (treatments, pilots, P0–P7, HR1–HR3, fact_supervision, SF1) all happens
directly under `C:\DaveLM-CADAVER` root, which imports the frozen v0.9-lineage model builder as a dependency.

### 2.2 Base architecture (unchanged since v0.7, carried through every treatment)

Baby is a **from-scratch, hand-written, decoder-only GPT-style Transformer** (no HF/`transformers` dependency):
8 blocks, hidden width 320, 10 attention heads (head_dim 32), FFN 1280 (4× width), context 256, ReLU
activation, an explicit (non-fused, mean/var computed manually to avoid a ROCm nondeterminism bug) pre-norm
LayerNorm, additive residual streams, learned absolute position embeddings, and **untied** input/output
embeddings. Total parameters: **10,594,944**, frozen at this size from v0.7 through v0.9 and through every
CADAVER treatment (which only ever bolt small modules — a few hundred thousand parameters — on top; the
Transformer blocks themselves are never touched by any treatment/HR/SF1 experiment). Tokenizer: 1,024-piece
BPE, unchanged since v0.7.

Eight separate experiments tried variants of **tied** input/output embeddings (with straight-through
estimators, variance/matched scaling, rotation/identity-aligned initialization) specifically to test whether
tying would let Baby generalize copying to never-trained ("reserved"/"independent") identity tokens. **All
eight failed** — either the positive control itself failed, or established-token copying regressed, or (in the
one case where established copying was fully recovered) transfer to never-routed tokens remained exactly 0%
in every condition. Baby's production models have used **untied** embeddings throughout.

A long chain of "treatment" experiments (5 through 13) progressively built and refined a bolt-on
retrieval/localization mechanism to solve **counterfactual query→answer binding** (which of two candidate
context rows/names is the correct answer to a query): early logit-bias/attention-shortcut designs (treatments
5–11) mostly failed at or near chance; Treatment-12 (an explicit, row-position-assisted query-conditioned
retrieval module) achieved verified causal 100% success; Treatment-13 removed the position scaffold and made
the model **learn** to localize the two candidate rows itself, culminating in the
"orthogonal shared-core + unbounded antisymmetric" variant reaching 99.06% exact-answer accuracy — the
**"graduation-level" checkpoint** that is the parent of all subsequent work (Pilot 1, HR1/HR2/HR3, SF1).

### 2.3 Developmental timeline (condensed)

1. **v0.9 gate (pre-Sep):** validated **geometry-invariant contextual copying** (COPY rung of the declared
   COPY→RETRIEVE→BIND→REASON capability ladder) — 95.6% on withheld geometry.
2. **Treatment 5–13 (Sep 2–4):** built and validated the **BIND** rung — counterfactual binding/localization —
   culminating in the graduation-level T13 checkpoint (99.06% exact, causally verified via ablation/forced-row/
   query-swap interventions).
3. **language_pilot_0 (unrestricted English fine-tune):** language perplexity improved sharply, but binding
   accuracy **degraded** (100%→91.25%), demonstrating that naive joint English/binding training damages the
   validated capability.
4. **language_pilot_1 (protected-block training):** freezing blocks 0–3 and the specialized T13 modules during
   English-only updates **preserved binding perfectly** (100%/97.5%→100%) at the cost of smaller language
   gains. This became the standing curriculum rule for all subsequent language work.
5. **P0–P7 (Sep 5):** an eight-stage language-capability ladder. P0–P5 each failed for a different reason
   (incoherent drift, narrow memorized templates, subject-substitution errors, verbatim corpus memorization).
   **P6's apparent 12/12 success is a content-independent fixed template** (an issue this audit's sub-agent
   identified directly from raw generations; no existing Baby document flags it). **P7 is the first genuine,
   non-templated, non-memorized narrow compositional success** (12/12 on a fixed panel, zero training-text
   overlap) — but a separate 8-prompt "naturalistic transfer" panel (ordinary, unconstrained English outside
   the trained templates) largely failed.
6. **Post-P7 report cards (v1–v3d, Sep 5):** an attempt to formally measure naturalistic/contextual-selection
   transfer repeatedly hit **construction bugs** — a parent-lineage misattribution (P7's true parent is P5, not
   P6 as originally recorded) and a 144/288-item answer-key defect in the v1 battery — both independently
   confirmed and formally withdrawn via erratum documents. Corrected batteries (v2, v3d) measured performance
   at **exactly chance** with near-zero decision margins, but v3d's own methodology was subsequently also
   found defective (prompt reuse, missing answer keys) and separately retracted. **Net honest status: broad
   naturalistic transfer remains unestablished — not proven-failed, not proven-passed.**
7. **HR1/HR2 (Sep 6, multiple seed variants):** discovered to have been trained with a **shifted-label causal
   training bug** (`y[i,1:len(z)] = z[1:]` instead of the correct `y[i,:len(z)-1] = z[1:]`), which trivializes
   next-token prediction into same-position token-copying. Their apparently excellent training loss was an
   artifact; correctly-scored, their real next-token ability is *worse than their own untrained parent*, and
   autoregressive generation collapses into fixed-token repetition loops. **HR1/HR2 must be treated as forensic
   artifacts, not valid language parents** — precisely per the mission's framing.
8. **HR3 (Sep 6):** a bug-free re-baseline (corrected alignment, block 3 additionally unfrozen). Confirmed
   non-pathological (normal sentence CE). But on both a matched-pair reversal test and a clean single-fact
   acquisition test, HR3 performs **identically to its Pilot-1 parent — exactly at chance** (50%, 0
   reversals). No new capability was established; the researchers explicitly stopped rather than pushing
   forward without new evidence.
9. **fact_supervision_87001 (+corrected v1–v8, Sep 5–6):** two genuine `STOP` episodes from combinatorial/
   design conflicts (not coding bugs) in constructing a clean two-fact battery with disjoint holdouts, plus
   further integrity findings discovered only after later audits (missing answer keys, prompt contamination,
   an "incomplete executable freeze" caught by an independent freeze-completion audit). The authoritative v8
   evaluation: **both** the Factual-trained and a matched Control-trained checkpoint scored ~50.5% (chance) on
   a 192-item battery — explicit classification **FAIL** — while binding stayed intact (80/80).
10. **single_fact_acquisition_sf1 (Sep 6):** the simplest possible factual interface (16 items, 4 families, 2
    actor-pairs). **Stopped by a pre-registered regression guard at update 100/200** before reaching its
    acquisition gate. Result was **asymmetric across the two actor pairs**: one pair (Mia/Nora) reached
    4/4 correct with properly balanced responses; the other pair (Alex/Owen) stayed at 2/4, with the model
    outputting a fixed default ("Alex") regardless of which actor was actually correct.
11. **sf1_readout_selection_forensic_v1 (Sep 6, evening):** on 256 unrelated contexts having nothing to do with
    the trained facts, SF1's update raised the four trained names' first-token logits by +3.6 to +4.8 on
    average (their combined probability mass rose ~75×), while true-next-token probability on ordinary
    unrelated text *fell*, and general TinyStories cross-entropy regressed on 78.4% of measured tokens. This is
    a **broad, context-independent prior boost to the four trained names**, not a context-conditioned
    relational-retrieval mechanism. On the Owen-correct training items specifically, the update *did* raise
    Owen's logit substantially (+12.15) and probability (+0.27), yet Alex still won top-1 in all 4 cases — a
    "signal increased but suppressed" pattern.
12. **sf1_component_swap_forensic_v1 (Sep 6, latest/most current):** a 2×2×2 factorial swap (upstream
    blocks/embeddings, final-norm, output head) localizes: (a) the **real, partial relational-selection gain
    (9/16→12/16 correct) is upstream-dominant and head/norm-independent** — it genuinely lives in transformer
    blocks 4–7; (b) the **broad name-prior pollution is a distributed upstream×head interaction**, i.e. partly
    in the output head too; (c) the **Owen-suppression/Alex-default bias could not be localized to any single
    component** by this intervention — every combination tested still failed on Owen-correct items. This is
    the most current and most mechanistically precise account of why Baby currently fails at ordinary factual
    acquisition.

### 2.4 What is genuinely established vs. not

**Established:** geometry-invariant contextual copying; counterfactual query→row binding/localization (surviving
protected-block language training essentially perfectly); a narrow, non-templated compositional-English
capability confined to a specific synthetic relation/vocabulary set (P7); a precise mechanistic diagnosis of the
HR1/HR2 training bug; a bug-free HR3 re-baseline; a partial, localizable (blocks 4–7) factual-selection gain in
SF1 that does not yet clear its own acquisition gate.

**Not established / open / retracted:** broad naturalistic (non-templated) English transfer; reliable
ordinary-English single- or two-fact selection above chance (repeatedly measured at ~50% across three
independent lineages/batteries); any reasoning or general conversational competence claim (never attempted;
explicitly disclaimed everywhere). P6's apparent success is likely spurious (a previously unflagged issue this
audit surfaced).

**Frozen scientific safeguards observed:** `FREEZE_RECEIPT.json`+detached SHA-256 checksums to certify
battery/bundle bytes before any model touches them; `independent_validator.py` scripts deliberately written
without importing the builder's own code (this caught the P7 answer-key defect and two combinatorial
impossibilities); `PREFLIGHT`/`PROTOCOL`/`MANIFEST` files declaring parent hash, tokenizer hash, and parameter
scope before execution; an explicit `STOP`-and-escalate mechanism used whenever a design conflict or integrity
gap is found (used at least 3 times); and additive-only erratum documents that withdraw prior interpretations
without altering historical artifacts. A separate independent audit (`FREEZE_COMPLETION_AUDIT_20260905.md`)
even caught a case where a "frozen" bundle had a receipt but was not actually executable as specified — i.e.,
the safeguards are actively used and have caught real problems, not just present as boilerplate.

---

## PART 3 — Comparative findings

For each finding: what Nemotron does, evidence, what Baby does instead, the Baby failure it targets, mechanistic
rationale, scale-transfer risk, whether Baby already has an equivalent, the smallest diagnostic, supporting/
falsifying evidence, and risk to Baby's proven capabilities. Findings are ranked **A (high value)**, **B
(diagnose first)**, **C (interesting/premature)**, **D (do not copy/irrelevant)**.

---

### Finding 1 — Preserve prior behavior when adding capability (distillation/replay-to-parent) — **Rank A**

1. **What Nemotron does:** Nemotron Elastic's central mechanism for adding/removing capability across model
   budgets is "knowledge distillation enabling simultaneous multi-budget optimization" inside "a two-stage
   training curriculum" that keeps nested submodels' behavior calibrated against the parent **[UPSTREAM,
   arXiv:2511.16664 abstract]**. The explicit design goal is that compressing/modifying a model to add new
   deployment configurations must not silently degrade its other, already-working behavior — the entire value
   proposition of "many-in-one" only works if each submodel is separately well-calibrated.
2. **Evidence it actually does this:** The abstract states nested submodels "perform on par or better than the
   SoTA in accuracy" after the elastic/distillation process, i.e. capability is not allowed to silently regress
   during the compression/adaptation step. **[INFERENCE]**: the exact loss formulation was not verified beyond
   the abstract in this pass.
3. **What Baby does instead:** Baby's only mechanism for preventing collateral damage during new-capability
   training is a coarse, static **parameter-freezing mask** (protected blocks 0–3, or 0–2 with block 3
   unfrozen in HR3) — a blunt instrument that either fully protects or fully exposes a block, with no
   fine-grained behavioral-preservation signal. Nothing in the reviewed code trains against a
   *distributional* target that says "keep your unrelated-context output distribution close to what it was
   before this update."
4. **The specific Baby failure this addresses:** the SF1 readout-selection forensic's **D3 finding** —
   training on 16 factual items produced a **broad, context-independent boost** to the four trained names'
   output-token prior across 256 completely unrelated contexts (probability mass +75×), while general
   next-token accuracy on ordinary text *regressed* (worse on 78.4% of measured tokens). This is exactly the
   kind of "new capability silently pollutes old behavior" failure that a distillation-to-parent/replay
   objective is designed to prevent, and it is also plausibly implicated in the unresolved
   "Alex-default" bias (component-swap forensic: broad prior pollution is `DISTRIBUTED_INTERACTION` between
   upstream blocks and the output head).
5. **Mechanistic reason it could help:** adding an explicit term that penalizes KL-divergence (or output-logit
   drift) between the in-training model and the frozen pre-treatment parent, evaluated on a fixed sample of
   *unrelated* contexts (not just the target facts), directly targets the exact quantity that regressed (D3).
   Because the effect measured is diffuse and non-selective (affects all four trained names roughly equally
   regardless of context), a global regularizer of this form is well-matched to a global, non-selective
   pathology — unlike the freezing mask, which cannot distinguish "acceptable capability gain" from
   "unwanted prior pollution" within the unfrozen blocks.
6. **Reasons it may NOT transfer to an ~11M-parameter Baby:** Nemotron's technique operates across
   *architecturally nested* submodels with billions of parameters and hundreds of billions of distillation
   tokens — a completely different regime. At Baby's scale, with only 16–192 training items, a KL-to-parent
   term evaluated on a small fixed unrelated-context sample could itself overfit to that sample, or could
   simply suppress *all* learning (including the genuine, partial, upstream-localized relational gain already
   observed in blocks 4–7), producing a Baby-specific failure mode with no large-model analogue to warn
   against it in advance.
7. **Does Baby already have an equivalent mechanism?** Partially — the block-freezing mask is a *structural*
   (all-or-nothing) analogue, but there is no *distributional*/replay-based analogue anywhere in the reviewed
   code. This is a genuine gap, not a redundant reinvention.
8. **Smallest diagnostic/bounded experiment (NOT executed):** Take the existing frozen SF1-update-100 training
   recipe and add a single additional loss term — KL(model_logits(unrelated_context) ‖ parent_logits(same
   context)) — computed on the *same* 256-context D3 replay set already used for measurement (or a disjoint
   held-out half of it, to avoid teaching-to-the-test), with a small weight swept over 2–3 values. Compare, at
   the same update count (100) and same acquisition-gate metrics already defined for SF1: (a) does the D3 prior
   inflation shrink relative to the un-regularized run? (b) does the partial Owen-correct relational gain
   (currently suppressed by the Alex-default) survive or also get suppressed by the KL term? This reuses SF1's
   own pre-registered gates and forensic instrumentation almost unchanged — it is a very small, cheap addition.
9. **Evidence that would support it:** D3-style prior inflation on held-out unrelated contexts shrinks
   substantially (e.g. combined probability mass increase drops from ~75× toward something close to 1×) while
   Owen-correct accuracy does not regress below its current (still-failing) baseline, and ideally the
   Owen-correct margin (currently −1.05 to −0.28 across component-swap conditions) moves further toward zero
   or positive.
10. **Evidence that would weaken/falsify it:** if adding the KL term causes acquisition accuracy on the 16
    training items to *regress* (i.e., the regularizer suppresses the genuine signal along with the pollution),
    or if D3-style prior inflation persists unchanged (implying the pollution is not being driven by the kind
    of update the KL term can suppress — e.g., if it is instead an artifact of the specific loss weighting or
    optimizer dynamics rather than a distributional drift the KL term can see).
11. **Risk to Baby's proven contextual-binding/language-retention capability:** low-to-moderate if scoped
    correctly — the existing protected-block mask (0–3, or updated per HR3) should remain in force, so the KL
    term would only apply within the already-unfrozen blocks 4–7 and the head, i.e., it operates inside the
    region already known to be where the change (both wanted and unwanted) occurs, not on the protected
    binding-critical blocks. Still, any new loss term is a new place for the regression-guard-triggering
    behavior seen in SF1 to reappear in a different form; the diagnostic should keep SF1's existing
    `STOP_REGRESSION` guard active.

---

### Finding 2 — Increase within-capability lexical/instance diversity to dilute frequency-prior shortcuts — **Rank B**

1. **What Nemotron does:** its post-training corpus is deliberately built from **many independently-generated
   synthetic variants of the same underlying seed material**, produced by multiple different teacher models
   (e.g. the same AoPS/AMC seed corpus reprocessed independently by DeepSeek-R1, then again by Qwen2.5-32B/
   Qwen2.5-Math-72B/Qwen2.5-72B-Instruct, then again by gpt-oss-120b+Qwen2.5-32B-Instruct — three separate
   "Synthetic Art of Problem Solving" dataset rows in the same table) **[UPSTREAM, model card dataset table]**.
2. **Evidence it actually does this:** directly visible in the model card's training-dataset table — the same
   named seed sources (OpenStax, AoPS/AMC, Stack Exchange, Common Crawl) recur across many rows with different
   generating teacher models and different token counts, i.e. deliberately broad surface-form diversity over a
   narrow set of underlying facts/seed documents. **[INFERENCE]**: NVIDIA does not explicitly state this
   diversity is intended to prevent frequency/shortcut bias; that specific causal claim is my inference from
   general training dynamics, not a documented NVIDIA rationale.
3. **What Baby does instead:** Baby's factual-acquisition curricula (`fact_supervision_87001` and `SF1`) use
   an intentionally **minimal, tightly controlled set of surface forms and identities** — e.g. SF1 uses exactly
   2 actor names per role-pair (Alex/Owen, Mia/Nora) across 16 training items total. This is a deliberate
   scientific-control choice (to keep the experiment interpretable), not an oversight — but the audit surfaced
   that it may itself be *inducing* the frequency-shortcut pathology under study: with only 2 candidate names
   per pair and asymmetric introduction order/likely token frequency, the base-rate imbalance between "Alex"
   and "Owen" may be large enough, relative to the weak relational signal, to dominate.
4. **The specific Baby failure this addresses:** the SF1 "Alex-default" bias — component-swap forensic found
   this bias is real, partially responsive to the relational signal (Owen's logit and probability both rise
   substantially on Owen-correct items) yet never wins, and could not be localized to any single architectural
   component ("MIXED_OR_AMBIGUOUS... not localized by this intervention").
5. **Mechanistic reason it could help:** if the bias is substantially a base-rate/frequency artifact of the
   tiny 2-name candidate pool (rather than a deep architectural incapacity), then widening the pool (more actor
   names per role, more surface phrasings of the same relation, roughly balanced first-appearance order) should
   directly reduce the magnitude of the frequency prior the relational signal has to overcome, without any
   architecture change at all.
6. **Reasons it may NOT transfer to Baby's regime:** this is the opposite of a scale-transfer risk — it is
   actually more native to Baby's regime than to Nemotron's (data diversity costs are trivial at Baby's data
   scale). The real risk is different: widening the candidate pool changes the *experimental design itself* —
   it stops being the same clean minimal-pair test that SF1's frozen pre-registration specifies, so it would
   need to be run as a **new, separately pre-registered experiment**, not a change to SF1 itself.
7. **Does Baby already have an equivalent mechanism?** No — no experiment reviewed varies candidate-pool size
   or surface-form diversity as an independent factor; every fact-acquisition/single-fact experiment reviewed
   (fact_supervision_87001 and its corrected variants, SF1) uses a small, fixed, symmetric-by-design candidate
   set. This is a genuine gap in what has been tried, not a re-discovery of an existing Baby mechanism.
8. **Smallest diagnostic/bounded experiment (NOT executed):** a new, small, pre-registered variant of SF1 that
   keeps the same relation/objective/curriculum (16-per-family style, protected-block masking, causal
   answer+EOS CE, same regression guard) but doubles or triples the number of distinct actor names per role
   (e.g. 4–6 names instead of 2), while holding total training-update budget fixed, and checks whether the
   Alex-style "always pick one fixed name" pathology reappears with a *different* single name, or whether it
   disperses (each name gets picked roughly at its own base rate rather than one name dominating).
9. **Evidence that would support it:** the fixed-default pathology weakens (no single name captures a
   disproportionate share of wrong answers) as candidate-pool size grows, at the same or better acquisition
   accuracy.
10. **Evidence that would weaken/falsify it:** the pathology persists at the same severity (some single name
    still wins nearly all ties) even with a larger, more balanced candidate pool — this would support the
    component-swap forensic's own tentative framing that the mechanism is not simply a base-rate artifact.
11. **Risk to Baby's proven capabilities:** minimal — this is a new, separate, small-scale experiment layered on
    the existing protected-block curriculum; it does not touch the binding/copy machinery or its frozen gates.

---

### Finding 3 — Reserved/inert vocabulary slots for future extensibility — **Rank C**

1. **What Nemotron does:** allocates ~100+ literal placeholder tokens (`<SPECIAL_18>` through beyond
   `<SPECIAL_249>`) in its 131,072-token vocabulary, alongside functional control tokens for chat, reasoning,
   and tool-calling, so that future capabilities can be added without resizing the embedding/output matrices
   **[LOCAL, GGUF tokenizer.ggml.tokens + UPSTREAM tokenizer_config.json]**.
2. **Evidence:** directly observed in both the GGUF tokenizer metadata and the HF tokenizer_config.json —
   tokens 18 onward are explicitly named as placeholders, distinct from the functional special tokens (0–17).
3. **What Baby does instead:** Baby's 1,024-piece BPE vocabulary has no reserved/placeholder slots documented in
   the reviewed tokenizer files; its "never-routed"/"reserved"/"independent" identity-token experiments
   (matched_scale_tied_copy, straight_through variants, aligned_output variants) instead tested whether an
   *existing* trained model could generalize copying to token IDs it had simply never been trained to emit as
   answers — a different question (post-hoc generalization) from Nemotron's approach (pre-allocated,
   intentionally-inert capacity).
4. **The specific Baby failure this addresses:** loosely related to the "0% transfer to never-routed/reserved
   identity tokens" result across all eight tied-embedding experiments, but this is not really the same
   problem — Nemotron's reserved tokens are inert until deliberately trained, not expected to generalize
   zero-shot either.
5. **Mechanistic reason it could help:** none demonstrated; this is an extensibility/engineering convenience for
   a production tokenizer roadmap, not a solution to Baby's demonstrated 0%-transfer finding.
6. **Reasons it may not transfer:** at 1,024 vocab pieces, Baby has essentially no room for large reserved
   blocks without materially shrinking usable subword coverage; the problem this solves for Nemotron (avoiding
   embedding-matrix resizes across a long product roadmap) does not obviously apply to a single active research
   codebase like Baby's.
7. **Does Baby already have an equivalent?** No, and there's no evidence it needs one yet.
8. **Smallest diagnostic:** N/A — no clear connection to a demonstrated Baby failure; not worth designing an
   experiment around at this time.
9–10. Not applicable given rank.
11. **Risk:** none identified; this finding is included for completeness of the comparison, not as an actionable
    item.

---

### Finding 4 — Hybrid Mamba-2/Attention SSM backbone, GQA, 262K context — **Rank D (irrelevant at Baby's scale)**

1. **What Nemotron does:** replaces all but 4 of 42 layers with Mamba-2 state-space blocks, uses grouped-query
   attention (8 KV heads for 40 query heads) on the remaining attention layers, and supports 262,144-token
   context, primarily to reduce per-token inference compute/memory for edge deployment **[UPSTREAM]**.
2. **Evidence:** config.json + GGUF header confirm the layer pattern and dimensions; the model card explicitly
   frames this as an inference-cost/edge-deployment optimization, not a capability/quality mechanism per se.
3. **What Baby does instead:** a small, dense, full-attention Transformer (8 layers, all standard multi-head
   attention, context 256).
4. **The specific Baby failure this addresses:** none of Baby's demonstrated failures (frequency-prior
   pollution, naturalistic-transfer gap, factual-selection chance-level performance) are inference-cost,
   memory-bandwidth, or long-context problems.
5. **Mechanistic reason it could help:** none identified for Baby's actual bottlenecks.
6. **Reasons it may not transfer:** Mamba-2/SSM training is materially harder to implement correctly, has its
   own well-known training-stability pitfalls, and would require rebuilding Baby's entire hand-written training
   stack; at 256-token context and ~11M parameters, none of the efficiency benefits (which come from avoiding
   O(n²) attention/KV-cache growth over very long sequences) are relevant.
7. **Does Baby already have an equivalent?** No, and it does not need one.
8–10. Not applicable.
11. **Risk:** would be a large, high-risk architectural rewrite with no plausible connection to any demonstrated
    Baby failure — explicitly **not recommended**.

---

### Finding 5 — Untied input/output embeddings — Baby's existing choice is corroborated, not a new recommendation — **Rank D (do not revisit)**

1. **What Nemotron does:** `tie_word_embeddings: false` **[UPSTREAM config.json]**, confirmed at the weight
   level by two distinct `token_embd.weight`/`output.weight` tensors in the GGUF **[LOCAL]**.
2. **Evidence:** direct config field + tensor inventory.
3. **What Baby does instead:** Baby's production models have always used **untied** embeddings (the base
   `DaveLM` class never ties them); eight separate, carefully controlled experiments testing various tied/
   scaled/aligned-tied variants (for a different purpose — generalizing to never-routed identity tokens) all
   failed to demonstrate any benefit, and none of the CADAVER treatment models use a tied configuration.
4. **Relevance:** this is not a Baby failure needing a fix — it is a case where Baby's already-established
   default (untied) matches the choice made by a much larger, heavily-evaluated production model, at a vastly
   different scale. This corroborates that continuing to invest in tied-embedding variants is unlikely to be
   fruitful and should be deprioritized, not revisited, absent new evidence specific to Baby's actual failure
   modes (which are about factual selection/frequency bias, not embedding tying).
5–11. Not applicable — this is a "do not pursue further" finding, included because the comparison surfaced it
    naturally, not because it is a new recommendation.

---

## PART 4 — Explicit call-outs required by the mission

### Nemotron features irrelevant at Baby's scale
- Mamba-2/SSM hybrid backbone, GQA, 262K context (Finding 4).
- FP8 training recipe (irrelevant; Baby's numerical-stability problem, if any, was already solved locally via
  the explicit-LayerNorm ROCm workaround, a different issue).
- The "Elastic" nested-submodel/zero-shot-extraction router itself (as a deployment mechanism) — Baby has no
  deployment-budget-family requirement; only the *distillation-for-calibration-preservation* idea underneath it
  (Finding 1) is plausibly relevant, not the router/nesting mechanism itself.
- Massive multilingual/tool-calling/reasoning-trace scaffolding — Baby has no coherent free-form generation yet
  (naturalistic transfer is unestablished); this is premature by a wide margin.

### Ideas Baby should NOT copy
- Do not attempt to switch to a Mamba/SSM backbone (Finding 4).
- Do not resume tied-embedding experimentation on the strength of "Nemotron uses X" reasoning — Nemotron uses
  the *same* untied choice Baby already made (Finding 5); this is a non-issue, not an opportunity.
- Do not adopt a large reserved-vocabulary-slot scheme without a demonstrated need (Finding 3).
- Do not adopt reasoning-trace/chat-template scaffolding before naturalistic language transfer itself is
  established — this would add complexity on top of an already-unestablished capability.

### Areas where Baby's existing approach appears preferable
- Baby's frozen-gate/preregistration/independent-validator/STOP methodology (Section 2.4) is materially more
  rigorous, in a mechanistic-verification sense, than anything documented in Nemotron's public materials, which
  rely on standard aggregate benchmark scores (BFCL, IFEval, GPQA, etc.) rather than causal
  ablation/intervention studies of the kind Baby routinely performs (e.g. Treatment-12's forced-wrong-row and
  query-swap causal validation, or the SF1 component-swap forensic). This is a genuine strength worth
  explicitly preserving as Baby's research culture scales up, not something to dilute in pursuit of
  Nemotron-style aggregate benchmarking.
- Baby's small-model, fully-interpretable, causally-verified approach to a novel mechanism (Treatment-12/13's
  localizer) is arguably more scientifically rigorous evidence of "why" a mechanism works than is available for
  any comparably-sized module inside Nemotron (which is treated as a black box in this audit — no Nemotron
  modeling-code-level causal ablation evidence was available to inspect, only the published architecture and
  aggregate eval numbers).

### Suspicious Baby design/training choices exposed by comparison
- The tiny (2-name-per-role) candidate pools used throughout fact_supervision_87001 and SF1, while good for
  controlled science, may themselves be *inducing* part of the frequency-prior pathology under study (Finding
  2) — worth treating as a confound in future single-fact-acquisition experiment design, not just a fixed
  methodological virtue.
- Baby's explicit (non-fused) LayerNorm is a legacy ROCm-portability workaround (v0.2.1), not a deliberate
  choice made for its own merits; this is not connected to any currently demonstrated failure, but its origin
  as a hardware workaround (rather than a considered normalization-strategy decision) is worth flagging so it
  is not mistaken for a validated design choice if normalization-related issues arise in the future.

### Overlooked Baby bottlenecks suggested by the comparison
- Baby has never (in the reviewed material) measured or regularized against **non-selective/context-independent
  side effects** of a targeted training update — i.e., there is no existing instrumentation in Baby's
  curriculum-design toolkit analogous to Nemotron Elastic's calibration-preservation goal, until the ad hoc D3
  diagnostic in the SF1 readout-selection forensic uncovered it after the fact. Building this kind of check
  into future acquisition-gate pre-registrations (not just as a post-hoc forensic) would likely surface similar
  pollution earlier in other lines of work (e.g. it was not checked in fact_supervision_87001, HR1/HR2/HR3, or
  the language P0–P7 ladder as far as this audit found).

---

## Summary judgment

Baby's currently demonstrated bottleneck — ordinary-English factual selection stuck at chance, with a
partially-diagnosed frequency-prior/relational-signal competition inside blocks 4–7 and the output head — is
**not** an architecture-capacity problem that a bigger/fancier backbone (Nemotron's Mamba hybrid, GQA, huge
vocab) would obviously fix, and the audit found no credible evidence that copying any of Nemotron's
scale-motivated architectural choices addresses it. The two findings that plausibly *do* connect to Baby's
actual demonstrated failure — (1) explicit preservation of prior/unrelated-context behavior when adding new
capability (mirroring Nemotron Elastic's distillation-based calibration-preservation goal), and (2) increasing
within-capability candidate/surface-form diversity to dilute a suspected base-rate/frequency shortcut (mirroring
Nemotron's deliberately diverse multi-teacher synthetic data construction) — are both modest, cheap, bounded,
and reuse Baby's own existing pre-registration/forensic instrumentation almost unchanged. Neither is guaranteed
to work; both are falsifiable with the smallest diagnostics described above, without touching Baby's proven
binding/copy capability or its frozen safeguards.
