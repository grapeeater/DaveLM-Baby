# Baby v0.10 foundation lessons

Status: initial evidence synthesis for a new lineage. This document separates
observations from interpretations and is based on the underlying CADAVER
artifacts, not only the old handoff summaries. Protected TEST / FINAL / SACRED
material was not opened.

## Scope and lineage boundary

Baby v0.9 / DaveLM-CADAVER is closed for scientific treatment. Its checkpoints
are evidence only. Baby v0.10 starts from fresh random weights. The evidence
sources used here include:

- `phase2a_t34_novel_copy_rehabilitation_v1/T34_TERMINAL_REPORT.md`,
  `T34_RESULT.json`, `TRAINING_DATA_SPEC.md`, and leakage/bigram audits;
- `research_root_cause_investigation_v1/` and its evidence ledger,
  falsification matrix, tokenizer analysis, and root-cause report;
- `research_in_context_induction_probe_v1/REPORT.md` and `ANALYSIS.json`;
- `research_l7h2_causal_ablation_v1/` and
  `research_l7h2_write_readout_trace_v1/FINAL_VERDICT.md`;
- T24, T28, T30, T31, T32, and T33 terminal protocols/reports;
- `baby_vnext_60m_design_v1/`, Phase1G records, tokenizer records, and the
  external-model archaeology under `research/external_model_archaeology/`.

## 1. Directly established facts

### 1.1 The v0.9 base family contains a trainable copy substrate

T34 started from existing v0.9 weights and changed no architecture, tokenizer,
or new module. With two independent seeds and an objective requiring random
novel contextual reproduction, held-out D1 novel top-1 rose from `0.0000` to
`0.9938` and `0.9975`; median target rank became `1.0`. Whole-span greedy
free-running exact was `0.8000` and `0.7650` on the training surface. The
broken-context free-running control was exactly `0.0000` on both seeds.

Source: `phase2a_t34_novel_copy_rehabilitation_v1/T34_TERMINAL_REPORT.md`
sections 1, 5, 8, 9, and 14.

### 1.2 T34 generalized across content, length, and prior support, but not surface

T34 held-out same-format K=40 decision top-1 was `0.920/0.920`. Low-support
versus high-support top-1 was approximately `0.989/0.996` and
`1.000/0.993`, respectively. Held-out marker/template decision top-1 was only
`0.020/0.010` short and `0.005/0.010` long; held-out-marker free-running exact
was `0.040/0.040`. Broken order was `0.0825/0.0838` versus approximately `0.99`
intact. The learned solution therefore was order-sensitive and contextual, but
keyed to the trained surface interface.

Source: T34 terminal report sections 5, 8, and 9.

### 1.3 The write measurement moved with behavior

The T34 novel write measured at the prior causal pathway increased from
`+0.200654` to `+2.822661/+2.903230`, a `14.07x/14.47x` increase. The
novel/familiar write ratio changed from `0.225` to approximately `1.55`.
The write-to-readout trace found that later blocks preserved or amplified the
signal and that final normalization and the native LM head expressed it rather
than selectively erasing it.

Sources: T34 terminal report section 6;
`research_l7h2_write_readout_trace_v1/FINAL_VERDICT.md` sections 1--8.

### 1.4 The earlier objective did not measure the required capability

Phase1G passed its language/generalization protocol and became the Phase2A
parent, but its gates were language CE and retention-oriented. It did not have a
novel-copy or template-generalization exit gate. The induction probe on the
frozen foundation reported `INDUCTION_ABSENT` for novel items under the old
objective, while the same foundation later learned novel copy under T34's
objective. That is direct evidence that a healthy CE trajectory is not a
capability certificate.

Sources: `baby_vnext_phase1g_language_v1/PHASE1G_LANGUAGE_PROTOCOL.md`;
`research_in_context_induction_probe_v1/REPORT.md` and `ANALYSIS.json`;
T34 terminal report.

### 1.5 The v0.9 binding sidecar was not a learned foundation skill

The Phase1G/v0.9 records state that the 984,321 binding/localizer parameters
were initialization-identical/untrained in the parent. T34 kept them frozen and
did not use QA/binding retention as a meaningful positive result. This matters
for v0.10: copying an untrained sidecar would not reproduce a proven capability.

Source: `CANONICAL_STATE.md`, Phase1G records, T34 section 12, and
`baby_vnext_60m_design_v1/PROVENANCE.json`.

### 1.6 The basic architecture and tokenizer facts are concrete

The validated v0.9/vNext base is a 12-layer, `d_model=640`, 10-head,
`d_mlp=2560`, context-256, learned-absolute-position, explicit pre-LayerNorm
decoder with ReLU MLP, untied biased LM head, and vocabulary 1024. The v0_7
tokenizer is byte-level BPE with 763 merges and `add_prefix_space=false`.

Sources: `baby_vnext_60m_design_v1/BABY_VNEXT_CONFIG.json` and
`research_root_cause_investigation_v1/TOKENIZER_LEXICAL_ANALYSIS.md`.

## 2. Strong evidence

1. The dominant T1--T33 failure was substantially objective/curriculum
   mismatch: T34 produced the largest behavioral change in the lineage while
   leaving the model family and tokenizer unchanged.
2. The T34 solution was not a fixed answer inventory or a pure language prior.
   Its audits reported fresh random target spans, no diagnostic/training
   overlap, zero target-bigram support, 368 distinct emitted tokens, broken
   context exact zero, and low-prior performance comparable to high-prior.
3. The relevant computation is not just “attend to a previous token.” The
   measured challenge is writing an arbitrary context identity with enough
   amplitude to beat native vocabulary competition and then using it
   autoregressively.
4. Surface markers are load-bearing because T34's learned route found the answer
   onset through a particular marker realization. Changing marker vocabulary
   without training that variation caused collapse even though content and
   lengths generalized.
5. Tokenizer fragmentation is an amplifier and a confounder, not a demonstrated
   root cure. The tokenizer diagnostic found shared first-token targets at
   `4.7%` exact versus `53.1%` for unique-first targets, but replacement and
   append studies did not establish that a larger vocabulary alone fixes native
   generation.
6. T33's rank-8 readout adapter was active and trainable but stayed in the
   35--40/128 exact regime. This closes that tested low-complexity readout
   intervention, not every possible architecture.

## 3. Plausible hypotheses

These are live design hypotheses, not findings.

- A model trained from scratch on many surface realizations can learn a
  structural rule such as “locate the value paired with the queried key,”
  instead of a marker-specific route.
- Surface diversity must be present in the optimization stream, not only in a
  post-training probe. Otherwise the first decisive answer-onset operation can
  remain tied to one marker family.
- A 1024-token fragmented tokenizer may make multi-token copying harder, but
  explicit random span training can teach the model to overcome that amplifier.
  Retaining it in v0.10 foundation v1 gives a clean test of curriculum before
  spending capacity and provenance on a tokenizer redesign.
- A mixed language/copy objective can retain natural language while training a
  stronger write pathway, provided language CE and structured capability are
  both gated during training.
- The useful substrate may be distributed across heads and layers. A future
  causal trace should measure function, not require an L7H2 replica.

## 4. Falsified hypotheses

- **“Baby fundamentally lacks a trainable substrate for novel contextual
  copying.”** Falsified by T34's two-seed same-surface result.
- **“The copy signal is erased downstream.”** Not supported after the
  write/readout trace; the measured contribution was preserved/amplified and
  the readout expressed it.
- **“A low-complexity native readout adapter is sufficient.”** Falsified for
  T33's rank-8 adapter under its frozen rule.
- **“Appending vocabulary rows alone creates native emission.”** Falsified by
  the append study's zero new-token greedy emissions.
- **“A single T34 template is enough for structural generalization.”** Falsified
  by the held-out marker collapse.
- **“Low training loss, familiar exact match, or representation alone proves
  downstream capability.”** Falsified as a scientific decision rule by the
  repeated v0.9 trajectory and T34's contrast.

## 5. Unresolved questions

1. Can a fresh v0.10 initialization learn held-out template/marker
   generalization at this scale, rather than relying on v0.9's already-formed
   substrate?
2. How much surface diversity is sufficient, and which variation axes are
   necessary rather than decorative?
3. Does autoregressive training on generated prefixes improve free-running
   retention without introducing a new shortcut?
4. How far can length generalization extend when both layout and markers change?
5. Can the v0_7 tokenizer support robust low-prior multi-token spans, or does a
   justified tokenizer change become necessary after curriculum evidence?
6. What causal subnetwork does a fresh model develop? No layer/head index is
   assumed in advance.
7. What is the smallest model/compute budget that can pass the full foundation
   gates? The current evidence does not justify blind scaling.

## 6. Lessons that directly affect v0.10 design

1. Define capability gates before training and run them during foundational
   stages. CE is a health metric, not a graduation gate.
2. Train copy with fresh, unbounded target identities. A finite answer
   inventory allows memorization and is scientifically insufficient.
3. Vary marker identities, separators, wording, order, positions, lengths,
   distractors, and prior-support strata in the optimization distribution.
4. Freeze a held-out surface family that is disjoint in marker identities and
   template composition. A same-format held-out set cannot reveal surface
   binding.
5. Include broken-context and broken-order controls. A model that emits the
   answer without the source or after order destruction has not learned the
   intended operation.
6. Record target logit/rank/margin, intact-versus-broken lift, first-error
   position, free-running exact, and retention at every checkpoint.
7. Preserve a clean foundation boundary. No binding, QA, or instruction stage
   is proposed until novel content, surface, length, low-prior, distractor,
   negative-control, and retention gates all pass on replicated fresh seeds.
