# HR-3 contextual-selection autopsy — bounded nonsacred diagnostic

Status: SCIENTIFIC_FORK_REQUIRES_DECISION. No new treatment designed, trained, or selected. No optimizer was created. No weights were changed. FINAL and sacred material were not accessed.

## Finding

Correctly aligned sentence loss improves while factual selection remains weak because those measurements reward different behavior. HR-1/HR-3 train independent short sentences with no preceding story context, then are tested on two-fact retrieval followed by a response cue. The training program contains no matched English factual task or counterfactual selection objective. This establishes missing direct supervision in these treatments; it does not establish that sentence LM training can never learn the capability.

## Exact exposure and objective

The frozen sentence pool contains 149,014 records. The 450 English batches present 28,800 distinct records once each: 672,319 content targets and 28,800 EOS targets. Zero scheduled records contain an internal sentence boundary under the authoritative builder regex. Maximum scheduled content length is 95 tokens. There are 867 question-mark records; they do not constitute a designed question/answer task.

The builder splits stories at punctuation-plus-whitespace and retains individually encoded sentences of at most 96 tokens. Each row gets a fresh BOS and final EOS. A batch of 64 sentences is 64 independent contexts; batching does not restore cross-sentence evidence. Zero designed English counterfactual families or response-selection batches are supplied. Incidental within-sentence relations remain possible and were not excluded by this count.

The 50 binding updates supply 1,600 structurally assisted document presentations. Their objective is mean answer CE plus 1.0536573711078283 times hard-min localization. English updates use token-mean causal CE. These losses alternate; their numerical coefficients and update ratio are not a measured ratio of effective gradient pressure. There is no English factual-selection term to downweight accidentally. Weak effective pressure for this behavior is supported; a quantitative causal claim that binding gradients suppress it is not.

## Complete matched-pair analysis from preserved raw scores

For each reversal pair, fix candidate identities A/B and compute d0=logP(A|context0)-logP(B|context0), d1 for reversed facts. The assignment-even component is b=(d0+d1)/2. The correctly oriented context component is s=q(d0-d1)/2, with q=+1 if A is correct under assignment0 and -1 otherwise. The two correct margins are s+qb and s-qb. Both are strictly positive exactly when s>|b|. This is a decomposition of observed contextual scores, not a no-context prior subtraction or calibrated endpoint.

| Checkpoint | Same candidate across reversal | Positive context direction | Mean abs bias | Mean signed context component | Word-only reversals | Absolute candidate mass, mean |
|---|---:|---:|---:|---:|---:|---:|
| Pilot1_parent | 11/12 | 6/12 | 2.6052 | 0.00501 | 0/12 | 2.08e-07 |
| HR1_causal_aligned_500 | 12/12 | 6/12 | 3.6273 | 0.00249 | 0/12 | 3.65e-07 |
| HR3_block3_update_500 | 12/12 | 6/12 | 3.6654 | -0.00432 | 0/12 | 2.75e-07 |

HR-1 and HR-3 choose the same name on all four items of every family (6/6 families): Cal, Cam, Cory, Ira, Ron, Sid. Correct identities are balanced, so this yields exactly 12/24 without retrieving the changing relation. Their choices are identical across checkpoints. Both choose the first-mentioned name on 12/24 items; neither universal first- nor last-mention preference alone explains the whole battery. Name identity and within-family mention position are not independently permuted here, so lexical versus family-specific position/relation preference cannot be causally separated. Candidate ordering is not passed to the forward computation.

All 12 candidate-score differences change numerically under reversal, but only 6/12 changes point in the correct direction for every checkpoint. This is contextual modulation without reliable contextual selection. Removing the final period yields zero complete reversals for all three checkpoints. Thus greedy decoding and period scoring do not conceal an already successful likelihood-based selector.

The historical field candidate_pair_probability_mass actually stores the softmax-normalized two-candidate preference distribution (sums to one). This autopsy separately computes true candidate-sequence mass exp(scoreA)+exp(scoreB), without EOS, from saved log scores. The historical artifact remains unchanged. Tiny mass shows these candidate sequences are individually improbable in the full vocabulary model, not that a calibrated pairwise preference is absent. Candidate lengths and EOS are not calibrated or altered.

## Bounded gradient and representation diagnostic

Selection was committed in DIAGNOSTIC_PLAN.json before inference: first lexicographic fact family F01, all four members; first literal English batch; first literal binding batch. All three models were evaluated with no dropout, no optimizer, and torch.autograd.grad only. Parameters were temporarily enabled for differentiation, including protected groups; no .grad buffers were accumulated. All state tensors were checked bitwise unchanged and checkpoint hashes were rechecked. This diagnostic does not replay any training update.

Candidate NLL is mean correct sequence NLL per candidate token, excluding EOS. Diagnostic pair loss is mean softplus(scoreWrong-scoreCorrect), with full candidate sums. Neither hypothetical loss was trained or adopted as a treatment.

| Checkpoint | Sentence CE, first training batch | F01 correct NLL/token | F01 pair loss | First binding batch objective | Cue final-norm relative change, q0 / q1 |
|---|---:|---:|---:|---:|---:|
| Pilot1_parent | 3.3020 | 7.3115 | 0.9344 | 0.020291 | 0.1074 / 0.1571 |
| HR1_causal_aligned_500 | 2.6939 | 6.7221 | 2.3277 | 0.000613 | 0.0722 / 0.0842 |
| HR3_block3_update_500 | 2.7515 | 6.9044 | 1.6155 | 0.001343 | 0.1474 / 0.1200 |

Both factual derivatives are finite and nonzero through embeddings, every block, final norm, and language head in all models. Specialized routing/retrieval gradients are zero on the ordinary English path, as expected. In HR-3, pair-loss gradient norms range 20.78–25.29 in blocks3–7 and 22.62 in the head. Existing sentence CE norms in those blocks are 0.249–0.291. These differently normalized, diagnostic losses must not be compared as actual optimizer update magnitudes; clipping and AdamW state also matter. Nonzero gradients establish connectivity, not useful semantic features or sufficient learnability.

HR-3 sentence-versus-pair gradient cosines in blocks3–7 are -0.098, -0.080, -0.049, -0.031, +0.029; head +0.008. This one-batch result does not demonstrate sustained destructive interference. The binding gradient is also nonzero and much smaller on this already-solved batch, but this cannot establish its cumulative training influence. Full per-group norms and cosines for all three checkpoints are preserved.

Fact reversal changes cue representations across layers and final logits. HR-3 final-norm relative changes for F01 are 0.147 and 0.120, compared with 0.107/0.157 in Pilot1 and 0.072/0.084 in HR-1. Merely different hidden vectors do not establish separable correct-answer semantics. No readout was fitted and no attention/representation mechanism has been inferred. This does not uniquely distinguish weak representations from an ineffective selection/readout.

## Historical explicit factual-supervision comparison

The preserved seed87002 report records factual 97/192 versus control 97/192 and parent 95/192; reversals 22/96, 11/96, 5/96 respectively; complete families zero for all. Factual object/predicate reversals were 13/48 and 9/48; control 9/48 and 2/48. All acquisition gates failed. The authoritative v8 harness uses correctly aligned answer-plus-EOS supervision: inputs=sequence[:-1], labels=[ignore]*(prefix_length-1)+candidate+[EOS]. Its failure cannot be dismissed as the HR copying-target bug. These older outcomes use a different battery and cannot be numerically pooled with readiness DEV. They support trying explicit contextual training, but do not establish that more answer CE alone will solve selection.

## Evidence ledger and decision

Established: isolated-sentence exposure; no designed English factual/counterfactual supervision in HR-1/HR-3; stable within-family candidate choices; zero HR-1/HR-3 reversals even under likelihood scoring; gradients reach currently trainable groups; no checkpoint modification.

Supported interpretation: the sentence-level objective and readiness retrieval task are poorly matched, and candidate bias dominates useful reversal-sensitive evidence. More block unfreezing is not the uniquely justified next step.

Unresolved: whether complete counterfactual answer-CE practice with appropriate language retention is sufficient, or whether explicit relative candidate supervision is needed; representation versus readout limitations; how much intervention data is required; aggregate gradient interference. Architectural incapacity, capacity ceiling, and a causal early-block bottleneck are unsupported.

Two materially different bounded treatments remain defensible:

1. Data/sampling-only intervention: complete, balanced two-fact counterfactual families with correctly aligned answer-plus-EOS full-vocabulary CE, alongside language retention and unchanged binding preservation. This changes task exposure without adding a new loss. Historical explicit CE failed, so success is not assured.
2. On exactly the same training families and retention schedule, add a prospective candidate-pair discrimination term to answer CE, testing whether relative selection pressure is necessary. CE must remain because pair-only scoring cannot by itself teach absolute answer likelihood or stopping. The coefficient and normalization are scientific choices that cannot be read from the autopsy gradients.

A same-data matched comparison between these objectives is the recommended next decision because it directly distinguishes the remaining explanations. It must be separately preregistered with fresh family-level training/DEV splits, semantic validation, fixed budget/weights/sampling and independent preservation checks before any update. Pilot1 is the defensible common starting point; existing HR-1/HR-3 remain baselines. No parameters, mixture, or treatment have been silently selected. Existing readiness gates remain intact.

STOP: the user explicitly requires a decision when materially different treatments remain defensible after bounded diagnostics. The evidence favors an explicit-supervision intervention class but does not uniquely pick its objective. No HR-4 was constructed or run.

## Measurement limits and provenance

Readiness DEV was already exposed and is development evidence, not a new independent confirmation set. These gradients are exploratory on one existing family; no population confidence intervals or mechanistic causal conclusions are attached. Source DEV loss averages four token-mean minibatch losses rather than globally weighting every token; preserved trajectory values retain that convention. Human-review fields and gates are unchanged. No sealed FINAL contents were needed for any check.

The non-circular receipt identifies the source, plan, raw diagnostics, source hashes and report. Old experiment files are unmodified. The autopsy outputs are read-only after sealing.
