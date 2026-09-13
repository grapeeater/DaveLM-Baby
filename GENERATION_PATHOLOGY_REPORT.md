# Read-only generation-pathology diagnostic

This report uses only nonsacred development material. The rollout battery is the five fixed smoke prompts used in `generation_pathology_hr123.json`; teacher-forcing uses the first 128 rows (3,796 supervised targets) of `human_readiness_hr1_seed87004_v8/ENGLISH_DEV.jsonl`. No final readiness item, sacred material, optimizer, or weight update was accessed.

## Teacher forcing versus rollout

Correctly aligned causal scoring (BOS + token sequence, target is the next token at the same output position) gives:

| checkpoint | loss | perplexity | teacher-forced top-1 | mean entropy | mean top-1 probability | mean gold probability |
|---|---:|---:|---:|---:|---:|---:|
| Pilot 1 | 3.3923 | 29.74 | 26.58% | 2.916 | 0.342 | 0.161 |
| HR-1 | 8.6381 | 5,642.5 | 0.58% | 2.009 | 0.736 | 0.0050 |
| HR-2 | 10.8060 | 49,314.2 | 0.55% | 0.644 | 0.919 | 0.0052 |

The historical HR-1/HR-2 DEV helper and English training loop use a different, shifted label layout. Recomputing that exact layout gives HR-1 loss 0.55538 (recorded 0.55536) and HR-2 loss 0.09255 (recorded 0.09239), with top-1 accuracy 94.97% and 99.45%. In that layout, at positions 1 through n the input is token `z[j]` and the target is also `z[j]`; the final EOS is paired with the zero padding input. It therefore measures an easy copy-like objective rather than ordinary next-token prediction.

The source is explicit: [HR1_TRAIN_REAL.py](C:/DaveLM-CADAVER/human_readiness_hr1_seed87004_v8/HR1_TRAIN_REAL.py:64-65) and [HR2_TRAIN.py](C:/DaveLM-CADAVER/human_readiness_hr2_seed87005_v7/HR2_TRAIN.py:64-65) assign `y[i,1:len(z)] = z[1:]`. The earlier validated language trainers use the causal alignment `y[i,:len(z)-1] = z[1:]` ([language_sentencebound_p5.py](C:/DaveLM-CADAVER/language_sentencebound_p5.py:29-30), [language_compositional_p7.py](C:/DaveLM-CADAVER/language_compositional_p7.py:35-37)). HR-2's unlikelihood term uses the same shifted `y` validity mask, so it was not a clean test of an anti-repetition objective.

## Autoregressive traces

All three checkpoints generated 32 tokens on every smoke prompt under greedy decoding; immediate EOS was 0/5 for each. HR-1 and HR-2 entered an adjacent-token loop on all five prompts at generated step 2, reached a three-token run at step 3, and remained on one token for all 32 steps. HR-1 loop tokens were `made`, `and`, `to`, `and`, and `.`, respectively. HR-2 loop tokens were `because`, `and`, `down`, `and`, and `.`. This is a fixed-token/punctuation attractor, not a delayed long-horizon failure.

Pilot 1 had no adjacent or triple token run on these five prompts. Its first repeated bigram appeared at steps 7–18 and first repeated trigram at steps 16–27; the outputs were still malformed or semantically weak, for example `The dog ... the big and fun. He wanted to the bird and wered ...` and `The small cat ... was veryoneyone ...`.

On the first five versus last five rollout steps, full-vocabulary entropy / top-1 probability were:

| checkpoint | entropy first→last | top-1 probability first→last | probability margin first→last |
|---|---:|---:|---:|
| Pilot 1 | 2.817 → 2.474 | 0.322 → 0.414 | 0.215 → 0.304 |
| HR-1 | 0.945 → 1.267 | 0.871 → 0.834 | 0.861 → 0.824 |
| HR-2 | 0.681 → 1.385 | 0.914 → 0.826 | 0.908 → 0.820 |

Thus HR-1/HR-2 are already highly peaked at or before degeneration. Their first-step EOS probabilities averaged 0.000189 and 0.000054 (mean ranks 357 and 314 of 1,024); Pilot 1 averaged 0.00571 (rank 400). The missing EOS is consistent with the shifted EOS supervision, but this is not by itself proof of a single internal mechanism.

## Decoding controls

The recorded `temp07_top20` baseline was deterministic top-20/temperature argmax, not sampling, and was exactly identical to greedy for all 15 checkpoint/prompt pairs. A separate deterministic top-20, temperature-0.7 sampling probe produced a triple repeat by step 3 on 5/5 HR-1 and 5/5 HR-2 prompts; Pilot 1 had repeated n-grams on 4/5 and one varied but still malformed output.

A repetition penalty of 1.2 left every HR-1 and HR-2 output unchanged (5/5 triple loops). A no-repeat-trigram constraint shortened some runs (mean lengths 27.4 for HR-1 and 29.0 for HR-2), but all five outputs still contained adjacent/triple runs beginning around steps 4–5 and became garbled fragments such as repeated subwords. These constraints suppress surface loops; they do not expose a coherent alternative continuation. Pilot 1 became somewhat less repetitive under the penalty, but remained malformed, so this is a decoding diagnostic only.

Raw traces and summaries are preserved in [generation_pathology_hr123.json](C:/DaveLM-CADAVER/generation_pathology_hr123.json), [generation_pathology_constraints.json](C:/DaveLM-CADAVER/generation_pathology_constraints.json), [generation_pathology_analysis.json](C:/DaveLM-CADAVER/generation_pathology_analysis.json), [generation_pathology_teacher_forced.json](C:/DaveLM-CADAVER/generation_pathology_teacher_forced.json), and [generation_pathology_eos.json](C:/DaveLM-CADAVER/generation_pathology_eos.json).

## Scientific conclusion and next direction

The strongest evidence is an implementation-level supervision/evaluation alignment defect in both HR-1 and HR-2. Their dramatic reported DEV losses are not evidence of improved ordinary causal next-token behavior; correctly aligned read-only scoring is worse than Pilot 1, while autoregressive rollouts immediately copy a dominant token. HR-2 therefore cannot establish that its unlikelihood term failed as a properly aligned anti-repetition treatment.

The first next intervention class is uniquely indicated as **training-side causal-alignment correction**: preserve Pilot 1, use the validated causal target/EOS placement, preserve binding rehearsal, and rerun a bounded treatment only after a corrected protocol and evaluator are frozen. A decoding-only intervention is insufficient on these traces, and no structural change is justified by this evidence. HR-1/HR-2 should not be used as the language parent for that correction because their aligned next-token behavior is already degraded. This report does not specify or run HR-3.

This conclusion does not establish general language incapacity, a capacity ceiling, or an architectural impossibility. It explains why the historical loss curves and free-running behavior diverged and identifies the cheapest decisive next test.
