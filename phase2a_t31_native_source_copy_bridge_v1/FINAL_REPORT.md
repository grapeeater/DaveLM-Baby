# T31 FINAL REPORT

## T31 STATUS
TERMINAL. Classification: **T31_EXPLICIT_SOURCE_TO_TOKEN_BRIDGE_INSUFFICIENT**. Seeds 880001 and 880002 independently terminal-failed under the sealed sequential rule; seed 880003 was not required. No protected panel was opened.

## Architectural change
A four-parameter, zero-initialized ordered source-copy logit bridge. Pointer weights over the two fact-clause source spans form a vocabulary distribution from each candidate's frozen answer-context token sequence (name plus period). At answer step j, centered log copy evidence is added directly to ordinary LM logits through copy_gate[j]. Gate zero is exactly neutral; EOS and later generation remain on the ordinary LM path. This is one native source-to-token path, not reranking, constrained decoding, or an oracle prefix.

## Frozen protocol
Parent Phase1G U6000 SHA c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1. Tokenizer v0_7 SHA e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b. Seeds 880001–880003; 1000 updates, 900 QA + 100 language; T24/T30 inherited scopes and losses; only copy_gate new. U0/U100/U300/U500/U750/U1000 DEV evaluations. T3 TEST, T2-EVAL-TEST, FINAL, sacred locked.

## Per-seed terminal results

| seed | update | pointer | native FC | exact+EOS | shared exact | unique exact | first-token-correct/diverge | post-fork | DEV CE | copy gate L2 | classification |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 880001 | 1000 | 102/128 | 89/128 | 35/128 | 1/64 | 34/64 | 46 | 1/30 | 1.255947 | 0.03258 | T31_FAIL_NO_REPRESENTATION |
| 880002 | 1000 | 111/128 | 90/128 | 35/128 | 2/64 | 33/64 | 45 | 2/30 | 1.255342 | 0.03257 | T31_REPRESENTATION_SUCCESS_OUTPUT_FAIL |
| 880003 | not required | — | — | — | — | — | — | — | — | — | not run by sealed rule |

*Final-status records omit gate telemetry; evaluation_1000 records contain the active gate vector (see artifact hashes). The route was trainable and finite; seed 880001/2 gate activation was not sufficient to improve native output.

Language CE and gap stayed within the inherited healthy range; binding/localizer and protected early blocks remained parent-identical. TRAIN16 remained intact under the inherited evaluator.

## U1000 interpretation
The explicit source-copy path did not produce a qualitative native-mouth departure: exact stayed at 35/35, shared-prefix exact remained 1–2/64, and divergence remained 45–46. Pointer/native forced choice and language stayed healthy. Thus direct source-to-vocabulary routing, in this implementation, is insufficient to convert retrieved identity into reliable greedy continuation. This does not show that identity information is absent; it shows that this bridge did not make Baby use it successfully.

## Mechanical incident
Seed 880001 stopped at U750 during the Phase-A→B scope transition because the copied runner's Phase-B count was initially stale. The count was corrected mechanically to the inherited base Phase-B scope plus four copy parameters, and the same rolling checkpoint was resumed. No scientific setting, seed, data, or weights were changed; the valid run completed U1000.

## Adjudication and stopping
Both first two seeds are terminal scientific failures (no hard-stop integrity breach). Per the frozen sequential rule, seed 3 was not launched. The explicit source-to-token bridge failed to move the native exact frontier. The current-architecture surgical treatment line is exhausted; no T32 is authorized in this task.

## Locks
T3 TEST: SEALED_UNOPENED. T2-EVAL-TEST, FINAL, sacred: LOCKED. No protected bytes loaded or scored.

