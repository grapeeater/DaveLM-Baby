# T20X GENERATION FAIL-MODE — FINAL

Status: **COMPLETE, read-only.** T20 U1000 greedy traces. TEST not loaded. No training. Did not parent T20 checkpoints.

Question: after readout-only CE collapsed exact to 9/10/13, did leftover revert to T17X TinyStories continuation or stay T18X `first_token_correct_then_diverge`?

## First generated token (128 DEV)
| seed | target BOS | not target BOS |
|---|---:|---:|
| 790001 | **42** | 86 |
| 790002 | **34** | 94 |
| 790003 | **37** | 91 |

T18X / T19X were 76–84 / 82–84–76. T20 lost about half the first-token answer-BOS channel.

## Remaining modes
Dominant: **`other_then_eos` 81 / 76 / 79**. Then `first_token_correct_then_diverge` 33 / 24 / 24. Unterminated-other **2 / 3 / 4** (T17X continuation is not back). Wrong-candidate 1–9.

Pointer-ok miss samples are short wrong spans then stop: `Ava.`→`Lema.`/`Butter.`/`Vork.`, `Wes.`→`Walt.`/`Walk.`/`escape.`, `Skye.`→`Sky.`. Not open TinyStories after `Answer:`.

**Observed in Baby.** Head-only CE does not keep T18’s first-token BOS and does not finish names. Residual is mostly a short wrong answer + EOS, plus leftover Sky/Walt diverge. Through-base CE is necessary to plant the first answer token. T19 already closed “raise suffix weight through the base.”

## Next
T21: T17/T18 two-phase, uniform `λ_ans=1.0` **mean** over the answer span, but **first answer token CE through the base** and **suffix+period+EOS CE stop-grad** through `language_head` only. Phase1G parent, seeds 800001–800003, same gates. Do not parent T14–T20. Do not retokenize.
