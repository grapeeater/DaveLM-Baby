# T21X GENERATION FAIL-MODE — FINAL

Status: **COMPLETE, read-only.** T21 U1000 greedy traces. TEST not loaded. No training. Did not parent T21 checkpoints.

Question: after hybrid CE moved exact 9–13 → 26–29, did first-token BOS recover toward T18X, and what is leftover?

## First generated token (128 DEV)
| seed | target BOS | not target BOS |
|---|---:|---:|
| 800001 | **79** | 49 |
| 800002 | **59** | 69 |
| 800003 | **72** | 56 |

T20X was 42/34/37. T18X was 76/77/84. First-token through-base restored most of the T18 BOS channel (800002 still short).

## Remaining modes
| seed | exact | diverge | other_then_eos | wrong cand / first |
|---|---:|---:|---:|---:|
| 800001 | 27 | **52** | 32 | 17 |
| 800002 | 29 | 30 | **41** | 28 |
| 800003 | 26 | **46** | 37 | 19 |

Unterminated-other 0. T17X continuation is gone. Dominant leftover on 2/3 seeds is T18X `first_token_correct_then_diverge` (`Wes.`→`Walt.`, `Ava.`→`York.` as other_then_eos).

**Observed in Baby.** T21 bought back the first-token stage T20 lost. Residual is T18X-style suffix collision plus leftover wrong-span-then-EOS. T18’s extra ~10 exact vs T21 is the suffix-through-base term, which still left diverge 40–44.

## Next
T22: first **two** answer tokens CE through the base; remaining name/period/EOS stop-grad. Tests whether T18’s +9 exact is the second name token (Sky/e, Wal/t) without T19’s full-suffix weight 3.0. Phase1G parent, seeds 810001–810003, same gates. Do not parent T14–T21. Do not retokenize.
