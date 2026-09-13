# T18X GENERATION FAIL-MODE — FINAL

Status: **COMPLETE, read-only.** T18 U1000 greedy traces. TEST not loaded. No training. Did not parent T18 checkpoints.

Question: after answer-span CE moved exact 0→36–40, what do remaining DEV fails emit?

## Confusion vs T17X
T17 U1000: exact 0, almost all `unterminated_other` story continuation; PNe 80–81/128.
T18 U1000: exact **36 / 36 / 40**; PNe drops to 48 / 42 / 43; PNE appears (35 / 33 / 36).

## First generated token (128 DEV)
| seed | target BOS | wrong-candidate BOS | other |
|---|---:|---:|---:|
| 770001 | **76** | 21 | 31 |
| 770002 | **77** | 22 | 29 |
| 770003 | **84** | 20 | 24 |

T17X first tokens were story pieces. T18 taught the answer-start on ~60% of DEV.

## Remaining modes (not exact)
Dominant: **`first_token_correct_then_diverge` 40 / 41 / 44**. Then `other_then_eos` 30 / 29 / 24. Wrong-candidate 8–14. Unterminated-other almost gone (1 / 0 / 0).

Hits decode as the full target (`" Ava."`, `" Mel."`, `" Ross."`). Typical PNe misses:

- `" Skye."` → `" Sky."` / `" Seth."`
- `" Sal."` → `" Salt."`
- `" Ava."` → `" York."` / `" Lois."` / `" He was a very happy."`
- `" Omar."` → `" York."`

**Observed in Baby:** residual exact failure is mostly **multi-token name completion / unigram collision after a correct first token**, plus some leftover story-then-EOS. It is not T17X’s open-ended TinyStories continuation.

**Hypothesized for Baby (Smol lesson 1, not a Smol result):** first-token readout and finishing+stopping the answer span are different stages. Uniform `λ_ans=1.0` already bought the first stage. More of the same uniform CE is the wrong next train.

## Next
T19: T18 recipe unchanged except **suffix-weighted** answer CE (weight 1.0 on the first answer token, 3.0 on the remaining name tokens + period + EOS). Phase1G parent, seeds 780001–780003, same gates. Do not retokenize. Do not copy Qwen/Smol.
