# T23X GENERATION FAIL-MODE — FINAL

Status: **COMPLETE, read-only.** T23 U1000 greedy traces. TEST not loaded. No training. Did not parent T23 checkpoints.

Question: did in-row unlikelihood change leftover vs T18X/T22X?

## First generated token (128 DEV)
| seed | target BOS | not target BOS |
|---|---:|---:|
| 820001 | **79** | 49 |
| 820002 | **75** | 53 |
| 820003 | **79** | 49 |

T18X was 76/77/84. T22X was 66/65/73. T23 recovered the T18 BOS channel.

## Remaining modes
| seed | exact | diverge | other_then_eos | wrong cand / first |
|---|---:|---:|---:|---:|
| 820001 | 34 | **45** | 26 | 23 |
| 820002 | 34 | **41** | 34 | 19 |
| 820003 | 38 | **41** | 28 | 21 |

Dominant leftover is still T18X `first_token_correct_then_diverge`. Samples: `Wes.`→`Walt.`, `Skye.`→`Sky.`, `Ava.`→`York.`.

**Observed in Baby.** Those leftovers are **off-row**. DI00 candidates are Ava/Wes — Walt is not in-row. DI01 candidates are Mel/Skye — Sky is not in-row. T23 never unlikelihood’d the tokens greedy actually emits.

Closed: raising `λ_unl` or repeating in-row unlikelihood. Do not parent 820001–820003.

## Next
T24: T18 full-span CE plus unlikelihood over the frozen **T3 train+dev name inventory** (not TEST names), excluding the gold name. Isolates rival-set: in-row pair (T23) vs all other known Baby names. Phase1G parent, seeds 830001–830003, same gates.
