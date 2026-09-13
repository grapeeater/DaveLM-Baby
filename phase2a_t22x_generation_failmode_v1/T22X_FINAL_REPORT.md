# T22X GENERATION FAIL-MODE — FINAL

Status: **COMPLETE, read-only.** T22 U1000 greedy traces. TEST not loaded. No training. Did not parent T22 checkpoints.

Question: T22 exact 21/31/26 did not recover T18. Is leftover still T18X/T21X `first_token_correct_then_diverge`, or did first-two through-base change the fail-mode split?

## First generated token (128 DEV)
| seed | target BOS | not target BOS |
|---|---:|---:|
| 810001 | **66** | 62 |
| 810002 | **65** | 63 |
| 810003 | **73** | 55 |

T21X was **79 / 59 / 72**. T18X was **76 / 77 / 84**. First-two through-base did **not** restore T18 BOS and lost T21’s 800001 BOS lead.

## Remaining modes
| seed | exact | diverge | other_then_eos | wrong cand / first |
|---|---:|---:|---:|---:|
| 810001 | 21 | **45** | **45** | 16 |
| 810002 | 31 | **34** | 34 | 28 |
| 810003 | 26 | **47** | 30 | 25 |

Unterminated-other 1/1/0. T17X continuation is gone. Leftover is the same T18X/T21X mix: `Wes.`→`Walt.`, `Skye.`→`Seth.`, `Ava.`→`York.` / `Vela.` / `Va.`.

**Observed in Baby.** First-two through-base did not change the leftover family. CE-span-count isolations are now closed:

| treatment | through-base answer CE | exact | representation |
|---|---|---|---|
| T20 | none (head-only) | 9/10/13 | 2/3 |
| T21 | first token | 27/29/26 | **3/3** |
| T22 | first two tokens | 21/31/26 | 2/3 |
| T18 | full span | **36/36/40** | 2/3 |
| T19 | full span, suffix weight 3.0 | 39/35/34 | **1/3** |

Closed: another CE-span-count or suffix-weight tweak finishes names. Do not relaunch T18–T22. Do not parent 810001–810003.

## Next
T23: T18 recipe (full-span through-base `λ_ans=1.0`) **plus** in-row wrong-candidate unlikelihood at aligned answer positions (`λ_unl=1.0`). Attacks rival completions (`Walt`/`Seth`/`York`) without more gold-span CE. Phase1G parent, seeds 820001–820003, same gates. Do not retokenize.
