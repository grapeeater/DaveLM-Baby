# Tokenizer failure mechanism

Status: reproduced from post-T28 `ITEMS.json` / `DETAILED_SUMMARY.json` (384 DEV observations, seeds 850001–850003). TEST was not loaded. v0_7 was not mutated.

## Reproduced rates

| Split | Exact | First-token-correct-then-diverge |
|---|---:|---:|
| Shared first token among DEV names (Omar, Opal, Sal, Skye) | 9/192 = **4.7%** | **43.2%** |
| Unique first token among DEV names (Ava, Mel, Ross, Wes) | 102/192 = **53.1%** | **21.4%** |

This matches the frozen post-T28 tokenizer diagnostic. Association, not tokenizer causality.

## Collision map (v0_7)

Only two DEV first-token collision groups exist. Longest shared prefix is one token; identity disambiguates at position 2.

| Family | Pieces | Exact | Dominant generated |
|---|---|---:|---|
| Sal `527,293` | ` S` + `al` + `.` | 6/48 | **Salt.** 36/48 |
| Skye `527,79,93,73` | ` S` + `k` + `y` + `e` + `.` | 2/48 | **Sky.** 28/48, Seth. 11/48 |
| Omar `536,81,285` | ` O` + `m` + `ar` + `.` | 1/48 | mixed; often never enters ` O` |
| Opal `536,84,293` | ` O` + `p` + `al` + `.` | 0/48 | mixed; Ross/York/Salt/Open |

Sal is a **string/token prefix of Salt**: greedy completes ` S`+`al` then emits `t` instead of `.`. Skye is the opposite **over-stop**: greedy completes ` S`+`k`+`y` then emits `.` instead of `e`.

## Unique-in-DEV is not unique-in-vocab

Wes is first-token-unique among the eight DEV names (` W` + `es`) but **Walt.** in 40/48. Walt shares first token `679` outside the DEV inventory. “Unique prefix” in the diagnostic is a DEV-panel contrast, not a global tokenizer uniqueness claim.

Mel (` M`+`el`) 44/48 exact and Ross (` R`+`os`+`s`) 44/48 exact show that multi-token names are not doomed when the continuation is not a high-prior English/name completion.

## Which factor is primary?

Not a single letter. Ranked on this panel:

1. **Token-prefix / stop-boundary interaction with language priors (A+B+E).** After a legal name prefix, greedy local argmax prefers a more complete word (`Salt`, `Sky`, `Walt`) or a high-prior other name (`York`).
2. **DEV-internal shared first token (A)** increases exposure (Sal/Skye, Omar/Opal) but Omar/Opal mostly fail before the shared prefix is even used.
3. **Length (C)** is mixed: 2-token Mel 91.7% vs Wes 0%; 3-token Ross 91.7% vs Omar ~2%.
4. **Frequency (D)** is mixed: zero Phase1G full-string count splits Ross (success) vs Omar/Opal/Wes (failure).

The suffix-state trace remains compatible: gold prefixes make later tokens readable in blocks 7–11; a wrong generated token collapses that path. Shared-prefix geometry raises the chance that greedy commits to a wrong second token.

## What this is not

Not proof that retokenizing DEV names would solve Phase 2A. Not a license to overwrite v0_7. Native 2-way forced-choice on T28 was already 82–88/128 while greedy exact was 35–40/128, so **candidate-constrained scoring already works far better than free greedy**. That gap is a decoding/continuation problem sitting on tokenizer geometry, not a missing pointer skill.
