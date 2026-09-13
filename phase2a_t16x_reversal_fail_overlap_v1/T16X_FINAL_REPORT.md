# T16X DEV REVERSAL-FAIL OVERLAP — FINAL

Status: **COMPLETE, read-only.** DEV `item_results` only. TEST not loaded. No training.

Question: are T16’s remaining reversal misses a stable hard set, and did Phase B actually recover pairs?

## Counts (failed reversal pairs / 64)

| run | fails | pass |
|---|---:|---:|
| T16 750001 U750 | 21 | 43 |
| T16 750002 U750 | 18 | 46 |
| T16 750003 U750 | 25 | 39 |
| T16 750002 U500 (pre-revert) | 24 | 40 |
| T14 730001 U750 | 18 | 46 |
| T14 730002 U750 | 15 | 49 |
| T15 740001 U750 | 26 | 38 |
| T15 740003 U750 | 21 | 43 |

## Phase B on the best T16 seed (750002)
U500→U750: **recovered 10**, still-fail 14, **new-fail 4**. Net +6 (40→46). Phase B is productive, not cosmetic, and is not a monotone subset shrink.

## Shared hard set
All three T16 U750 seeds miss the same **9** pairs. Those 9 are **template-disjoint families only**: DT00 (4), DT01 (3), DT07 (2). Templates: locative 5, owns 4. Zero shared fails on `active`.
750002’s 18 fails are mostly locative (11) and owns (6). T14 730002’s 15 fails are mixed (locative 6, owns 5, active 4). Shared T16-750002 ∩ T14-730002 = 10 pairs.

## Verdict
1. Cutting the 8–11 scaffold at U500 left Phase B starting at 35–40 rev; Phase B can recover pairs but not enough to clear 48.
2. The leftover misses concentrate on **template-disjoint locative/owns**, including a 9-pair set that no T16 seed solved.
3. That does **not** yet justify changing the tokenizer or copying Qwen. It also does **not** justify parenting 730002.

## Next
T17: same two-phase recipe, but **do not cut the scaffold until U750** (T14’s representation horizon), then Phase B 250 with 8–11 reverted. Tests whether T16 failed because Phase A was too short, without changing data mix or pointer keys. Record the DT locative/owns hard set for a later treatment if T17 still stalls on those families.
