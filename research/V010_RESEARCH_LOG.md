# Baby v0.10 running research log

This log is append-only. It does not rewrite frozen v2R4 receipts or Gate L/C/R.

v2R5 remains **absent in GitHub / unresolved / pending**.

---

## Milestone 2026-09-16 — emission source: copy works, selection does not

### CURRENT BEST DIAGNOSIS

Baby v0.10 at v2R4 U16000 has a working **copy-a-value-span-from-context** mechanism on the training surface. She does **not** reliably **select the queried pair** among competing in-context values. Held-out exact=0 is a mix of weaker inventory copy, first-pair fallback, and held-out separator/EOS OOD. Architecture is not the immediate claim.

Split:

- **A** identify internally: partial. When she copies a competitor, the queried first token is almost never rank 1 and is typically rank 2 on train novel.
- **B** bind/select queried item: failed. Dominant train-novel error is emitting another pair's full value (52/96).
- **C** copy selected payload: supported. 93/96 train-novel greedy outputs are some in-context value span.
- **D** free-generate that payload: supported. TF value = free value.
- **E** separator/EOS: train OK; held-out exact killed by OOD seps `{82–85}`.
- **F** new surfaces: held-out inventory copy drops to 55/96 and first-pair bias appears.

### EVIDENCE FOR IT

Locked from frozen metrics+panels via `emission_source_report`:

| panel | queried copy | competitor copy | off-inventory | inventory copy |
|---|---:|---:|---:|---:|
| primitive_keyed | 64 | 0 | 0 | 64/64 |
| short_keyed | 54 | 10 | 0 | 64/64 |
| same_surface_novel | 41 | 52 | 3 | 93/96 |
| heldout_surface | 23 | 32 | 41 | 55/96 |

Train novel: competitor copies have `rank1_when_competitor_copy = 0` and median target rank **2**. Queried copies have median rank **1**.

Chance-level 3-pair queried rate (7/21) is **not** "she cannot copy." She copies a real pair; the queried one is selected at ~1/3 with a first-slot tilt on 4-pair (`P(ok|render_pos=0)=7/11`).

### EVIDENCE AGAINST IT / CAVEATS

- First-pair bias is only moderate on train novel (`P(emit first)=0.427` vs `P(query first)=0.323`). Not a pure first-slot machine.
- Nearest-pair-to-query is weaker than first-pair (`P(emit nearest)=0.29` on novel).
- 4-pair first-slot 7/11 is a small bucket.
- Held-out 2-pair first-slot is stronger (`12/21` vs `2/17`) but still n-limited.
- Competitor rank 2 does not prove the full queried *span* is represented, only the first token.
- v2R5 unknown. Single seed.

### WHAT WAS FALSIFIED

- Previous (including this agent's first autopsy) reading of "3-pair at chance ⇒ copy failed / random tokens" is **false**. Failures are mostly **wrong-pair copies**.
- "Knows but cannot emit the value" remains false for the payload.
- Pure nearest-to-query positional shortcut is false as the main recipe.
- Always-copy-first-pair is false as a complete train-surface account.

### WHAT REMAINS UNKNOWN

- Query-swap follow vs stuck-on-old (needs U16000 weights).
- Whether moving the queried pair to body-first **causes** the 4-pair boost (body-reorder probe; needs weights).
- Marker-swap vs sep-swap causal split (needs weights).
- Value-absent: does original-answer copy die when the span is gone (needs weights).
- Whether rank-2 queried first token is accompanied by the rest of the span in later TF steps when a competitor was greedily chosen.
- Induction: 48/64 primitive emissions are some source token, only 19 the aligned continuation — separate offset/binding problem.
- v2R5.

### NEXT EXPERIMENT

1. **Already implemented, needs Fan Diesel weights:** `diagnose_checkpoint` now includes query-swap, marker/sep swap, value-absent, and **body-reorder query-first/last**.
2. **Already run here, no weights:** emission-source census (this milestone).

Fan Diesel body-reorder is the highest-information *causal* positional test: if query-first lifts 4-pair queried copy a lot and query-last drops it, selection is partly a body-slot prior, not only a missing bind operator.

### WHY HIGH INFORMATION

It distinguishes B (wrong pair chosen among copied candidates) from C (cannot copy) without another training run, and it makes the checkpoint probes about *selection* rather than generic accuracy.

---

## Milestone 2026-09-16 — TF lock, chance tests, probe-limit, induction EOS glue

### CURRENT BEST DIAGNOSIS

Baby has a **copy-a-present-value-span** skill on the train surface. The remaining keyed bottleneck is **first-token selection among in-context values**, not span tracking and not free-generation of a selected span.

Once the gold first token is teacher-forced, the rest of the queried value **and** the train separator/EOS are rank-1 almost always, even on rows where greedy copied a competitor. She can identify and copy the queried payload internally. She does not reliably **choose** it.

Queried-copy rates at 2/3/4 pairs are **not** above 1/K at p<0.05. The earlier "2-pair binding signal" is withdrawn.

Held-out failure is a mixture of weaker inventory copy (especially long values), first-token selection failure, and **train-separator glue** (all 41 off-inventory held-out emissions contain a train sep). Induction immediate-EOS is **not** generic collapse: 22/22 cases are "context already ends with this item's separator → emit EOS."

Curriculum: 1-pair copy saturates first; multi-pair inventory copy saturates at U12000 on the n=16 probe; selection never lifts on the matched slice. Architecture is still not the next claim.

### EVIDENCE FOR IT

- Train novel TF continuation: queried 41/41 full lock; competitor 51/52 rest-value lock and 51/52 full lock.
- Held-out queried: 23/23 rest-value lock, **0/23** full lock (sep/EOS OOD).
- Greedy first token ∈ inventory first tokens: 95/96 train novel; first tokens unique 96/96.
- Chance tests: 2-pair 19/31 two-sided p=0.281; 3-pair 7/21 p=1.0 vs 1/3; 4-pair 15/44 p=0.222 vs 1/4; short 2-pair 18/28 p=0.185.
- Probe limit: U6000–U15500 scored n=16; U16000 scored full panels. Matched first-16 novel queried 6→5 from U15500→U16000. Inventory copy 16/16 from U12000.
- Held-out off-inventory 41/41 contain a train separator; value_length=10 inventory copy 0/12.
- Induction EOS: last token == own separator in 22/22 immediate-EOS rows; 0/64 full-induction EOS when last token is not that separator.
- Query-before-body (`keyed_5`) queried rate 5/12 vs 36/84 after-body. Not the main recipe.

### EVIDENCE AGAINST IT / CAVEATS

- 4-pair first-slot queried copy 7/11 vs ~0.25 on other slots is a possible body-slot prior. n is small; body-reorder still needs weights.
- Held-out 2-pair first-slot 12/21 vs 2/17 is stronger than train 2-pair (which is ~0.60 in both slots). Surface change may engage a first-slot heuristic that train 2-pair does not need.
- TF lock is gold-teacher-forcing, a state greedy often never visits after a wrong first token. It measures A, not B.
- Competitor rank distribution on train is not always 2 (hist includes 3, 4, 5, and a few large ranks). "Runner-up" is typical, not universal.
- Single seed. v2R5 unknown. Query-swap still unscored.

### WHAT WAS FALSIFIED

- This branch's earlier claim that 2-pair queried copy is a real binding signal.
- Reading U16000 41/96 as a late capability jump versus U15500 (confounded by probe_limit=16 vs full panel).
- "Immediate EOS on induction = generic collapse / language death." It is trailing-separator suffix glue.
- "3-pair at chance means she cannot copy" (already falsified by emission source; restated).
- Query-must-follow-body as the train-surface recipe (`keyed_5` is statistically the same, n=12).

### WHAT REMAINS UNKNOWN

- Query-swap: follow new value vs stuck on old vs copy some other inventory value.
- Whether body-reorder **causes** the 4-pair first-slot tilt.
- Marker identity vs separator identity as the held-out copy drop.
- Value-absent knockout (train-novel copy of novel spans almost has to be from context, but it is still the right causal control).
- Why held-out long values lose even inventory copy.
- Induction offset binding when the trailing-sep trap is removed (3/64 correct among last≠sep full induction).
- v2R5.

### NEXT EXPERIMENT

Highest-information remaining test that needs weights: **query-swap** on same-surface novel and short 2-pair, read as:

- `query_swap_follow_new_value`
- `emitted_original_value_span` (stuck-on-old)
- `competitor_copy` of a third value
- `rest_value_tf_lock` on the **new** gold span

Predictions:

- Follow high, stuck-old low → some query-conditioned selection exists; then body-reorder and pair-count pressure matter.
- Follow low, stuck-old high → first-token selection ignores the query; copies a previously highlighted span.
- Follow low, stuck-old low, inventory copy stays high → copies *a* span, not the old one and not the new one; still not binding.
- New-gold `rest_value_tf_lock` high with follow low → A holds for the swapped target; B is still the bottleneck.

Body-reorder, marker/sep swap, and value-absent remain on the same Fan Diesel command. Do not train yet.

### WHY THAT EXPERIMENT HAS HIGH INFORMATION

Every data-only test still leaves open "maybe a hidden query cue we did not swap." Query-swap is the cleanest causal intervention on B that does not update weights. Chance tests already removed the excuse that 2-pair is solved.

