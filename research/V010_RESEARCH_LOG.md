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
