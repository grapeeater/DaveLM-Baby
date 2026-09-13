# Bounded shared-plus-antisymmetric collapse autopsy (read-only)

## Sources and boundary

Only the saved per-document retention audits were read; no checkpoint was loaded and retention was not rerun.

Champion audit SHA-256: `5363a0d432e815ae2a15701d14ac05650a98e738fd48ebae6736d40730fab20c`  
Bounded audit SHA-256: `977ebd2a7a28dc8f0f38adff28f3c051aa5fa54a8aaf346082a85fc50ea63544`

The complete per-document collapse records, transition records, and seven BOTH_DISTINCT answer-error records are in `BOUNDED_SHARED_ANTISYMMETRIC_COLLAPSE_AUTOPSY.json`.

## 1. Matched transition census

| Champion → bounded | Documents | Answer transition counts |
|---|---:|---|
| BOTH_DISTINCT → BOTH_DISTINCT | 254 | 247 (1→1), 7 (1→0) |
| BOTH_DISTINCT → EXACTLY_ONE | 12 | 7 (1→1), 5 (1→0) |
| BOTH_DISTINCT → SLOT_COLLAPSE | 24 | 12 (1→1), 12 (1→0) |
| EXACTLY_ONE → BOTH_DISTINCT | 22 | 13 (1→1), 9 (0→1) |
| EXACTLY_ONE → SLOT_COLLAPSE | 8 | 2 each of 0→0, 0→1, 1→0, 1→1 |

Thus 22/30 former wandering cases were repaired, 8/30 became collapse, while 36/290 former successes were damaged (24 collapse, 12 EXACTLY_ONE). Net BOTH_DISTINCT change: −14 (290→276). New collapse was quartet-stable in eight quartets: `qt_000016`, `qt_000019`, `qt_000025`, `qt_000047`, `qt_000050`, `qt_000055`, `qt_000064`, `qt_000073`. The 22 repairs occurred in six quartets: `qt_000012`, `qt_000040`, `qt_000042`, `qt_000052`, `qt_000066`, `qt_000068`.

## 2. Collapse characterization

All 32 collapse documents are listed individually in the JSON artifact with quartet, layout, member, query slot, orientation, true positions, champion/bounded coordinates, answer result, and row weights.

Aggregate: 8 collapsed onto `true0`, 24 onto `true1`, and 0 onto a non-source. Using the saved row-position convention, this is 16 query-relevant versus 16 non-query-relevant cases; query slot and orientation are each exactly balanced (16/16). All 32 collapsed onto the physically earlier source in these saved records. Layout counts: p20_b8 8, p16_b12 8, p24_b10 4, p32_b12 4, p24_b8 4, p32_b8 4. There is therefore a physical/source-index preference, but no query-slot or orientation preference. Localization probabilities on each true source were not saved.

## 3–4. Rho saturation and shared-score override

**NOT AVAILABLE.** The authoritative audits save top coordinates and downstream row weights, not hidden states, `S`, pre-tanh `a`, `tanh(a)`, `R`, scorer logits, or localization probabilities. Obtaining those quantities would require a new retention forward evaluation, which is outside this autopsy. Consequently, saturation, `|Delta_S|>|Delta_R|`, and the claim that the shared score overrode assignment cannot be directly tested here.

## 5. Fate of the original 30 EXACTLY_ONE cases

22 became BOTH_DISTINCT, 8 became SLOT_COLLAPSE, and none remained EXACTLY_ONE or became NEITHER. The intervention therefore repaired a substantial fraction locally but converted the remainder into same-source collisions rather than eliminating failure.

## 6. Seven BOTH_DISTINCT answer errors

The seven records are listed in the JSON artifact. Six retained the query-correct localized row yet emitted an incorrect target (mostly the distractor or another vocabulary token), so their failure is downstream retrieval/value representation or answer decoding and is unresolved from saved fields. One (`qt_000075:o1_k0`) selected the non-query row (`query_row_correct=false`), providing a direct query-row-selection error. In all seven, selected-row correctness remained true; the saved audit does not expose enough internal value/answer decomposition to distinguish the six downstream cases further. The corresponding champion documents were correct.

## 7. Comparison with other interventions

Nonsource suppression repaired 12 but damaged 48, reduced old filler attractors, created new attractors, and introduced four collapses plus downstream query-selection errors. Source-recognition supervision repaired 18/30 but damaged 32, created no collapse, and concentrated new errors at slot-1 position 22 while downstream conditional correctness stayed perfect. The bounded scorer repaired 22/30 but damaged 36, created 32 same-source collapses, and reduced conditional answer correctness to 269/276. Commonly, each auxiliary/parameterization change reshaped a quartet-stable localization basin rather than producing uniform improvement. The failure modes are intervention-specific: filler attractor relocation, position-22 slot-specific wandering, and bounded-scorer source collision respectively.

## Final diagnosis

**ESTABLISHED:** The bounded scorer both repairs and damages cases. It repairs 22 old EXACTLY_ONE documents but turns 8 into collapse, damages 36 former BOTH_DISTINCT documents, and creates 32 quartet-stable collapses. Collapses are onto true sources, balanced by query slot/orientation, and all occur at the earlier physical source in the saved coordinate convention. Six of seven conditional answer errors occur despite query-correct row selection.

**SUPPORTED:** The bounded parameterization introduced a distinct-coverage/assignment-capacity bottleneck: limiting slot disagreement plausibly converts independent wandering into same-source selection. This is consistent with the new collapse pattern and the regression, but scorer saturation or shared-score domination is not directly demonstrated.

**UNRESOLVED:** Whether `R` is saturated, whether `|Delta_S|>|Delta_R|` causes the collisions, and whether the six downstream errors arise in value retrieval or the answer head. No further treatment is designed here.

No training, optimizer creation, checkpoint continuation, or retention rerun occurred. Authoritative predecessor artifacts were not modified.
