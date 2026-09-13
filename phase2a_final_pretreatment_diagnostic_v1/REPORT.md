# PHASE2A FINAL PRE-TREATMENT DIAGNOSTIC

## 1. LEDGER AMENDMENT
- Level A generator defect: no true assignment counterfactual; prior A-based reversal/counterfactual claims are compromised.
- The prior B/last 0.63–0.78 representation result is withdrawn as non-reproducible. Corrected B-only independent implementations report 0.500.
- T2 remains `PHASE2A_T2_FAIL`; T2-EVAL-TEST remains unopened.
- Binding/localizer was untrained, chance-level, and disconnected from ordinary scoring because `layout=None`.

## 2. OLD PIPELINE DISCREPANCY ROOT CAUSE
The preserved high B/last path used final/last representations with row-level candidate-pair/query-slot labels and pair-key folds. Same family/twin/lexical-pair rows could cross folds, and the label encoded candidate orientation rather than an independently keyed assignment. The corrected B-only path uses explicit identity rows, independently keyed labels, verified positions, and grouped family/twin splits; it returns 0.500. Surviving artifacts do not preserve every intermediate preprocessing byte, so no stronger historical claim is made.

## 3. AUTHORITATIVE PIPELINE VERSION/HASH
The new implementation scripts and all generated artifacts are hashed in `SHA256SUMS.txt`; source hashes are in `PROVENANCE.json`. The historical row-probe source is `C:\DaveLM-CADAVER\treatment4_query_separability_probe.py` (SHA256 `7db79d2001770aa3b39721dc2e7d16d00dda2aa07ca93a1698c4a6fe799c8d67`).

## 4. PIPELINE VALIDATION
Positive separable control=1.0; shuffled-label control=0.46875; leakage injected=1.0; leakage removed=0.5625; family-grouped nuisance=0.5. All controls pass. Native re-extraction max logit and hidden-state errors are both 0.0. Input checkpoint/tokenizer hashes match their frozen records.

## 5. CLEAN CORPUS / GENERATOR AUDIT
`CORPUS.json` contains 128 newly generated nonsacred rows: 16 explicit families, 8 assignment/query/order variants per family, fixed candidate order, 64 answer-reversing twins and 64 query counterfactuals. Correct index is 64/64. Identity, family, twin, template, prompt length, and token IDs are recorded; all prompts are <=256 tokens. No T2-EVAL-TEST, FINAL, sacred, or locked outcomes were read or used.

## 6. FROZEN REPRESENTATION PROTOCOL
For Pilot1 U0 and T2 U100, hooks capture each actual Transformer block output and final norm at the final unpadded prompt token. This is the causal state whose logits predict the first answer token; answer-token states are excluded. The primary ridge probe uses family-disjoint 12/4 folds with twins together; deterministic name-disjoint results are also recorded.

## 7. RESULTS
Family-disjoint accuracies are near chance across depth (Pilot1 final_norm 0.56; T2 U100 final_norm 0.53; other layers 0.47-0.56), with shuffled controls near chance. Twin-difference norms increase with depth but do not establish relational decoding. Aligned language CE on 128 authorized windows is Pilot1 1.17229 and T2 U100 1.19759. See `LOCALIZATION_RESULTS.jsonl`, `REPRESENTATION_RESULTS.json`, `COUNTERFACTUAL_GEOMETRY.json`, `ITEM_RESULTS.jsonl`, and `LANGUAGE_CE.json`.

### U0/U100 layer results

| checkpoint | block_0 | block_3 | block_7 | final_norm |
|---|---:|---:|---:|---:|
| Pilot1 U0 | 0.50 | 0.47 | 0.53 | 0.56 |
| T2 U100 | 0.50 | 0.47 | 0.56 | 0.53 |

## 8. OPTIONAL NONLINEAR BOUND
Because linear/geometric results were null and controls passed, one preregistered fixed quadratic ridge bound was run on final_norm. Pilot1=0.50 (shuffle 0.50); T2 U100=0.5625 (shuffle 0.50). This small bound does not establish nonlinear relational signal.

## 9. ADJUDICATION
**D1_LINEAR_RELATIONAL_SIGNAL_ABSENT**. No reliable linear relational assignment signal is decoded under the valid family/twin-disjoint pipeline; the secondary nonlinear bound is not distinguishable from its shuffled control.

## 10. CONFIDENCE / LIMITATIONS
This is a bounded decodability result for this corpus, two checkpoints, selected positions, and low-capacity probes. It does not prove absence of nonlinear/distributed code, identify where learning occurred, or establish held-out behavioral competence. Locked T2-EVAL-TEST, transfer/copy panels, FINAL, and sacred material remain untouched.

## 11. SINGLE NEXT TRAINING-TREATMENT DIRECTION
The next action is a training treatment, not another diagnostic chain: explicit representation-forming contextual supervision using true assignment/query counterfactual pairs, while retaining current language and binding protections.

No optimizer was created, no weights were changed, and no training/inference on locked material occurred.

PHASE2A_FINAL_PRETREATMENT_DIAGNOSTIC_COMPLETE
