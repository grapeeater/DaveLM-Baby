# Proposed protocol: v2R4 first-token selection treatment

Status: **FROZEN PROPOSAL. NOT LAUNCHED.**  
Name: `BABY_V010_V2R4_FIRST_TOKEN_SELECTION`  
Parent evidence: Fan Diesel isolation probe of verified U16000 seed 106001  
Does not replace frozen Gate L/C/R  
Does not open TEST / FINAL / SACRED  
Does not authorize v2R6, seed 106002, architecture rewrite, tokenizer change, or sidecar enablement

## Why this is specific enough to design, not launch

Isolation on the hashed v2R4 U16000 checkpoint (`94b3a9da…17827`):

1. Changing only the query does **not** move greedy first-token selection above 1/K (novel 2/3/4-pair follow 16/31, 8/21, 12/44).
2. Teacher-forced continuation of the **new** gold value after its first token locks 94/96 (31/31, 21/21, 42/44).
3. Inventory copy remains 92/96 after query-swap. More copy loss is the wrong objective.
4. On 2-pair misses the queried first token is typically rank 2; the original first token is rank 1. The queried candidate often has a signal and still loses.
5. Body-reorder query-first lifts matched-parent 22/65 → 36/65 (2-pair 10/16 → 15/16). A first-slot prior exists and must not be the training cheat code.

This is Fork 4 of [`design/V010_CONDITIONAL_NEXT_STEPS.md`](V010_CONDITIONAL_NEXT_STEPS.md): query-swap does not follow, new-gold rest-lock stays high.

## Objective

Make greedy **first-token selection** query-conditioned among in-context value first tokens, without teaching more span copying and without weakening held-out exact.

Success is **not** a Gate C headline. Success is query-swap follow above 1/K with payload continuation remaining locked.

## Training distribution (if later launched)

- **No 1-pair keyed attractor.** Primitive keyed 1-pair saturates and hides B. Minimum two pairs on every keyed optimization item.
- Include 3-pair and 4-pair in the train stream, not only as eval.
- **Query-swap twins:** same body, two different query keys, opposite gold first tokens. Both are optimization targets.
- **Render-order randomization** including query-last. Do not allow “copy body-first pair” to solve the loss.
- Keep train-surface markers/separators for this treatment. Do not retokenize. Do not mix held-out separators into a silent Gate C bypass.
- Language stream retention unchanged. Do not open protected eval.

## Optimization (if later launched)

Prefer a **forced-choice / contrastive first-token** term over another generation-only copy loss:

- At the first answer position, among the in-context value-first-token set, the queried first token should be rank-1.
- Optionally a margin against the best competing inventory first token.
- Teacher-forced remainder of the gold value may stay as an auxiliary (it already locks). Do not up-weight it as if copy were the failure.
- Do not train to emit held-out separators as a way to “fix exact.”

Exact loss coefficients, batch mix, and step budget are **not** specified here. They require a later preregistration if someone launches. This file only freezes the *target mechanism*.

## Diagnostic scores (not graduation gates)

Must be reported separately:

- query-swap follow / stuck-old / other-competitor / off-inventory
- queried first-token vocab rank and inventory rank
- `rest_value_tf_lock` on the queried gold
- value-span exact vs free-running exact (separator/EOS)
- body-reorder query-first vs query-last
- pair-count 2/3/4 vs 1/K
- value-absent original-span copy (must stay ~0)

Do not compare n=16 interim slices to full-panel n without matching.

## Graduation

Frozen Gate L/C/R stay. A treatment that raises query-swap follow but still fails Gate C is a **mechanism win**, not a graduation. Held-out exact remains a separator-OOD plus copy-entanglement problem; do not quietly drop it.

Two-seed graduation remains required for any later claim of a passing protocol. This proposal is single-parent-seed evidence.

## Explicitly not authorized

- Launching this treatment
- Weakening Gate L/C/R
- Starting v2R6 / seed 106002
- Editing the v2R4 checkpoint, frozen panels, or frozen metrics
- Interpreting v2R5 absence as pass or fail
