# Phase 2A Treatment 3 â€” design

## 1. Treatment question
The validated staircase found no reliable linear relational assignment signal in Babyâ€™s ordinary hidden states, and the nonlinear bound did not separate from shuffled controls. T3 tests whether direct base-model representation-forming supervision can create a counterfactually valid relational state and improve native candidate selection.

## 2. Prior evidence and scope
Phase2A-v1 full answer-span CE failed and was dominated by answer formatting. T2 replaced that with a length-normalized hardest-distractor margin and stopped at U100; confidence rose without reliable reversal/generalization. The final diagnostic is valid and D1. These results justify one explicit representation intervention; they do not justify a routing-only treatment, an architectural change, or another diagnostic chain.

## 3. Objective choice
**Selected A: parameter-free queryâ†’candidate contrastive pointer.** A is the smallest formulation that places unavoidable gradient on Babyâ€™s base hidden state while remaining valid under assignment, query, fact-order, name, and family counterfactuals. B is rejected because fixed prototypes impose an arbitrary identity geometry and can reward name priors. C is rejected because a trainable classifier could absorb the task without changing Babyâ€™s representation; even if used, it would require a second raw-hidden-state proof.

For each corrected QA row, compute normalized cosine similarity between the query decision state and each candidateâ€™s contextual mention state, then apply a two-candidate softmax loss. There is no trainable pointer head, no negative bank, and no stop-gradient. Both query and candidate states receive gradients. Equal/common-mode vectors give equal similarities and retain nonzero loss; assignment/query twins reverse the target, so a fixed name preference cannot minimize the loss across a family.

## 4. Exact location and anchors
Use `final_norm` at the final unpadded prompt token in the BOS-prefixed sequence (index `len(prompt_token_ids)`), the state whose language head predicts the first answer token. Candidate anchors are the mean of all token states in the first defining factual mention span for each candidate, including all name subtokens and excluding punctuation. The answer-token state is never used. Exact spans and token IDs are recorded by the corrected generator.

## 5. T2 native objective retained
T3 is T2â€™s exact length-normalized hardest-distractor margin (`M=1.0`) plus `1.0 * L_ptr`. T2â€™s no-prompt-CE/no-continuation-CE/no-EOS/no-KL semantics remain unchanged. Language rehearsal remains the same 9:1 cadence. Binding/localizer parameters remain frozen; no auxiliary binding objective is added.

## 6. Corrected data
The generator must produce genuine assignment and query counterfactuals, answer-reversing twins, independent name swaps, balanced answer-name marginals, fixed candidate order, balanced fact order and candidate position, and no phrase-slot/object-position shortcut. The prospective training set is 48 complete families (384 rows). DEV and TEST are 16 complete families each (128 rows), with disjoint families and names; TEST is sealed and scored once only after a valid DEV terminal commit. Exact identities are selected mechanically from an authorized lexical allowlist after excluding frozen identities and panels; no outcomes are consulted.

Each family crosses assignment, query, and fact order across eight variants. Structural name-swapped families keep the same relation/template while changing names. This makes pointer supervision face the same counterfactual reversals that defeated the prior shortcut pipelines.

## 7. Scope, schedule, and optimizer
Parent is the untouched Phase1G U6000 best checkpoint (`c5406f8053ec...`). Use three independent seeds 620001â€“620003, each with fresh optimizer state and identical treatment. Run 500 updates: 450 QA and 50 language, nine QA then one language. QA batches contain 32 rows; language batches contain 64 windows. Use AdamW `lr=5e-5`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.05`, no scheduler, global clip `2.0`. All base parameters are trainable; all 984,321 binding/localizer parameters are frozen.

## 8. Evaluation and gates
Evaluate U0, U100, U300, and U500. The primary DEV representation gate requires at least 96/128 pointer retrieval, 48/64 assignment reversals, 8/16 complete families, and at least 48/64 on both name-disjoint and template-disjoint subsets. Native DEV requires at least 96/128 forced-choice, 48/64 reversals, and 80/128 exact answer+EOS. TRAIN16 remains a preserved control with 16/16 correct, 16/16 exact answer+EOS, 8/8 reversals, and 4/4 complete families; a failure is a retention failure. Language must remain within own U0 +0.25 nat and terminal train/DEV gap must be <=0.5. Binding is a preservation check relative to U0 on each existing nonsacred pool; there is no claim of binding acquisition because the frozen binding module is untrained.

Shortcut audits report name, candidate-order, fact-order, token-length, recency, and template predictors. A held-out result explained by those nuisance variables is classified `T3_SHORTCUT_FAILURE`. U100 is descriptive; only integrity, nonfinite state, or catastrophic retention failure can stop a run early.

## 9. Prospective classifications
- `T3_FULL_SUCCESS`: at least two of three runs pass representation/name/family gates, native output gates, language, binding, and shortcut controls.
- `T3_REPRESENTATION_SUCCESS_OUTPUT_FAIL`: representation and generalization gates pass, but native output remains below gate.
- `T3_FAIL_NO_REPRESENTATION`: held-out relational representation/reversal gates fail.
- `T3_SHORTCUT_FAILURE`: apparent success is explained by nuisance leakage or family/twin contamination.
- `T3_LANGUAGE_REGRESSION`: relational progress occurs with language retention failure.
- `T3_HARD_STOP`: integrity or mechanical failure prevents interpretation.

No historical T2-EVAL-TEST, FINAL, sacred, or locked transfer panel is opened by T3 design.

## 10. Implementation plan
Create a versioned T3 bundle containing the corrected generator, deterministic family/name manifests, token-span maps, T2 runner pinned by hash, a minimal pointer-loss adapter, evaluator adapters, schedule, checkpoint/restart controller, and preflight tests. Verify U0 on all inherited rulers before step 1. Run the three branches sequentially, persist raw per-row results before aggregation, and freeze classification before any optional TEST scoring. This is an implementation plan only; no code, optimizer, checkpoint load, or training update was performed in this design pass.

## 11. Risks and limits
The pointer term may align contextual mention states without making the native language head use them; this is why representation and native-output gates are separate. The small T3 corpus tests relational acquisition, not broad language or human readiness. Multi-token names require exact span bookkeeping. Binding preservation is a regression constraint, not a new binding result.
