# SF4 final report: hard stop before branching

**SF4_HARD_STOP_COMMON_STATE_REPRODUCTION_MISMATCH**

The prospective common-state test failed. No control continuation, treatment continuation, or annealed update ran. No branch receipt was issued. This is an infrastructure-validity failure, not a treatment acquisition failure.

## Decision and prospective design

Selected one late English learning-rate schedule variable: constant5e-5 control versus linearly decaying5e-5-to-zero over90 English updates after100. Both would keep binding LR5e-5, exact SF2 factual CE+parentKL, frozen data/batches, parameter scope and restored optimizer/RNG state.
Baby evidence favored this bounded test: the lone historical SF2 error was a -0.030-nat near-tie after an Owen-side shift; head swapping showed upstream-led movement, so a head constraint was less direct. Margin losses add objective assumptions; replay and KL changes target retention that SF2 already preserved; extra constant CE adds budget. Architecture/corpus changes were unsupported. Comparative models were background only.
Two checkpoints do not establish monotonic overshoot. The LR hypothesis was prospective and tentative. It changes integrated step size and LR-scaled decay as well as schedule shape, so even a completed success would not uniquely establish a settling mechanism.
The design predesignated common_a, required two fresh100-update replicas to match exactly, then required identical real branch loads and identical first same-LR continuation update101. Protocol, sources, LR records and gates were sealed before either common prefix ran.

## Preflight and common-state evidence

Static/dependency checks passed:65 pinned input/source/runtime hashes, all200 frozen schedule rows, masking, KL-pool exact-array disjointness, inherited semantic/overlap validation, runtime and mock restart/negative-state test. No locked panel was parsed or scored.
Common_a and common_b each passed update0 reproduction and common100 safety gates. Their Python/CPU/GPU RNG states, completed update, English index, scope and provenance matched exactly. All update0 raw acquisition, D3 and binding/language outputs were byte-identical.
At update100, 334/414 model tensors differed; max absolute delta=0.000911831855774, RMS=2.84642933546e-06. Optimizer differing fields=668/1002.
The first **logged** difference was update12: {'kl': 2.9802322387695312e-08}. 89/100 metric records differed. This does not locate the first changed parameter or low-level operator.
The earlier forensic A/B equality remains historical evidence. This fresh pair did not meet that guarantee despite matching recorded versions and deterministic-algorithm mode. The specific cause is unresolved; no driver/kernel or training-mechanism claim is warranted.

## Results (common-prefix validation only)

| Replica | Correct | Exact+EOS | Reversals | Families | DEV CE | PPL | D3 name mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| common_a | 11/16 | 10/16 | 3/8 | 1/4 | 3.5740517378 | 35.660789 | 0.0097196371 |
| common_b | 11/16 | 10/16 | 3/8 | 1/4 | 3.5740479231 | 35.660653 | 0.0097202265 |

Both replicas, on each nonsacred pool separately:80/80 answer,80/80 BOTH_DISTINCT, zero collapse,20/20 quartets,40/40 strict reversals; queried-row and downstream correctness conditional on BD both1.0.

## Frozen gates and stopping

- Acquisition at200:16/16 correct AND16/16 exact AND8/8 reversals AND4/4 families. **NOT EVALUATED**.
- Language:CE<=own common0CE+0.25. Both common100 prefixes **PASS**.
- Binding:each pool answer>=76,BD>=76,collapse0. Both common100 prefixes **PASS**.
- D3:mean four-name mass<=0.01. Both common100 prefixes **PASS**; this prospective safety gate does not rewrite SF2.
- Common exact model/optimizer/RNG/controller equality: **FAIL**. Mandatory hard stop before branching.
- Control/treatment endpoint classification: **NOT RUN**. No effect estimate.
- All HELDOUT/ALTERNATE/COPY/COMPETING panels: **LOCKED/UNSCORED**. FINAL/sacred untouched.

## Narrow interpretation and recommended next action

This study establishes that a fresh fully state-checked branch prerequisite did not pass on the recorded current stack. Similar aggregate scores cannot substitute for the frozen numerical guarantee. The LR hypothesis remains untested. SF2 and SF3 classifications are unchanged.
Exact recommended next action: prospectively instrument the unchanged common-prefix calculation around the first logged divergence (updates11-12), recording forward outputs, gradients and post-Adam tensors to identify the earliest non-reproducible operation. Review that bounded reproducibility diagnostic before authorizing it; do not tune LR or relax equality from these outcomes. No such replay or second experiment was run here.

## Artifact identities

Protocol receipt: faa35f4556d88179ae5e6b9abac7738cff51cf74453264abd456c2e1511f3baf
Protocol manifest: 512518c4580de7b7222b1930d476a6fb4525625be134259f0f38648c389a3b84

common_a checkpoint: C:\DaveLM-CADAVER\sf4_common_state_lr_anneal_v1\common_a\checkpoint_100.pt
0c929dcc9ab1db21b5e18f0ba70caedda9da4bb134f16896e5f1bc8b95db461b
Full restart SHA256: 43c921283013ff88e6a92a30c24ddbc14fe6c0d02c8eda64b02f8b9b0d60e8b6

common_b checkpoint: C:\DaveLM-CADAVER\sf4_common_state_lr_anneal_v1\common_b\checkpoint_100.pt
3dfd1ca2df088e860ee22a198a3feb4a163df9bf2200d5fd19880205c3bbdb1c
Full restart SHA256: 6d2ae12dc911669d7cc453587e6fcd65d517e7e609235db9309263ed6ddaa35d

Detailed state and optimizer deltas: MISMATCH_ACCOUNTING.json. Raw metrics and predictions remain in common_a/common_b. Sources, protocol and inputs are listed in PRETRAIN_SHA256SUMS.txt; final outputs are listed in SHA256SUMS.txt.

Dave-coded: the two starting runs look the same on the small scorecard, but they are not the same numerical state. We obeyed the stop gate. Neither annealing nor its control continuation has been tested.
