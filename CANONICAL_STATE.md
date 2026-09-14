# DaveLM/Baby canonical state

**Authoritative current-state record — 2026-09-14.** If an older report, protocol,
or handoff conflicts with this document on current state or an already-adjudicated
historical correction, this document wins unless a newer explicitly authoritative
document supersedes it. Historical artifacts remain preserved as evidence; this is
not a rewrite of their provenance.

## Identity

Baby vNext was trained **from scratch**. The current model has 61,520,385 trained
parameters: 60,536,064 base-model parameters and 984,321 binding/localizer
parameters. It is a decoder-only model with 12 layers, `d_model=640`, 10 heads,
context 256, vocabulary 1024, and DaveLM tokenizer v0_7.

## Critical correction: Phase 1 is not Phase 1G

These are distinct runs and must not be conflated.

| Run | U0 DEV CE | Best / selected result | Final U6000 DEV CE | Status |
|---|---:|---:|---:|---|
| Original Phase 1 | — | 1.3750959888100625 at U3000 | 1.4820435523986817 | Substantial post-U3000 overfit |
| Phase 1G | 7.086638051271438 | 1.2040123894810677 at U6000 | 1.2040123894810677 | Decisive PASS; Phase 2A parent |

Phase1G U6000 `best.pt` is the preserved Phase 2A parent:
`c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.
Phase1G is complete and established the language foundation.

## Critical correction: binding/localizer

The 984,321 vNext binding/localizer tensors were **not inherited as a learned
synthetic-binding skill**. They were found initialization-identical/untrained in
vNext: vNext inherited the machinery design, not the old 10M Baby's trained
synthetic-binding competency. Early Phase 2A sidecar/localizer work found
chance/dead routing under ordinary scoring. Future work must not describe the
frozen sidecar as a previously learned reliable retrieval subsystem.

## Phase 2A goal

Teach contextual/relational binding and native factual answer generation while
preserving Phase1G language behavior. TEST, FINAL, and sacred material remain
outside normal development work.

## Established findings

- Relational representation and native exact generation are separable in this
  protocol. The T17 family repeatedly produced useful relational representation
  without reliable exact native output.
- Full answer-span CE opened a generation channel, but first-token correctness
  can greatly exceed complete answer-plus-EOS generation.
- Exact native generation remains far below the frozen success bar.
- The CE-span-count family, increased suffix weighting, T23 in-row unlikelihood,
  T24 inventory-wide unlikelihood, T28 suffix hard-negative margin, and T29
  TRAIN-identity disambiguation-fork hinge are closed.
- T25 supported a teacher-forced versus free-running continuation/exposure gap.
  T26, T26B, and T27 were resource/engineering stops, not scientific disproofs.
- T28 FAST V2 was a valid completed experiment and is terminally
  `T28_SUFFIX_MARGIN_VALID_COMPLETION / OUTPUT_FAIL`.

## T28 terminal results

| Seed | DEV CE | Pointer | Native forced-choice | Exact + EOS |
|---:|---:|---:|---:|---:|
| 850001 | 1.253619 | 112/128 | 88/128 | 36/128 |
| 850002 | 1.259366 | 108/128 | 82/128 | 35/128 |
| 850003 | 1.256593 | 116/128 | 82/128 | 40/128 |

All three completed U1000. Language, TRAIN16, and protected binding remained
healthy; protected data remained locked.

## Current bottleneck

Strong relational representation has repeatedly been demonstrated under the
current Phase 2A protocol. Native exact generation remains unresolved.
First-token-correct-then-diverge and continuation failures are strongly implicated;
T25 supports an autoregressive continuation/exposure-gap hypothesis.
Architectural, routing, and tokenizer explanations remain hypotheses until directly
tested. Do not call representation "solved."

## Closed doors — do not repeat casually

- Sidecar/localizer-only routing and the name-span `BindingLayout` path
- Paired-reversal batching and the closed scalar-LR sweep
- CE-span-count variants and higher suffix weighting
- T23 in-row and T24 inventory-wide unlikelihood
- T28 suffix hard-negative margin
- T29 TRAIN-identity disambiguation-fork hinge
- Retokenization to conceal generation failures
- Parenting known OUTPUT_FAIL descendants without a new, explicit lineage question
- Lowering the 2/3 replication bar

## Protected data

TEST, FINAL, and sacred material remain sealed/locked. T3 TEST remains
`SEALED_UNOPENED`; no Phase 2A treatment has earned authorization to open it.

## Post-T28 diagnostic result

The two required DEV-only diagnostics are complete. Shared first-token name
geometry is strongly associated with exact-generation failure, while length and
frequency alone are insufficient explanations. In first-token-correct-then-diverge
cases, correct suffix tokens are often directly decodable through Baby's frozen
normalized readout in blocks 7–11 under gold prefixes, but much less often after a
wrong free-running token enters the prefix. See
`PHASE2A_POST_T28_DIAGNOSTIC_SYNTHESIS.md`.

## Tokenizer / output mouth-repair (2026-09-13)

Decode-only study `tokenizer_repair_v1` on the three T28 FAST V2 checkpoints.
Inventory-free beam width 4 did not close the shared-prefix exact gap (4.7% greedy
→ 4.2% beam). An 8-name DEV inventory rerank reached 52.6% shared-prefix exact but
is not native generation and left Wes at 0/48. Tokenizer v0_7 was not changed.
See `tokenizer_repair_v1/TOKENIZER_REPAIR_RESULTS.md`.

Option C (`tokenizer_append_extension_v1`) was a valid completion with no native-exact
effect. Option B (`tokenizer_replacement_study_v1`) is
`TOKENIZER_REPLACEMENT_NOT_JUSTIFIED`. Tokenizer engineering is paused.

## Prefix rescue (2026-09-13)

Forward-pass oracle study `phase2a_post_t28_prefix_rescue_v1` on the 124 DEV
first-token-correct-then-diverge T28 cases. Single-token gold repair then free greedy
recovered exact+EOS 108/124 (87.1%; seeds 32/39, 37/40, 39/45). Wrong-matched and
prefix-compatible-wrong controls recovered 0% exact. Classification:
`EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES`. This is an oracle intervention, not native
generation. Activation patching was not run. No next treatment was designed.

## T29 terminal results (2026-09-14)

`phase2a_t29_disambiguation_fork_v1`: teacher-forced `L_fork` at the first TRAIN-identity
unique-commit token (`lambda_fork=0.5`, margin 1.0) on the T24 recipe. Seeds
860001–860003 all completed U1000.

| Seed | DEV CE | Pointer | Native forced-choice | Exact + EOS |
|---:|---:|---:|---:|---:|
| 860001 | 1.254286 | 110/128 | 88/128 | 35/128 |
| 860002 | 1.254771 | 111/128 | 87/128 | 35/128 |
| 860003 | 1.255805 | 114/128 | 88/128 | 35/128 |

Language and pointer/family representation held (3/3). Shared-prefix exact 2/2/0 of 64
did not beat T28 4.7%. Overall exact 35/35/35 did not beat T18/T24/T28. Classification:
`T29_REPRESENTATION_SUCCESS_OUTPUT_FAIL`. The TRAIN disambiguation-fork objective is
closed. Tokenizer lineage remains paused. TEST remains `SEALED_UNOPENED`. T30 is not
authorized. Full tokenizer replacement or from-scratch retrain requires owner review.

## T30 persistent pointer route (2026-09-14)

T30 was the single owner-authorized architectural successor after T29. It added a zero-initialized 640-parameter per-channel gate that makes the existing query-weighted fact-clause pointer available at every answer prediction position. All three Phase1G-parent branches completed U1000 with binding/localizer locks and protected data intact. The route gate activated in every branch, but native exact generation did not improve beyond the established ceiling.

Terminal classification: `T30_REPRESENTATION_SUCCESS_OUTPUT_FAIL` (2/3 branches representation-positive; 0/3 native full-success). Final DEV exact+EOS was 31/36/38 of 128; shared-prefix exact 2/3/3 of 64; first-token-correct-then-diverge 45/42/41. Parent SHA `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. T3 TEST, T2-EVAL-TEST, FINAL, and sacred remain sealed/locked. T30 artifacts are in `phase2a_t30_persistent_pointer_route_v1`; do not launch T31 under this authorization.
