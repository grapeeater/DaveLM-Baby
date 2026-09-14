# DaveLM/Baby canonical state

**Authoritative current-state record — 2026-09-13.** If an older report, protocol,
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
  T24 inventory-wide unlikelihood, and T28 suffix hard-negative margin are closed.
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

T29 remains unplanned and unauthorized. Full tokenizer replacement or from-scratch
retrain requires owner review.
