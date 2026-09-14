# Phase 2A post-T28 diagnostic synthesis

## Protocol integrity and access

Both protocols were frozen before scoring. Inputs were the 128-row T3 DEV panel,
the frozen DaveLM v0_7 tokenizer, the Phase1G language-train token stream for
frequency counts, and the three terminal T28 checkpoints. T3 TEST, T2-EVAL-TEST,
FINAL, sacred material, and historical locked panels were not loaded. No model
weights changed; no optimizer, gradient, or training step was used.

An initial suffix trace incorrectly omitted final normalization before
intermediate unembedding. It is preserved under
`invalid_pre_norm_omission_attempt` and excluded. The corrected trace applies the
actual Baby final norm before every frozen-head projection. A failed unbatched
tokenizer execution emitted no results; equal-length prefix batching restored the
same frozen greedy semantics.

## Findings

Tokenizer structure is a supported contributor to exact-output failure. Shared
first-token names were 4.7% exact versus 53.1% for unique-first-token names
(odds ratio 0.043), and divergence was doubled. Four-token Skye was especially
poor, but length and frequency were not sufficient explanations: same-length and
zero-corpus-occurrence names had sharply different outcomes. Sal→Salt and
Skye→Sky were frequent, replicated cases; Omar/Opal provided a second shared-prefix
collision group.

In frozen first-token-correct-then-diverge cases, correct suffix tokens become
directly decodable through the frozen normalized readout mainly in blocks 7–11
under gold prefixes. At the native final state, target top-1 was 66.6% with gold
prefixes and 26.5% with generated prefixes. The gap replicated across seeds and
appeared only after an actual wrong token entered the generated prefix; subsequent
errors usually compounded. This supports prefix-conditioned continuation failure.

## Supported, weakened, inconclusive

- **Supported:** tokenizer shared-prefix geometry is strongly associated with
  exact failure; gold-prefix suffix information is often directly decodable late;
  free-prefix divergence sharply reduces that direct decodability.
- **Weakened:** a single explanation based only on answer length, token rarity,
  or full-string rarity.
- **Inconclusive:** tokenizer causality; whether early-layer information exists in
  a nonlinear/distributed form; whether late directly decodable states are
  causally used; exact-success and wrong-first internal traces, which were outside
  the frozen suffix protocol.

These diagnostics do not authorize retokenization, architecture changes, T29,
or protected TEST access.

## Artifacts

- `phase2a_post_t28_tokenizer_diagnostic_v1/{PROTOCOL.json,RUN_DIAGNOSTIC.py,ITEMS.json,DETAILED_SUMMARY.json,TOKENIZER_REPORT.md,SHA256SUMS.txt}`
- `phase2a_post_t28_suffix_state_trace_v1/{PROTOCOL.json,RUN_DIAGNOSTIC.py,TRACE.json,DETAILED_SUMMARY.json,SUFFIX_REPORT.md,SHA256SUMS.txt}`

