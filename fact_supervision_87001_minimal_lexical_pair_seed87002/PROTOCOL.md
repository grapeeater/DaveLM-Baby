# Minimal-lexical paired diagnostic (Fork L2) - pre-execution freeze

ONE: emit only the correct mapping row (48 items).
TWO: emit both competing mapping rows (96 items).
Row syntax: `<name> <predicate> the <description>`; original English query unchanged (`The person who <predicate> the <description> was`).

Scoring: reuse the frozen candidate conditional LL over response tokens + EOS; signed margin; correct iff margin>0, tie==0; greedy argmax<=32 stopping at EOS. No normalization, ranks/top-k, perplexity, thresholds, significance tests.

Report ONE and TWO separately (overall and object/predicate): exact correct/total, ties, complete families, correct-index/name slices, row-order slice (TWO), frozen margin summaries, greedy exact; plus a descriptive ONE-vs-TWO difference and a comparison with the already-frozen A/C and gloss aggregates. Original two-fact relational reversal/query dimensions do not transfer unchanged.

No checkpoint was loaded and no inference was run during this freeze.
