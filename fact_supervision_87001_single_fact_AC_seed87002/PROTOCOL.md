# Single-fact A+C diagnostic (seed-87002) — pre-execution freeze

A: retain the queried factual sentence and original query/candidates; remove the competing factual sentence.
C: retain the queried factual sentence, add one neutral sentence mentioning the other candidate name (fixed template `{name} sat down.`), balanced so the factual sentence appears first for half and second for half.

Scoring: reuse EVALUATE.py candidate conditional LL over response tokens + EOS; signed margin; correct iff margin>0, tie==0; greedy argmax<=32 stopping at EOS. No normalization, ranks/top-k, perplexity, or thresholds.

Report A and C separately and their paired difference: exact correct/total, ties, object/predicate strata, candidate-index/name balance, frozen margin summaries, greedy exact, and (C) sentence-order balance. The original two-fact relational reversal and query dimensions do not transfer unchanged.

No checkpoint was loaded and no inference was run during this freeze.
