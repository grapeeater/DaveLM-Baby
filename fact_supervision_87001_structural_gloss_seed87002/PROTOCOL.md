# Structural-gloss companion diagnostic (Fork B) — pre-execution freeze

Replace the two English fact sentences with structural rows `N <name> P <predicate> O <description>` and keep the original English query unchanged (`The person who <predicate> the <description> was`).

Both competing mappings and both candidate names are present; the correct actor is the unique row whose (predicate, description) equals the query. Row order is balanced and does not change the answer key.

Scoring: reuse the frozen candidate conditional LL over response tokens + EOS; signed margin; correct iff margin>0, tie==0; greedy argmax<=32 stopping at EOS. No normalization, ranks/top-k, perplexity, thresholds, significance tests, or mechanistic intervention.

Report (per checkpoint, overall and object/predicate strata): exact correct/total, ties, families-complete, correct-index/name slices, row-order slice, frozen margin summaries, greedy exact; plus a descriptive gloss-vs-A / gloss-vs-C difference using the already-frozen A/C aggregates. The original two-fact relational reversal and query dimensions do not transfer unchanged.

No checkpoint was loaded and no inference was run during this freeze.
