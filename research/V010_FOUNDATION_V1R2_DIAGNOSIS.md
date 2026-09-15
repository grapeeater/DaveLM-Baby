# v1R2 diagnosis and v2 hypothesis

The v1R2 terminal runs were mechanically valid and replicated the same
scientific result across seeds 101001 and 101002: language CE reached 2.4592
and 2.3648, while contextual copy stayed at approximately zero top-1/exact
performance. The final panel also showed low greedy output diversity (12 and
20 tokens), indicating that the structured loss was being reduced without
learning the intended identity-transport operation.

The implementation audit found no protected-data access, parent checkpoint,
panel overlap, internal target-bigram support, or fixed keyed target duplicate.
The main design difference from the successful T34 experiment is therefore
the starting state and curriculum: v1R2 exposed a fresh model immediately to
long, high-entropy multi-key retrieval and induction mixed together. T34 had a
pretrained substrate and trained on a much simpler family-A decision and
family-B retrieval objective with a lower learning rate.

The v2 hypothesis is narrow: a fresh model can first learn an elementary
one-token contextual continuation when the task is shorter and the wrapper is
varied, then use that substrate to acquire short content-addressed retrieval,
then expand to the full challenge. This is a curriculum hypothesis, not a
claim that the architecture has been proven capable. It is frozen and will be
accepted only if both fresh v2 seeds pass the original final gates.
