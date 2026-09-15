# Baby v0.10 foundation v2R1 diagnostic adjudication

Protocol: `BABY_V010_FOUNDATION_V2R1`  
Seed: `103001`  
Updates: 2,000  
Parent checkpoint: none

The identity-only loss correction removed the deterministic separator/EOS
shortcut, but did not produce the primitive capability. At update 2,000,
language DEV CE was 2.604; primitive-induction first-token top-1, teacher-
forced exact, and free exact were all 0 on the 16-item interim probe, with
median target rank 134. Primitive keyed, short keyed, same-surface induction,
and full keyed panels were also 0 exact; greedy output diversity was 1 token.
The structured identity loss remained near the random-token regime (~6.1-6.4)
rather than decreasing through contextual learning.

This is not a final capability-gate verdict because the protocol specified a
diagnostic boundary rather than 6,000-update graduation. It is nevertheless a
clear failure of the v2R1 diagnostic hypothesis at the planned boundary. The
combined evidence from v1R2 and v2R1 is now:

1. language-only learning reaches a healthy short-run CE;
2. 500 language warmup updates are not enough to support fresh contextual copy;
3. removing target-suffix shortcuts is necessary but insufficient; and
4. the fresh model has not yet received the 6,000-update language foundation
   used by the successful v0.9/T34 parent lineage.

The next preregistered hypothesis is therefore a longer fresh language
foundation, using the verified Phase1G 6,000-update / effective-batch-64
schedule, followed by an identity-only primitive stage at a lower structured
learning rate. No v0.9 checkpoint is imported, and no final gate is weakened.
