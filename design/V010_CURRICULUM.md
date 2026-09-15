# Baby v0.10 foundational curriculum

Status: frozen curriculum v1.

The curriculum makes the capabilities required downstream visible before
graduation. All structured train examples are generated from fresh RNG streams;
evaluation panels are independently generated and hash-sealed before the first
optimizer update.

## Stages

### Stage 0 — language base and measurement

Train ordinary causal language modeling on the authorized Phase1G language
stream while measuring held-out language CE, train/DEV gap, and generation
health. This establishes a healthy base but does not imply copy ability.

### Stage 1 — exact contextual reproduction

Mix induction continuation and keyed span retrieval. Values are sampled from a
large token pool whose internal bigrams have zero support in the language stream.
Keys never occur inside values. The model must read an arbitrary context-specified
identity rather than retrieve a fixed answer inventory.

### Stage 2 — structural generalization during training

Every batch randomizes a surface realization: marker vocabulary, separators,
prefix/suffix wording tokens, pair count, queried-pair position, context order,
distractor count, answer position, and length. Training includes both ordinary
teacher-forced copy and short free-running prefix-continuation loss. The held-out
panel changes the marker set and templates entirely.

### Stage 3 — challenge and retention

Continue the mixture while increasing unseen lengths, low-prior spans, distractor
counts, and order changes. Measure all foundation probes every evaluation point.
Select a terminal checkpoint only from the predeclared gates; never cherry-pick a
checkpoint after seeing a favorable metric.

## Anti-shortcut design

For every generator, the question is “how could the model cheat?” Fixed answer
inventories are defeated by fresh random spans. Position shortcuts are defeated
by variable layouts and query order. Frequency shortcuts are defeated by
low-support and high-support strata. Marker memorization is defeated by disjoint
held-out marker families. Bigram and target overlap audits are mandatory.

The evaluator also runs: broken-context, broken-order, shuffled-distractor,
constant-position, fixed-inventory, and answer-prior controls. A low loss without
these controls is not a capability result.

## Mixture schedule

The first diagnostic protocol uses 4000 updates: 500 language-only warm-up,
1000 contextual reproduction, 2000 surface-general mixture, and 500 challenge /
retention. During mixed stages, 20% of updates are language replay and 80% are
structured tasks. The same frozen probes run at U0 and every 250 updates.

The budget is deliberately a diagnostic first run. It is long enough to reveal
whether the diverse objective trains the capability and short enough to stop
without wasting a full v0.9-sized run on a bad generator.
