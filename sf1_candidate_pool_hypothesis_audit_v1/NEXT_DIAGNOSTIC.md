# Next diagnostic decision

## Immediate next action: upstream-localization forensic

Freeze and execute a read-only `SF1_UPSTREAM_LOCALIZATION_FORENSIC_V1` before any candidate-diversity training study.

### Authorized inputs

- authoritative Pilot1 checkpoint;
- preserved SF1 update100 checkpoint;
- exact 16 SF1 TRAIN records;
- exact frozen 256 unrelated D3 positions;
- existing architecture/runtime and tokenizer.

No SF1 HELDOUT, ALTERNATE, COPY, COMPETING, readiness FINAL, or sacred material should be loaded or scored.

### Intervention

At the embedding boundary and after each transformer block 0–7, construct both directional prefix/suffix functional hybrids:

- Pilot1 prefix through boundary `k` followed by the SF1 suffix;
- SF1 prefix through boundary `k` followed by the Pilot1 suffix.

At each boundary retain both Pilot1 and SF1 final output paths as separately reported readouts. Baseline all-Pilot1 and all-SF1 paths must reproduce the already-frozen D3/D4 values within a preregistered numerical tolerance before interpretation. Embedding-only, complete-prefix, and complete-suffix endpoints must be included so no boundary is inferred from a partial sweep.

For every TRAIN item report first-token and four-token correct-minus-distractor margins, full-vocabulary top-1, candidate mass, exact answer+EOS, and groups Mia/Nora, Alex/Owen, Alex-correct, and Owen-correct. For D3 report per-name logits/probabilities/ranks, combined name mass, entropy, EOS, and true-next-token behavior.

Also record native Pilot1/SF1 answer-position hidden-state contrasts for every matched reversal at every boundary. These descriptive contrasts must not be called causal without the corresponding suffix intervention.

### Decision signatures

- If a narrow boundary makes SF1 prefixes sufficient and Pilot1 prefixes necessary for the Alex default while relational reversal contrast remains absent, that supports an upstream lexical/object shortcut and raises the value of the conditional-diversity study.
- If correct relational contrast emerges early but a later SF1 suffix collapses Owen behind Alex, prioritize that later selection/readout interaction rather than target diversity.
- If effects are distributed across boundaries or cross-model suffixes are unstable, classify the localization as mixed; do not force a layer claim.
- If native baseline reproduction fails, stop without interpreting hybrids.

This diagnostic does not prove or falsify pool diversity directly. It determines whether another training experiment is justified and which upstream failure signature it should be expected to change.

## Why a read-only widening test is not available

No existing checkpoint differs from SF1 only in target-name diversity. Adding extra distractors at evaluation would change the inference choice set, not the training distribution. Scoring the locked COMPETING panel is unauthorized and would still not create a training-pool counterfactual. The older factual checkpoint used wider pair coverage but also changed context structure, training duration, and task difficulty. It cannot identify the pool effect.

## Deferred matched conditional-diversity study

If upstream localization supports a lexical/object shortcut, preregister this two-arm treatment. Do not describe it as an explicit candidate-set intervention because candidates remain absent from the training input.

### Manipulated variable

Conditional target-name diversity per object family:

- **Fixed-pair arm:** each object is permanently associated with one two-name pair across predicates and cues.
- **Rotating-pair arm:** the same names, objects, predicates, cues, record count, and per-name exposure are used, but pair-to-object assignments rotate prospectively so every object is exposed to all names.

### Minimal balanced construction

Use eight tokenizer-eligible names, four objects, two predicates, and two frozen cue frames. For every `(object, predicate, cue)` cell render the two assignment reversals for its assigned pair, producing exactly 32 unique training records per arm.

- Fixed arm: one fixed perfect matching of the eight names is repeated across the four `(predicate, cue)` cells, with one pair assigned permanently to each object.
- Rotating arm: four prospectively fixed edge-disjoint perfect matchings are assigned one-to-one to the four `(predicate, cue)` cells using a deterministic Latin-square mapping. Across those four cells every object sees all eight names once, each name appears exactly once per cell and four times overall, and no unordered pair repeats.

This gives both arms exactly 32 records; four target occurrences per name; eight per object; sixteen per predicate; sixteen per cue; two assignment reversals per cell; and identical total response-token/EOS supervision. The same materialized schedule must repeat corresponding records equally. Name selection and matching must be frozen before checkpoint loading.

Tokenizer eligibility requires identical response-token length, exact boundary concatenation, round-trip decoding, no special-token collision, and matched prompt-start/leading-space structure. Because both arms use the same eight names, corpus frequency and token geometry are held constant across arms. Pair assignments must balance shared-subtoken similarity so no object or arm receives systematically easier name pairs.

### Policies held identical

Use the authoritative Pilot1 parent, SF1 causal response+EOS objective, SF1 protected English scope, binding scope and rehearsal, optimizer, learning rate, 9:1 cadence, update budget, seed policy, update-100 regression guard, TinyStories monitoring, binding gates, and persistence. Any departure requires a new scientific decision.

### Prospective primary evidence

At update 0 and the frozen monitoring points, report training-family assignment accuracy, both-correct reversal pairs, complete families, exact answer+EOS, first-token and sequence margins, full-vocabulary candidate/name mass, and concentration of wrong answers on a single default name. Compare arms at the family level. Preserve the same frozen unrelated D3 sample, TinyStories aligned CE, and both nonsacred binding pools as diagnostics/gates.

The diversity hypothesis is supported only if the rotating arm prospectively reduces fixed-default concentration and improves reversal-family acquisition relative to the fixed arm without worse language or binding regression. Equal performance, a merely different default name, or improvement explained only by changed full-vocabulary name mass weakens it. Held-out transfer remains a separate later claim.

## Ordered recommendation

1. Run the upstream-localization forensic.
2. Review whether it reveals a lexical/object shortcut signature.
3. Only then freeze the matched conditional-diversity study above if the evidence still warrants training.

No treatment, dataset, or evaluation was executed in this audit.

