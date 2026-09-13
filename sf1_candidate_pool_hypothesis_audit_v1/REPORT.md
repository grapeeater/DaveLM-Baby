# SF1 candidate-pool hypothesis audit v1

## Scope

This was a read-only diagnostic-design audit. It inspected preserved SF1 construction and execution artifacts, the readout/selection forensic, the component-swap forensic, Pilot1 language corpora, the completed wider factual-supervision experiment, and the Nemotron comparative audit. It did not load a checkpoint, run inference, score a locked SF1 panel, train, modify an existing artifact, or access FINAL/sacred material.

## Classification

**ALREADY_WEAKENED**

The broad idea that low lexical diversity can encourage shortcuts remains plausible. The specific claim that SF1's Alex default arose from an imbalanced two-name candidate pool is already weakened by the actual design and results:

1. The distractor candidates were evaluation metadata. They were not presented to the model during SF1 training. Each training sequence contained the factual prompt, the correct four-token name completion, and EOS.
2. SF1 answer exposure was exactly balanced. Each of the 16 training records appeared 360 times; Alex, Owen, Mia, and Nora were each the supervised answer 1,440 times.
3. Correct candidate index was balanced 2,880/2,880, and all response completions had four tokens.
4. Pilot1 had substantial inherited name priors, but their magnitude did not predict which pair SF1 learned. Mia was much more favored over Nora than Alex was over Owen, yet SF1 correctly selected both Mia and Nora while failing all four Owen-correct records.
5. The two actor pairs were perfectly confounded with different object vocabularies. Alex/Owen occurred only with `small drum` and `wooden boat`; Mia/Nora occurred only with `soft scarf` and `round plate`. Pair-specific success therefore cannot be attributed uniquely to pool size or name frequency.
6. The component-swap forensic placed the discrete 9/16 to 12/16 training-item gain upstream of final normalization/readout. The SF1 head amplified unrelated-context name mass only in interaction with SF1 upstream representations. This is inconsistent with a simple output-row/base-rate account, though it remains compatible with an upstream shortcut learned from a narrow design.

Claude's proposal survives only in a narrower form: **fixed target-name/object partitioning and low conditional target diversity may have encouraged an upstream shortcut**. Current evidence neither confirms nor cleanly falsifies that data-structure hypothesis.

## What SF1 actually exposed

SF1 TRAIN contains 16 single-fact cloze records:

- two predicates: `found`, `carried`;
- four objects, two assigned permanently to each actor pair;
- two assignment reversals for every predicate/object combination;
- fixed candidate metadata order within each pair: Alex/Owen and Mia/Nora;
- four correct records per name.

The 200-update schedule contains 180 English and 20 binding updates. Every English batch contains all 16 records exactly twice in shuffled order. Thus every record receives 360 presentations and every name receives exactly 1,440 correct-answer presentations.

The English loss construction is:

`BOS + prompt + correct candidate tokens + EOS`

with causal loss only on the four response tokens and EOS. The incorrect candidate is absent. Consequently, a future experiment should call the manipulated variable **conditional target-name diversity** or **name/object pairing topology**, not candidate-set size unless candidates are explicitly included in the model input or loss.

## Tokenization and inherited priors

| name | prompt-start tokens | response tokens | response length | exact name occurrences in Pilot1 TRAIN | Pilot1 unrelated-context mean probability | Pilot1 training-context mean probability |
|---|---|---|---:|---:|---:|---:|
| Alex | `[37,290,92]` | `[314,290,92,18]` | 4 | 85 | 0.00008198 | 0.00003265 |
| Owen | `[51,91,274]` | `[536,91,274,18]` | 4 | 7 | 0.00001773 | 0.00001421 |
| Mia | `[49,77,69]` | `[925,77,69,18]` | 4 | 1,190 | 0.00077599 | 0.00030860 |
| Nora | `[50,276,69]` | `[512,276,69,18]` | 4 | 0 | 0.00003154 | 0.00002921 |

The corpus counts use case-insensitive exact word boundaries over the preserved 9,000-story Pilot1 training file. Token-sequence counts tell the same directional story: leading-space full-name stems occurred 68/3/965/0 times for Alex/Owen/Mia/Nora. Individual token-ID frequencies are not clean name-frequency measures because the BPE pieces also occur inside unrelated words.

Pilot1's unrelated-context mean first-token logits were Alex -2.019, Owen -2.748, Mia -0.425, and Nora -2.487. On the 16 factual prompts, the corresponding means were -1.701, -2.528, 0.446, and -1.908. The Mia/Nora inherited gap was therefore substantially larger than the Alex/Owen gap. Nevertheless, SF1 reached 8/8 on Mia/Nora and 4/8 on Alex/Owen, always choosing Alex in the latter pair.

All four candidates have equal four-token completion lengths. Every prompt actor spelling and its response spelling share two internal name tokens but use different first tokens because of the leading-space boundary. Mia and Nora additionally share the final `a` name piece; Alex and Owen do not. This is another pair-specific difference, although it does not by itself explain first-token selection.

## Existing comparative evidence

The earlier factual-supervision treatment used all four names, all six unordered name pairs, equal name frequency, 48 training families, and 384 factual records. Its factual checkpoint still failed the distinct two-fact Primary acquisition gate: 97/192 item correctness, 22/96 successful reversals, and 0/24 complete families. That experiment differs in task difficulty, training duration, and context structure, so it is not a clean pool-size test. It does show that four globally balanced names and cross-pair exposure did not automatically create contextual selection.

## Diagnostic priority

The upstream-localization forensic has higher information value now. The component swap established that the factual gain is upstream-dominant but bundled embeddings, positions, and all eight blocks. A layer-boundary localization can determine where the Alex/Owen context differential and the unrelated name prior emerge using only the already-authorized 16 TRAIN records and 256 D3 positions. It is cheaper, remains read-only, and can distinguish an early lexical/object shortcut from a later response-selection interaction before committing to another training run.

The candidate-diversity question does not logically require upstream localization, but localization should precede it because the simple frequency account is already weakened and no existing checkpoint provides a clean read-only pool-diversity counterfactual.

## Conclusion

The evidence says SF1 learned an upstream change that solved the Mia/Nora training reversals, strengthened Owen evidence without overcoming Alex, polluted unrelated contexts with name probability, and damaged ordinary language loss. Equal SF1 target frequency rules out unequal SF1 exposure as the direct explanation. Inherited name priors and pair-specific token/object structure remain plausible contributors, but the successful reversal of an even larger Mia/Nora prior prevents a simple base-rate explanation.

Claude's hypothesis survives scrutiny only after being narrowed from “two-name candidate frequency” to “low conditional target diversity and fixed name/object topology may encourage an upstream shortcut.” It is not yet strong enough to displace the planned read-only upstream-localization forensic.

