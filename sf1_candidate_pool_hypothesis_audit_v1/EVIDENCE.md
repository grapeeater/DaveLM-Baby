# Evidence ledger and provenance

## Established from frozen SF1 artifacts

- `TRAIN.json` has 16 records: four each with Alex, Owen, Mia, and Nora correct.
- Alex/Owen and Mia/Nora are fixed candidate-metadata pairs. Candidate order never changes, while correct index is exactly balanced.
- Alex/Owen are coupled exclusively to `small drum`/`wooden boat`; Mia/Nora to `soft scarf`/`round plate`.
- Both predicates occur equally for both names and both objects in each pair.
- All four response strings contain exactly four tokenizer tokens including the period.
- Every English update contains each record twice; 180 English updates yield 360 presentations per record and 1,440 supervised answer presentations per name.
- The training objective contains only the correct response and EOS. No distractor candidate or explicit candidate set is supplied to the model.
- Pilot1 sequence-likelihood behavior on TRAIN favored Alex in 4/4 Alex-correct records and Owen in only 1/4 Owen-correct records. It favored Mia in 4/4 Mia-correct and Nora in 0/4 Nora-correct records.
- SF1 update100 changed those outcomes to Alex 4/4, Owen 0/4, Mia 4/4, Nora 4/4; full-vocabulary top-1 and exact answer+EOS were 12/16.
- On Owen-correct records, Owen's mean first-token logit rose by 12.150, but Alex remained top-1.
- On 256 unrelated positions, Pilot1-to-SF1 mean first-token logit changes were Alex +4.197, Owen +4.818, Mia +3.561, and Nora +4.475. Combined name probability rose from 0.000907 to 0.067833.
- Component swaps showed all P-upstream conditions at 9/16 sequence correctness and 0/16 top-1/exact, while all S-upstream conditions were 12/16 on all measures. Final-norm and head source did not change this discrete pattern.

## Evidence against a simple frequency/base-rate account

- SF1 supervised-target frequency is exactly equal across names.
- Every name is introduced in update 1 because every English batch contains every training item twice; within-batch serialization order cannot create sequential “introduction” under the batch-mean loss.
- Pilot1 TinyStories TRAIN exact name counts are Alex 85, Owen 7, Mia 1,190, Nora 0. The Mia/Nora imbalance is larger than Alex/Owen, yet SF1 overcame the Mia default and not the Alex default.
- Pilot1 unrelated-context mean probabilities show the same mismatch: Mia/Nora ratio about 24.6, Alex/Owen ratio about 4.6. On factual prompts the ratios are about 10.6 and 2.3 respectively.
- SF1 output-head name-token row deltas were not full-vocabulary outliers, and head-only swaps did not recover the factual gain.
- A historically separate four-name/all-six-pair factual treatment failed its harder two-fact acquisition gate. It is supporting context, not a matched causal comparison.

## Evidence compatible with a broader diversity/topology hypothesis

- SF1 uses only four target labels and partitions them into two noncommunicating object-linked groups.
- Each object identifies the two-name response subset before the actor is interpreted.
- The upstream component could therefore encode an object-to-subset shortcut plus a within-pair default.
- The two pair groups differ in object words, name-token geometry, shared suffix structure, and inherited priors. Existing SF1 data cannot separate these factors.
- The unrelated-context name prior and language damage show that narrow correct-name supervision changed behavior beyond factual contexts.

## Not established

- Candidate-pool size has not been manipulated independently.
- No read-only comparison demonstrates what SF1 would have learned from a wider target-name pool.
- The Alex default is not established as a frequency effect, tokenizer effect, object-family effect, positional effect, or single-layer mechanism.
- The four-name factual-supervision failure does not prove diversity is useless for the easier single-fact task.
- Nemotron's large-scale synthetic diversity is external precedent, not evidence about Baby's causal failure.

## Key input identities

| artifact | SHA-256 |
|---|---|
| Nemotron comparative `REPORT.md` | `d6f26f918780334603ee9a00d659defd339518b42660e6f5ec99ee450276ef0b` |
| Nemotron `RECOMMENDATIONS.md` | `daa0ec8868fb8c7c7a55e64ec085db932bce4fe42a4fdbed0c77a45a461389ad` |
| Nemotron `SOURCES.md` | `43a817782af1b8dea090865b2904e66e3b47c91f4c925bbf01ab3b449c601c55` |
| Nemotron `INSPECTED_PATHS.md` | `4419feedd1e8f01dfbce2ac6bff94a5943337f4ede274038e5c98e409e3253f3` |
| SF1 `TRAIN.json` | `08318d68e0d1b005ae340448af9dabb933013af08046a9cf8fd3ab64153ed9dd` |
| SF1 `PROTOCOL.json` | `5c236478bb7370f514745bcc32463f4e0c860e25f65bbfffa2ceed1d6ccd785f` |
| SF1 `SCHEDULE.json` | `91dad3510e7997b0d55ab53e6189db9337857d582c9399a720723412f70a7ab6` |
| Readout/selection `REPORT.md` | `6200282eff0b8a1d3cd557be5d9e25780d21fe79534b5f7df146f817fd154610` |
| Readout D3 Pilot1/SF1 | `6f4f6ec7e6c5744331c03f99403a68ae2f67b145c9d6cee330d2403a14af7570` / `1e6a0c2f4e7ee834c71efd21de72ec1a8e92128d5fe9e6ce4ce16cf46dca2a7e` |
| Readout D4 Pilot1/SF1 | `beea9c0d29ab1912986ca13e4bc49ce90562e3f096baffb025d92cfc35de1fcf` / `21680ef85378f2a1cf3b219238195f8802ffe80c4e0b76d1e881affcb3f1fa08` |
| Component-swap `REPORT.md` | `b9b679137dff2a148b74fe24fe04de9e085988be49f3e2365fb8a10e86581d68` |
| Component-swap factorial effects | `096325c98c8dc20e0fd8251d764a7125ac739fa989099a9b31c32038eebff555` |
| Earlier factual-treatment `REPORT.md` | `7d2624964b595e79307789de54fa1f42c89c71c0fcf0cdaad91641039553a530` |
| Earlier factual-treatment families | `20fa8830cb6d71480c9834707b37d0cc670dd34e76b2a03f6e966dab03e2ce56` |
| Tokenizer | `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b` |
| Pilot1 TinyStories TRAIN/DEV | `450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c` / `deff4fc7ed18e6e1f0b6f32faccc80f1eb44d0e58f3f38d512ffdfab798d6ac4` |

Locked SF1 HELDOUT, ALTERNATE, COPY, and COMPETING behavior was not scored or inspected. FINAL and sacred material were not accessed.

