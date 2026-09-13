# P7 lineage and report-card erratum — 2026-09-05

This new amendment supersedes erroneous conclusions and documentation only.
Historical archives, checkpoints, v1 battery, execution outputs and the integrity
stop audit remain unmodified.

## Actual immediate P7 parent

The evidence identifies `C:\DaveLM-CADAVER\language_sentencebound_p5\latest.pt`
as the immediate P7 parent, not P6.

- P7 training source explicitly sets PARENT to language_sentencebound_p5/latest.pt.
  Source SHA-256: bab7f60082bd38201454f256d5eaeaf79afc22fae187b1a2d85ce0da83a46f21.
- P7 RESULTS.json records parent_sha256
  da009100412fba2adcf58823aefd37cc9486ef7aff39672361ce6111773c6a3b.
  RESULTS.json SHA-256: 8379fb6b7e742e5b1320d675a392a8f22f911164362616e9a3f512bcda331901.
- The current P5 checkpoint hashes to that exact parent_sha256.
- The current P6 checkpoint instead hashes to
  e708f00bdab4c4325382f3ceb724ad377360daf8ff2f6c14278cc17e692a9930.

Both ARCHIVE_RECORD.md and ARCHIVE_RECORD.json incorrectly named P6. Their parent
hash was correct for P5. The historical source and run artifact agree with the
actual P5 file hash. This audit did not deserialize any checkpoint or instantiate
a model. It establishes recorded immediate lineage from source/run provenance;
it is not a reconstruction of training from tensor differences.

Original P7 and read-only archived P7 both remain:
d41ed1186cc945aa05dbd2ba3086fac08ff3b97035c9532e4fa70e78b149a20e.

## Withdrawn v1 interpretation

The reported controlled accuracy, answer-oriented margins, reversals and complete
families from post_p7_language_report_card_v1_seed8380 are invalid as measures of
the intended task because 144/288 keys contradict the rendered facts. Prior claims
that those metrics demonstrated transfer failure are withdrawn. No corrected
scores are substituted. Original raw observations and naturalistic continuations
remain historical evidence; their existence does not validate the defective keys.
The original corrected P7 12/12 panel remains a narrow milestone. Broad contextual
transfer remains unestablished, rather than established as a failure by v1.

## New prospective replacement

post_p7_language_report_card_v2_seed8391 contains newly rendered contexts with
distinct object colors, independent semantic validation and matched transformations.
The exposed v1 texts were read solely as contamination references, not reused or
repaired. V2 controlled name completion measures factual selection under its
specific QA/cloze formats; it does not by itself measure full-sentence generation.
Generation is reserved for the separate 24-prompt naturalistic panel.

No model behavior, inference, training, treatment or parent selection occurred.
