# Frozen English context battery — pre-execution receipt

Status: FROZEN_ENGLISH_CONTEXT_BATTERY_READY_FOR_EXECUTION

Approved protocol materialized and independently validated. No scientific changes and no mechanical corrections. Zero checkpoint deserializations, model instantiations, forward calls, optimizer creations, backward passes, behavioral scores, or sacred-exam access.

Raw complete-family counts/proportions, reversal profiles, and likelihood-margin summaries are the future headline behavioral evidence. The finite-population interval is secondary and applies only to the enumerated eligible lexical-family universe, never Baby’s general English capability.

## Material counts

| Task/frame | Items |
|---|---:|
| location/core | 128 |
| location/frame_holdout | 128 |
| location/qa | 128 |
| naming/core | 64 |
| possession/core | 128 |
| possession/frame_holdout | 128 |
| possession/qa | 128 |

832 distinct contextual prompts; 208 logical prior comparisons, representing 33 distinct prior strings and 128 distinct prior prompt/candidate-pair combinations.

40 distinct sampled lexical families; 104 family/frame groups. Every group contains eight complete assignment/query/order items and four matched reversal pairs: 416 reversal pairs total.

In every family/frame: assignment, query, fact order, correct candidate, correct-candidate mention rank, and queried-fact rank have counts 4/4. Each candidate is correct four times and incorrect four times, and appears first/second four times. Each of the four entity/value associations occurs four times. All 832 answer keys agree with both q XOR assignment and an independent lookup in the rendered mapping.

## Tokenization and overlap

All 2,080 prompt/candidate boundary checks passed. All 5,200 prompt/candidate/completed-string encode/decode checks passed. Candidate words are two tokens; the terminal period is token18; completed candidates are exactly three tokens. Contextual completed input length, including BOS2, is 26–30 tokens. No padding, truncation, extra EOS, or document token is specified. ByteLevel processing; byte_fallback=false.

| Candidate | Spaced word token IDs | Completed token IDs |
|---|---|---|
| Ben | [532, 274] | [532, 274, 18] |
| Sam | [527, 339] | [527, 339, 18] |
| Tim | [525, 337] | [525, 337, 18] |
| Tom | [525, 295] | [525, 295, 18] |
| bed | [281, 278] | [281, 278, 18] |
| cave | [511, 403] | [511, 403, 18] |
| room | [942, 295] | [942, 295, 18] |
| shop | [373, 1001] | [373, 1001, 18] |
| yard | [715, 609] | [715, 609, 18] |

The 9,000/1,000 story partition was reproduced from the hashed source file and all 10,000 stored story token arrays were retokenized successfully. All 18 variable vocabulary entries met the approved frequency/length rules.

Normalized exact-string checks found zero TRAIN/DEV matches for 55 atomic facts, 160 two-fact contexts, 832 whole prompts, and 1,664 prompt-plus-candidate strings. Normalization is whitespace collapse, strip, and casefold. Exact whitelist-template and ordinary-phrase counts are recorded in PREFLIGHT.json; no absence of semantic paraphrases or near-duplicates is claimed.

## Selected family identities

| Family | Entities | Values |
|---|---|---|
| naming:f000 | bear / cat | Ben / Sam |
| naming:f001 | bear / cat | Ben / Tom |
| naming:f002 | bear / cat | Sam / Tim |
| naming:f003 | bear / fish | Ben / Sam |
| naming:f004 | bear / fish | Sam / Tim |
| naming:f005 | cat / fish | Ben / Tim |
| naming:f006 | cat / fish | Ben / Tom |
| naming:f007 | cat / fish | Sam / Tim |
| possession:f000 | Ben / Sam | bag / book |
| possession:f001 | Ben / Sam | bag / hat |
| possession:f002 | Ben / Sam | car / toy |
| possession:f003 | Ben / Tim | bag / hat |
| possession:f004 | Ben / Tom | bag / book |
| possession:f005 | Ben / Tom | bag / hat |
| possession:f006 | Ben / Tom | ball / toy |
| possession:f007 | Ben / Tom | book / hat |
| possession:f008 | Sam / Tim | bag / hat |
| possession:f009 | Sam / Tim | book / hat |
| possession:f010 | Sam / Tom | bag / hat |
| possession:f011 | Sam / Tom | ball / car |
| possession:f012 | Sam / Tom | ball / toy |
| possession:f013 | Sam / Tom | book / hat |
| possession:f014 | Sam / Tom | car / toy |
| possession:f015 | Tim / Tom | ball / car |
| location:f000 | Ben / Sam | bed / room |
| location:f001 | Ben / Sam | cave / yard |
| location:f002 | Ben / Sam | shop / yard |
| location:f003 | Ben / Tim | cave / yard |
| location:f004 | Ben / Tim | shop / yard |
| location:f005 | Ben / Tom | cave / shop |
| location:f006 | Ben / Tom | shop / yard |
| location:f007 | Sam / Tim | cave / shop |
| location:f008 | Sam / Tim | cave / yard |
| location:f009 | Sam / Tim | shop / yard |
| location:f010 | Sam / Tom | bed / room |
| location:f011 | Sam / Tom | cave / shop |
| location:f012 | Sam / Tom | cave / yard |
| location:f013 | Tim / Tom | bed / room |
| location:f014 | Tim / Tom | cave / yard |
| location:f015 | Tim / Tom | shop / yard |

## Verified input and checkpoint byte hashes

| Input | SHA-256 |
|---|---|
| tokenizer | `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b` |
| source_corpus | `94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4` |
| train | `450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c` |
| dev | `deff4fc7ed18e6e1f0b6f32faccc80f1eb44d0e58f3f38d512ffdfab798d6ac4` |
| pilot1_train_copy | `450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c` |
| pilot1_dev_copy | `deff4fc7ed18e6e1f0b6f32faccc80f1eb44d0e58f3f38d512ffdfab798d6ac4` |
| pilot0_manifest | `03b251542818cb436290d7de7690f25978d4f518350d8f6736c1b8ccd64c0708` |
| pilot1_manifest | `83c2eef6260bf616be61d88f144eb2e823bba11353a69fba663d8e288361f5e2` |
| checkpoint/graduate (unchanged before/after) | `fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430` |
| checkpoint/pilot0 (unchanged before/after) | `769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5` |
| checkpoint/pilot1 (unchanged before/after) | `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb` |

## Common nonsacred binding references

| Reference | Documents / quartets | SHA-256 |
|---|---|---|
| pilot0_dev | 80 / 20 | `30bfbcbe1b11d3d2d427a631562f43b24dbb97f6e595164516d40553792edc4e` |
| pilot1_dev | 80 / 20 | `3b6774b2cadeb7818d59c8a4f4f0b69b2cd85744efce91af562d9bd6f2a54ea1` |

Both sets are internally unique and have zero full-token-array intersection with each other. Each set also has zero intersection with both pilots’ rehearsal pools and the four checked T5/T10/T11/T13 prior pools. All six checked file hashes, array counts, and zero intersections are recorded in PREFLIGHT.json. Sacred pools were not opened; historical sacred-disjointness claims were only read from existing manifests. No binding evaluation was performed.

## Ordinary language path isolation

Static source/AST review confirms the only model inference expression in the frozen scoring primitive is model.base_model(input_ids). There is no wrapper forward, localizer/retrieval call, checkpoint loading, optimizer, backward call, or import-time execution. Correct causal-shift indexing and FP32-model/FP64-log-probability scoring were source-reviewed. The existing final_norm capture hook returns None without modifying outputs. No dynamic model execution is claimed or permitted by this receipt.

## Independent review

An independent read-only audit reconstructed family selection and all rendered items, answer keys, balancing and prior links; retokenized all 2,080 boundaries; recounted the lexicon; checked corpus absence, six training-disjointness comparisons, source hashes, and UTF-8/LF encoding. Result: PASS; no discrepancies or corrections.

## Exact frozen scientific-artifact byte hashes

| Artifact | SHA-256 |
|---|---|
| PROTOCOL.md | `ced4af59529bd2ede5540dd6697b9ecd7dfba176075ceb902000caea1b255f77` |
| MANIFEST.json | `777213e605dadc0f86bbe330608a11d6e00a8f09e1264fb5895092531e12be9c` |
| LEXICON.json | `6376b5a8612682c17f06a5af80419dc20e1dd02fe539940903c957bb6458cbdd` |
| FAMILIES.json | `c21006dd345219bdb50eec5b01870206c51aa12cf48b5178eeefa03a747f31f9` |
| ITEMS.jsonl | `5e6669275562044012113a353996e627a1b9faad28b6b4664328d27e268a509a` |
| PRIORS.jsonl | `da476f147a40d7bc3e37187e0d98811a526af0ce59b28a395bc70826828e7902` |
| PREFLIGHT.json | `fcdcd782db1fc6f425c13e93ca4dc8947a836e3503013b8635d51edb7c2db42f` |

The detached SHA256SUMS.txt receipt hashes all seven scientific artifacts, both frozen implementation sources, and this receipt. It does not contain its own hash; its actual byte hash is reported in the completion message. This avoids circular/self-referential hashes. All sealed files use UTF-8 without BOM and LF line endings, and are marked read-only.

FROZEN_ENGLISH_CONTEXT_BATTERY_READY_FOR_EXECUTION
