# Prospective English context-sensitivity characterization protocol

Protocol version: 1.0. Seed: 8380. Status: approved specification for a pre-execution frozen battery.

## Purpose and scope

This protocol tests whether ordinary causal-LM predictions respond correctly to counterfactual changes in short English contexts. Possession/association cloze and location cloze are the primary tests. Naming, QA, alternate frames, and synthetic binding remain separate diagnostics.

The purpose is not to determine whether Baby “understands English.” It is to determine whether the language-trained checkpoints acquired measurable context-sensitive ordinary-English behavior, whether that behavior survives counterfactual reversal, how it differs across Graduate/Pilot 0/Pilot 1, and what the results make useful to investigate next.

Reporting clarification approved at implementation: raw complete-family counts/proportions, reversal profiles, and likelihood-margin summaries are the headline behavioral evidence. The finite-population confidence interval is secondary and applies only to the explicitly enumerated eligible lexical-family universe. It must not be presented as uncertainty over Baby’s general English capability.

This document is the scientific specification. Numerical observations in Section 1 labeled prior read-only preflight are inherited from the approved protocol and must be reproduced by the materialization preflight. Current material-integrity validation and receipts belong in PREFLIGHT.json and MANIFEST.json. No checkpoint inference, behavioral scoring, generation, optimizer creation, training, or backward pass is authorized during the materialization/freezing turn.

## 1. Tokenizer / corpus preflight

Use the existing tokenizer:

`C:/DaveLM-v0.9/tokenizer/v0_7/davelm_tokenizer.json`

SHA-256: `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.

It uses ByteLevel processing with `byte_fallback=false`. Do not describe this implementation as using byte fallback.

### Answer-token decision

Use matched two-token answer words. Familiar single-token answers are too scarce to provide a suitable inventory for these tasks. For example, `" dog"` is token 870, but names and most ordinary locations are multi-token. Do not change the semantic task to accommodate the single-token vocabulary.

Prior read-only preflight reported the following encodings and frequencies. Counts use case-sensitive matches bounded on both sides by the absence of an ASCII letter. Thus possessive occurrences such as `Tom's` contribute to the count for `Tom`. Story counts count each document once.

| Word | Without leading space | With one leading space | TRAIN occurrences / stories | DEV occurrences / stories |
|---|---|---|---:|---:|
| Ben | [38,274] | [532,274] | 4,168 / 546 | 344 / 46 |
| Sam | [55,339] | [527,339] | 2,136 / 313 | 189 / 29 |
| Tim | [56,337] | [525,337] | 2,453 / 360 | 288 / 40 |
| Tom | [56,295] | [525,295] | 3,991 / 536 | 531 / 65 |
| bed | [70,278] | [281,278] | 683 / 408 | 50 / 31 |
| cave | [398,403] | [511,403] | 183 / 64 | 13 / 5 |
| room | [313,295] | [942,295] | 1,306 / 801 | 113 / 72 |
| shop | [87,76,1001] | [373,1001] | 149 / 77 | 31 / 13 |
| yard | [93,609] | [715,609] | 225 / 137 | 20 / 17 |

Other variable words:

| Context/query word | TRAIN occurrences | DEV occurrences |
|---|---:|---:|
| bear | 1,278 | 76 |
| cat | 1,121 | 125 |
| fish | 680 | 58 |
| bag | 307 | 30 |
| ball | 1,376 | 142 |
| book | 373 | 42 |
| car | 1,461 | 166 |
| hat | 380 | 51 |
| toy | 1,700 | 187 |

All these words also encode as two tokens with a leading space.

### Lexical eligibility rules

The fixed vocabulary must satisfy all of the following:

- At least 100 TRAIN occurrences in at least 50 TRAIN stories.
- Exactly two tokens when encoded with one leading ASCII space.
- Exact encode/decode round trip.
- No special token within a word.
- Within each paired vocabulary set, the higher TRAIN occurrence count is at most twice the lower count.
- No selection based on checkpoint outputs.
- DEV frequencies are reported, not used to optimize pair selection.

These are accessibility and matching rules, not capability thresholds.

### Answer boundaries

Prompts end without trailing whitespace. Every scored completion is:

`one ASCII space + answer word + "."`

For example, `" Tom."` is `[525,295,18]`. Every answer therefore contains exactly three scored tokens, including the common period.

The period establishes an explicit word/answer boundary. Score the complete sequence; do not compare only the first token. `Tim` and `Tom` share first token 525.

For every prompt/candidate combination, require:

\[
\operatorname{encode}(P+C)=\operatorname{encode}(P)\,\Vert\,\operatorname{encode}(C).
\]

Prior read-only preflight reported that contextual-item boundary checks passed and that completed inputs, including BOS, contain 26–30 tokens, below the 256-token limit. Reproduce this validation during materialization.

Retain word-only scores, excluding the period, as a diagnostic for effects attributable to completion termination.

### Corpus overlap

Frozen overlap normalization: case-fold text and collapse whitespace. Report exact substring matches under that normalization.

Prior read-only preflight reported:

- None of the 55 distinct constructed atomic fact sentences occurred exactly in TRAIN or DEV.
- None of the 160 distinct two-fact contexts occurred.
- None of the 832 complete test prompts occurred.
- Consequently, none of their completed versions occurred exactly.

This establishes exact-string absence under that normalization, not absence of related stories or semantic paraphrases.

Frame familiarity varies. Prior preflight reported TRAIN counts of 4 for `is called`, 11 for `has the`, 22 for `is in the`, and 1 for `the person with`. The reserved location phrase `can be found in the` occurs zero times in TRAIN and DEV. This difference must inform interpretation of the frame diagnostic.

The source files and selection procedure are documented in:

`C:/DaveLM-CADAVER/language_pilot_0_tinystories_seed8380/CORPUS_MANIFEST.json`.

## 2. Minimal contextual-dependency probe

Use naming association with a paraphrased query:

```text
The bear is called Ben. The cat is called Sam.
The name of the bear is
```

Candidates: `" Ben."`, `" Sam."`.

The matched reversal is:

```text
The bear is called Sam. The cat is called Ben.
The name of the bear is
```

This tests retrieval of a supplied identity association and avoids repeating the same complete clause as the answer prompt. “Minimal” refers to the relation being expressed; the counterbalancing remains as complete as in the relational tests.

Fixed animal vocabulary: `bear, cat, fish`.

Fixed name vocabulary: `Ben, Sam, Tim, Tom`.

Use 8 lexical families × 8 items = 64 items.

This stage is diagnostic. Its results must never prevent evaluation or reporting of later stages.

## 3. Relational cloze — primary tests

Use exactly these two primary constructions:

| Test | Fact sentences | Query prefix | Answer candidates |
|---|---|---|---|
| Possession/association | `Ben has the bag. Sam has the book.` | `The person with the bag is` | `" Ben."`, `" Sam."` |
| Location | `Ben is in the bed. Sam is in the room.` | `Ben is in the` | `" bed."`, `" room."` |

Possession measures having/association, not legal ownership. It avoids the stronger inference that having an object necessarily means owning it.

Possession queries an object to recover a person. Location queries a person to recover a place. Differences between these tests cannot be attributed purely to the semantic relation: query direction and wording also differ.

Use 16 lexical families per relation, each containing eight items:

- Possession: 128 primary items.
- Location: 128 primary items.

### Complete family construction

Let the entity pair be \(E_0,E_1\), the value pair \(V_0,V_1\), and assignment \(a\in\{0,1\}\).

The facts express:

\[
E_i\rightarrow V_{i\oplus a}.
\]

Cross three binary factors:

| Factor | Values |
|---|---|
| Assignment a | Original / reversed |
| Query q | First identity / second identity |
| Fact order o | Fact for E0 first / fact for E1 first |

This produces eight items.

- Naming and location query \(E_q\); correct answer is \(V_{q\oplus a}\).
- Possession queries \(V_q\); correct answer is \(E_{q\oplus a}\).
- Print the fact for \(E_o\) first, followed by the other fact.

For possession, the four assignment/query combinations before order balancing are:

| Assignment | Query | Correct answer |
|---|---|---|
| E0→V0, E1→V1 | V0 | E0 |
| Same | V1 | E1 |
| E0→V1, E1→V0 | V0 | E1 |
| Same | V1 | E0 |

Evaluate both fact orders for every row.

## 4. Explicit QA — separate diagnostic

Use exactly the same facts, assignments, queries, orders, lexical families, and candidate sets as the corresponding primary cloze test. Replace only the query prefix:

| Relation | QA prefix |
|---|---|
| Possession | `Who has the {queried_object}?` |
| Location | `Where is {queried_person}?` |

Append the same leading-space, word, and period completion.

- Possession QA: 128 items.
- Location QA: 128 items.

Report each QA result separately from its cloze counterpart. There is no combined cloze/QA endpoint. Successful cloze with weak QA remains evidence of context-sensitive continuation and identifies an additional question-format difficulty.

## 5. Checkpoint comparison

Use these exact checkpoints:

| Checkpoint | Path | SHA-256 |
|---|---|---|
| Canonical Graduate | `C:/DaveLM-CADAVER/treatment13_orthogonal_shared_unbounded_seed8380/checkpoints/orthogonal_shared_unbounded/seed_8380/latest.pt` | `fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430` |
| Pilot 0 | `C:/DaveLM-CADAVER/language_pilot_0_tinystories_seed8380/pilot_run/checkpoints/seed_8380/latest.pt` | `769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5` |
| Pilot 1 | `C:/DaveLM-CADAVER/language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt` | `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb` |

For every English item in the later execution:

- Call only the ordinary `base_model` causal-LM path.
- Supply no query-position metadata, answer-position metadata, localized slots, or retrieved residual.
- Use evaluation mode and inference mode.
- Use FP32 model computation, with autocast and TF32 disabled.
- Compute log-softmax and score accumulation in FP64.
- Use the same device and implementation for all checkpoints.
- Process unpadded sequences individually; no context truncation.
- Prepend exactly one BOS token, ID 2.
- Append no EOS or document marker.
- Create no optimizer and perform no backward pass.
- Perform no generation or decoder search.
- Verify checkpoint hashes before and after execution.

The comparison is behavioral. Different scores cannot isolate the causal effect of block protection, language adaptation, rehearsal material, or synthetic-to-English transfer.

## 6. Required counterbalancing and shortcut controls

The eight-item family guarantees:

- Each candidate is correct four times and incorrect four times.
- Each entity is associated with each value.
- Both query identities occur equally.
- Both sentence orders occur equally.
- The queried association appears first and second equally.
- The correct candidate is the earlier and later candidate mention equally.
- Both candidates appear once in each factual context.
- Matched reversals use identical candidate sets.
- Reversal pairs preserve the query and fact-order condition.
- Paired words have equal token lengths and bounded frequency differences.

Semantic categories remain grammatical: people are not substituted into object positions. “Swapping roles” means swapping identities within those categories and changing their assignments.

### No-context priors

For every family, query identity, and evaluated frame, score the query prefix alone with the same candidate pair. This produces 208 logical prior comparisons per checkpoint. Identical computations may be cached, but preserve their links to every relevant family.

These priors measure candidate preferences under the query wording. They are diagnostic because removing facts also changes prefix length and position.

### Explicit shortcut reference

Report predicted performance of three deterministic rules on the generated answer keys:

- Always choose the same candidate.
- Choose the first candidate mentioned in the facts.
- Choose the last candidate mentioned in the facts.

All achieve zero complete-family success. A first/last-mention rule can achieve 50% successful reversal pairs on naming or location because candidate order reverses with the assignment. Reversal-pair success alone is therefore insufficient. Complete-family success is essential.

Do not claim the battery excludes every possible shallow algorithm. Success may reflect a narrow learned rule for these constructions; that is within the intended characterization.

Exclude distractors from this experiment. Distance/interference robustness would be a separate subsequent protocol.

## 7. Scoring

Let \(P\) be the prompt tokens, including BOS, and let candidate \(c\) contain tokens \(t_1,\ldots,t_L\). Here \(L=3\).

Candidate conditional log-likelihood:

\[
\ell(c\mid P)=\sum_{j=1}^{L}\log p_\theta(t_j\mid P,t_1,\ldots,t_{j-1}).
\]

Use natural logarithms and the ordinary vocabulary softmax. Score every candidate token at its correctly shifted causal prediction position. Do not average by length: the candidates have equal token counts, and the sum is the conditional log-probability of the specified completed answer.

For a fixed candidate ordering \(c_0,c_1\):

\[
d_i=\ell(c_0\mid P_i)-\ell(c_1\mid P_i).
\]

Correct-minus-incorrect margin:

\[
m_i=\ell(c_{\mathrm{correct}}\mid P_i)-\ell(c_{\mathrm{incorrect}}\mid P_i).
\]

Item correctness:

\[
I_i=\mathbf1[m_i>0].
\]

Exact ties count as incorrect and are reported separately.

Reversal correctness pairs the two assignments at fixed family, query, order, and frame:

\[
R_{f,q,o}=I_{f,0,q,o}\,I_{f,1,q,o}.
\]

Both opposed contexts must prefer their correct answer. Merely shifting toward the correct answer does not suffice.

Complete-family correctness:

\[
F_f=\prod_{a,q,o}I_{f,a,q,o}.
\]

All eight items must be correct.

### Two primary endpoints, separately for each primary relation

1. Complete-family proportion:

\[
\widehat F=\frac1n\sum_f F_f.
\]

2. Mean within-family reversal success:

\[
\widehat R=\frac1n\sum_f\frac14\sum_{q,o}R_{f,q,o}.
\]

Do not combine possession and location into one primary score. Headline evidence is raw complete-family counts/proportions, reversal profiles, and likelihood-margin summaries; the finite-universe interval is secondary.

### Diagnostics

Report:

- Item accuracy and tie counts.
- Distribution of family reversal scores: 0, 1/4, 1/2, 3/4, 1.
- Median family-mean margin.
- Median family-minimum margin.
- Raw candidate log-probabilities and candidate-pair probability mass.
- Word-only versions of the margins, excluding the period.
- Query-only priors.
- Signed changes across matched reversals.

Candidate-pair mass matters: preferring the correct answer within this pair does not establish that the model would freely generate it. If word-only and terminated-answer results differ substantially, describe that as completion-boundary sensitivity and retain both results.

### Calibration decision

Do not subtract priors. The primary question concerns the checkpoint’s actual contextual preference. Prior subtraction would change that question and could credit a contextual shift while the model still prefers the wrong answer.

## 8. Generalization split

Reserve one alternate query frame per relation, using the same sampled lexical families and factual contexts:

| Relation | Primary query | Reserved query |
|---|---|---|
| Possession | `The person with the {object} is` | `The {object} is with` |
| Location | `{person} is in the` | `{person} can be found in the` |

- Possession reserved frame: 128 items.
- Location reserved frame: 128 items.

Freeze these frames before checkpoint scoring. Evaluate them in the same execution, but report them separately. No revisions are permitted after viewing primary results.

Use the term “robustness to the specified alternate query frames.” There is no claim of broad linguistic generalization or of vocabulary unseen during training.

Do not add a lexical holdout. The limited matched vocabulary and small experiment favor keeping vocabulary constant while changing the query construction.

The reserve is a separate reporting stratum, not a training/test split: no checkpoint learns from the core battery. Its unfamiliar constructions make failure less diagnostic than success.

## 9. Common synthetic-binding reference

Use both existing nonsacred DEV sets:

- `C:/DaveLM-CADAVER/language_pilot_0_tinystories_seed8380/binding_dev.json`: 80 documents, 20 quartets.
- `C:/DaveLM-CADAVER/language_pilot_1_early_block_protection_seed8380/binding_dev.json`: 80 documents, 20 quartets.

Prior read-only comparison reported zero full-token-array overlap between the two DEV sets, and zero overlap between their union and either pilot’s rehearsal set. Reproduce those checks.

Evaluate both sets identically on all three checkpoints. Report the two panels separately; do not use historical scores as substitutes for common evaluation.

Use the existing structurally assisted binding path for this reference only. Report:

- Answer exactness.
- BOTH_DISTINCT.
- Slot collapse.
- Complete quartets.
- Strict reversal both-correct.
- Queried-row selection conditional on BOTH_DISTINCT.
- Downstream answer correctness conditional on BOTH_DISTINCT.

Do not treat the redundant “selected row belongs to either true source” measure as independent evidence.

Before execution, verify recorded nonsacred provenance and disjointness against known training pools. Do not open the sacred exam.

A new binding set would serve independent confirmation but is unnecessary for this bounded characterization. These existing sets are development references with prior research exposure.

## 10. Statistical inference

Use no graduation-style pass/fail threshold. The primary sampling unit is a complete lexical family. Its eight members are not independent observations.

### Finite eligible universes

Construction samples lexical families uniformly without replacement from these enumerated universes:

| Task | Eligible families N | Sampled families n |
|---|---:|---:|
| Naming | 18 | 8 |
| Possession | 36 | 16 |
| Location | 24 | 16 |

Any sampling inference concerns only these vocabulary combinations under the fixed frames. It does not concern all English, all names, or training-seed variation.

### Secondary confidence intervals

For complete-family success, report x/n and a two-sided 95% exact finite-population interval. The raw count/proportion is headline; this interval is secondary and must never be described as uncertainty over Baby’s general English capability.

For each possible number K of successful families in the eligible universe:

\[
X\sim\operatorname{Hypergeometric}(N,K,n).
\]

Retain K values satisfying both:

\[
P_K(X\ge x)\ge .025,\qquad P_K(X\le x)\ge .025.
\]

Report the minimum and maximum retained K/N.

This avoids pretending that the eight responses within a family are independent. Lexical reuse does not invalidate this finite-universe sampling interpretation, but sharply limits its scope.

For reversal rates and margins, report family distributions and descriptive summaries. Do not create a large grid of small-sample significance tests.

### Paired checkpoint comparisons

For each relation and frame, report:

- Difference in observed complete-family proportions.
- Families successful for both checkpoints, only the first, only the second, or neither.
- Per-family reversal-score differences.
- Median paired difference in family-mean margin.

Use identical families in every comparison. Do not label a checkpoint a statistically established general winner, infer a protection effect, or select a training parent automatically.

Do not use 1/256 as a chance complete-family rate. It assumes independent item guesses and does not describe relevant shortcut behavior.

## 11. Predefined interpretation

“Stronger” means the reported family/reversal profile and margins, with their counts and uncertainty—not an undisclosed threshold. Any quoted finite-universe uncertainty remains limited as specified in Section 10.

| Observed pattern | Supports | Does not support | Most useful next question |
|---|---|---|---|
| Graduate weak; both pilots stronger | Increased tested context sensitivity accompanies the language-trained checkpoint histories | English updates alone caused it; broad English competence | How robust is it across the reserved frames and additional combinations? |
| Pilot 0 stronger than Pilot 1 | Pilot 0 performs better on these matched English tasks | Freezing caused Pilot 1’s weakness | Is the difference explained by learning progress or by the protection procedure under matched training conditions? |
| Pilot 1 stronger than Pilot 0 | Better contextual discrimination can coexist with worse TinyStories perplexity | Protection caused improvement; synthetic binding transferred | Does that advantage replicate with matched rehearsal and controlled training histories? |
| Both pilots strong on cloze, weak on QA | Context-sensitive continuation with additional question-format difficulty | Cloze success is invalid; conversational competence | Which question-format requirement limits accessible answers? |
| Perplexity improves, reversals remain weak | Better distribution prediction without reliable tested counterfactual selection | Pure memorization; absence of all semantic learning | Do errors follow priors, query wording, or association selection? |
| All three weak | Failure on this battery | Architectural impossibility, insufficient parameter count, or inability to acquire binding | Are the frames accessible, and do margins show contextual modulation without correct selection? |
| Graduate unexpectedly strong | Some tested ordinary-path behavior predates the pilots | Pilot-acquired capability or broad language competence | Verify path isolation and scoring, then examine reserved-frame performance and a later independent replication |
| Naming weak, relational cloze stronger | Relational success despite a weaker naming diagnostic | Contradiction or grounds to discard cloze results | Does naming wording or corpus familiarity explain the ordering? |
| Naming strong, relational cloze weak | A simpler identity dependency succeeds while the tested relational constructions do not | Universal binding or a known architectural barrier | Which added demand—relation wording, query direction, or role selection—is responsible? |

Additional rules:

- Positive margin shifts without correct reversals demonstrate contextual modulation, not successful contextual selection.
- Success confined to one presentation order demonstrates order-sensitive behavior.
- Core success with reserve failure supports the core construction; reserve results limit demonstrated robustness.
- Separate possession and location interpretations because their query directions differ.
- No failure changes the previously established synthetic graduation criteria.

T10/T11 failures and this protocol cannot establish mathematical incapacity of the backbone.

## 12. Mechanistic follow-up

No mechanistic analysis is scheduled in this protocol.

The present decision concerns whether behavior exists and where it fails. Logit-lens or attention inspection would not presently discriminate the main alternatives more effectively than the matched behavioral results.

If later work asks which representation causally mediates a demonstrated reversal, a separately designed counterfactual residual-patching study could become useful. A predicted answer change would support causal influence of the patched site under that intervention. A null result would not establish absence of contextual information: the site could be wrong, information distributed, or the intervention ineffective.

Readable intermediate logits or recognizable attention patterns alone remain correlational. Their absence—including absence of an apparent induction head—does not establish impossibility.

## 13. Final construction and frozen artifact specification

### Exact lexical pair lists

Names:

`Ben/Sam, Ben/Tim, Ben/Tom, Sam/Tim, Sam/Tom, Tim/Tom`

Animals:

`bear/cat, bear/fish, cat/fish`

Objects:

`bag/book, bag/hat, ball/car, ball/toy, book/hat, car/toy`

Locations:

`bed/room, cave/shop, cave/yard, shop/yard`

These lists satisfy the approved eligibility rules. Do not substitute additional words during execution.

### Deterministic family selection

For each task independently:

1. Form the Cartesian product of its ordered pair lists:
   - Naming: animal pair × name pair.
   - Possession: name pair × object pair.
   - Location: name pair × location pair.
2. Sort the resulting tuples lexicographically.
3. Initialize a fresh Python 3.12 `random.Random(8380)`.
4. Sample without replacement: 8 naming families, 16 possession families, 16 location families.
5. Sort the sampled tuples lexicographically.
6. Assign sequential family IDs within task.
7. Enumerate a, q, o in lexicographic order over {0,1}³.
8. Generate all specified frames for the same sampled families.

No seed search or resampling after model inspection is permitted. Materialize every selected family identity and every exact item; the frozen exam must not exist only as this algorithm.

### Exact prompt serialization

`first fact + one ASCII space + second fact + LF newline + query prefix`

Each fact ends with a period. The query prefix has no trailing whitespace. Candidate strings supply the leading space and final period.

| Task/frame | Fact for entity Ei | Query |
|---|---|---|
| Naming | `The {animal} is called {name}.` | `The name of the {queried_animal} is` |
| Possession core | `{person} has the {object}.` | `The person with the {queried_object} is` |
| Possession reserve | Same | `The {queried_object} is with` |
| Possession QA | Same | `Who has the {queried_object}?` |
| Location core | `{person} is in the {place}.` | `{queried_person} is in the` |
| Location reserve | Same | `{queried_person} can be found in the` |
| Location QA | Same | `Where is {queried_person}?` |

### Final counts

| Stratum | Families | Items |
|---|---:|---:|
| Naming diagnostic | 8 | 64 |
| Possession primary cloze | 16 | 128 |
| Location primary cloze | 16 | 128 |
| Possession reserved frame | 16 | 128 |
| Location reserved frame | 16 | 128 |
| Possession QA | 16 | 128 |
| Location QA | 16 | 128 |
| Contextual total per checkpoint | 40 distinct lexical families | 832 |
| Query-only prior comparisons | — | 208 logical comparisons |

There are 1,040 English candidate-pair comparisons per checkpoint plus the separate 160-document synthetic reference. No raw-generation panel is included.

### Pre-execution validation

Require all of the following before checkpoint scoring:

- Exact tokenizer, corpus, and checkpoint hash matches.
- Unique contextual prompts and stable family IDs.
- Eight complete members per family/frame.
- Four matched reversal pairs per family/frame.
- Correct counts for every balancing factor: assignment, query, fact order, correct candidate, mention order, and candidate roles.
- Verified answer keys from the assignment equations.
- Candidate token IDs, token-boundary and round-trip assertions for every prompt/candidate pair.
- Equal three-token completed candidates.
- Maximum completed input length including BOS and answer; no input exceeds 256 tokens.
- Corpus-overlap results reproduced under the specified normalization.
- No intersection with training material claimed absent unless mechanically checked.
- Both nonsacred binding-reference disjointness/provenance checks.
- English evaluator path isolated from specialized retrieval.
- Identical runtime/scoring configuration across checkpoints.

A failed validation is an implementation or material-integrity failure. Stop and report it; do not silently drop items, replace vocabulary, or change scoring.

Implementation clarification: if an obvious mechanical implementation error has exactly one specification-preserving fix, correct it, rerun affected validation, and document the correction. If implementation reveals a scientific ambiguity, multiple plausible fixes, a provenance problem, a failed material-integrity check, or anything requiring a protocol change, stop and report instead of choosing a new design.

### Pre-execution artifacts

| Artifact | Required contents |
|---|---|
| PROTOCOL.md | Approved protocol version, endpoint hierarchy, interpretation rules |
| MANIFEST.json | File hashes, checkpoint registry, tokenizer/corpus hashes, seed, runtime settings and versions |
| LEXICON.json | Word counts, document counts, bare/leading-space/terminated token IDs, eligibility |
| FAMILIES.json | Eligible universes, selected families, canonical ordering |
| ITEMS.jsonl | Exact prompts, candidates, token IDs, task/frame, family, assignment/query/order, correct index |
| PRIORS.jsonl | Query-only prompts, candidates, links to contextual families |
| PREFLIGHT.json | Balancing, tokenization, overlap, length, disjointness, and path-isolation checks |

Hash actual UTF-8 file bytes using LF line endings and no BOM. Avoid ambiguity between in-memory text hashes and Windows-written bytes. Report exact SHA-256 for every frozen artifact, including the manifest’s actual bytes in the external receipt rather than requiring an impossible self-referential hash.

### Later execution outputs

- `RAW_SCORES.jsonl`: checkpoint hash, item ID, per-token log-probabilities, candidate sequence scores, word-only scores, margins, ties.
- `FAMILY_RESULTS.csv`: item/reversal/family outcomes and family margin summaries.
- `BINDING_REFERENCE_RESULTS.json`: results separately for both nonsacred DEV sets.
- `REPORT.md`: all predefined strata, exact counts, secondary complete-family finite-universe intervals, paired checkpoint comparisons, prior/shortcut diagnostics, and interpretation; headline raw counts/proportions, reversal profiles, and margins.
- Manifest recording output hashes and checkpoint hashes after evaluation.

Publish every predefined result, including failures. No post-output vocabulary changes, omitted families, calibration changes, new gates, or automatic choice of a future training parent.

No English battery checkpoint evaluation or behavioral scores are authorized during the freezing turn. The sacred graduation exam must not be opened or evaluated.
