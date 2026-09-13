# Prospective post-P7 language report card v2

Seed 8391. This new version supersedes v1 for future measurement; it does not repair
or reuse exposed v1 items. V1 controlled-transfer conclusions are withdrawn because
144 answer keys contradicted their prompts. Historical files remain preserved.

## Construction and scientific scope

Sam and Tom, both present in P7 training, are the matched candidate subjects.
Found and carried are the two factual relations. Objects and colors come from P7.
Sample 20 distinct object-pair/color-pair tuples from the sorted-by-construction
Cartesian universe using Python Random(8391). Full details and selected identities
are materialized in scripts, FAMILIES.json and ITEMS.jsonl. Sampling is performed
once before any model behavior. Distinct colors make these fresh factual contexts,
not v1 contexts with a new label or an irrelevant prefix.

Near-distribution: lexical families 0-11, 8 members each (96). This is familiar
vocabulary recombination with a two-fact QA demand, NOT a direct replication of
P7 single-prompt sentence generation. Counterfactual: families 12-19 (64), same
active frame, an independent lexical panel focused on reversal. Every controlled
section is fully counterfactual; there is no claim that these two sections isolate
different mechanisms. Surface: families 0-7 in passive wording and cloze queries
(64); both fact and query wording change, so causes cannot be separated.
Distractor: families 0-7 with a third named entity and irrelevant sentence placed
before OR after the two facts independently of all other factors (128).
Compare these diagnostics to matched near-distribution families 0-7 only.
Naturalistic: 24 separately fixed story starts, generation only, no candidate key.

Each base family crosses assignment x queried relation x intact fact order:
2 x 2 x 2 = 8. Distractor families also cross placement (16 members). Assignment
swaps agents only; object descriptions, predicates and query stay unchanged.
Query change selects the other fact. Order reversal changes only intact sentence
order, never fact meaning. Both candidates occur once in the relevant facts;
Mia is the distractor, never an answer. Correct agent, queried object, queried
action, relevant first/second mention, recency and assignment are balanced within
each family. Colors/object sampling is enumerated, not claimed population-balanced.
The builder resolves answers from rendered text. The independent validator uses
token-position grammar parsing, does not import builder logic, checks every key
and transformation, and rejects every deliberately inverted key.

## Frozen scoring and runtime for a later authorized execution

Use ordinary base causal LM only, eval/inference mode, BOS=2, EOS=3, float32 model,
no autocast or TF32, one sequence at a time without padding/truncation. Save device,
torch/tokenizers versions and model source hashes. Use float64 log_softmax on CPU
over float32 logits. With P=prompt tokens and C=candidate tokens, score
L(C|P)=sum_j log_softmax(logits[BOS,P,C] at position len(P)+j)[C_j], j=0..2.
Thus the first candidate is predicted at the final prompt position. Candidates
are exactly leading-space Sam/Tom plus period; each has three tokens. Do not score
BOS or append/score EOS. Do not length-normalize or subtract priors. Margin is
L(correct)-L(other); >0 correct, ==0 tie and incorrect. Preserve token log scores,
candidate log scores and margins. No generation for controlled items.

Headlines separately per section: complete-family counts/proportions (all 8 or
16 correct), distribution of reversal successes per family, mean within-family
reversal fraction, and margin min/median/mean/max. Reversal pairs fix query, order
and placement and vary assignment: four per ordinary family, eight per distractor
family; success requires both items correct. Diagnostics: item counts/accuracy,
ties, family mean/minimum margin, paired surface/distractor versus matched active
family outcomes. No inferential pass/fail gate or broad-English confidence interval.
Representative controlled examples: every member of lexically first family ID
within each section, chosen before results. Do not cherry-pick.

Naturalistic: greedy argmax, BOS plus exact prompt, maximum 32 new tokens; stop
after EOS or the first generated period/question-mark/exclamation-mark token.
This is a punctuation boundary, not a judgment of grammatical completeness.
Save all generated token IDs including stop token and verbatim decoded continuation
with special tokens omitted; never strip whitespace or repair text. Save stop
reason. Human-review fields, separately for every prompt: grammatical completeness
(complete sentence/fragment/unclear); subject consistency (consistent/conflicting/
not assessable); object consistency (consistent/conflicting/not assessable);
relation/action appropriateness (plausible/conflicting/unclear); repetition failure
(yes/no/unclear, repeated content without progress); contextual contradiction
(yes/no/unclear with quoted evidence); truncation (32-token cap before punctuation
or EOS, mechanical flag). Empty output is recorded explicitly; it is not success.
Unspecified new facts are not automatically contradictions. Semantic judgments
await the user's review; do not invent automatic success labels. Multiple coherent
continuations are possible, intentionally. No grammar-scoring model is used.

## Audits and interpretation

Require zero normalized duplicate or previously exposed exact prompts, zero full
prompt/completion overlap against listed sources, valid token roundtrips/boundaries,
and length within 256 including BOS/answers or generation budget. Decode corpus
JSON text fields before normalization; do not search serialized escapes as prose.
Report atomic fact overlap diagnostically; familiar facts do not demonstrate novel
semantic knowledge. No exact overlap does not establish absence of paraphrase or
semantic contamination. No model output influenced selection. Neither controlled
success nor naturalistic plausibility establishes broad comprehension, conversation,
reasoning, capacity limits or synthetic-to-English transfer. Controlled failures
confound QA format with factual selection and cannot erase P7's original 12/12.
No inference, training, parent selection or sacred binding access during freeze.
