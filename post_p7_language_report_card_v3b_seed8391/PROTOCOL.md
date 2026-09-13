# Post-P7 English context report card v3

Prospective, read-only battery. Seed 8391. No checkpoint is loaded during construction.
The v1 battery remains defective and excluded; v2 remains historical and is not reused.

Each matched semantic item has two complete factual statements, one queried predicate,
one queried object, one correct agent, and the same candidate identities/order in three
formats. Only the response cue changes. A predicate is never converted to another
semantic relation: found remains found, carried remains carried.

Primary matched battery: 8 families, 8 semantic assignments/query/order combinations
per family, in continuation, declarative-cloze, and explicit-QA formats (192 items).
Four families are near-distribution, two counterfactual-focused, and two conservative
surface-form families. A separate four-family distractor diagnostic has 64 items and
is excluded from the primary 192. Sixteen naturalistic prompts are generation-only.

Controlled scoring uses ordinary base-model causal-LM conditional sequence likelihood.
Candidates are equal-token-length leading-space names with period. For prompt P and
candidate C, LL(C|P)=sum_j log p(C_j|P,C_<j); margin is LL(correct)-LL(incorrect).
Positive margin is correct; zero is a tie and incorrect. No generation is used for
controlled scoring, no prior subtraction, and no length normalization.

Naturalistic generation is a separate descriptive diagnostic. Human semantic fields
are HUMAN_REVIEW_REQUIRED unless frozen before inference and cannot affect ranking.
All construction, parsing, tokenization, overlap, uniqueness, balancing, and provenance
checks must pass before freezing. Sacred binding material is never accessed.
