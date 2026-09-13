# Continuation experiment log

## Parent and reason

Parent: Language Pilot 0 (`769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5`). The frozen English battery found chance-level counterfactual selection in all three checkpoints, while Pilot 0 had the lowest reported TinyStories perplexity. This continuation tests whether more ordinary causal-LM exposure alone is sufficient for a coherent response.

## Treatment

One isolated 1,000-update language-only continuation on the existing TinyStories TRAIN stream, seed 8380, batch 64, context 256, AdamW learning rate 3e-4, weight decay 0.05, gradient clipping 2.0. All specialized T13 parameters were frozen. No frozen checkpoint or English battery artifact was modified.

## Result

The continuation evaluator’s DEV perplexity changed from 2.2217 to 1.7011. Fixed greedy probes became more fluent locally, but remained repetitive and semantically unreliable: “The dog is a big boy” changes the subject category, and multiple prompts repeat park/toy clauses. No output satisfies the predeclared standard of a complete context-appropriate response. This is a failed coherence attempt, despite lower perplexity.

## Next decision

Do not treat the perplexity gain as success. The next intervention should add a small, explicitly held-out controlled sentence/response set with anti-repetition and subject/reference consistency checks, while retaining a TinyStories perplexity diagnostic and the nonsacred binding references. The frozen English characterization remains the baseline and is not reopened.
