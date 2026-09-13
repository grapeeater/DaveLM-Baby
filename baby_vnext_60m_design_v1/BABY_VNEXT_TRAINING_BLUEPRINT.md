# Baby vNext training blueprint

This is an ordered blueprint, not authorization to train. Every phase needs its own prospective protocol, hashes, restart policy, and stop rules before optimizer creation.

## Initialization

Start from scratch using the frozen architecture and tokenizer. Record the initialization seed, deterministic runtime settings, full initialized checkpoint hash, model-state digest, and config/tokenizer hashes. Do not morph Research Baby weights. Preserve Research Baby as the comparison lineage.

## Phase 1 — language acquisition

Train the full base model on the existing authorized TinyStories pipeline with causal next-token loss. Keep the binding adapter inactive during pure language batches. Use aligned DEV CE/perplexity and fixed nonsacred generation diagnostics. Do not access factual transfer, FINAL, or sacred panels.

The Pilot1 source contains 9,000 documents and 3,576,861 encoded stream tokens. Pilot1 used 900 language updates x 64 windows x 256 predictions = 14,745,600 supervised next-token positions. That was about 1.36 presented positions per 10.84M trained parameter. Replaying it unchanged for vNext would be only 0.24 positions per 61.52M parameter and is an obvious undertraining risk.

The bounded initial recommendation is **6,000 language updates at effective batch 64 and context 256**, or 98,304,000 supervised positions. That is about 1.60 presented positions per vNext parameter and roughly 27.5 passes over the current encoded stream. It is a minimum capacity-test budget, not a claim of compute-optimal pretraining. Modern scaling heuristics would imply much more data; the present corpus may overfit before that point.

Freeze evaluation points and checkpoint selection before training. A sensible prospective schedule is fixed-step evaluation every 500 language updates, with no same-run hyperparameter changes. The exact optimizer and LR schedule require separate approval because directly copying Pilot1's 3e-4 constant LR to a 5.7x larger model is not mechanically guaranteed to be appropriate.

## Phase 2 — binding acquisition

Use the established nonsacred T13 rehearsal construction through `legacy_t13_layout`. Train the two-slot adapter and whatever base scope is prospectively frozen. Preserve the answer CE plus hard-min permutation-invariant localization objective and report answer exactness, BOTH_DISTINCT, collapse, reversal pairs, and complete quartets on both nonsacred pools independently.

Language regression must be evaluated alongside binding. A binding graduate is not accepted if it destroys the Phase-1 language baseline.

## Phase 3 — language/binding consolidation

If binding acquisition passes, run an interleaved consolidation phase so language and binding coexist. Pilot1's 9:1 cadence is evidence for the cadence, but the old "freeze blocks 0-3" rule does not map automatically to 12 layers. Freeze the vNext scope prospectively after a zero-update gradient and parameter audit. Do not infer a six-block freeze simply by proportion.

## Phase 4 — narrow factual acquisition

Only after language and both binding pools pass, run the existing controlled TRAIN16 acquisition lesson using the low-dose factual-margin recipe as the first comparison. Preserve full causal alignment, exact answer+EOS scoring, matched reversals, complete families, language retention, D3, and both binding pools. The treatment begins from the accepted vNext language/binding parent.

## Phase 5 — widening frontier

Open DEV_SURFACE and DEV_ORDER only under a preregistered factual protocol. Compare the same coexistence dimensions that closed the 10M series: exact factual generation, surface generalization, source/order behavior, D3, language, and binding. Historical locked transfer and FINAL remain closed until their existing authorization rules are met.

## Fan Diesel execution strategy

Float32 weights, gradients, and Adam moments occupy about 939 MiB. A simultaneous no-grad teacher adds about 235 MiB before activations and framework workspace. This leaves substantial nominal room in 16GB, but activations dominate at batch 64.

Start the future hardware preflight at language microbatch 8 with eight-step accumulation to effective batch 64, then test microbatch 16. Keep full-batch loss normalization mathematically equivalent. Do not promise binding batch 32 until the real 193-token binding path is measured; use accumulation only if its objective reduction is proven equivalent. Activation checkpointing is optional and should be enabled only if measured memory requires it. Keep float32 as the baseline; mixed precision is deferred until independently validated on ROCm.

For parent-KL factual stages, avoid two trainable activation graphs. Run the parent under `no_grad`; if memory still binds, precompute and hash teacher logits for the frozen retention positions or evaluate the teacher sequentially. That choice must be frozen as training infrastructure, not changed after results.

