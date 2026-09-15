# Baby v0.10 curriculum v2R4

Protocol: `BABY_V010_FOUNDATION_V2R4`  
Status: preregistration draft; freeze before launch

## Motivation

v2R3 established that a 20% language-retention mix and the U6000 transition
could preserve the fresh language foundation, but its frozen blocks 0--6 did
not acquire reliable primitive copying. The next iteration changes one causal
factor: optimization scope. The architecture, tokenizer, panels, masks,
retention probability, thresholds, and stage boundaries remain fixed.

## Frozen design

- Start from a fresh random v0.10 initialization; load no v0.9 or v2R3
  checkpoint.
- Train the same 6,000-update Phase1G-style language foundation.
- At U6000, unfreeze every model parameter. Use a layerwise structured
  optimizer: `1.5e-5` for token/position embeddings and blocks 0--6, and
  `3.75e-5` for blocks 7--11, final norm, and language head.
- Keep a 20% language-retention batch probability in every capability stage.
- Keep primitive and short stages answer-span-only masked. Keep full-stage
  answer/separator/EOS masking unchanged.
- Keep the same frozen foundation_v2 panels, audit, model family, tokenizer,
  context length, and capability gates.

## Causal question

Does the v2R3 failure come from freezing the lower contextual computation rather
than from insufficient language retention or an absent architectural substrate?

Support requires the lower-scope run to preserve language while improving
intact primitive copying without comparable broken-context or repetition
success. Failure means the optimization-scope hypothesis is not supported and
does not authorize a threshold change.

## Diagnostic and continuation rule

The first diagnostic is fresh seed `106001` through U8000. Evaluate at U6000,
U6500, U7000, U7500, and U8000. Continue to the full U16000 schedule only if
language retention remains within Gate R and primitive intact panels show a
reproducible positive signal with no obvious shortcut collapse. If supported,
run fresh seed `106002` under this same frozen protocol. No checkpoint is
selected from a diagnostic merely because it has the best metric.

