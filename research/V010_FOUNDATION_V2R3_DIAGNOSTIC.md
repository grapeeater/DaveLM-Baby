# Baby v0.10 foundation v2R3 diagnostic

Status: terminal diagnostic failure; preserved for iteration design.

Protocol: `BABY_V010_FOUNDATION_V2R3`  
Run: fresh seed `105001`, `8000` updates  
Run directory: `runs/diagnostic_v2r3_seed105001_8000/`  
Protected material opened: no  
Parent checkpoint: none

## Preregistered question

Would a 6,000-update fresh language foundation followed by (a) freezing blocks
0--6, (b) training the upper-block scope used in the T34 analysis, and (c)
mixing 20% language-retention batches preserve language while allowing the
primitive identity curriculum to acquire a copy signal?

The protocol was frozen before the run. Gates and panels were not changed after
observing the result.

## Direct result

The retention hypothesis was supported, but the capability hypothesis was not.
Language DEV CE stayed near the pre-structured baseline throughout the primitive
stage. The structured objective produced a weak and unstable rank/logit signal,
but not reliable contextual reproduction. The run therefore did not support
continuation as a successful v2R3 foundation and is not eligible for
graduation, selection, or checkpoint reuse as a capability result.

| update | stage | language DEV CE | primitive induction top-1 | primitive induction median rank | primitive keyed top-1 | short keyed top-1 | same-surface novel top-1 | held-out surface top-1 | broken-context top-1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6000 | primitive transition | 1.1900 | 0.0000 | 277 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6500 | primitive | 1.2033 | 0.0000 | 135 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 7000 | primitive | 1.2013 | 0.0000 | 256 | 0.0625 | 0.0000 | 0.0000 | 0.0625 | 0.1250 |
| 7500 | primitive | 1.2051 | 0.1875 | 148 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.1250 |
| 8000 | short transition | 1.2064 | 0.0625 | 128 | 0.1875 | 0.0000 | 0.0000 | 0.0000 | 0.1250 |

At the endpoint, teacher-forced exact and free-running exact were zero on the
novel and held-out surface monitors. The context-lift proxy had free-running
difference zero. The endpoint metrics used a 16-example monitoring subset; the
full frozen panels remain the adjudication source for later terminal runs.

## What is directly established

- The v2R3 fresh language foundation was healthy: DEV CE was 1.1900 at U6000
  and 1.2064 at U8000.
- The 20% language-retention mix plus the frozen prefix prevented the immediate
  v2R2 language collapse. Retention batches stayed around 1.0--1.1, and DEV CE
  remained within the Gate R allowance.
- Primitive answer-only training produced a nonzero but unstable rank/logit
  movement. It did not produce robust top-1 reproduction or exact copying.
- Same-surface novel-copy top-1 remained zero at every listed endpoint. The
  held-out-surface result never exceeded 0.0625 and ended at zero.
- Broken-context top-1 became nonzero while intact copy remained weak. This is
  consistent with a generic-token/repetition shortcut or weak unconditioned
  preference, not evidence of causal contextual use.
- No protected TEST, FINAL, or SACRED material was opened. No v0.9 checkpoint
  was loaded. The run used a fresh v0.10 initialization.

## Strong evidence and diagnosis

The v2R3 intervention fixed the specific v2R2 forgetting failure but overfixed
the optimization scope. The upper-block-only scope could preserve the language
substrate, yet it did not provide enough trainable adaptation for the randomly
initialized model to learn the contextual mapping under the new objective.

The structured loss fell from its initial value but remained around 5.3--6.4,
close to a high-entropy identity objective rather than a solved answer-span
objective. Generated sequences frequently repeated a small number of tokens;
the broken-context rows sometimes received higher target scores than intact
rows. These observations make an upper-block bottleneck plus shortcut pressure
more plausible than simple insufficient update count.

This is not evidence that the Baby architecture lacks a copying substrate. T34
already demonstrated the opposite in the v0.9 lineage. It is evidence that the
v2R3 fresh-initialization curriculum did not make the useful substrate
trainable under the frozen-prefix constraint.

## Falsified or unsupported hypotheses

- “20% language retention alone is enough to make the frozen T34-like upper
  scope learn primitive copying.” Unsupported and rejected by this run.
- “A weak rank improvement is sufficient evidence of contextual reproduction.”
  Rejected; exact and held-out behavior remained at zero.
- “Broken-context top-1 movement is evidence that the model has learned the
  intended retrieval relation.” Rejected; broken-context movement occurred
  without intact-copy success and with repetitive outputs.

## Next justified iteration

The next diagnostic should preserve the successful retention mechanism while
testing whether the frozen-prefix bottleneck is causal. It should unfreeze all
blocks after U6000 with a smaller learning rate on blocks 0--6 than on blocks
7--11, retain the 20% language mix, keep the same panels and gates, and use the
same U6000/U8000 diagnostic checkpoints. This is an incremental optimization
scope test, not an architecture replacement and not a threshold change.

The next run must be preregistered before launch. It should be considered
supported only if language retention remains within Gate R and primitive
same-surface plus held-out-surface behavior rises together without broken-context
success or repetition collapse.

## Reproducibility

- Metrics: `runs/diagnostic_v2r3_seed105001_8000/metrics.jsonl`
- Run configuration: `runs/diagnostic_v2r3_seed105001_8000/RUN_CONFIG.json`
- Frozen protocol: `data_specs/V010_FOUNDATION_PROTOCOL_V2R3.json`
- Frozen panel audit: `data/generated/foundation_v2/panels.AUDIT.json`
- Trainer: `src/baby_v010/train_v2r3.py`
