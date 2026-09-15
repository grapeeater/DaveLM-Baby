# Baby v0.10 curriculum v2R2

Protocol: `BABY_V010_FOUNDATION_V2R2`  
Status: frozen before the v2R2 diagnostic

## Motivation

The v1R2 and v2R1 fresh starts both reached low language CE quickly but failed
to learn identity transport. The authoritative Phase1G record shows that the
same architectural family was language-trained for 6,000 updates with an
effective batch of 64, warmup/cosine schedule, and all base parameters before
T34 acquired contextual copy. v2R2 tests whether that missing fresh-start
substrate is causal.

## Stages

| updates | stage | objective |
|---:|---|---|
| 0-5,999 | language foundation | full-vocabulary language CE, effective batch 64, Phase1G warmup/cosine schedule |
| 6,000-7,999 | primitive identity | staged induction/retrieval, answer-span-only mask, structured LR 3.75e-5 |
| 8,000-9,999 | short retrieval | short induction/retrieval, answer-span-only mask, same structured LR |
| 10,000-15,999 | full foundation | full varied retrieval/induction, answer+separator+EOS mask, same structured LR plus 20% language retention |

The first run is a diagnostic through update 8,000. It is not a graduation
run; it asks whether 6,000 fresh language updates produce a measurable
primitive identity signal. If it supports the hypothesis, the same frozen
protocol continues to the full 16,000-update run and is replicated at a second
fresh seed. If it fails, a major architecture replacement is not implied; the
next step must be a mechanistic optimization study.

## Controls

The v2 frozen panels and their audits are reused unchanged. Answer-only masks
apply only to training loss; evaluation still scores full answer, separator,
and EOS behavior. No panel answers are used in training, and all final gates in
`V010_CAPABILITY_GATES.md` remain unchanged.
