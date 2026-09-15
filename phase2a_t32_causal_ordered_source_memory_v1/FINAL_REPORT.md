# T32 TERMINAL REPORT

## Classification

**T32_EXPLICIT_ORDERED_SOURCE_MEMORY_INSUFFICIENT**. The first two required
seeds completed the frozen 1000-update schedule with finite metrics and intact
language/binding scope. The predeclared 2-of-3 rule therefore made seed 3
NOT_REQUIRED after two independent scientific failures.

## Results

| seed | exact+EOS | shared-prefix exact | unique-prefix exact | first-token-correct-then-diverge | post-fork completion | pointer | native FC | DEV CE | memory gate L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 890001 | 35/128 | 0/64 | 35/64 | 41 | 0/25 | 112/128 | 79/128 | 1.254763 | 0.05395 |
| 890002 | 35/128 | 2/64 | 33/64 | 35 | 2/17 | 110/128 | 82/128 | 1.254309 | 0.05639 |
| 890003 | NOT_REQUIRED | NOT_REQUIRED | NOT_REQUIRED | NOT_REQUIRED | NOT_REQUIRED | NOT_REQUIRED | NOT_REQUIRED | NOT_REQUIRED | NOT_REQUIRED |

The memory route was mechanically active, but the terminal memory ON/OFF
ablation was effectively unchanged: seed 890001 exact 35/128 in both modes,
seed 890002 exact 35/128 in both modes; language CE was identical in each
pair. Pointer and native metrics were also unchanged except a single
first-token-divergence count difference in seed 890001. Thus activation did
not establish causal native use.

## Integrity

Parent SHA256: `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.
Tokenizer v0_7 SHA256:
`e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.
The T32 preflight proved exact zero-gate parent neutrality (max logit
difference 0.0), nonzero pre-gate signal, and nonzero memory-gate gradient.
Both runs kept binding/localizer tensors intact and never loaded T3 TEST,
FINAL, sacred, or any locked historical panel.

## Interpretation

T32 did not break the historical 35–40 exact ceiling and did not improve
shared-prefix native generation. The result is a valid negative test of this
final-block ordered K/V memory interface, not evidence that all source-memory
architectures fail. Per the frozen failure branch, no T32b or T33 is run.

The frozen descriptive telemetry pass measured mean answer-position memory
attention mass of 1.0 per head (10.0 summed across ten heads) and mean
correct-source pointer weights of 0.661 (890001) and 0.669 (890002). The
branch was populated and selected source weights were nontrivial, yet
neutralizing it did not change native behavior.

**NEXT_OWNER_PLANNED_ACTION = TARGETED_EXTERNAL_MODEL_ARCHAEOLOGY_EXACTLY_3_MODELS**

T3 TEST remains `SEALED_UNOPENED`.
