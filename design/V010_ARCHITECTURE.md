# Baby v0.10 architecture

Status: prospective v0.10 foundation design, frozen before the first fresh
training run.

## Decision

Use the validated Baby vNext decoder-only family with a fresh random
initialization:

| component | v0.10 foundation choice |
|---|---:|
| layers | 12 |
| model width | 640 |
| attention heads | 10, head dimension 64 |
| MLP width | 2560 |
| activation | ReLU |
| context | 256 learned absolute positions |
| normalization | explicit pre-LayerNorm |
| attention | fused QKV, causal SDPA with reference fallback |
| dropout | 0.05 embedding, attention, and residual |
| vocabulary | 1024, v0_7 tokenizer for foundation v1 |
| output head | untied linear head with bias |
| binding sidecar | disabled in foundation v1 |

The base transformer is approximately 60.5M parameters. A binding/localizer
sidecar is intentionally not present in the foundation model: v0.9 records show
that its 984,321 tensors were initialized machinery, not a learned binding
competency, and the foundational gates do not require it. A later binding stage
may be proposed only after the foundation gates pass.

## Why this is not a cargo-cult architecture change

T34 moved the v0.9 base model from zero to near-perfect held-out novel copy with
no new module, no new parameters, and no tokenizer change. That is direct
evidence that the basic family contains a trainable copy/induction substrate.
T33's adapter futility and the v0.9 mechanistic studies do not justify replacing
the backbone. v0.10 therefore changes the training distribution and evaluation
before changing the model family.

## Intentional differences from v0.9

1. Fresh random initialization; no v0.9 checkpoint, optimizer state, or trained
   sidecar is loaded.
2. The untrained binding sidecar is omitted from the foundational base model.
3. The first tokenizer is retained for controlled comparison, while lexical
   fragmentation and bare/leading-space variants are audited as first-class
   metrics.
4. The structured objective is a mixture of many generated surface forms,
   variable lengths, key/value orderings, separators, marker vocabularies, and
   distractor layouts. No single T34 template is authoritative.
5. Capability probes and negative controls run during every stage, with frozen
   exit gates defined before training.

## Instrumentation contract

The model exposes ordinary logits and hidden states. The v0.10 evaluator records
target probability, target rank, target logit, target-versus-competitor margin,
intact-versus-broken-context lift, teacher-forced span exact, free-running exact,
first-error position, language CE, and retention. Attention-head causal traces
are optional diagnostics and never substitute for behavioral gates; L7H2 is not
assumed to recur at the same index.
