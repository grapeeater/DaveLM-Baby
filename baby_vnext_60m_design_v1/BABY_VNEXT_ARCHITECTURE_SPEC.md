# Baby vNext 60M architecture specification

Status: **frozen candidate; no training authorized or performed**

## Decision

Baby vNext is a **cleaner capacity successor**. It preserves the vocabulary, context window, decoder-only pre-norm Transformer semantics, activation, learned positions, dropout, untied readout, and two-slot binding principle. The principal scientific change is model capacity and its allocation.

The selected candidate is 12 blocks at width 640. This doubles residual width and adds four layers. Ten heads preserve the old head count while increasing head dimension from 32 to 64. The MLP remains four times the residual width. This gives more room both within a token state and across sequential computation without adopting a new model family.

## Frozen candidate

| Property | Candidate |
|---|---:|
| Trained parameters | 61,520,385 |
| Base-model parameters | 60,536,064 |
| Binding/structural parameters | 984,321 |
| Blocks | 12 |
| Residual width | 640 |
| Attention heads | 10 |
| Head dimension | 64 |
| MLP width | 2,560 |
| Vocabulary | 1,024 |
| Context | 256 |
| Activation | ReLU |
| Norm | explicit LayerNorm, pre-norm, epsilon 1e-5 |
| Position | learned absolute |
| Embedding/readout | untied |
| Dropout | 0.05 embedding / attention / residual |
| Retrieval width | 128 |

The configuration hash is recorded in `VALIDATION_RESULTS.json` and the sealed manifest.

## Exact parameter accounting

| Component | Parameters |
|---|---:|
| Token embedding | 655,360 |
| Position embedding | 163,840 |
| Attention per block | 1,639,040 |
| MLP per block | 3,280,000 |
| Norms per block | 2,560 |
| One block | 4,921,600 |
| All 12 blocks | 59,059,200 |
| Final norm | 1,280 |
| Untied language head | 656,384 |
| **Base total** | **60,536,064** |
| Orthogonal localizer | 1,281 |
| Retrieval projections | 983,040 |
| **Binding total** | **984,321** |
| **Trained total** | **61,520,385** |

The physical module count and independent analytical count agree exactly.

## Candidate comparison

| Candidate | Shape | Base | Binding | Total | Decision |
|---|---|---:|---:|---:|---|
| Balanced | 12 x 640, 10 heads, MLP 2560 | 60,536,064 | 984,321 | 61,520,385 | **Selected** |
| Deeper/narrower | 16 x 560, 10 heads, MLP 2240 | 61,593,184 | 753,761 | 62,346,945 | Rejected: more serial cost and less state width for the same budget |
| Wider/shallower | 10 x 704, 11 heads, MLP 2816 | 61,168,768 | 1,172,865 | 62,341,633 | Rejected: less added depth for relation/source composition |

The balanced candidate is the least opinionated allocation: it adds both representational width and computation depth while keeping head count, expansion ratio, and all task-facing dimensions stable.

## Architecture decisions beyond size

| Decision | Classification | Rationale |
|---|---|---|
| Configuration-driven dimensions | REQUIRED | The old implementation scattered shape assumptions through model and experiment code. |
| Explicit binding layout interface | STRONGLY JUSTIFIED | It removes the fixed row/value offset from the model while preserving the proven two-slot mechanism. |
| Fused QKV plus PyTorch SDPA | STRONGLY JUSTIFIED | Same attention parameterization/math; much lower Python overhead and a practical path on the AMD stack. Reference attention remains available for equivalence tests. |
| Existing tokenizer | REQUIRED for this capacity test | Preserves token-level evaluators and isolates architecture/capacity. |
| Learned absolute positions and context 256 | KEEP | No evidence makes a positional redesign necessary for the measured frontier. |
| RoPE, RMSNorm, gated MLP, tied head, longer context | DEFER | Each is a separate architectural variable. |
| Larger/retrained tokenizer | DEFER | Multi-token name collisions are real, but changing tokenizer with capacity would confound this test and invalidate frozen token identities. |

## Binding migration

The general mechanism is preserved: query state localizes two source slots, retrieves the corresponding value representation, and injects it into the answer state before the language head.

The vNext core accepts explicit query, answer, key, value, and candidate-validity tensors. It no longer assumes `ROW_VALUE_OFFSET=4`, a fixed document length, or that all positions before the query are eligible. The historical offset-four geometry lives only in `legacy_t13_layout`, an evaluator/data adapter. The orthogonal two-slot localizer remains because that is the validated capability. Generalizing to an arbitrary number of slots would change the mechanism and is deferred.

This is option **B: generalize the coupling while preserving the functional principle**.

## Tokenizer decision

The existing 1,024-token byte-level BPE is preserved at SHA-256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`. Input embeddings and output weights remain untied. The implementation verifies distinct storage.

This leaves the known multi-token-name issue in place intentionally. A later tokenizer experiment may be valuable, but it must be a separately controlled generation change.

## Initialization and checkpoint strategy

vNext should train **from scratch**. Width, MLP size, head geometry, depth, base embedding shape, readout shape, and binding projections all differ. Weight morphism would add many arbitrary choices; partial transplantation would preserve only fragments; distillation would add a training objective. None is needed to test the capacity hypothesis cleanly.

Initialization uses deterministic PyTorch module defaults under an explicitly recorded construction seed. Checkpoints store the complete config and its hash, tokenizer hash, schema version, initialization seed, update count, and model-state digest. The validation checkpoint is an initialized, zero-update artifact.

## Capacity-test interpretation

A successful vNext does not prove parameter count was the sole 10M limitation. It shows that this larger allocation, under its frozen training protocol, moves the measured coexistence frontier. A failure does not prove scale is irrelevant unless the larger model received an adequate training budget and passed its language/binding acquisition gates.

