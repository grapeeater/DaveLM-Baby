# Baby vNext implementation report

## Scope

A new implementation was created only under `C:\DaveLM-CADAVER\baby_vnext_60m_design_v1`. No Research Baby source, checkpoint, dataset, evaluator, or historical study was edited.

The audit's claimed `C:\DaveLM-v0.9\v0_8\model.py` path does not physically exist. The operative architecture is the frozen `v0_7.model.DaveLM`, also imported by the physical `v0_8_2` wrapper. The v0.8 configuration exists and confirms the 8 x 320 settings. This path correction does not change the audited architecture facts.

## Implementation files

| File | Purpose |
|---|---|
| `baby_vnext/config.py` | Frozen, validated, serializable model and binding configuration |
| `baby_vnext/model.py` | Decoder-only pre-norm model with fused QKV, reference/SDPA attention, and explicit LayerNorm |
| `baby_vnext/binding.py` | Dimension-parameterized orthogonal localizer, retrieval path, explicit geometry, legacy layout adapter |
| `baby_vnext/checkpoint.py` | Versioned config-bound checkpoint schema and exact state digest |
| `baby_vnext/parameter_count.py` | Independent analytical and physical parameter accounting |
| `validate_design.py` | Full zero-training mechanical validation |

## Historical safeguards

The implementation imports no historical training script and writes only in the new directory. Validation recorded SHA-256, size, and modification time for the authoritative model, tokenizer, Pilot1 sources/data/checkpoint, T13 source/checkpoint, and audit before and after testing. Every record was unchanged.

No locked panel, FINAL artifact, or sacred material was opened. No real Baby example was used for backward testing.

## Shape generalization

Model dimensions, head dimension, layer count, MLP width, vocabulary, context, dropout, norm epsilon, output tying, attention backend, slot count, and retrieval width now come from one validated configuration.

The core binding path takes explicit layout tensors. The historical constants `ROW_VALUE_OFFSET=4` and two-row document positions are not embedded in the core forward pass. Two slots remain an explicit configuration invariant because the recovered orthogonal localizer and preservation metrics validate two slots; changing that is outside this capacity test.

## Mechanical validation

| Check | Result |
|---|---|
| Instantiate 61,520,385-parameter candidate | PASS |
| Analytical vs physical counts | PASS, exact |
| Expected tensor shapes | PASS |
| Tiny forward pass | PASS on RX 9060 XT |
| Tiny causal language backward | PASS; every intended base group received gradient |
| Tiny binding backward | PASS; localizer, all retrieval projections, and base received gradient |
| Weights unchanged across backward | PASS; identical model-state SHA-256 |
| Causal masking | PASS; altered suffix changed earlier logits by 0.0 |
| SDPA vs reference attention | PASS; maximum logit difference `8.344650268554688e-07`, tolerance `2e-05` |
| Config save/reload | PASS |
| Initialized checkpoint save/load | PASS; exact model-state digest |
| Tokenizer identity/vocabulary | PASS |
| Input/output weights untied | PASS |
| Historical artifacts unchanged | PASS |
| Optimizer created or stepped | NO |
| Training on real data | NO |

## Runtime

- Python 3.12.14
- PyTorch 2.12.0+rocm7.14.0
- tokenizers 0.23.1
- AMD Radeon RX 9060 XT detected and used for the tiny forward/backward smoke
- deterministic algorithms enabled

PyTorch warned that flash and memory-efficient attention are experimental on this AMD GPU unless its experimental AOTriton switch is enabled. The SDPA call completed and matched the reference implementation. The warning means the eventual training preflight must benchmark actual memory/throughput without silently enabling an experimental kernel.

## Known implementation risks

1. No trained vNext checkpoint exists, so capability is entirely unmeasured.
2. The SDPA smoke proves correctness on a tiny input, not full-batch memory or speed.
3. The preserved tokenizer retains multi-token name collisions.
4. The two-slot localizer remains appropriate for the regression lab but is not a general natural-language memory system.
5. The existing language corpus is small relative to 61.5M parameters and will be heavily replayed under the bounded initial budget.
6. Exact training scopes, accumulation semantics, and curriculum cadence still require a separate prospective training freeze.

