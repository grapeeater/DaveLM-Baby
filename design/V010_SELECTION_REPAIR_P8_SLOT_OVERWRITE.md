# Selection repair P8: local unpaired-key slot scorer

Protocol id: `V010_SELECTION_REPAIR_P8_SLOT_OVERWRITE`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by P7 **NULL**. Freeze-backbone held induction. Bilinear
`Q(h0[gen])·K(h0[t])` did not find the query: gen is a filler ~50 tokens
later. The query occurrence is the **unpaired key** (not followed by its
value). P2 not launched. No TEST.

## 1. Hypothesis

**H1.** A local scorer `s_t = Linear([h0[t]; h0[t+1]])` with causal softmax
at `gen` will put mass on the unpaired query key. Identity overwrite with
that mass implements D3b replace after gate-on and raises long-gap gold.

Matched control: same module, **λ_ptr = λ_gate = 0**, Baby frozen.

## 2. Why this class

Not P7 λ. Not bilinear-from-gen. P7 showed the pointer must see the *key's
local continuation*, which Q(gen) never observes.

## 3. Identity

Same parent, S2, diet, freeze-Baby, overwrite LR **1e-3**, λ schedule,
SUCCESS/REGRESSION/futility bars as P7, except:

| item | value |
|---|---|
| train seed | `220001` (replicate `220002` generated, not launched unless SUCCESS) |
| data seed | `220100` |
| bootstrap | `220300` |
| pointer | local `[h0[t]; h0[t+1]]` scorer, causal softmax |
| denied | panels, S1, S2, D1b, P1/P3/P4/P5/P6/P7 exact inputs |

Step-0 long-gap hits must stay within 8 of 74.

## 4. Frozen

P7 decision rules reused unchanged. Not to be changed after results are seen.
