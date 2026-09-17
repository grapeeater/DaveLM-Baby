# Selection repair P10: aux-only overwrite (CE stop-grad)

Protocol id: `V010_SELECTION_REPAIR_P10_AUX_ONLY`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by P9 **MECHANISM SUPPORTED, DOSE INSUFFICIENT**. P9's slot
overwrite moved mass 0.003→0.568, L0 cosine 0.042→0.228, and long-gap
hits 74→84 vs matched λ=0 (CI [+0.023, +0.073], induction held). Gate[gen]
stalled ~0.29 after λ_gate=1. Language-only steps in P9 were pure CE
through overwrite. This protocol keeps the P9 operator, freeze, λ, and
800-update budget; the sole change is that **CE does not update
overwrite**. P2 not launched. No TEST. Do not unfreeze Baby. Do not raise
λ_ptr or λ_gate.

## 1. Hypothesis

**H1.** P9's gate stall is caused by CE gradients into overwrite (especially
language-only batches, which have no pointer/gate term and want gate→0).
Training overwrite on pointer+gate aux only will let gate[gen] rise toward
1, raise L0 query cosine, and lift long-gap gold vs matched λ=0.

**H0.** Removing CE from overwrite does not raise gate or cosine beyond P9;
the stall is not CE fight.

## 2. Why this class

Not P9 λ. Not unfreezing. Not a new pointer. P9 already showed this
operator moves selection; the remaining causal question is why the write
saturates at a ~30% blend.

## 3. Identity

Same as P9 except:

| item | value |
|---|---|
| train seed | `240001` (replicate `240002` generated, not launched unless SUCCESS) |
| data seed | `240100` |
| bootstrap | `240300` |
| CE → overwrite | **false** (logits from `hidden.detach()`; language steps skip backward) |
| denied | panels, S1, S2, D1b, P1/P3/P4/P5/P6/P7/P8/P9 exact inputs |

Unchanged: LocalSlotOverwrite, Baby frozen, overwrite AdamW 1e-3,
λ_ptr=1, λ_gate=1 after 200, gate bias −4, diet 8/3/5, language p=0.20,
MAX_UPDATES=800, 200 futility mass < 0.05, 400 futility excess gain < +0.02
and cosine gain < +0.05 and mass < 0.50.

SUCCESS / REGRESSION / MECHANISM bars unchanged from P9. Step-0 hits
within 8 of 74.

## 4. Frozen gates

SUCCESS if all hold at the adjudicated step:

- treatment long-gap excess ≥ 0.10
- treatment − control excess ≥ 0.07
- bootstrap 95% CI lower bound > 0
- L0 cosine ≥ 0.40 **or** cosine gain ≥ 0.15
- no retention-panel drop > 0.05 (primitive_induction, primitive_keyed,
  short_keyed, rest_lock)
- negative controls pass (value_absent, broken_context, broken_order)

MECHANISM SUPPORTED, DOSE INSUFFICIENT if retention holds and
(mass ≥ 0.50 or cosine ≥ 0.20 or cosine gain ≥ 0.10) but SUCCESS fails.

REGRESSION if a retention panel or negative control fails.

NULL if excess gain < 0.02 and cosine gain < 0.05 and mass < 0.05.

These rules are not to be changed after results are seen.

## 5. Control

Matched schedule, λ_ptr=λ_gate=0, CE still detached from overwrite. Control
overwrite therefore receives no gradient (aux zero, CE detached). That is
the correct match for "aux-only training vs the same detached graph."

## 6. Hard stops

Language DEV CE > initial + 0.20; nonfinite loss/grad; disk < 10 GiB;
wall clock 7200 s.

## 7. Promotion / TEST

No promotion of U16000. No TEST. Replication seed 240002 only if SUCCESS.
