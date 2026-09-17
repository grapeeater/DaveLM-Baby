# Selection repair P11: gen-only aux overwrite

Protocol id: `V010_SELECTION_REPAIR_P11_GEN_ONLY`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by P10 **REGRESSION**. P10 confirmed CE-through-overwrite pinned
P9's gate, and that unmasking the gate globally rewrites the residual
stream (cosine 0.89, CE 4.76, hits 74, induction collapsed). This
protocol keeps P10's aux-only update and hard-masks the write to
`len(input)-1`. P2 not launched. No TEST. Do not unfreeze Baby. Do not
raise λ.

## 1. Hypothesis

**H1.** A gen-index-only identity overwrite, trained aux-only, will open
gate[gen] toward 1 without scrambling non-gen residuals, raise L0 cosine
between post-overwrite gen and the untouched query residual, and lift
long-gap gold vs matched λ=0, while holding language CE and
primitive_induction.

**H0.** Spatial restriction does not produce a selection lift (pointer
mass still too diffuse, or L0 query identity is still insufficient).

## 2. Why this class

Not P10 λ. Not unfreezing. Not a new pointer. P10 isolated the missing
locality of the D3b write. P9 showed the same pointer can move hits when
the write is partial and the backbone is intact.

## 3. Identity

Same as P10 except:

| item | value |
|---|---|
| train seed | `250001` (replicate `250002` generated, not launched unless SUCCESS) |
| data seed | `250100` |
| bootstrap | `250300` |
| write mask | **gen index only** (`LocalSlotOverwrite(gen_only=True)`) |
| pack hook | sets `overwrite.gen_index` from `len(input)-1`; consumed each forward |
| denied | panels, S1, S2, D1b, P1/P3–P10 exact inputs |

Unchanged: Baby frozen, overwrite AdamW 1e-3, CE↛overwrite, λ_ptr=1,
λ_gate=1 after 200, gate bias −4, diet 8/3/5, language p=0.20,
MAX_UPDATES=800, 200 futility mass < 0.05, 400 futility excess gain <
+0.02 and cosine gain < +0.05 and mass < 0.50.

SUCCESS / REGRESSION / MECHANISM bars unchanged from P9/P10. Step-0 hits
within 8 of 74.

## 4. Frozen gates

SUCCESS if all hold at the adjudicated step:

- treatment long-gap excess ≥ 0.10
- treatment − control excess ≥ 0.07
- bootstrap 95% CI lower bound > 0
- L0 cosine ≥ 0.40 **or** cosine gain ≥ 0.15
- no retention-panel drop > 0.05
- negative controls pass

MECHANISM SUPPORTED, DOSE INSUFFICIENT if retention holds and
(mass ≥ 0.50 or cosine ≥ 0.20 or cosine gain ≥ 0.10) but SUCCESS fails.

REGRESSION if a retention panel or negative control fails, or language
DEV CE hard-stop.

NULL if excess gain < 0.02 and cosine gain < 0.05 and mass < 0.05.

These rules are not to be changed after results are seen.

## 5. Control

Matched schedule, λ_ptr=λ_gate=0, CE detached, gen-only mask. Control
overwrite receives no gradient.

## 6. Hard stops

Language DEV CE > initial + 0.20; nonfinite loss/grad; disk < 10 GiB;
wall clock 7200 s.

## 7. Promotion / TEST

No promotion of U16000. No TEST. Replication seed 250002 only if SUCCESS.
