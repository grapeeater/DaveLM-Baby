# PHASE 2A T4 BINDING SIDECAR — FINAL REPORT

Status: **TERMINAL — study classification `T4_FAIL_NO_ROUTING` (3/3).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
Train only the 984,321-param OrthoLocalizer + retrieval sidecar with `BindingLayout` on the T3 disjoint train/DEV. Parent = untouched Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Base LM frozen. 500 QA updates, seeds 630001–630003.

## U500 results

| seed | loc | native FC | exact | loc rev | nat rev | loc families | DEV CE | base intact | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 630001 | 64/128 | 62 | 0 | 6/64 | 0/64 | 0/16 | 1.204012 | true | `356a3404…903eaf55` |
| 630002 | 64/128 | 62 | 0 | 7/64 | 1/64 | 0/16 | 1.204012 | true | `bedb8b35…65ba8b00` |
| 630003 | 64/128 | 64 | 0 | 6/64 | 2/64 | 0/16 | 1.204012 | true | `ec5d7366…dd67597` |

Language gate held exactly (U0 reproduction). Binding weights **did change** vs parent (max |Δ| ≈ 0.013; three distinct checkpoint hashes; `same_bind=false`, `same_base=true`). This is not a no-update infrastructure failure. Localization/native stayed at chance for 500 updates.

## Frozen-gate adjudication
Localization 96/128, loc reversals 48/64, complete families 8/16, native FC 96/128, native reversals 48/64: **all fail on all seeds**. Language preservation: **pass**. Study class **`T4_FAIL_NO_ROUTING`**. No T4 TEST panel existed; T3 TEST remains sealed.

## Interpretation vs T3
T3 (base trainable, binding frozen) moved pointer retrieval to 97–102/128 but failed reversals/exact and crossed CE 1.30. T4 (binding trainable, base frozen) preserved language and did not create routing from frozen Phase1G mention states under this name-span layout. Binding sidecar-only is not sufficient; base-only pointer is not sufficient for counterfactual completeness.

## Next
T5: Phase1G parent, freeze transformer blocks 0–3 (Phase1G/SF8 English scope), train remaining base + binding, 9:1 language rehearsal, T3 pointer + T2 margin + T4 localization, hard language stop DEV CE > 1.30. Same T3 train/DEV; T3 TEST unopened. Bundle `phase2a_t5_latebase_binding_v1`.
