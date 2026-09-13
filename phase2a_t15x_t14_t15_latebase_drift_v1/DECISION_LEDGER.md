# T15X DECISION — T14 vs T15 late-base drift

T15 held language 3/3 and missed representation 0/3. Qwen D1 suggested late modules can be training scaffolds. T15X tests the Baby-native version of that claim with weight drift only.

Result: T15 8–11 bit-identical to Phase1G. T15 4–7 RMS matches T14. Δ-cosine T15-vs-T14 ≈ T14-vs-T14. Failure is not under-updating 4–7. Next is a two-phase scaffold treatment, not more T15 / not more scalar LR.
