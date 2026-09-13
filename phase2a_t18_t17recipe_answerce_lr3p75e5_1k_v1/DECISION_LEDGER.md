# T18 DECISION — T17 recipe plus answer-span CE

T17 cleared language-safe representation 2/3 with the two-phase U750 scaffold. Native/exact failed: FC 86–90, native rev 24–29, exact 0. T17X showed greedy generation continues Phase1G stories after `Answer:` (e.g. `Tom!"\nAnna and Ben run to the`) even on the 80–81/128 rows that are already pointer-correct and native-forced-choice-correct. QA updates were pointer+margin only; `language_head` has no pointer gradient.

T18 keeps Phase1G parent, fact-clause / 9:1 / 3.75e-5 / Phase A 750 / revert-freeze 8–11 / Phase B 250 / same mix / same gates. Seeds 770001–770003. The only added term is teacher-forced CE on the correct candidate tokens + EOS (`λ_ans=1.0`), answer span only.

Support: native FC ≥96 and/or exact ≥80 on ≥2/3 while language and representation hold.
Refute: language regression ≥2/3, representation collapse ≥2/3, or exact stays 0.

Do not parent T14/T15/T16/T17/730002/760001–760003. Do not reweight templates. Do not open TEST unless gates later authorize it. Do not copy Qwen.

## TERMINAL 2026-09-12T14:38:37Z
Study `T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL`. Language 3/3. Representation 2/3 (770001, 770003). Exact 36/36/40 (from T17’s 0). Native FC 86/81/86, native rev 23/18/22. ACQ16 exact pass 3/3. Answer CE opened a channel; it did not convert DEV native/exact. Do not parent T18 checkpoints. Next: T18X fail-mode vs T17X continuation baseline.
