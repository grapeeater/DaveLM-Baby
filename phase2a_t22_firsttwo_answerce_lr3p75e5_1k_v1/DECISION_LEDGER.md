# T22 DECISION — first two answer tokens through-base

T21 held language 3/3 and representation 3/3. Exact 27/29/26 (T20 9/10/13, T18 36/36/40). T21X BOS 79/59/72. Leftover is T18X diverge (Wes→Walt) plus leftover other_then_eos (Ava→York). T19 closed suffix-weight 3.0. T18 already did full-span through-base.

T22 = T17 two-phase + uniform mean `λ_ans=1.0`, first TWO answer tokens CE through the base, tail (rest of name + period + EOS) stop-grad. Phase1G parent. Seeds 810001–810003. Same gates. No T19 suffix weight. No retokenize.

Support: exact rises toward T18 36–40 and/or diverge falls, without language fail or representation <2/3.
Refute: exact stays ~26–29 (T18’s +9 is not the second token), or representation drops to 1/3.

Do not parent T14–T21 OUTPUT_FAIL. Do not copy Qwen/Smol. Do not open TEST unless gates authorize it.

## Pause (user order; do not resume until CONTINUE)
Physically quiescent 2026-09-13 after U550 `rolling_restart` persist (Phase A). Watchdog 23396 and 810001 runner stopped. 810002/810003 never started. No T23. See `T22_PAUSE.md`.

## Terminal (2026-09-13T06:14:55Z)
User CONTINUE. Resumed from U550 SHA `ff1000a8…` unchanged. `STUDY_COMPLETE T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL`. Language 3/3. Representation **2/3**. Exact **21/31/26** — no T18 recovery, no lift vs T21 27/29/26. Closed: first-two through-base recovers T18 exact. Do not parent 810001–810003. Next: T22X.

