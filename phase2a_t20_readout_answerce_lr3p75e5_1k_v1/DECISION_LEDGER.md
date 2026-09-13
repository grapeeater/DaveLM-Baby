# T20 DECISION — readout-only answer CE

T19 held language 3/3, missed representation 2/3 (1/3), and did not lift exact (39/35/34 vs T18 36/36/40). T19X leftover is still Sky/Walt/York after a correct first token. Answer CE through late-base is the suspected representation tax. `language_head` is untied.

T20 = T17 two-phase + T18 uniform λ_ans=1.0, but answer CE applies `hidden.detach()` before `language_head`. Pointer still trains 4–11 / 4–7. Phase1G parent. Seeds 790001–790003. Same gates. No suffix weight. No retokenize.

Support: representation returns to ≥2/3 and/or exact rises without language fail.
Refute: representation still 1/3, or exact stays ~35–40 with representation 2/3 (then suffix completion is not a head-only problem).

Do not parent T14–T19 OUTPUT_FAIL. Do not copy Qwen/Smol. Do not open TEST unless gates authorize it.

## Terminal (2026-09-12T21:11:25Z)
Watchdog 23460 `STUDY_COMPLETE T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL`. Language 3/3. Representation **2/3**. Exact **9/10/13** (collapsed vs T18 36/36/40). ACQ16 fail 3/3. Support: representation recovered from T19’s 1/3. Refute: suffix completion is not head-only; through-base CE was the T18 exact channel. Do not relaunch T20. Do not parent 790001–790003. Next: T20X fail-mode.
