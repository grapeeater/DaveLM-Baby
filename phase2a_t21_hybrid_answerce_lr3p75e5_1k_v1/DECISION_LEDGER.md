# T21 DECISION — hybrid first-token-through-base + suffix readout CE

T20 recovered representation 2/3 and collapsed exact to 9/10/13. T20X first-token BOS 42/34/37 (T18X was 76–84). Dominant leftover `other_then_eos` 81/76/79, not T17X continuation. T19 closed suffix-weight 3.0 through late-base.

T21 = T17 two-phase + uniform mean `λ_ans=1.0`, first answer token CE through the base, suffix+period+EOS CE on `language_head(hidden.detach())`. Pointer still trains 4–11 / 4–7. Phase1G parent. Seeds 800001–800003. Same gates. No T19 suffix weight. No retokenize.

Support: first-token BOS and/or exact rise from T20 without language fail or representation falling below 2/3.
Refute: BOS stays ~35–42, or exact stays ~9–13 with BOS recovered (finishing still needs through-base suffix at λ=1.0), or representation drops to 1/3 (even first-token CE taxes like T19).

Do not parent T14–T20 OUTPUT_FAIL. Do not copy Qwen/Smol. Do not open TEST unless gates authorize it.

## Terminal (2026-09-13T00:42:34Z)
Watchdog successor 29316 `STUDY_COMPLETE T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL`. Language 3/3. Representation **3/3**. Exact **27/29/26** (above T20 9/10/13, below T18 36/36/40). Support: first-token through-base restores a generation channel without T19 tax. Refute: it does not recover T18 exact. Do not relaunch T21. Do not parent 800001–800003. Next: T21X fail-mode.
