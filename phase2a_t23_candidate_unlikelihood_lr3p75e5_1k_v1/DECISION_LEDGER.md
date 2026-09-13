# T23 DECISION — in-row candidate unlikelihood

Earned by T22 / T22X. CE-span-count isolations are closed. Leftover is T18X-style rival completions after `Answer:`.

T23 = T17/T18 two-phase from Phase1G + T18 full-span through-base `λ_ans=1.0` + **in-row wrong-candidate unlikelihood** `λ_unl=1.0` at aligned answer positions. Fact-clause, 9:1, lr 3.75e-5, seeds 820001–820003. Same gates. No suffix-weight 3.0. No head-only. No first-only / first-two split. No retokenize.

Closed if this does not lift exact above T18 36–40 while holding language + representation ≥2/3: “rival-token unlikelihood finishes names.” Do not parent 820001–820003 on OUTPUT_FAIL. Do not unseal TEST unless native gates pass.
