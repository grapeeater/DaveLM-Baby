# T24 DECISION — train+dev name-inventory unlikelihood

Earned by T23 / T23X. In-row unlikelihood never saw Walt/York on Wes/Ava items. Sky is a tokenizer collision, not a corpus name.

T24 = T18 two-phase + full-span `λ_ans=1.0` + unlikelihood `λ_unl=1.0` over the frozen 24-name T3 train+dev inventory (no TEST). Seeds 830001–830003. Same gates. Do not raise `λ_unl`. Do not reopen CE-span-count. Do not retokenize.
