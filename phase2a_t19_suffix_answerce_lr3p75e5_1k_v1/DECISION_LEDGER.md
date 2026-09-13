# T19 DECISION — suffix-weighted answer CE

T18 held language 3/3 and representation 2/3 and moved exact 0→36–40. Native FC did not lift. T18X showed the leftover is not T17X story continuation: first token is already the answer BOS on 76–84/128 rows. Dominant miss is `first_token_correct_then_diverge` (Skye→Sky, Sal→Salt). Observed in Baby. Smol’s “generation is a later stage” is only analogical.

T19 keeps Phase1G / T17 two-phase / 9:1 / 3.75e-5 / same mix / same gates. Seeds 780001–780003. The only change: answer-span CE weights the first answer token 1.0 and all later answer tokens + period + EOS 3.0.

Support: DEV exact rises materially (and/or diverge-after-BOS falls) on ≥2/3 while language and representation hold.
Refute: language regression ≥2/3, representation collapse, or exact stays ~36–40.

Do not parent T18/T17 OUTPUT_FAIL checkpoints. Do not retokenize. Do not copy Qwen or Smol. Do not open TEST unless gates authorize it.

## TERMINAL 2026-09-12T18:07:47Z
Study `T19_FAIL_NO_REPRESENTATION`. Language 3/3. Representation **1/3** (only 780001). Exact 39/35/34 — no lift vs T18. Suffix-weight 3.0 closed. Next: T19X then readout-only answer CE (stop-grad base).
