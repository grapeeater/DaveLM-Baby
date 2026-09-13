# T16 DECISION — two-phase 8–11 scaffold

T15 froze 8–11 from Phase1G and missed representation 0/3 while holding language. T15X showed 4–7 still moved as much as T14; 8–11 were bit-identical. T14X showed that *after* 4–11 joint training, putting 8–11 back to Phase1G improved language and reversals on the only T14 hit. Those are not parentable patches.

T16 is the prospective trained version: Phase A = T14 scope for 500 updates (8–11 plastic); after the U500 eval, revert 8–11 to Phase1G, freeze them, rebuild AdamW, Phase B = T15 scope for 250 updates. Same gates, Phase1G parent, seeds 750001–750003. Do not parent T14/T15/730002. Do not open TEST unless this frozen protocol later authorizes it after gates pass.

## Terminal
`T16_FAIL_NO_REPRESENTATION` 3/3. Language held 3/3 (CE 1.241–1.245, gap 0.368–0.371); Phase B improved language vs U500. Representation 0/3 (rev 43/46/39). Phase A ended too early (U500 rev 35–40). Do not parent. Do not relaunch T16. Next is a read-only DEV reversal-fail overlap, then one action.
