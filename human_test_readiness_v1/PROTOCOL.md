# HUMAN-TEST-READY v1 protocol
Parent: Language Pilot 1 (immutable checkpoint hash 2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb).
Development diagnostics may be repeated; FINAL_ITEMS.jsonl is sealed before any candidate tuning and is opened once only after development criteria pass.
Development criteria: (1) >=80% non-EOS on 20 short prompts, (2) >=70% human-readable complete sentences, (3) >=60% prompt-relevant responses, (4) >=70% exact controlled fact-selection on 20 matched items with >=8/10 reversals, (5) >=70% elementary instruction items, (6) >=60% two-turn continuity, (7) both nonsacred binding pools >=76/80 answer and BOTH_DISTINCT with zero collapse. These are engineering criteria, not graduation claims.
Final gate: on sealed final items, >=75% for each controlled stratum (generation coherence/relevance, fact selection, instruction, continuity), no immediate-EOS majority, and both binding pools pass the preregistered 76/80, BD 76/80, collapse 0 rule. Human scoring is frozen before opening; raw generations are retained.
No sacred material. No v1.0 claim. All checkpoint and artifact hashes are recorded in MANIFEST.json.
