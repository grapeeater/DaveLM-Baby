STUDY: POST_T28_CAUSAL_PREFIX_RESCUE_V1
PRIMARY QUESTION: when Baby emits the first wrong answer token after a correct answer prefix, does repairing ONLY that token and returning to free greedy recover the exact answer?
POPULATION: DEV first-token-correct-then-diverge 39/40/45 pooled 124 (all rows kept)
CHERRY PICK SAL/SKYE: NO
GOLD RESCUE EXACT+EOS: 108/124 (87.1%) pooled; 32/39 (82.1%); 37/40 (92.5%); 39/45 (86.7%)
NEXT-TOKEN RECOVERY: 117/124 (94.4%)
REMAINING-SUFFIX EXACT: 108/124 (87.1%)
EOS / PERIOD+EOS: 124/124 / 124/124
CONTROL A GOLD REPAIR: primary, above
CONTROL B WRONG-MATCHED EXACT+EOS: 0/124 (0.0%)
CONTROL C NO INTERVENTION EXACT+EOS: 0/124 (0.0%; expected)
CONTROL D PREFIX-COMPATIBLE EXACT+EOS: 0/55 feasible (0.0%); 69 infeasible
SHARED-FIRST GOLD EXACT+EOS: 73/83 (88.0%)
UNIQUE-FIRST GOLD EXACT+EOS: 35/41 (85.4%)
FAMILY Sal: 36/36 (100%); Skye 30/39 (76.9%); Omar 3/4; Opal 4/4; Wes 34/40 (85.0%)
AT-DIVERGENCE GOLD: mean P=0.062, mean rank=88.9, top1=0/124
AT-DIVERGENCE WRONG: mean P=0.840, mean rank=1.0, top1=124/124
AFTER GOLD REPAIR NEXT: mean P=0.866, mean rank=1.15, top1=117/124 (94.4%)
INTERNAL STATE: blocks 7-11 frozen-head next-gold top1 ~85-97% after gold token vs ~0-3% after wrong token; no trained probe
ACTIVATION PATCHING: NOT RUN (gold rescue large+replicated; study stopped)
CLASSIFICATION: EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES
EARNED HYPOTHESIS: YES — first divergent token is a causally dominant contributor to these diverge failures under a single-token oracle; not native generation; does not authorize T29/TEST/tokenizer mutation/patching
TOKENIZER LINEAGE: PAUSED
T29: NOT AUTHORIZED / NOT LAUNCHED
TEST / FINAL / SACRED: SEALED
TRAINING / OPTIMIZER / CHECKPOINT / TOKENIZER WRITES: NONE
GIT COMMIT: pending
PUSH STATUS: pending
