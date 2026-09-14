OPTION C STATUS: APPEND_ONLY_EXTENSION_VALID_COMPLETION / NO_EFFECT
CANDIDATES BUILT: 7 (v0_7 baseline + 6 TRAIN-only BPE variants: 1536, 2048, 4096, prefix-space 2048, titlecase-x4 2048, whitespace-first 2048)
BEST CANDIDATE: v0_7_baseline (best new: bpe_bl_1536, still worse)
V0_7 SHARED-PREFIX RATE: 0.500 of DEV names (Omar/Opal, Sal/Skye); generation exact on those items remains the T28 4.7% figure
BEST CANDIDATE SHARED-PREFIX RATE: 0.500 (bpe_bl_1536); titlecase_x4 0.625
V0_7 FIRST-ID-DISAMBIGUATION: group-conditional mean 1.50 (Omar/Opal/Sal/Skye at 2)
BEST CANDIDATE FIRST-ID-DISAMBIGUATION: 1.75 for bpe_bl_1536 (worse); 1.50 for bpe_ws_2048 / bpe_bl_4096 (no gain)
V0_7 TOKENS/NAME: 2.75 mean (DEV)
BEST CANDIDATE TOKENS/NAME: 3.00 (bpe_bl_1536); 2.62 (bpe_ws_2048 / 4096)
LANGUAGE COMPRESSION CHANGE: worse; v0_7 2.201 bytes/token vs 3.45–4.00 for new candidates
TRAIN-SIDE GENERALIZATION: some TRAIN collision drop (0.750→0.375) from seeing TRAIN names in BPE corpus
DEV-SIDE GENERALIZATION: none; Sal/Skye and Omar/Opal remain shared-first on every candidate
KNOWN COLLISION FAMILY EFFECT: Sal/Salt and Skye/Sky prefix geometry preserved; Wes not repaired; titlecase_x4 added Wes to a collision group
REPLACEMENT CLASSIFICATION: TOKENIZER_REPLACEMENT_NOT_JUSTIFIED
CHECKPOINT COMPATIBILITY: none; new IDs cannot load T28/Phase1G
FRESH MODEL LINEAGE REQUIRED: yes IF replacing, but replacement is not justified
FULL RETRAIN RECOMMENDED: NO
FULL RETRAIN AUTHORIZED: NO
TEST / FINAL / SACRED: SEALED
T29 STATUS: NOT AUTHORIZED
GIT COMMIT: pending
PUSH STATUS: pending
OWNER DECISION REQUIRED: YES
