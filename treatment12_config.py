"""Frozen configuration for Treatment-12 query-conditioned mapping retrieval."""

from pathlib import Path

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")
SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
SEED = 8380
NAME = "treatment12_query_conditioned_mapping_retrieval"

OUT_ROOT = CADAVER_ROOT / "treatment12_query_conditioned_mapping_retrieval_seed8380"

SOURCE_T11_ROOT = CADAVER_ROOT / "treatment11_answer_only_strict_counterfactual_binding_seed8380"
TRAIN_POOL_PATH = SOURCE_T11_ROOT / "treatment11_training_quartet_pool.json"
RETENTION_POOL_PATH = SOURCE_T11_ROOT / "treatment11_retention_quartet_pool.json"
SCHEDULE_PATH = SOURCE_T11_ROOT / "treatment11_frozen_schedule.json"

EXPECTED_TRAIN_POOL_SHA256 = "8f5d60de10fe2fdc8e772a9c1fc3e9f07861edd1583d7c413a095c2f55c6903c"
EXPECTED_RETENTION_POOL_SHA256 = "c341d7308b145bd3c633b62d56e01391f4b05635bfcf2edfb146bcd9f2de4e69"
EXPECTED_SCHEDULE_SHA256 = "f784cce5ddc6da9cc2b0a8e3a05194b55ae657ba95879ee15015e7320af6092b"

START_CHECKPOINT_PATH = (
    SOURCE_ROOT / "experiments" / "minimal_contextual_binding" / "checkpoints"
    / "treatment_one_mapping" / "seed_8380" / "latest.pt"
)
START_CHECKPOINT_SHA256 = "345984c52a06db5f988aaf4cd47963eea0e9d77489cee94dbb10816af2f5443e"

PREFLIGHT_RESULT_PATH = OUT_ROOT / "treatment12_preflight_result.json"
CHECKPOINT_ROOT = OUT_ROOT / "checkpoints" / "query_conditioned_mapping_retrieval" / "seed_8380"
FINAL_CHECKPOINT_PATH = CHECKPOINT_ROOT / "latest.pt"
TRAINING_RESULT_PATH = OUT_ROOT / "treatment12_training_result.json"
TRAINING_METRICS_PATH = OUT_ROOT / "treatment12_training_metrics.jsonl"
FINAL_AUDIT_RESULT_PATH = OUT_ROOT / "treatment12_final_retention_audit.json"

MAX_STEPS = 1000
BATCH_SIZE = 32
QUARTETS_PER_STEP = 8
DOCUMENT_LENGTH = 193
VOCAB_SIZE = 1024
EMBED_SIZE = 320
RETRIEVAL_DIM = 64

EXPECTED_TRAIN_QUARTETS = 200
EXPECTED_TRAIN_DOCUMENTS = 800
EXPECTED_RETENTION_QUARTETS = 40
EXPECTED_RETENTION_DOCUMENTS = 160
EXPECTED_PRESENTATIONS_PER_QUARTET = 40
EXPECTED_SUPERVISED_ANSWER_DECISIONS = MAX_STEPS * BATCH_SIZE

LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.05
GRADIENT_CLIP_NORM = 2.0
