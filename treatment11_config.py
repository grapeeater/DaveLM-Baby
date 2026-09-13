"""Frozen configuration for Treatment-11 answer-only counterfactual binding."""

from pathlib import Path

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")
SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
SEED = 8380
NAME = "treatment11_answer_only_strict_counterfactual_binding"

OUT_ROOT = CADAVER_ROOT / "treatment11_answer_only_strict_counterfactual_binding_seed8380"
SOURCE_T10_ROOT = CADAVER_ROOT / "treatment10_strict_counterfactual_binding_seed8380"
SOURCE_T10_TRAIN_POOL = SOURCE_T10_ROOT / "treatment10_training_quartet_pool.json"
SOURCE_T10_RETENTION_POOL = SOURCE_T10_ROOT / "treatment10_retention_quartet_pool.json"
SOURCE_T10_TRAIN_POOL_SHA256 = "03ed49abcf6458362fe4f29b44ec2a5e2bfa9e05ad6788b9d2e6ff37c7bcfc17"
SOURCE_T10_RETENTION_POOL_SHA256 = "48731fbc1a37b5902f4827069e51a7dff1bd1d19c3ff1ed2d29b8e0d0a996de6"

START_CHECKPOINT_PATH = (
    SOURCE_ROOT / "experiments" / "minimal_contextual_binding" / "checkpoints"
    / "treatment_one_mapping" / "seed_8380" / "latest.pt"
)
START_CHECKPOINT_SHA256 = "345984c52a06db5f988aaf4cd47963eea0e9d77489cee94dbb10816af2f5443e"

TRAIN_POOL_PATH = OUT_ROOT / "treatment11_training_quartet_pool.json"
RETENTION_POOL_PATH = OUT_ROOT / "treatment11_retention_quartet_pool.json"
GENERATOR_RESULT_PATH = OUT_ROOT / "treatment11_generator_result.json"
SCHEDULE_PATH = OUT_ROOT / "treatment11_frozen_schedule.json"
PREFLIGHT_RESULT_PATH = OUT_ROOT / "treatment11_preflight_result.json"

CHECKPOINT_ROOT = OUT_ROOT / "checkpoints" / "answer_only_strict_counterfactual_binding" / "seed_8380"
FINAL_CHECKPOINT_PATH = CHECKPOINT_ROOT / "latest.pt"
TRAINING_RESULT_PATH = OUT_ROOT / "treatment11_training_result.json"
TRAINING_METRICS_PATH = OUT_ROOT / "treatment11_training_metrics.jsonl"
FINAL_AUDIT_RESULT_PATH = OUT_ROOT / "treatment11_final_retention_audit.json"

MAX_STEPS = 1000
BATCH_SIZE = 32
QUARTETS_PER_STEP = 8
EXPECTED_TRAIN_QUARTETS = 200
EXPECTED_TRAIN_DOCUMENTS = 800
EXPECTED_RETENTION_QUARTETS = 40
EXPECTED_RETENTION_DOCUMENTS = 160
EXPECTED_DOCUMENT_LENGTH = 193
EXPECTED_VOCAB_SIZE = 1024

LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.05
GRADIENT_CLIP_NORM = 2.0
