"""Shared configuration for the Treatment-10 strict counterfactual binding curriculum."""

from pathlib import Path

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")
SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

SEED = 8380
NAME = "treatment10_strict_counterfactual_binding"
OUT_ROOT = CADAVER_ROOT / "treatment10_strict_counterfactual_binding_seed8380"

TOKENIZER_PATH = (
    SOURCE_ROOT / "tokenizer" / "v0_7" / "davelm_tokenizer.json"
)
TOKENIZER_SHA256 = (
    "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
)

START_CHECKPOINT_PATH = (
    SOURCE_ROOT
    / "experiments"
    / "minimal_contextual_binding"
    / "checkpoints"
    / "treatment_one_mapping"
    / "seed_8380"
    / "latest.pt"
)
START_CHECKPOINT_SHA256 = (
    "345984c52a06db5f988aaf4cd47963eea0e9d77489cee94dbb10816af2f5443e"
)

TRAIN_POOL_PATH = OUT_ROOT / "treatment10_training_quartet_pool.json"
RETENTION_POOL_PATH = OUT_ROOT / "treatment10_retention_quartet_pool.json"
SCHEDULE_PATH = OUT_ROOT / "treatment10_frozen_schedule.json"
GENERATOR_RESULT_PATH = OUT_ROOT / "treatment10_generator_result.json"
PREFLIGHT_RESULT_PATH = OUT_ROOT / "treatment10_preflight_result.json"

CHECKPOINT_ROOT = OUT_ROOT / "checkpoints" / "strict_counterfactual_binding"
SEED_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "seed_8380"
FINAL_CHECKPOINT_PATH = SEED_CHECKPOINT_ROOT / "latest.pt"
TRAINING_RESULT_PATH = OUT_ROOT / "treatment10_training_result.json"
TRAINING_METRICS_PATH = OUT_ROOT / "treatment10_training_metrics.jsonl"
FINAL_AUDIT_RESULT_PATH = OUT_ROOT / "treatment10_final_retention_audit.json"

RAW_DOCUMENT_TOKEN_COUNT = 192
MODEL_VISIBLE_DOCUMENT_LENGTH = 193  # BOS + 192 raw tokens.

KEY_COUNT = 5
VALUE_COUNT = 5
RETENTION_COUNT = 40
RETENTION_SUBSET_STEP = 5

MAX_STEPS = 1000
BATCH_SIZE = 32
DOCS_PER_STEP = BATCH_SIZE
QUARTETS_PER_STEP = DOCS_PER_STEP // 4
QUARTET_DOCS = 4
ORIENTATIONS = (1, 2)

LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.05
GRADIENT_CLIP_NORM = 2.0
ANSWER_LOSS_LABEL = "ordinary autoregressive full-vocabulary causal cross entropy"
