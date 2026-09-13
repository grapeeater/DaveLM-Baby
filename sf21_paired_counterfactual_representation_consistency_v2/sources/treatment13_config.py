"""Frozen configuration for Treatment-13 learned mapping-row localization (preflight only)."""

from pathlib import Path

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")
SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
SEED = 8380
NAME = "treatment13_learned_mapping_row_localization"

OUT_ROOT = CADAVER_ROOT / "treatment13_learned_mapping_row_localization_seed8380"
SOURCE_T11_ROOT = CADAVER_ROOT / "treatment11_answer_only_strict_counterfactual_binding_seed8380"
SOURCE_T11_TRAIN_POOL = SOURCE_T11_ROOT / "treatment11_training_quartet_pool.json"

START_CHECKPOINT_PATH = (
    SOURCE_ROOT / "experiments" / "minimal_contextual_binding" / "checkpoints"
    / "treatment_one_mapping" / "seed_8380" / "latest.pt"
)
START_CHECKPOINT_SHA256 = "345984c52a06db5f988aaf4cd47963eea0e9d77489cee94dbb10816af2f5443e"

TRAIN_POOL_PATH = OUT_ROOT / "treatment13_training_quartet_pool.json"
RETENTION_POOL_PATH = OUT_ROOT / "treatment13_retention_quartet_pool.json"
GENERATOR_RESULT_PATH = OUT_ROOT / "treatment13_generator_result.json"
PREFLIGHT_RESULT_PATH = OUT_ROOT / "treatment13_preflight_result.json"
PREFLIGHT_REPORT_PATH = OUT_ROOT / "treatment13_preflight_report.md"

BATCH_SIZE = 32
DOCUMENT_LENGTH = 193
EMBED_SIZE = 320
VOCAB_SIZE = 1024
RETRIEVAL_DIM = 64
ROW_VALUE_OFFSET = 4  # public grammar: value token sits exactly 4 positions after its source key

TRAIN_PREFIX_RANGE = (8, 12)
TRAIN_BETWEEN_RANGE = (3, 6)
RETENTION_COMBOS = [
    (16, 8), (20, 8), (24, 10), (28, 10),
    (32, 12), (16, 12), (24, 8), (32, 8),
]
QUARTETS_PER_TRAIN_COMBO = 10
QUARTETS_PER_RETENTION_COMBO = 10
KEY_COUNT = 5
VALUE_COUNT = 5
