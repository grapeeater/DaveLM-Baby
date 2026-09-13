"""SF2 KL retention pool builder (M2) - pure data, no torch, no checkpoint load.

Builds KL_POOL.json + KL_POOL_MANIFEST.json from the Pilot1 language_train
corpus using the exact frozen construction rule (FROZEN_PROTOCOL_DRAFT.md §6).

Run BEFORE any checkpoint is loaded and BEFORE training.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER")
TRAIN_SRC = ROOT / "language_pilot_1_early_block_protection_seed8380" / "language_train.jsonl"
TRAIN_SRC_SHA = "450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c"
DEV_MONITOR = ROOT / "single_fact_acquisition_sf1_seed87011" / "data" / "ENGLISH_DEV.jsonl"
DEV_MONITOR_SHA = "2053a8036d7d0f02ba56ab28f739df95a1743760d0f8c980e112d6667b38104e"
SF1_TRAIN = ROOT / "single_fact_acquisition_sf1_seed87011" / "TRAIN.json"
SF1_TRAIN_SHA = "08318d68e0d1b005ae340448af9dabb933013af08046a9cf8fd3ab64153ed9dd"
OUT_DIR = Path(r"C:\DaveLM-CADAVER\sf2_kl_parent_retention_run_v2")

KL_POS_PER_UPDATE = 160
MIN_POOL = KL_POS_PER_UPDATE * 180  # 28,800


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    assert sha256(TRAIN_SRC) == TRAIN_SRC_SHA, "language_train.jsonl hash mismatch"
    assert sha256(DEV_MONITOR) == DEV_MONITOR_SHA, "ENGLISH_DEV.jsonl hash mismatch"
    assert sha256(SF1_TRAIN) == SF1_TRAIN_SHA, "TRAIN.json hash mismatch"

    src = load_jsonl(TRAIN_SRC)
    dev = load_jsonl(DEV_MONITOR)
    dev128 = dev[:128]
    assert len(dev128) == 128
    train = json.loads(SF1_TRAIN.read_text(encoding="utf-8"))
    assert len(train) == 16

    monitor_arrays = [list(r["token_ids"]) for r in dev128]
    train_prompts = [list(r["prompt_token_ids"]) for r in train]

    retained = []          # rows retained, in file order, each = token_ids list
    retained_orig_index = []  # original file line index (0-based) for provenance
    rejected_empty_or_long = 0
    disjointness_violations = 0
    for idx, row in enumerate(src):
        ids = list(row["token_ids"])
        if not (2 <= len(ids) <= 254):
            rejected_empty_or_long += 1
            continue
        # Disjointness: exact token-array equality with any first-128 monitor row
        # or any SF1 TRAIN prompt array.
        if any(ids == m for m in monitor_arrays) or any(ids == t for t in train_prompts):
            disjointness_violations += 1
            continue
        retained.append(ids)
        retained_orig_index.append(idx)

    assert disjointness_violations == 0, (
        f"CONTAMINATION: {disjointness_violations} retained-row token arrays match a "
        f"monitor/TRAIN array; STOP."
    )
    assert len(retained) > 0, "no retained rows"

    # Row-major pool P of (row_idx_in_retained, col). col in 0..len(ids) inclusive
    # (each predicts z[col+1] of z=[2]+ids+[3]; count = len(ids)+1 per row).
    entries: list[list[int]] = []
    for i, ids in enumerate(retained):
        for col in range(len(ids) + 1):
            entries.append([i, col])

    pool_size = len(entries)
    assert pool_size >= MIN_POOL, (
        f"|P|={pool_size} < {MIN_POOL}; enlarge by retaining more rows. STOP."
    )

    manifest = {
        "rule": {
            "source": str(TRAIN_SRC),
            "source_sha256": TRAIN_SRC_SHA,
            "retention_filter": "rows with 2 <= len(token_ids) <= 254, in file order",
            "disjointness": (
                "exact token-array inequality vs first-128 ENGLISH_DEV.jsonl monitor rows "
                "and vs all 16 SF1 TRAIN prompt_token_ids arrays; violations => STOP"
            ),
            "ordering": "row-major over retained rows in file order",
            "aligned_form": "z=[2]+token_ids+[3]; labeled columns j=0..len(token_ids) "
                            "(column j predicts z[j+1]); count per row = len(token_ids)+1",
            "per_update_rotation": "English update u in 1..180 -> start=((u-1)*160) mod |P|; "
                                   "take next 160 entries of P (wrap at end)",
            "lambda_kl": 1.0,
        },
        "counts": {
            "source_rows": len(src),
            "dev_monitor_rows_checked": 128,
            "sf1_train_prompts_checked": 16,
            "rejected_empty_or_long": rejected_empty_or_long,
            "disjointness_violations": disjointness_violations,
            "retained_rows": len(retained),
            "pool_entries": pool_size,
        },
        "minimum_pool_required": MIN_POOL,
    }

    pool = {"rows": retained, "entries": entries, "count": pool_size}

    (OUT_DIR / "KL_POOL.json").write_text(
        json.dumps(pool, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8", newline="\n",
    )
    (OUT_DIR / "KL_POOL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "pool_sha256": sha256(OUT_DIR / "KL_POOL.json"),
        "manifest_sha256": sha256(OUT_DIR / "KL_POOL_MANIFEST.json"),
        **manifest["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
