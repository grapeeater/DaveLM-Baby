# DaveLM Baby (CADAVER working tree)

Private research tree for **DaveLM Baby**: a ~61.5M vNext language model (Phase 2A) plus closed historical lineages. This is **not** a public product repo. Intended GitHub home (not created yet): **`github.com/grapeeater/DaveLM-Baby`** (private). Directory names on disk are scientific identifiers; do **not** rename or reorganize frozen experiment folders.

Authoritative live status: [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md) and [`CURRENT_STATE.md`](CURRENT_STATE.md). Agent rules: [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md). Closed-line ledger: [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md). Local binaries: [`artifacts/LOCAL_ONLY.md`](artifacts/LOCAL_ONLY.md).

## What this is

Baby’s current architecture is the frozen bundle `baby_vnext_60m_design_v1` (61,520,385 params). The language parent is Phase1G U6000 `best.pt` (local-only; SHA-256 in `artifacts/LOCAL_ONLY.md`). Phase 2A treatments T3–T23 are **closed**. **T24 is live** (`phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1`, seeds 830001–830003).

The 10.6M SF1–SF21 series is closed history. Nested `DaveLM-v0.9/` is a stale snapshot; Git tracks **only** `DaveLM-v0.9/tokenizer/`.

## Protected TEST (do not open)

| | |
|---|---|
| Bytes (local-only) | `phase2a_t3_rebuilt_study_v1/data/qa_test.jsonl` |
| Seal (in git) | `phase2a_t3_rebuilt_study_v1/data/TEST_SEAL.json` |
| SHA-256 | `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb` |
| Status | `SEALED_UNOPENED` |

Do **not** load, score, copy, or print `qa_test.jsonl` (or any other `qa_test.jsonl`) until terminal DEV representation **and** native gates pass. Other `qa_test.jsonl` copies exist under superseded T1/T2 ctxbind trees; they are also gitignored.

## What Git does not hold

Checkpoints (`*.pt` and friends), all `*_run_seed*` trees, `sf2_runtime/`, `archive/`, venv/pycache, the rest of nested `DaveLM-v0.9/`, census dumps, logs, `.env` / `.aider*` / `.config/`, `mistral_baby_history.json`, huge `d3_update*_RAW.json`, and **all TEST jsonl bytes**. Reproduce from local paths + SHA-256 in `artifacts/LOCAL_ONLY.md`.

## Current scientific head (read, do not run)

- Architecture: `baby_vnext_60m_design_v1/`
- Language streams / Phase1G protocol: `baby_vnext_phase1g_language_v1/`
- Authoritative T3 corpus (train/dev/panels; TEST sealed): `phase2a_t3_rebuilt_study_v1/`
- Live treatment: `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1/`
- Tokenizer (vendored copy): `DaveLM-v0.9/tokenizer/v0_7/davelm_tokenizer.json`
- Optional advisory (markdown only, no weights): `docs/reference/QWEN3_1P7B_BASE_REFERENCE_STUDY_V1.md`

Runners still hard-code `C:\DaveLM-CADAVER\...` and often `C:\DaveLM-v0.9\tokenizer\...`. Path refactor is **out of scope** for this onboarding commit.

## Standing prohibitions

- Do not relaunch T4–T23. Do not parent T14–T23 OUTPUT_FAIL checkpoints (or 730002 / 820001–820003).
- Do not lower the v1.0 / 2/3 representation bar.
- Do not retokenize. Do not copy Qwen or Smol weights into this tree.
