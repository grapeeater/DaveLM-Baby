# Sources

## AUTHORITATIVE UPSTREAM FACT (NVIDIA-published)

1. NVIDIA. "NVIDIA-Nemotron-3-Nano-4B-BF16" model card.
   https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (fetched live).
2. NVIDIA-Nemotron-3-Nano-4B-BF16 `config.json` (architecture hyperparameters).
   https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/config.json
3. NVIDIA-Nemotron-3-Nano-4B-BF16 `tokenizer_config.json` (special tokens, chat template markers).
   https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/tokenizer_config.json
4. NVIDIA-Nemotron-3-Nano-4B-BF16 `generation_config.json`.
   https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/generation_config.json
5. Taghibakhshi et al. (NVIDIA), "Nemotron Elastic: Towards Efficient Many-in-One Reasoning LLMs,"
   arXiv:2511.16664 (Nov 2025). https://arxiv.org/abs/2511.16664
6. NVIDIA, "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models,"
   arXiv:2504.03624 (Apr 2025, rev. Sep 2025). https://arxiv.org/abs/2504.03624
7. Model card References list (cited but not independently re-fetched full-text in this pass — titles/claims
   attributed to the model card, not independently re-verified against the papers' full text):
   - "NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model"
     (NVIDIA research technical report PDF)
   - "NVIDIA Nemotron 3: Efficient and Open Intelligence," arXiv:2512.20856
   - "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic
     Reasoning," arXiv:2512.20848
   - "Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic
     Reasoning" (NVIDIA research technical report PDF)

## OBSERVED LOCALLY (this machine)

8. `C:\Users\jdman\.lmstudio\hub\models\nvidia\nemotron-3-nano-4b\manifest.json`, `model.yaml`, `README.md`
   (LM Studio's own catalog metadata for the model, not NVIDIA's original card).
9. `C:\Users\jdman\.lmstudio\models\lmstudio-community\NVIDIA-Nemotron-3-Nano-4B-GGUF\NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf`
   — GGUF header + 36 metadata KV pairs + 263-tensor tensor-info table, parsed with a purpose-built
   from-scratch reader (no third-party `gguf` library available in this environment). This is a
   community (`lmstudio-community`) GGUF re-packaging/quantization of the NVIDIA BF16 release, not an
   NVIDIA-published artifact itself — flagged accordingly in REPORT.md.

## Baby (DaveLM) — all OBSERVED LOCALLY, primary source code and experiment artifacts

10. `C:\DaveLM-v0.9\*` (authoritative current v0.9 repository — see INSPECTED_PATHS.md for the specific
    files read).
11. `C:\DaveLM-CADAVER\*` (root-level research/forensic tree: treatment4–treatment13, two_mapping_*,
    language_pilot_0/1, language_continuation/consistency/mixed/storybound/unlikelihood/sentencebound/
    compositional P0–P7, post_p7_language_report_card v1–v3d, fact_supervision_87001 and its corrected
    v1–v8 variants, human_readiness_hr1/hr2/hr3 (all seed variants), single_fact_acquisition_sf1,
    sf1_readout_selection_forensic_v1, sf1_component_swap_forensic_v1, and the root-level diagnostic
    scripts `GENERATION_PATHOLOGY_REPORT.md`, `hr_generation_pathology.py`, etc.)
12. `C:\DaveLM-CADAVER\archive\DAVELM_P7_MILESTONE_seed8380\ARCHIVE_RECORD.md`.
13. `C:\DaveLM-CADAVER\forensics\FREEZE_COMPLETION_AUDIT_20260905.md`.

All Baby-related facts in REPORT.md are OBSERVED LOCALLY by construction (Baby is not a public model);
the OBSERVED LOCALLY / AUTHORITATIVE UPSTREAM FACT / INFERENCE labels in REPORT.md are applied only to the
Nemotron side, as instructed.

## Note on evidentiary confidence

- Items 1–4 and the config/tokenizer JSON files are first-party NVIDIA-hosted artifacts fetched directly from
  huggingface.co at research time — treated as AUTHORITATIVE UPSTREAM FACT.
- Item 9 (the GGUF file) is a **third-party re-packaging** (`lmstudio-community`) of NVIDIA's released
  weights. Its metadata (architecture dims, tokenizer vocab/merges, quantization scheme) is treated as
  OBSERVED LOCALLY and cross-checked against item 2 (`config.json`) where overlapping fields exist; where the
  two agree (block_count/num_hidden_layers=42, embedding/hidden_size=3136, attention.head_count=40,
  vocab_size=131072, ssm.state_size=128, ssm.conv_kernel=4, n_groups/ssm.group_count=8), this is noted as
  strong corroboration. Any GGUF-only field not present in `config.json` (e.g. the per-layer
  `hybrid_override_pattern`-derived feed_forward_length/attention.head_count_kv arrays) is OBSERVED LOCALLY
  only and not independently confirmed against an NVIDIA document.
- Items 5–7 (arXiv papers) are read only at abstract/title level in this pass (fetched via arXiv abstract
  pages); full-text methodological claims (e.g. exact training curricula, exact distillation losses) beyond
  what the abstract states are marked INFERENCE or explicitly flagged as "not independently verified beyond
  the abstract" in REPORT.md.
