# Phase 2A T3 — Architecture/Config Provenance Note (prospective)

Status: **resolved — hash/serialization difference only (Resolution A). No semantic mismatch.**

## Discrepancy under audit
- Sealed vNext design receipt `config_sha256`: `8d7e1cbc604c9bf9f1942d4afc01817b20af17d8805a984f7ae57b1254f957b4`
- Physical `BABY_VNEXT_CONFIG.json` raw-bytes SHA-256: `3078bbc7cca4d0d74b2c2befdeb1bdcd068f948c134497c603a09e03b7415480`

## Root cause
Two different, both-correct hash functions over the same file:
- `3078bbc7…` = raw file-bytes SHA-256 of `baby_vnext_60m_design_v1\BABY_VNEXT_CONFIG.json` exactly as written by `BabyVNextConfig.save()` (JSON indent=2, `sort_keys=True`, trailing newline). This value is the integrity entry for that file in the sealed design `SHA256SUMS.txt`.
- `8d7e1cbc…` = semantic canonical hash `BabyVNextConfig.sha256()` = SHA-256 of `json.dumps(to_dict(), sort_keys=True, separators=(",", ":"))` (see `baby_vnext/config.py:87-91`). This value is the architecture-config identity recorded in the design `FREEZE_RECEIPT.json` (`config_sha256`), the Phase1G config (`architecture.config_sha256`), and the initialized checkpoint payload (`config_sha256`).
The two differ only by serialization format (pretty-printed file bytes vs compact canonical JSON), not by any configuration value.

## Evidence
- `BabyVNextConfig.load(BABY_VNEXT_CONFIG.json).sha256()` = `8d7e1cbc…` (semantic).
- Raw file SHA-256 = `3078bbc7…`; the same `3078bbc7…` appears as the `BABY_VNEXT_CONFIG.json` line in the sealed design `SHA256SUMS.txt`.
- Phase1G initialized checkpoint `seed_610001_initialized.pt`: `schema_version = baby_vnext_checkpoint_v1`, `config_sha256 = 8d7e1cbc…`, and its embedded `config` dict equals the current file's `to_dict()` exactly.
- Phase1G language config `architecture.config_sha256 = 8d7e1cbc…`.
- Phase1G U6000 parent `best.pt` provenance `architecture_receipt_sha256 = 776859cb…`, equal to `baby_vnext_60m_design_v1\FREEZE_RECEIPT.sha256`.
- Strict load of `best.pt` state_dict into `BabyVNextWithBinding(BabyVNextConfig.load(BABY_VNEXT_CONFIG.json))` succeeds; 146 state-dict keys; base 60,536,064; binding 984,321; total 61,520,385.
- Design receipt `manifest_sha256 = a4a9fb21…`; `initialized_model_state_sha256 = c2313015…` (design-level init, distinct from the Phase1G run init by design).

## Authoritative architecture/config identity (for T3 and all downstream phases)
| Item | Value |
|---|---|
| Architecture bundle | `C:\DaveLM-CADAVER\baby_vnext_60m_design_v1` |
| Config file | `BABY_VNEXT_CONFIG.json` |
| Config file raw-bytes SHA-256 (manifest) | `3078bbc7cca4d0d74b2c2befdeb1bdcd068f948c134497c603a09e03b7415480` |
| Config semantic canonical SHA-256 (receipt) | `8d7e1cbc604c9bf9f1942d4afc01817b20af17d8805a984f7ae57b1254f957b4` |
| Architecture receipt SHA-256 | `776859cb1bf671634ae2d7e8dfab7f7cb8e517ce802d30d3cb3733b6b734b368` |
| Design manifest SHA-256 | `a4a9fb219f082e1936bf7341f42d82354fc2c7cf8d6f9b0d03c84e815117cad4` |
| Tokenizer SHA-256 | `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b` |
| Parent (Phase1G U6000 best.pt) SHA-256 | `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1` |
| Parent model-state digest | `0101cec9e3220c1de1a0d8abc4e241e2fad32127b7358bc5a2944a34c5a5b953` |
| vocab / context | 1024 / 256 |
| d_model / layers / heads / d_mlp | 640 / 12 / 10 / 2560 |
| binding: slots / retrieval_dim | 2 / 128 |
| base / binding / total params | 60,536,064 / 984,321 / 61,520,385 |

## Resolution
**Resolution A — hash/serialization difference only.** The sealed receipt hash and the physical-file hash refer to the same architecture configuration; the semantic configuration that constructed the authoritative Phase1G U6000 parent is exactly `BabyVNextConfig.load(BABY_VNEXT_CONFIG.json)`, whose canonical hash is `8d7e1cbc…`. No historical artifact was modified. T3 may proceed using this authoritative identity.
