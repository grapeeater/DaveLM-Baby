# DaveLM project structure

Project **DaveLM** · family **Mangaris** · development codename **Baby**.
Active architecture **RNC (Racina Neural Core)**; current research model
**RNC-1, 91,246,848 parameters**. MRCN-Alpha is the preserved historical lineage.

## Published repository

This existing repository now publishes the RNC source alongside preserved museum
history. Its default branch is `v1.0`; the separate immutable `v1.0` tag still
identifies the earlier museum release.

```text
rnc/
  rnc/                  model, tokenizer codec, data, trainer, checkpoint and generation code
  configs/              authoritative first-run configuration
  tokenizer/rnc_bpe_8k/ exact active tokenizer and metadata
  tests/                existing health-check source
  docs/                 architecture, tokenizer, training and publication notes
  reports/              selected saved results and summaries
  reproducibility/      small dataset/run records and checkpoint hash sidecars
  tools/                existing inspection and reporting helpers
docs/history/           dated Baby milestones and historical research references
src/baby_v010/           preserved museum source
tokenizer/              preserved museum tokenizer
runs/actual_baby/       previously published historical museum artifacts
```

RNC checkpoints, corpus text, prepared token streams, virtual environments,
caches, private credentials and temporary GPU files are excluded. Original
historical Git LFS records are retained; this update adds no model weights.

## Fan Diesel workspaces (2026-10-16)

| Local path | Role |
| --- | --- |
| `C:\DaveLM-Training\rnc\` | Active RNC laboratory; source and original data/run records |
| `C:\DaveLM-Experiments\` | Separate local workspace for experimental descendants; existing Copy Goblin stays independent |
| `C:\DaveLM_ARCHIVE\MRCN-Alpha_LAB_20261016\` | Preserved historical lab, including certified S1/S2 components, failed runs and evidence |
| `C:\DaveLM-v1.0\` | Restored historical museum checkout; existing working-tree differences preserved |
| `C:\DaveLM_ARCHIVE\` | Archive index, museum masters, historical evidence and relocation receipts |

The Training root's `rebuild` and `checkpoints` entries are junctions into the
archived MRCN-Alpha lab. They do not identify active RNC training outputs.
Historical documents may retain pre-relocation paths; the archive's
`PATH_MAP.json` records the relocation.

The experiments workspace is not contained in the Training workspace and was
not a separate Git repository at the start of this update. Its pre-existing
`copy_goblin_227k_001` is a separately trained toy copy model, not an RNC descendant.
The completed organization adds `RNC/Instruct`, `RNC/Reasoning`, `RNC/Coder`,
`RNC/Writer`, `RNC/Personal` and `RNC/Other`, each containing only a README.
Copy Goblin stays in place and retains its independent purpose.

The [family tree](MANGARIS_FAMILY_TREE.md) explains the lineage and future plan.
`docs/experiments/` publishes only the eight new README files from the local
experiments workspace, byte-identical. It contains no model files and does not
relocate the experiments workspace into Training.

Instruct, Reasoning, Coder, Writer and Personal are future descendant concepts.
They should eventually originate from a verified common foundation. The future
Adam & Eve checkpoint has not been created or graduated. No specialist training
or infrastructure is part of this repository update.
