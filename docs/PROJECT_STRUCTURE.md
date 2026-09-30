# DaveLM project structure

DaveLM's local project uses separate workspaces for the museum snapshot,
active research, readable historical notes, and future descendant pointers.
The workspaces are not all published in this Git repository.

## This Git repository

This repository preserves the earlier 61.5M-parameter PropMatch/stack2
graduate. The `v1.0` tag is an immutable historical tag for that snapshot. The
tag is not the Mangaris Pioneer v1.0 release and does not identify the newer
118M S1 graduate.

- `src/baby_v010/`: earlier museum model and supporting code.
- `runs/actual_baby/`: earlier model weights and sealed evaluation evidence.
- `tokenizer/`: earlier snapshot tokenizer.
- `scripts/`: museum snapshot checks.
- `docs/`: publication-facing project history and notes.

The earlier repository checkpoint publication uses Git LFS. The 118M S1
checkpoint stays in the separate lab workspace and is described by its hash
and provenance; this documentation change publishes no new weights.

## Separate local workspaces

| Workspace | Purpose and current state |
| --- | --- |
| `DaveLM-Training/` | Active research lab. Contains the MRCN-Alpha 118M Baby source, frozen specifications, evaluations, and authoritative S1 runs. The canonical u4700 S1 checkpoint lives here. |
| `DaveLM_ARCHIVE/` | Human-readable ecosystem index, Baby timeline, Mangaris Zero history, cleanup notes, and S1v3g/h/i/j history. |
| `DaveLM-Clean/` | Clean-lineage and ancestry pointer documentation. |
| `DaveLM-v1.0/` | Legacy museum workspace corresponding to this repository; retained because scripts and historical references use its path. |
| `DaveLM-Personal/` | Personalization ancestry pointer; no S1 personalization run is claimed. |
| `DaveLM-Experiments/Coder/`, `Instruct/`, `Reasoning/`, `Writer/` | Future specialist ancestry/pointer preparation only. No specialist weights or trained specialist models were created. |
| `DaveLM-v0.10/` | Legacy stripped workspace, retained for historical path references. |
| `Mangaris Zero` records | Archive/graveyard history for pre-Pioneer experiments and failed developmental work; not a released model class. |

These names describe local workspace roles, not folders bundled in this Git
checkout. Individual machines may place them in different paths.

## Names used in project records

| Name | Meaning |
| --- | --- |
| DaveLM | Project and model ecosystem |
| Mangaris | Model family |
| Pioneer, Lite, Core, Forge, Atlas | Model classes; Pioneer remains a future decision |
| MRCN-Alpha, MRCN-Beta | Architecture generations |
| Baby | Permanent codename for the original model and eventual Pioneer lineage |
| Coder, Instruct, Reasoning, Writer | Future specialist concepts, not trained releases |

For the current milestone and exact parent lineage, see
[`history/S1_GRADUATION.md`](history/S1_GRADUATION.md).
