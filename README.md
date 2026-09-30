# DaveLM · Mangaris · Baby

DaveLM is a model research project. **Mangaris** is its model family, and
**Baby** is the permanent codename for the original model line. Baby's current
118M-parameter research line uses the **MRCN-Alpha** architecture lineage.

> **S1 PASSED — BABY GRADUATED THIS STAGE.**
>
> On September 30, 2026, Baby passed the frozen S1 single-token-copy curriculum
> and certification at update **u4700**.

The two gated panels each scored **128/128** on teacher-forced content, EOS,
and exact free generation, plus **64/64** counterfactual pairs. The frozen
certification also passed. The run stopped at u4700 under its first-pass
graduation rule; u4701–u4800 were not run. All 12 S1v3h novel-position errors
were resolved. The nongated identity-stress panel remains an open frontier.

**S1 is one foundational stage, not completion of Baby's curriculum. It does
not declare Mangaris Pioneer v1.0 complete or released.** S1 demonstrates a
narrow single-token copy capability; it does not make Baby a general
conversational, coding, reasoning, or instruction-following assistant.

## Canonical S1 lineage

```text
S1v3g u3200 ──┬──→ S1v3h attempt 005 u4000 ──→ S1v3j revision 2 u4700 (S1 PASS)
              └──→ S1v3i u3200–u4000 (failed sister arm; not an ancestor)
```

The graduating run continued from S1v3h. S1v3i branched separately from
S1v3g and is retained as a failed comparison, not as a parent of the graduate.
See [the S1 graduation record](docs/history/S1_GRADUATION.md) for the gate
results, checkpoint provenance, and verification details.

## Canonical checkpoint and reproducibility

The canonical graduating checkpoint is held in the separate research lab
workspace, not in this repository:

```text
Lab-relative path:
rebuild/baby_reincarnation_118m_001/runs/S1v3j_single_token_copy_anneal_from_s1v3h_001/S1_single_token_copy_graduated.pt

Update: 4700
Stage: S1_single_token_copy
Size: 1,418,883,555 bytes
SHA256: 200e02188063885560edb4de0d6e0f05054fa75c236c0264d4a2c461ec51771b
```

Canonical release checkpoints are immutable masters: record their provenance
and verify their hashes, then conduct experiments on separate descendants.
Never overwrite or silently replace a master copy. The S1 checkpoint is not
included in this Git update.

## Names and scope

| Name | Meaning |
| --- | --- |
| DaveLM | Overall project and model ecosystem |
| Mangaris | Model family |
| Pioneer, Lite, Core, Forge, Atlas | Model classes; Pioneer remains a future project decision |
| MRCN-Alpha, MRCN-Beta | Architecture generations |
| Baby | Permanent codename for the original model and eventual Pioneer lineage |
| Mangaris Zero | Historical graveyard for pre-Pioneer experiments; not a released model class |
| Coder, Instruct, Reasoning, Writer | Future specialist concepts and branches; no specialist models have been trained |

S1 tests whether Baby can find a requested token in a sequence and copy that
single-token answer across frozen heldout and novel-position panels. It is an
experimental milestone in a larger foundational curriculum. Identity-stress
remains unresolved. Later stages, model-class graduation, and release status
require their own evidence and decisions.

## Project structure

This Git repository preserves the earlier **61.5M-parameter DaveLM v1.0
museum snapshot**. Its immutable `v1.0` tag records a different, earlier
PropMatch/stack2 graduate. That historical snapshot is **not** the 118M S1
graduate and is **not Mangaris Pioneer v1.0**. Its existing checkpoint
publication uses Git LFS; this S1 documentation update adds no model weights.

The current S1 lab and readable archive are separate local trees. The
[project-structure guide](docs/PROJECT_STRUCTURE.md) explains which material
belongs to this repository and which remains in the lab, archive, or pointer
workspaces.

| In this repository | Purpose |
| --- | --- |
| `src/baby_v010/` | Legacy museum model, runtime, evaluations, and historical helpers |
| `runs/actual_baby/` | Earlier museum graduate weights and sealed exam evidence |
| `tokenizer/` | Earlier museum snapshot tokenizer |
| `scripts/` | Museum snapshot verification tools |
| `docs/history/` | Project naming, structure, and current S1 graduation record |
| `ENV_SETUP.md` | Environment instructions for the legacy museum snapshot in this repository |

## License and notices

This project is **source-available under the DaveLM Proprietary Research and
Evaluation License v1.0** (`LicenseRef-DaveLM-Research-1.0`), not an
open-source license. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Separately identified
third-party material remains under its own terms.

## Legacy museum snapshot

The earlier stack2 graduate's environment and immutable artifact details are
recorded in [`ENV_SETUP.md`](ENV_SETUP.md),
[`GRADUATE_SHA256SUMS.txt`](GRADUATE_SHA256SUMS.txt), and
[`MANIFEST.txt`](MANIFEST.txt). They apply to that tagged museum snapshot, not
to Baby's current 118M MRCN-Alpha S1 research line. The historical `v1.0` tag
is preserved and is not moved by this documentation update.
