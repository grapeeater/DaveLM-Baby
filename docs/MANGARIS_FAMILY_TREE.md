# Mangaris family tree

**DaveLM** → project; **Mangaris** → model family; **RNC (Racina Neural Core)**
→ active architecture; **Baby** → development codename; **RNC-1** → current
91,246,848-parameter research model.

```text
DaveLM / Mangaris
  Preserved historical research
    61M museum graduate (immutable v1.0 tag)
    MRCN-Alpha (certified S1/S2 milestones and later research)
  Active RNC lineage (random initialization; independent weights)
    Baby / RNC-1: TinyStories plus seven bounded curriculum sessions
    Frozen Session 007 common ancestor
      Instruct  [frozen starting copy; not trained]
      Reasoning [frozen starting copy; not trained]
      Coder     [frozen starting copy; not trained]
      Writer    [frozen starting copy; not trained]
    Personal  [future direction; no copy]
    Other     [future direction; no copy]
```

The tree expresses project relationships, not a graduation claim. The Session 007 checkpoint is the frozen common ancestor for four independent specialist starting copies. It is not itself evidence that those specialists have trained. RNC curriculum results show bounded performance on generated panels; they do not establish general conversation, broad reasoning, or factual reliability. See [the session results and limits](RNC_CURRICULUM_PROGRESS.md) and [the checkpoint provenance](SESSION_007_COMMON_ANCESTOR.md).

## Separate local workspaces

```text
C:\DaveLM-Training\
  rnc\                         active RNC laboratory
C:\DaveLM-Experiments\
  README.md
  copy_goblin_227k_001\         pre-existing independent toy model; preserved
  RNC\
    README.md
    CommonAncestor_Session007\
    Instruct\StartingCopy_Session007\
    Reasoning\StartingCopy_Session007\
    Coder\StartingCopy_Session007\
    Writer\StartingCopy_Session007\
    Personal\README.md   future direction only
    Other\README.md      future direction only
C:\DaveLM_ARCHIVE\
  MRCN-Alpha_LAB_20261016\      preserved historical laboratory
```

The frozen common ancestor and four starting copies are complete and hash-verified locally. Each has its own checkpoint file, frozen run configuration, tokenizer files and PROVENANCE.json. No branch training has started. Personal and Other remain README-only future directions. The active laboratory, historical archive and separate experiments workspace retain their boundaries.

The pre-existing Copy Goblin remains an independent 227,536-parameter toy copy
experiment. It was not moved under RNC, renamed or republished. Its entry
in the copied experiments overview identifies local-only material; its files
and weights are not part of this documentation snapshot.

See [the experiments documentation snapshot](experiments/README.md),
[specialist roles](experiments/RNC/README.md),
[project structure](PROJECT_STRUCTURE.md), and
[historical transition](history/RNC_TRANSITION.md).
