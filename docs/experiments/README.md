# DaveLM experiments and the Mangaris family

**DaveLM** is the project; **Mangaris** is the model family; **RNC (Racina Neural Core)** is the active architecture; **Baby** is the development codename. The current RNC-1 research model has 91,246,848 parameters.

RNC starts from random initialization with its own tokenizer. It has no weight-inheritance edge from the historical MRCN-Alpha models. The seven completed curriculum sessions show bounded results on generated English tasks; they do not establish general conversational competence.

## Frozen Session 007 ancestor

The completed Session 007 checkpoint is the frozen common ancestor for four independent local specialist starting copies. Its source SHA-256 is dc0c60766460c57db52f1bfcc0ab58c449b3a36fda3bee66db8c7861efe4832a. Instruct, Reasoning, Coder and Writer copies each matched its checkpoint, config and tokenizer hashes and passed a read-only load check. No specialist training has started. Details are in [the common-ancestor record](../SESSION_007_COMMON_ANCESTOR.md) and [RNC workspace overview](RNC/README.md).

Personal and Other remain future directions without a Session 007 copy. The broader Adam & Eve common foundation has not been named as graduated. The frozen ancestor designation records lineage and does not establish a capability level.

The local experiments workspace stays at C:\DaveLM-Experiments, separate from C:\DaveLM-Training. Its RNC tree has the common-ancestor copy and these four starting-copy folders:

- [Instruct](RNC/Instruct/README.md): instruction following and useful question answering.
- [Reasoning](RNC/Reasoning/README.md): task-specific reasoning work.
- [Coder](RNC/Coder/README.md): code generation and understanding.
- [Writer](RNC/Writer/README.md): writing, revision, and narrative work.

This Git repository contains their README documentation, not their weights or datasets. The [curriculum report](../RNC_CURRICULUM_PROGRESS.md) describes measured results and their limits.

## Existing experiment preserved

copy_goblin_227k_001 is an independent 227,536-parameter toy copy model from historical qualification work, with its own source, checkpoint and evidence. It shares no canonical weights with Baby and is not an RNC specialist or common ancestor. It stays at its original path without reclassification.
