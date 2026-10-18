# RNC specialist starting copies

These are four isolated local starting copies of Baby’s frozen Session 007 common ancestor. Each copy has a verified independent checkpoint file with SHA-256 dc0c60766460c57db52f1bfcc0ab58c449b3a36fda3bee66db8c7861efe4832a. The exact source, run config, tokenizer, and read-only load verification are recorded in [the common-ancestor record](../../SESSION_007_COMMON_ANCESTOR.md).

| Starting copy | Intended research direction | Status |
| --- | --- | --- |
| [Instruct](Instruct/README.md) | Instruction following and useful question answering | Frozen; no training |
| [Reasoning](Reasoning/README.md) | Reasoning tasks with task-specific evaluations | Frozen; no training |
| [Coder](Coder/README.md) | Code generation and understanding | Frozen; no training |
| [Writer](Writer/README.md) | Writing, revision, and narrative tasks | Frozen; no training |

On Fan Diesel the folders are C:\DaveLM-Experiments\RNC\Instruct\StartingCopy_Session007, Reasoning\StartingCopy_Session007, Coder\StartingCopy_Session007, and Writer\StartingCopy_Session007. Each stores checkpoint.pt, CHECKPOINT.json, CONFIG.json, tokenizer files, and PROVENANCE.json. The checkpoint includes model weights, architecture config, optimizer state, and RNG state. Source checkpoint size is 1,095,096,203 bytes; run-config SHA-256 is f27145bf5ee1192ead0df5d0092eef9317ad93b96b610ce63be158d0567254de; tokenizer SHA-256 is d37360ce034918fcc2dc9be5fee17903244001eb307a1154ac662a5ee10ff6d3.

Fan Diesel verified each copy’s checkpoint, config, and tokenizer hashes against the source and recorded a read-only CPU load pass on 2026-10-17. The model weights and local copies are not included in this Git repository. These folders name future research directions; they are not trained or evaluated specialists.

Personal and Other remain future directions without a Session 007 starting copy. The independent Copy Goblin project remains outside this lineage.
