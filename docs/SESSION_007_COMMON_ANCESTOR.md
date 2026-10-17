# Session 007 frozen common ancestor

Completed RNC training checkpoint selected as the frozen common ancestor for the Instruct, Reasoning, Coder, and Writer starting branches.

## Source identity

- Source directory: `runs/curriculum_ch4_007/ckpt_0000100`
- Checkpoint file: `checkpoint.pt`
- Checkpoint SHA-256: `dc0c60766460c57db52f1bfcc0ab58c449b3a36fda3bee66db8c7861efe4832a`
- Size: 1,095,096,203 bytes
- Run step: 100 (run parent step: 12,100)
- Model parameters: 91,246,848
- Tokenizer SHA-256: `d37360ce034918fcc2dc9be5fee17903244001eb307a1154ac662a5ee10ff6d3`

The checkpoint contains model tensors, step, configuration, RNG state, and AdamW optimizer state. The optimizer is embedded in `checkpoint.pt`; it is not a separate file. Its architecture configuration is embedded in the checkpoint and also recorded beside the source as `runs/curriculum_ch4_007/CONFIG.json`. The tokenizer files are `tokenizer/rnc_bpe_8k/tokenizer.json` and `tokenizer/rnc_bpe_8k/TOKENIZER_CONFIG.json`.

To load or continue the model, use the matching RNC loader/model/config/training code from the training workspace, the checkpoint, its frozen run configuration, and the pinned tokenizer. Training also requires the data and lesson assets referenced by that run configuration; those are not part of these checkpoint copies.

## Local copies

The frozen ancestor and four independent local starting copies are under `C:\\DaveLM-Experiments\\RNC`:

- `CommonAncestor_Session007`
- `Instruct/StartingCopy_Session007`
- `Reasoning/StartingCopy_Session007`
- `Coder/StartingCopy_Session007`
- `Writer/StartingCopy_Session007`

Each local `PROVENANCE.json` records the source path and hashes. All copies matched the source checkpoint, run config, and tokenizer hashes and passed a read-only CPU load check. The source checkpoint and historical checkpoints were left unchanged.

This commit records provenance only. It contains no checkpoint weights or datasets. The common-ancestor designation does not claim branch training, graduation, or capability results.
