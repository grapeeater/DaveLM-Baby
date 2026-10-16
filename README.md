# DaveLM · Mangaris · Baby

**RNC — Racina Neural Core — is DaveLM's active development architecture.**
DaveLM is the project, Mangaris is the model family, and Baby is the development
codename. The current research model is **RNC-1: 91,246,848 parameters**.

RNC begins a new weight lineage from random initialization with its own tokenizer.
The historical MRCN-Alpha models, certified S1/S2 accomplishments, earlier 61M
museum model, Git history and immutable `v1.0` release tag remain preserved.

## Current verified result

RNC-1 completed its first fixed-budget TinyStories run on Fan Diesel's AMD Radeon
RX 9060 XT: **3,000 optimizer steps, 98,304,000 training-input tokens**.
Saved records show held-out loss falling from **9.1523 to 1.5221 nats/token**;
final perplexity was approximately **4.58**. Saved samples show learned English
sentence patterns and short story continuations, with repetition and causal
inconsistencies still present.

This demonstrates language-model learning. General conversational competence,
instruction following and broad factual accuracy have not been established.
The future **Adam & Eve common-foundation checkpoint has not been created or
graduated**. RNC-1's first-run checkpoint is not that future foundation.

The existing health-check record reports **91 tests passed**. This publication
update reviewed saved results and file integrity; it launched no training,
evaluation or GPU health-check jobs.

- [First-run result and scope](rnc/reports/FIRST_RUN_SUMMARY.md)
- [Saved machine-generated report](rnc/reports/run_rnc1_tinystories_shakedown_001.md)
- [Original logs, config and checkpoint hash records](rnc/reproducibility/first_run/)

## Active implementation

RNC-1 is a decoder-only transformer: 12 layers, width 768, 12 attention heads,
SwiGLU width 2048, RMSNorm, RoPE, QK normalization, tied embedding/readout,
8,192-token byte-level BPE vocabulary and 512-token context.

| Repository path | Purpose |
| --- | --- |
| [`rnc/`](rnc/) | Active RNC implementation, config, tokenizer, tests and documentation |
| [`rnc/reproducibility/`](rnc/reproducibility/) | Small original records; datasets and checkpoints stay local |
| [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | Publication layout and local workspace boundaries |
| [`docs/history/`](docs/history/) | Preserved Baby history and milestones |
| `src/baby_v010/`, `tokenizer/`, `runs/actual_baby/` | Historical museum content already present in Git history |

The active local laboratory is `C:\DaveLM-Training\rnc`. Start with its
[workflow guide](rnc/README.md), [architecture](rnc/docs/RNC_ARCHITECTURE.md),
[tokenizer notes](rnc/docs/TOKENIZER.md) and [training record](rnc/docs/TRAINING.md).
Publication details and reproducibility limits are in
[PUBLICATION.md](rnc/docs/PUBLICATION.md).

The RNC workflow uses one configuration, tokenizer, trainer and GPU. S1/S2/S3
treatment gates, Baby University, agent orchestration and their approval chains
are retired for RNC; their historical evidence remains preserved.

## Preserved historical lineage

- [Baby, From Day One](docs/history/BABY_FROM_DAY_ONE.md): the existing canonical
  history through S1 graduation, retained unchanged as a dated historical record.
- [S1 graduation](docs/history/S1_GRADUATION.md): single-token copy graduation at
  u4700; this was a bounded copy milestone.
- [Historical S2 accomplishments and the RNC transition](docs/history/RNC_TRANSITION.md).
- The immutable Git **tag** `v1.0` remains the earlier PropMatch/stack2 museum
  release. The current default **branch** also happens to be named `v1.0`;
  adding RNC commits to that branch does not move or redefine the release tag.
- Original museum hashes remain in [GRADUATE_SHA256SUMS.txt](GRADUATE_SHA256SUMS.txt).

Historical S1/S2 certifications belong to MRCN-Alpha. RNC inherits project history
and engineering lessons, but no historical or third-party pretrained weights;
those certifications are not RNC capability claims.

## License

Source-available under the DaveLM Proprietary Research and Evaluation License
v1.0 (`LicenseRef-DaveLM-Research-1.0`): see [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). TinyStories provenance identifies
CDLA-Sharing-1.0; its dataset is not included in this update.
