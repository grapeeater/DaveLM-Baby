# Phase-1 language validation

Status: **PASS before seal**

## Identity and provenance

- Architecture receipt SHA-256: `776859cb1bf671634ae2d7e8dfab7f7cb8e517ce802d30d3cb3733b6b734b368`
- Architecture config SHA-256: `8d7e1cbc604c9bf9f1942d4afc01817b20af17d8805a984f7ae57b1254f957b4`
- Tokenizer SHA-256: `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`
- Pilot 1 train source SHA-256: `450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c`
- Pilot 1 DEV source SHA-256: `deff4fc7ed18e6e1f0b6f32faccc80f1eb44d0e58f3f38d512ffdfab798d6ac4`
- Initialized checkpoint SHA-256: `039e8efe7353d5065c06c2519053695e5944d58eeee2765594a8dfadb80d821c`

## Mechanical checks

| Check | Result |
|---|---|
| Runner imports and parses | PASS |
| Frozen config loads | PASS |
| Architecture chain verifies | PASS |
| Model instantiates/initialized checkpoint loads | PASS |
| Initialized checkpoint records zero updates | PASS |
| Tokenizer identity | PASS |
| Train/DEV sources and frozen streams resolve | PASS |
| Literal 6,000 x 64 schedule resolves | PASS |
| Evaluation selection resolves | PASS |
| Parameter scope: 60,536,064 trainable / 984,321 frozen | PASS |
| LR boundary values | PASS |
| Restart state round-trip | PASS |
| Wrong schedule/protocol provenance rejected | PASS |
| Resume controller targets completed update + 1 | PASS by code-path validation |
| Missing scheduled evaluation can run without replay | PASS by code-path validation |
| Static required-function completeness | PASS |
| Placeholder/TODO/NotImplemented branch scan | PASS; none present |
| Optimizer created during validation | NO |
| Optimizer steps during validation | 0 |
| Training performed | NO |

The frozen U0 initialized-model baseline is DEV CE **7.086638** / PPL **1195.88**, train CE **7.087802** / PPL **1197.27**, with DEV-minus-train gap **-0.001164**. Ten fixed greedy diagnostics were generated only from the disposable zero-update initialization; all were non-immediate-EOS and are descriptive rather than a readiness gate.

One mechanical defect was found before sealing: Windows does not permit `fsync` through a read-only file handle. The atomic torch-save helper was corrected to reopen the completed temporary file in read/write binary mode before `fsync` and atomic replacement. No scientific configuration or model state changed. The complete bounded validation was rerun after the correction.

The sealed controller must re-run this same validation path against the physical receipt and manifest before future training can begin.
