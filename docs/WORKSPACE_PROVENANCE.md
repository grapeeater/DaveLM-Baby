# v0.10 workspace provenance

## Path discrepancy

The requested path `C:\\DaveLM0.10` did not exist when work began. The
initialized writable workspace was `C:\\DaveLM-v0.10`, which is therefore the
working authoritative path for this session. No trained v0.9 checkpoint is used
as a parent.

## Existing contents at initialization

The directory was not empty: it contained uncommitted T31/T33-era scratch
artifacts and state notes. They are preserved in place and are not imported by
the v0.10 package. New v0.10 work lives in the clean `src/`, `configs/`,
`data_specs/`, `evaluations/`, `research/`, `design/`, `runs/`, `checkpoints/`,
and `tests/` structure. The scratch artifacts are not treated as v0.10
scientific results.

## Reused components

- The core decoder-only model implementation is a deliberate reuse of the
  validated Baby vNext implementation family from CADAVER. It is copied into
  `src/baby_v010/` and initialized afresh.
- The v0_7 byte-level BPE tokenizer is reused as a measurement baseline, not as
  a trained model artifact. Its fragmentation risks are explicitly audited.
- The Phase1G language token stream is used only as a non-protected language
  retention source during development; it is never used for structured
  evaluation targets.

No TEST, FINAL, SACRED, or protected answer panel is read by the v0.10
pipeline.
