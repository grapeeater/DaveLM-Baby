# SF3 preflight

Status: **PASS — PROSPECTIVE FREEZE AUTHORIZED FOR THE SINGLE SF3 RUN**

The frozen design uses Pilot1 as parent, replays the exact SF2 data/objective/scope/optimizer schedule through
update 100, requires exact equality with the preserved SF2 update-100 model tensors, and then changes only the
late English learning rate. The immutable held-out transfer/copy panels stay locked unless all fixed update-200
gates pass. FINAL and sacred material are excluded.

Required pre-training checks:

- Verify the copied SF1 scientific bundle and its non-circular receipt/manifest.
- Verify parent, tokenizer, SF2 KL pool, D3 selection, SF2 update-100 reference, and pinned source hashes.
- Verify Python 3.12.14, torch 2.12.0+rocm7.14.0, tokenizers 0.23.1, deterministic GPU availability.
- Verify the exact 200-update materialized schedule, 180 English/20 binding cadence, all identities, padding,
  aligned masking, scope switching, and authoritative binding objective wiring.
- Verify the LR function: historical `5e-5` through update 100, binding `5e-5` throughout, and 90 post-100
  English values decreasing linearly from `5e-5` to `0`.
- Reproduce update-0 acquisition, aligned language CE, both nonsacred binding gates, D3, and zero baseline KL.
- Scan the controller for placeholders, locked-panel reads before gate, FINAL/sacred references, or extra
  treatment branches.
- Seal the protocol, controller, copied scientific payload, KL pool, and provenance before optimizer creation.

## Results

- Copied SF1 receipt/manifest: PASS, all 20 listed payload files.
- Parent, tokenizer, SF2 update-100 reference, KL pool/manifest, D3 selection, and pinned sources: PASS.
- Runtime: Python 3.12.14, torch 2.12.0+rocm7.14.0, tokenizers 0.23.1, AMD Radeon RX 9060 XT: PASS.
- Schedule: 200/200 records resolved; 180 English and 20 binding; cadence and all materialized identities: PASS.
- Masking: exactly four answer tokens plus EOS supervised; context and padding ignored; maximum model input 39: PASS.
- Scope: frozen/training parameter classification matches SF2 on the actual uninitialized architecture: PASS.
- LR schedule: first 100 updates reproduce SF2; 90 late English rates strictly decrease from `5e-5` to `0`;
  binding remains `5e-5`: PASS.
- Controller syntax, placeholder scan, transfer-lock ordering, and absence of FINAL/sacred path reads: PASS.
- Actual sealed controller pre-parent-load path: PASS (`SF3_PRE_PARENT_LOAD_PASS`).
- Actual update-0 baseline reproduction: 9/16 correct, 0/16 exact; aligned CE 3.3906604052 / PPL 29.68555;
  D3 name mass 0.0009072396; both nonsacred binding pools 80/80 answer, 80/80 BOTH_DISTINCT,
  zero collapse; baseline KL below `1e-9`: PASS.

Static receipt SHA-256: `1bb86c4317f96068c76a980491f4c8ce7b051036e96b82ce69b1f1339928687f`.
Baseline result SHA-256: `495fa83fd5fd65bd3191da0e19914dfbb91a59be80a51446785d301dcbe3e29e`.

The update-100 exact SF2 tensor-reproduction check remains a mandatory in-run pre-treatment gate. The annealed
learning rates cannot begin if that gate fails. No optimizer update had occurred when this preflight was signed.
