# Read-only freeze-completion audit: fact_supervision_87001_corrected_v3

Audit scope: unchanged preserved bundle `C:\DaveLM-CADAVER\fact_supervision_87001_corrected_v3`. No checkpoint was loaded, no model was instantiated, no optimizer was created, no inference/training occurred, and sacred material was not accessed.

## Hash discrepancy

The physical `SHA256SUMS.txt` currently hashes to:

`d78a807280bbf7ca80ef725ede117e50f8741ab00bcde357c2696a15b4c41960`.

Its 21 listed file hashes all match their physical files, including the current `FREEZE_RECEIPT.json` hash. The bundle is therefore byte-consistent with the externally supplied `d78a…` receipt.

`FREEZE_RECEIPT.json` contains `93112b625c8a4812aff719422c46859f0bc20047c3a65a9636e39bd8fe4f8e2f`. Source `fact_supervision_87001_corrected_v3/finalize_freeze.py` (SHA-256 `ad7fbb2adbf507751064ab69b97f1afbbe6acc9963a2b8717ef998d0e2d6b7bd`) shows the cause: it first wrote a checksum file, wrote a receipt containing that checksum’s hash, rewrote `SHA256SUMS.txt` to include that receipt, then rewrote the receipt again and finally rewrote `SHA256SUMS.txt`. Thus `93112b…` is the intermediate checksum-file hash produced immediately before the final receipt rewrite; it is not a different hashing domain and no alternate physical file with that hash remains. `d78a…` is the final physical checksum-file hash.

All physical bundle files remain identical to the version represented by the current detached `d78a…` checksum. No file was changed during this audit. The receipt’s internal field is stale, but correcting it would modify the preserved bundle and is not a read-only mechanical completion.

## Executable-completeness classification

Classification uses A/B/C/D exactly as requested.

| Required component | Class | Evidence and conclusion |
|---|---|---|
| Per-update arm identity | A (partial) / D (executable) | `SCHEDULES.json` records arm labels, but no runnable consumer or durable execution schedule is present. |
| Per-update English/binding identity | A (partial) / D (executable) | Generic `kind` labels exist; concrete execution semantics are absent. |
| English batch example identities/order | D | `SCHEDULES.json` contains no example IDs or ordering. Frozen `ITEMS.jsonl` has rows but no selected batch partition/order. |
| Binding rehearsal batch identities/order | D | `BINDING_REFERENCES.json` identifies pools only; no 50-batch materialized schedule exists. |
| Repeated-family presentation rule | C | The prose gives presentation counts, but no unique family permutation/shuffle convention is frozen. Existing Pilot 1 uses a distinct schedule procedure. |
| Response masking | A (helper only) / D (trainer) | `harness.py` (`89a951d7c2058c2e67ce0f02bfc8039920c7a96bcd4053135b636304aefe6c10`) contains `prepare_example`; no integrated trainer uses it. |
| Padding masking | A (helper only) / D (trainer) | `harness.py` has `pad_batch`; no frozen batch construction invokes it. |
| BOS/EOS handling | A (helper only) / D (trainer) | `harness.py` records BOS/EOS masks, but no executable training path is frozen. |
| Factual/control response construction | A | `ITEMS.jsonl`, `FAMILIES.json`, and `independent_validator.py` are present and hashed. |
| Optimizer class/settings | A (declarative) / D (executable) | `PROTOCOL.md` states AdamW, 5e-5, weight decay .05, clipping 2.0; no trainer or parameter-group/state policy exists. |
| Learning-rate behavior | A (declarative) / D (executable) | Constant LR is stated, but no executable scheduler/optimizer is present. |
| English parameter scope | A (helper/declarative) / D (complete) | Scope is stated and `harness.py` has a helper; no integrated model trainer or exact parameter inventory is frozen. |
| Binding parameter scope | A (declarative) / D (complete) | Full-scope intent is stated; no executable transition is present. |
| Scope-transition gradient clearing | A (helper only) / D (trainer) | `harness.py` clears gradients, but no training loop calls it. |
| Optimizer state across scope transitions | D | No policy specifies whether frozen parameters remain optimizer members, how state is initialized, or how state behaves on reactivation. |
| Answer causal-CE implementation | B | Uniquely recoverable from Pilot 1 `run.py` SHA-256 `5a29ef7f4e96dde93e2feb8199669cb4f69d86609081334ad40a99159e7feda5`; its indexed full-vocabulary CE is explicit. Materialization would require pinning that source and wiring the new rows. |
| Hard-min permutation-invariant localization | B | Pilot 1 `run.py` same hash contains `loc_loss` and `LAM=1.0536573711078283`; exact formula is recoverable. It still requires explicit source pinning/wiring. |
| DEV evaluation points | D | No trainer/evaluator specifies executable evaluations at 0/100/500. |
| Endpoint behavior | A (declarative) / D (executable) | 500 updates is stated, but no endpoint checkpoint/evaluation implementation exists. |
| Checkpoint persistence | D | No frozen trainer persistence path or serialization implementation. |
| Durable per-unit journal/restart semantics | A (helper only) / D (complete) | `harness.py` contains `Journal`, but no runner integration or restart artifact is frozen. |
| English evaluation/scoring | A (helper only) / D (complete) | Candidate scoring helper exists; no complete evaluator over this study’s endpoints is frozen. |
| Seeds/RNG usage | A (labels only) / C | Seeds 87002/87003 appear in `SCHEDULES.json`, but random generators, streams, and ordering are not defined. Existing Pilot 1 uses `random.Random(8380)` and a torch generator, which is not uniquely implied by this bundle. |
| Runtime/tokenizer provenance | A | `PREFLIGHT.json`, `LEXICON.json`, source provenance, tokenizer hash, and package versions are present. |
| Parent identity | A | `MANIFEST.json` records Pilot 1; physical parent hash independently matches `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`. |

## Authoritative recoverable sources (B)

The only uniquely recoverable implementation fragments are the Pilot 1 binding objective and the existing causal answer CE in:

`C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\run.py`  
SHA-256: `5a29ef7f4e96dde93e2feb8199669cb4f69d86609081334ad40a99159e7feda5`

The source explicitly defines `loc_loss`, `LAM`, answer CE indexing, and binding tensors. Pinning it would be a mechanical source-copy/reference operation only for those fragments. It does not uniquely recover the new 500-update English/control schedule, batch ordering, optimizer-state policy, DEV harness, or durable runner.

Related but non-authoritative predecessor sources were inspected for comparison:

- Pilot 1 preparation: `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\prepare.py`, SHA-256 `b4e77c2d169d6bb7d37ad342ab12d820d9d1c21c16dbee0d8d2187627c276021`.
- T13 trainer: `C:\DaveLM-CADAVER\treatment13_train.py`, SHA-256 `299488cf8cd8de578d4eedc2e722d0986072ab2b27a472a94a20aad346b10917`.
- T13 model: `C:\DaveLM-CADAVER\treatment13_model.py`, SHA-256 `b7eea7b14c0193ce701e0be1d4f52a58ec48c99557e163e4d48068db1d12fc70`.

These sources implement different experiments and settings. Reusing them for the approved pair requires scientific choices wherever their conventions differ from the frozen prose.

## Schedule uniqueness

The frozen datasets and seeds do **not** uniquely determine concrete batches. Multiple plausible choices remain for family permutation, repeated presentations, batch grouping, control/factual co-location, RNG stream, and ordering. `SCHEDULES.json` contains 500 generic records per arm but no row/example IDs, no quartet IDs, and no order-defining RNG material. It therefore cannot be treated as a reproducible schedule.

## Files required for a genuinely executable corrected freeze

A corrected bundle would need, at minimum:

- corrected internally consistent receipt and detached checksum;
- materialized 500-update schedules for both arms, with exact English/binding identity and ordered example IDs;
- exact binding rehearsal batch schedule;
- pinned trainer source and model-construction source;
- integrated factual/control response masking and padding implementation;
- explicit BOS/EOS and loss-index records;
- explicit optimizer parameter membership and state-transition policy;
- DEV evaluator and fixed endpoint evaluator;
- checkpoint persistence and durable journal/restart implementation;
- source hashes, runtime inventory, and a pre-execution compatibility receipt;
- complete output schema and post-run checksum procedure.

## Final determination

The hash issue has a mechanical historical cause, but the executable-completeness gap is not mechanically recoverable. Copying Pilot 1 or T13 code would import conventions that are not uniquely specified for this experiment, especially schedule/RNG/batching and optimizer-state behavior. A corrected fully executable freeze cannot be produced with zero new scientific decisions.

