# SF2 KL-to-parent retention run v2 — final report

- Run id: `sf2_kl_parent_retention_run_v2`
- Frozen protocol: `C:\DaveLM-CADAVER\sf2_kl_treatment_preflight_review_v1\FROZEN_PROTOCOL_DRAFT.md`
- Runtime: Python 3.12.14 (standalone build, relocated), torch 2.12.0+rocm7.14.0, tokenizers 0.23.1, CUDA
  (AMD Radeon RX 9060 XT). Venv: `C:\DaveLM-CADAVER\sf2_runtime\sf2venv`.
- Preflight: PASS (environment gate, artifact hashes, retention-pool disjointness, update-0 baseline
  reproduction incl. D3 mass 0.000907 and KL==0).
- Training: ran the full frozen budget of 200 updates (180 English + 20 binding). Final status:
  `SF2_ACQUISITION_FAIL` (endpoint gate unmet at 200); no earlier stop fired; transfer panels remained locked.

## Results table

| Metric | Pilot1 (u0) | SF1 u100 | SF2 u100 | SF2 u200 |
|---|---|---|---|---|
| Acquisition correct | 9/16 | 12/16 | 11/16 | 15/16 |
| Exact answer+EOS | 0/16 | 12/16 | 10/16 | 15/16 |
| Reversals | 1/8 | 4/8 | 3/8 | 7/8 |
| Complete families | 0/4 | 2/4 | 1/4 | 3/4 |
| Owen-correct | 1/4 | 0/4 | 0/4 (mgn -0.224) | 4/4 (mgn +0.516) |
| Aligned CE (PPL) | 3.391 (29.7) | 4.653 (104.9) | 3.574 (35.7) | 3.565 (35.3) |
| D3 four-name mass | 0.000907 | 0.0678 | 0.00971 | 0.00696 |
| Binding (both pools) | PASS | PASS | PASS | PASS |

## Interpretation

- The KL-to-parent retention term, at the frozen lambda=1.0, suppressed the SF1 name-prior pollution on
  unrelated contexts by ~7-10x (D3 mass 0.0069-0.0097 vs SF1 0.068), held ordinary-language aligned CE near the
  parent (3.57 vs SF1's 4.65; the SF1 language stop never tripped), and preserved binding exactly (80/80/80 both
  pools).
- The SF1 Alex-default on Owen-correct items was eliminated by update 200 (Owen-correct 0/4 -> 4/4, positive mean
  sequence margin). At update 100 acquisition was one item behind SF1 (11 vs 12 correct), i.e., a small early
  cost of the constraint, but the run continued to 200 and reached 15/16 — it did not reach the pre-registered
  all-AND endpoint (16/16 correct AND 16/16 exact AND 8/8 reversals AND 4/4 families). The single residual error
  is an Alex-correct item now producing Owen (over-generalization after the default was removed), not the old
  Alex default.
- Preregistered signatures: S1 MET, S2 MET, S3 NOT MET (11/16@100, 15/16@200), S4 PARTIAL (by-200 clauses met),
  S5 MET. Falsifiers F1-F4 not observed (F4 not applicable).
- Verdict: the evidence SUPPORTS the KL-retention mechanism for the demonstrated harms (pollution, distributed
  language damage, and the Alex/Owen default), and shows conditional factual learning is achievable under the
  constraint; the pre-registered acquisition endpoint was not established within the 200-update budget. This is
  training-family evidence only; all transfer panels remained locked and no held-out/generalization claim is made.
