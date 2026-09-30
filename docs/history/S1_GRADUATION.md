# Baby's S1 graduation

**Result: S1 PASSED — BABY GRADUATED THIS STAGE.**

On September 30, 2026, Baby passed the frozen `S1_single_token_copy`
curriculum and its certification at update **u4700**. This is a major
foundational milestone. It is not completion of the full foundation, and it
does not declare Mangaris Pioneer v1.0 complete or released.

## Model and run

| Field | Value |
| --- | --- |
| Project | DaveLM |
| Family | Mangaris |
| Codename | Baby |
| Architecture lineage | MRCN-Alpha, 118M-parameter research line |
| Graduated stage | `S1_single_token_copy` |
| Formal run | `S1v3j_single_token_copy_anneal_from_s1v3h_001` |
| Graduating update | u4700 |
| Continuation updates | 700 (u4001–u4700) |
| Stop rule | Frozen first gate pass and required certification |

The treatment stopped automatically after the first passing evaluation and
certification. Updates u4701–u4800 were not run.

## Frozen gate results

Primary evaluation and certification both passed every frozen gate:

| Panel | Metric | Threshold | Primary | Certification | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Heldout combinations | Teacher-forced content | ≥0.99 | 1.00 (128/128) | 1.00 (128/128) | PASS |
| Heldout combinations | EOS | ≥0.99 | 1.00 (128/128) | 1.00 (128/128) | PASS |
| Heldout combinations | Free exact, including EOS | ≥0.95 | 1.00 (128/128) | 1.00 (128/128) | PASS |
| Heldout combinations | Counterfactual pairs both exact | ≥0.95 | 1.00 (64/64) | 1.00 (64/64) | PASS |
| Novel positions | Teacher-forced content | ≥0.99 | 1.00 (128/128) | 1.00 (128/128) | PASS |
| Novel positions | EOS | ≥0.99 | 1.00 (128/128) | 1.00 (128/128) | PASS |
| Novel positions | Free exact, including EOS | ≥0.95 | 1.00 (128/128) | 1.00 (128/128) | PASS |
| Novel positions | Counterfactual pairs both exact | ≥0.95 | 1.00 (64/64) | 1.00 (64/64) | PASS |

All 12 novel-position errors recorded for the S1v3h parent were resolved by
u4700. Identities 802, 501, 312, 645, and 901 remained correct through every
scheduled evaluation and certification.

The nongated identity-stress panel remains unsolved: content/free exact was
0/128 and counterfactual pairs were 0/64; EOS was 128/128. This preserved
frontier is not an S1 gate failure and must not be represented as solved.

## Canonical lineage

```text
S1v3g u3200 ──┬──→ S1v3h attempt 005 u4000 ──→ S1v3j revision 2 u4700 (graduate)
              └──→ S1v3i u3200–u4000 (failed sister arm; not an ancestor)
```

The authorized parent was S1v3h attempt 005 at u4000, checkpoint SHA256:

```text
e5fd823f820e6c0bb30ef54bc5c7c10b271edcc0077c3fba2102469dd9e0681c
```

The S1v3h parent traces back to S1v3g u3200, checkpoint SHA256:

```text
35d53511286fd8dc80792ca4d78692fdf37f770c4989963025245487fa5df949
```

S1v3i was a failed cosine sister arm from S1v3g u3200. It is retained as
history and is not part of the graduating ancestry.

## Canonical checkpoint

The canonical master is in the separate `DaveLM-Training` lab tree. Its
path below is relative to that tree; the checkpoint is not copied into this
legacy museum repository.

```text
rebuild/baby_reincarnation_118m_001/runs/S1v3j_single_token_copy_anneal_from_s1v3h_001/S1_single_token_copy_graduated.pt

Stage: S1_single_token_copy
Update: 4700
Size: 1,418,883,555 bytes
SHA256: 200e02188063885560edb4de0d6e0f05054fa75c236c0264d4a2c461ec51771b
```

The final model and optimizer tensors were finite; optimizer, RNG, and
continuation state were present. Full restoration verification passed. Returned
run artifacts and checkpoints were hash-verified. No safety abort occurred.

## Interpretation and limits

The result shows that this frozen run met the S1 gates. The late annealing
branch resolved the parent's observed novel failures, but this single
comparison cannot isolate learning-rate decay from additional training or
establish a universal optimization mechanism. Later foundation stages and any
Pioneer class/release decision require separate evidence and approval.

The old tagged 61.5M DaveLM v1.0 museum snapshot in this Git repository is a
separate historical line. It is neither the 118M S1 graduate nor Mangaris
Pioneer v1.0.
