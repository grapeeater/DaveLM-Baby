# SF3 prospective treatment v1 — final report

## Headline

**SF3 stopped at its mandatory update-100 replay gate before the treatment variable activated.** The replay
matched SF2's visible TRAIN acquisition counts exactly (11/16 correct, 10/16 exact, 3/8 reversal pairs, 1/4
complete families), but it did not reproduce the preserved SF2 update-100 model tensors exactly: 334/414 tensors
differed, with maximum absolute difference `0.0009046793`. The frozen protocol required exact equality. No
post-100 annealed English update ran, so there is no valid SF3 treatment outcome and no basis to compare the
annealing hypothesis against SF2.

Classification: **`SF3_HARD_STOP_UPDATE100_REPLAY_MISMATCH`**.

## Chosen treatment and why

The prospective treatment changed only the English learning rate after global update 100. Updates 1-100 were to
replay SF2 at `5e-5`. The remaining 90 English learning rates were frozen to decrease linearly from `5e-5` to
zero; binding updates remained at `5e-5`. SF2's data, forward parent-KL, causal CE, binding objective, parameter
scope, optimizer, seed, batching, schedule, gates, and transfer lock were unchanged.

Baby's own residual autopsy controlled the choice: SF2 recovered all four Owen-correct items but ended with one
Alex-correct item only `-0.030` nats across the decision boundary, while all four Alex/Owen cell midpoints had
moved mildly toward Owen. Every item appeared twice in every batch, excluding exposure/recency imbalance. A
late step-size test was therefore narrower than changing the objective, head, data mixture, or architecture.

Alternatives were rejected prospectively:

- Head freezing was less direct because upstream state caused most boundary movement and the head only tipped
  the last near-tie.
- Pairwise/contrastive loss changed objective semantics without evidence that SF2 lacked discrimination.
- More constant-rate CE risked continuing the same Owen-side sweep.
- Language replay was not targeted because SF2 already retained language and D3 substantially better than SF1.
- Modified/staged KL changed a successful retention mechanism without evidence that KL caused the residual.
- Architecture/tokenizer/norm changes were unsupported.

Comparative-model anatomy was background only; it did not select the treatment.

## Preflight

Static and update-0 checks passed before execution:

- Pilot1 SHA-256: `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`.
- Tokenizer SHA-256: `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.
- Runtime: Python 3.12.14, torch 2.12.0+rocm7.14.0, tokenizers 0.23.1, AMD Radeon RX 9060 XT.
- Copied SF1 receipt/manifest and SF2 KL pool/manifest: PASS.
- Materialized schedule: 200/200 records; 180 English, 20 binding; all identities resolved.
- Correct causal masking: four answer tokens plus EOS supervised; context/padding ignored.
- Scope, binding implementation, KL teacher separation, controller integrity, and transfer locks: PASS.
- Update 0: 9/16 correct, 0/16 exact; aligned TinyStories CE `3.3906604`, PPL `29.6856`;
  D3 four-name mass `0.00090724`; each binding pool 80/80 answer, 80/80 BOTH_DISTINCT, zero collapse.
- Sealed receipt SHA-256: `bb677918631a3143d9e3bf86dfc45e0cf6eeddc134acb7ce3df05109fb632f35`.
- Sealed manifest SHA-256: `d257582b042b6a452cfe39c1fd975c4488515aaf04edef9fbf2df6e4ff01b835`.

The protocol prospectively deferred one indispensable determinism check until update 100 because SF2 did not
preserve its update-100 optimizer/RNG state. It required replaying the first half from Pilot1 and comparing all
model tensors before the new schedule could begin.

## Replay result and frozen gates

| Evidence/gate | Pilot1 / historical | SF1 u100 | SF2 | SF3 replay | Result |
|---|---:|---:|---:|---:|---|
| TRAIN correct | 9/16 | 12/16 | 11/16 at u100; 15/16 at u200 | 11/16 at u100 | behavioral count matches SF2 u100 |
| Exact answer+EOS | 0/16 | 12/16 | 10/16; 15/16 | 10/16 | behavioral count matches SF2 u100 |
| Reversal pairs | 1/8 | 4/8 | 3/8; 7/8 | 3/8 | behavioral count matches SF2 u100 |
| Complete families | 0/4 | 2/4 | 1/4; 3/4 | 1/4 | behavioral count matches SF2 u100 |
| Exact u100 tensor replay | n/a | n/a | reference | 334/414 mismatch | **FAIL / HARD STOP** |
| Update-200 acquisition | n/a | n/a | 15/16, 15/16, 7/8, 3/4 = FAIL | not reached | not evaluated |
| Language/D3/binding at u100 | parent values above | historical | SF2 passed | not run after hard stop | not evaluated |

The first numerical divergence in logged training metrics appeared at update 2 as a KL difference of roughly
`1.4e-9`; differences accumulated despite deterministic-algorithm mode. Across the first 100 records, only one
metric record was byte-numerically identical, mean per-record maximum numeric difference was about `0.000763`,
and the largest logged metric difference was about `0.01242`. At update 100, binding loss was `0.00683208` in
the replay versus `0.00683664` historically. This is consistent with numerically diverging deterministic replay,
but the present run does not establish its low-level cause.

The protocol forbade relaxing the equality criterion after seeing this result. Continuing would have mixed the
intended LR manipulation with an unreproduced optimizer/model trajectory.

## What is and is not learned

Established:

- The prospective design and update-0 baseline were internally valid and uncontaminated under the checks run.
- Re-executing the pinned first 100 updates reproduced SF2's aggregate TRAIN behavior, but not its tensors.
- The annealing treatment never activated; 100/200 replay updates were committed, zero annealed updates ran.
- No locked transfer/copy panel, FINAL, or sacred material was accessed.
- Pilot1 and the preserved SF2/SF1 artifacts remain unchanged. SF2 remains `SF2_ACQUISITION_FAIL`.

Not established:

- Whether late English LR annealing helps, hurts, or leaves the residual unchanged.
- An SF3 acquisition, language, D3, binding, or transfer result.
- Any held-out factual generalization, broader language competence, or architectural conclusion.

## Artifact identities

- SF3 directory: `C:\DaveLM-CADAVER\sf3_prospective_treatment_v1`.
- Replay checkpoint: `run\checkpoint_100.pt`, SHA-256
  `a5f83a666dd947b9b86159d94516c3cc650a79582034f0653f264e24377a3e0a`.
- Preserved SF2 update-100 reference: SHA-256
  `ec5ce8964855d2e735439291fbabcbb4d2caf4bc4ace84ae8f2d49363ae74ab4`.
- Controller: SHA-256 `fcd974e54fb64fb27655021551a62771c4a0ea01b137f42485976216a719b769`.
- Protocol: SHA-256 `7b92763f8d73967024ea9a9415a77dab48b7de49aaf073d7f613d38ebeb12e10`.

## Dave-coded interpretation

`DAVE: THE IDEA WAS CLEAN, BUT THE REPLAY WAS NOT IDENTICAL. THE NEW TREATMENT NEVER STARTED. DO NOT SCORE THIS AS
AN SF3 WIN OR LOSS, AND DO NOT MOVE THE GATE.`
