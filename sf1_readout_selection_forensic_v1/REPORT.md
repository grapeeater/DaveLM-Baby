# SF1 readout-selection forensic audit v1

This is one read-only audit of the authoritative Pilot1 parent and preserved
SF1 update100 checkpoint. It used only the two checkpoints, SF1 `TRAIN.json`,
and the first 128 records of the authorized aligned TinyStories DEV source.
No optimizer, gradient, weight mutation, locked transfer/copy panel, readiness
FINAL, or sacred material was used.

## ESTABLISHED

The inputs and identities were verified before comparison:

- Pilot1: `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt`, SHA-256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`.
- SF1 update100: `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011_run\checkpoint_100.pt`, SHA-256 `550ce4306b450c2bdb64e57314e45ec7db5bb2ee68fc0978ed9d3617f4fc1e7e`.
- Tokenizer: `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json`, SHA-256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.
- Aligned DEV source: `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\data\ENGLISH_DEV.jsonl`, SHA-256 `2053a8036d7d0f02ba56ab28f739df95a1743760d0f8c980e112d6667b38104e`.
- Runtime: Python 3.12.14, torch 2.12.0+rocm7.14.0, tokenizers 0.23.1, CUDA device; complete input/source hashes are in `PROVENANCE.json`.

The model is `Treatment13Model` with 8 decoder blocks, 320-dimensional
embeddings, vocabulary 1024, 256 positions, and an untied output head. The
base model has 10,594,944 parameters and the complete T13 wrapper has
10,841,346 parameters. The embedding and output-head sizes are 327,680 and
328,704 parameters respectively. The architecture and tied-weight check are
in `ARCHITECTURE.json`.

SF1 English updates trained embeddings, positions, blocks 4–7, final norm, and
the language head; blocks 0–3 and specialized localization/retrieval were
frozen for those updates. Binding updates restored the full 334-parameter-set
scope. Consequently, the endpoint delta includes binding-update drift in
every group and cannot be used to isolate English-only drift. `PARAMETER_SCOPE.json`
and `OPTIMIZER_METADATA.json` preserve the authoritative scope and AdamW
metadata (lr 5e-5, betas 0.9/0.999, eps 1e-8, weight decay .05, no scheduler).

### D1 parameter delta census

Aggregate endpoint relative deltas (Pilot1 -> SF1) were:

| group | relative delta | delta L2 | RMS delta |
|---|---:|---:|---:|
| blocks 0–3 combined | 0.000368 | 0.42369 | 0.000151 |
| blocks 4–7 combined | 0.003041 | 3.50061 | 0.001248 |
| embeddings | 0.000594 | 0.32553 | 0.000569 |
| positional | 0.000989 | 0.26112 | 0.000912 |
| final norm | 0.003381 | 0.06682 | 0.002641 |
| output head | 0.061789 | 1.86661 | 0.003256 |
| localizer/retrieval | 0.006895 | 0.11088 | 0.000223 |

The complete per-tensor census and every block/attention/MLP subgroup are in
`PARAMETER_DELTAS.jsonl` and `PARAMETER_AGGREGATES.json`. The smaller early
block drift is consistent with English-time protection, but it is not proof
that those blocks were unchanged during the intervening binding updates.

The output-head row audit did not show four name rows dominating the vocabulary:
the all-row delta-L2 median was 0.06018 (p10 0.04080, p90 0.06954). First-token
rows were at approximately the 34th percentile (Alex), 47th (Owen), 40th
(Mia), and 34th (Nora). All name-token rows and deterministic common/rare DEV
controls are in `OUTPUT_HEAD_ROW_AUDIT.json`.

### D3 unrelated-context prior audit

Exactly 256 positions were selected deterministically and hashed before model
loading. The rule and the selected IDs are in `D3_SELECTION.json` and
`D3_SELECTION_RECEIPT.json`.

On these contexts, mean first-token logit changes for the four trained names
were Alex +4.197, Owen +4.818, Mia +3.561, and Nora +4.475. Their mean rank
changes were −347, −430, −217, and −382, respectively. Mean combined
four-name probability rose from 0.000907 to 0.067833. Entropy fell from 3.003
to 2.491, EOS probability rose from 0.000220 to 0.004604, and mean true-next
token probability fell from 0.1401 to 0.0794.

The same unrelated contexts produced heterogeneous control changes: common
control-token logit deltas ranged from −1.189 to +1.938 (the period token's
probability rose by 0.176 on average), while rare controls rose by about
1.027–2.392 logits with tiny probability changes. This establishes a broad
output-distribution/name-prior shift on unrelated language positions; it does
not establish that the change is context-conditioned or relational.

### D4 SF1 training-item readout

The 16 allowed SF1 TRAIN records were scored only through `base_model` with
the full four-token candidate sequence likelihood. Pilot1 had 9/16 positive
correct-minus-distractor margins and 0/16 full-vocabulary first-token wins.
SF1 had 12/16 positive margins and 12/16 full-vocabulary wins.

| subgroup | Pilot1 positive / n | SF1 positive / n | SF1 full-vocabulary top-1 |
|---|---:|---:|---:|
| Alex/Owen | 5/8 | 4/8 | 4/8 |
| Mia/Nora | 4/8 | 8/8 | 8/8 |
| Alex-correct | 4/4 | 4/4 | 4/4 |
| Owen-correct | 1/4 | 0/4 | 0/4 |

On Owen-correct rows, SF1 increased Owen's first-token logit by 12.150 on
average and its probability by 0.2665, but all four still selected Alex as
full-vocabulary top-1 and all four retained a negative sequence margin
(mean −0.2730). Under the preregistered descriptive rule this is
`STAGE_B_SIGNAL_INCREASED_BUT_SUPPRESSED`. The raw item tables are
`D4_PARENT.jsonl` and `D4_SF1.jsonl`; the complete comparison is in
`D4_SUMMARY.json`.

### Language-loss decomposition

Correctly aligned causal CE was recomputed on 3,796 supervised tokens from the
same first 128 DEV rows. SF1−Pilot1 delta CE had mean +1.2624, median +1.2451,
standard deviation 1.8656, p10 −1.0136, p25 +0.1767, p75 +2.3997, p90
+3.5426, p95 +4.3171, and p99 +6.0493. 21.6% of tokens improved and 78.4%
worsened. The largest increases and improvements are recorded with their row
contexts in `LANGUAGE_LOSS_SUMMARY.json`; raw token-level records are in
`LANGUAGE_LOSS.jsonl`. The correlation between D3 combined-name-mass change
and joined DEV token CE change was 0.1441.

## SUPPORTED

- SF1 made its largest endpoint changes in the output head and blocks 4–7,
  with much smaller aggregate changes in blocks 0–3; this is compatible with,
  but does not prove, the documented English-time mask because the endpoint
  also contains binding updates.
- SF1 produced a broad unrelated-context increase in the four trained-name
  logits/ranks. Controls changed nonuniformly, so the effect cannot be reduced
  to a name-only or context-specific shift from this sample.
- On the 16 training items, the Mia/Nora pair became strongly separable, while
  the Alex/Owen pair retained an Alex default. Owen evidence increased but was
  suppressed in the exact sense defined above.
- The aligned TinyStories regression is broad across this 128-row sample
  (most supervised tokens worsened), with a mixed tail of improvements; it is
  not an isolated single-token effect.

## NOT ESTABLISHED

- These measurements do not establish relational generalization, held-out
  transfer, general English competence, conversational ability, architectural
  impossibility, or a causal mechanism for the SF1 failure.
- Training-item D4 logits describe evidence on the training records only; they
  cannot establish genuine relational generalization.
- The endpoint deltas cannot identify whether changes occurred on English or
  binding updates, and no gradient/Adam diagnostic or component swap was run.
- The deterministic D3 sample and output-head audit do not prove absence of
  semantic or near-duplicate contamination beyond the authorized inputs.

All raw data, scripts, source/input hashes, and the non-circular output receipt
are in this directory. No treatment was recommended or executed.

SF1_READOUT_SELECTION_FORENSIC_COMPLETE
