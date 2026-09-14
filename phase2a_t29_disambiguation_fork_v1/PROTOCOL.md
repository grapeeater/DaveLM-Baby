# T29 disambiguation-fork protocol (frozen)

Authorized by `TREATMENT No29.txt`. One study, three seeds, no post-hoc lambda/margin tuning.

## Hypothesis

Baby often has enough downstream capacity to finish the correct answer if she takes the correct native token at the first identity-disambiguating fork. Train that decision. Do not redesign the tokenizer. Do not open TEST. Do not use constrained decoding.

## Parent and recipe

- Parent: Phase1G U6000 `best.pt` SHA-256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`
- Tokenizer: frozen v0_7
- Mechanical runner: T24/T28-fast teacher-forced path
- T28 suffix-vocab margin: **off**
- Inherited T24 unlikelihood: on (not the T29 question)
- LR `3.75e-5`, 1000 updates, phase A then U750 revert, 9:1 language, batch 32
- Seeds `860001`, `860002`, `860003` with T28 batch schedules remapped 1:1

## L_fork

`relu(1.0 - logit(gold) + max competitor logits)` at one index, `lambda_fork=0.5`.

Fork rule (TRAIN only):

1. Identities = T3 `audits.train_names` (16). DEV/TEST never used to build the rule.
2. A row has a fork only if another TRAIN identity shares a **nonempty** token prefix.
3. `fork_index` is the first token that uniquely commits gold among TRAIN identities.
4. Competitors are those prefix-compatible next tokens, not the full vocabulary.
5. Unique-first-token TRAIN names get **no** invented fork.

Same rule is applied to DEV identities **only at evaluation**.

## Gates

Unchanged T24/T18 gates. Exact native bar remains 80/128. TEST stays sealed unless those gates pass on DEV with 2/3 replication, which this protocol does **not** pre-authorize.

## After terminal

Adjudicate, report, update canonical docs, push safe artifacts. T30 is not authorized by this treatment.
