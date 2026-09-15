# Baby v0.10 curriculum v2

Protocol: `BABY_V010_FOUNDATION_V2`  
Status: frozen before the v2 training runs

## Why v1R2 failed

Both v1R2 seeds passed language DEV CE but failed every contextual-copy gate.
The structured loss was finite, yet target ranks stayed far from one and greedy
outputs collapsed to a small set of structural tokens. The v1R2 generator
asked a fresh model to solve multi-key, multi-token retrieval, variable-length
contexts, induction, low-prior targets, and changing wrappers concurrently.
The next hypothesis is therefore about curriculum difficulty, not about
weakening the final test or replacing the architecture.

## Frozen stages

| updates | stage | structured distribution | purpose |
|---:|---|---|---|
| 0-499 | language warmup | language CE only | establish ordinary LM and stable representations |
| 500-1799 | primitive induction | 80% one-token induction, 20% one-key retrieval; short random spans and many train wrappers | establish contextual identity transport |
| 1800-3499 | short retrieval | 35% induction, 65% one/two-key retrieval; 1-4 token answers | add content-addressed selection without full entropy |
| 3500-5999 | full foundation | 25% induction, 75% multi-key/multi-token retrieval; variable lengths, low-prior items | expand the capability while retaining the primitive |

After warmup, 80% of updates are structured and 20% remain language CE. The
optimizer, model family, tokenizer, and final held-out gates are unchanged in
kind from v1R2; only the pre-gate training scaffold and the frozen v2 panels
change. Both runs start from fresh random initialization and no v1R2
checkpoint is reused.

## Anti-shortcut controls

- Values are newly sampled from the language-derived span bank for each training
  item; no finite answer table is supplied.
- Every value span has no internal bigram supported by the frozen language
  stream, and keyed target spans are unique within the frozen final panels.
- Keys, values, markers, and separators occupy disjoint roles.
- Train and held-out surface marker/separator IDs are disjoint, including from
  key and filler roles; the panel audit checks both directions.
- Source lengths, query positions, wrapper variants, filler placement, key
  order, number of records, and answer lengths vary.
- The final panels include same-surface controls, unseen lengths, low-prior
  values, distractor records, broken-context controls, and broken-order
  controls. No final panel is used as a training target.

## Falsification condition for this iteration

If both fresh seeds again pass language modeling but fail the primitive
induction panel, the staged-scaffold hypothesis is not supported and the next
decision must be based on a mechanistic/optimization study rather than another
blind increase in structured-task complexity. A final foundational pass still
requires the frozen gates in `design/V010_CAPABILITY_GATES.md`; no intermediate
stage is allowed to substitute for them.
