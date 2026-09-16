# Fan Diesel handoff — v2R4 isolation diagnostic

Copy/paste for Codex on Fan Diesel. This is read-only. It does not train Baby.

## Preconditions

1. Workspace: `C:\DaveLM-v0.10` (or the current v0.10 checkout that contains this branch).
2. Confirm v2R5 is **not** writing `runs\structured_v2r4_seed106001_from6000_terminal\`.
3. Confirm the terminal checkpoint exists and hashes.
4. Do not open TEST/FINAL/SACRED.
5. Do not launch training.
6. v2R5 remains unresolved unless a sealed receipt is already on disk; do not guess.

## 1. Data-only autopsy (no weights)

```powershell
$env:PYTHONHASHSEED=0
python -B -m src.baby_v010.autopsy_v2r4 `
  --metrics runs\structured_v2r4_seed106001_from6000_terminal\metrics.jsonl `
  --panels data\generated\foundation_v2\panels.json `
  --out runs\v2r4_isolation_autopsy
```

Expect `status: V2R4_INDEPENDENT_AUTOPSY` and the locked census in `research\V010_V2R4_INDEPENDENT_AUTOPSY.md` (primitive keyed all `pair_count=1`; held-out value 23/96 exact 0/96; 3-pair novel 7/21).

## 2. Write isolation copies (does not touch frozen panels.json)

```powershell
python -B -m src.baby_v010.isolation_transforms `
  --panels data\generated\foundation_v2\panels.json `
  --out runs\v2r4_isolation_panels
```

After this, `git status` must still show `data\generated\foundation_v2\panels.json` unmodified.

## 3. Checkpoint probe (requires local weights)

```powershell
certutil -hashfile runs\structured_v2r4_seed106001_from6000_terminal\checkpoint_16000.pt SHA256
python -B -m src.baby_v010.diagnose_checkpoint `
  --checkpoint runs\structured_v2r4_seed106001_from6000_terminal\checkpoint_16000.pt `
  --expect-sha256 94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827 `
  --panels data\generated\foundation_v2\panels.json `
  --out runs\v2r4_isolation_probe `
  --device cuda
```

If the checkpoint is absent, the command writes `CHECKPOINT_ABSENT` and exits 0. That is not a capability result.

If the SHA does not match, the command refuses to score.

## Bring back

From `runs\v2r4_isolation_probe\PROBE.json`:

- `same_surface_novel_pair_count_2/3/4` `value_ok_rate` versus `chance_1_over_k`
- `short_keyed_pair_count_1/2` `value_ok_rate`
- `query_swap_same_surface_novel` and `query_swap_short_keyed`: `query_swap_follow_new_value` and `emitted_original_value_span`
- `marker_swap_keep_sep_same_surface_novel` `value_ok_rate` vs parent novel 41/96
- `sep_swap_keep_markers_same_surface_novel` `value_ok_rate` and `free_exact_rate`
- `value_absent_*` `value_ok_rate` versus broken-context 16/64
- `body_reorder_query_first_same_surface_novel` / `query_last` `value_ok_rate`

Also confirm: no training occurred; frozen panels hash unchanged; v2R5 still marked pending unless you attach a real sealed receipt.

### How to read the new fields

- `inventory copy` / competitor vs queried: already computable from the data-only autopsy `emission_source` block; expect novel 41/52/3.
- Body-reorder: compare `body_reorder_query_first_same_surface_novel.value_ok_rate` to parent novel 41/96 restricted to the same moved items. A large lift supports a first-slot prior. A null supports missing query-key binding rather than slot bias.
- Query-swap: follow-new-value is the binding test. Stuck-on-old-value is copy-without-query.

## Do not do

- v2R6 / seed 106002 / another 16k train
- weaken Gate C because separators were OOD
- retokenize, enable sidecar, or open protected eval
- point `--out` at a live v2R5 run directory
