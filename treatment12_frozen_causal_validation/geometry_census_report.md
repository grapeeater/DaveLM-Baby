# T12/T11 frozen geometry census

Outcome-blind structural census. No model, logits, hidden states, or outcomes used.

## Provenance

- train documents: 800
- retention documents: 160
- train pool sha256: `8f5d60de10fe2fdc8e772a9c1fc3e9f07861edd1583d7c413a095c2f55c6903c`
- retention pool sha256: `c341d7308b145bd3c633b62d56e01391f4b05635bfcf2edfb146bcd9f2de4e69`

## Unique layout signatures

| set | count |
|---|---|
| train layout signatures | 20 |
| retention layout signatures | 20 |
| retention signatures also seen in train | 20 |
| retention signatures NOT seen in train | 0 |

Full signatures (layout + query slot + orientation + mapping order): train=80, retention=80, retention-not-in-train=0.

## Absolute position reuse

| variable | train distinct | retention distinct | retention missing from train |
|---|---|---|---|
| q | 8 | 8 | [] |
| answer_causal | 8 | 8 | [] |
| k0 | 5 | 5 | [] |
| v0 | 5 | 5 | [] |
| k1 | 5 | 5 | [] |
| v1 | 5 | 5 | [] |

All absolute q/k/v/answer positions reused in retention: True

## Relative distance reuse

| variable | train distinct | retention distinct | retention missing |
|---|---|---|---|
| q_to_k0 | 4 | 4 | [] |
| q_to_k1 | 4 | 4 | [] |
| q_to_v0 | 4 | 4 | [] |
| q_to_v1 | 4 | 4 | [] |
| k0_to_v0 | 1 | 1 | [] |
| k1_to_v1 | 1 | 1 | [] |
| row0_to_row1_spacing | 1 | 1 | [] |
| row0_val_to_row1_val_spacing | 1 | 1 | [] |
| query_to_answer | 1 | 1 | [] |

Relative-distance combination reuse: True

## Deterministic geometry relationships (train only)

- layout signatures total: 20
- signatures with a single query slot: 0 -> query_slot deterministic given layout: False
- signatures with a single orientation: 0 -> orientation deterministic given layout: False
- distinct target-row values per layout signature (set sizes): [2]

Retention contains genuinely unseen layout geometry: False
