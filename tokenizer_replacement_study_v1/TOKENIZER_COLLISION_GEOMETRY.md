# Tokenizer collision geometry

DEV inventory (after freeze): Ava, Mel, Omar, Opal, Ross, Sal, Skye, Wes.
TRAIN inventory: Bea, Gia, Gus, Helen, Hope, Ida, Jan, Lois, Marie, Mark, Roy, Ruth, Seth, Vic, Walt, York.
Answer surface: leading space + name (T3 convention). `[sp]` is the byte-level space mark.

## DEV shared-first-token groups

| Candidate | Collision groups | Shared-name rate |
|---|---|---:|
| v0_7 | Omar/Opal ; Sal/Skye | 4/8 = 0.500 |
| bpe_bl_1536 | Omar/Opal ; Sal/Skye | 0.500 |
| bpe_bl_2048 | Omar/Opal ; Sal/Skye | 0.500 |
| bpe_bl_2048_prefix_space | Omar/Opal ; Sal/Skye | 0.500 |
| bpe_bl_4096 | Omar/Opal ; Sal/Skye | 0.500 |
| bpe_ws_2048 | Omar/Opal ; Sal/Skye | 0.500 |
| bpe_bl_2048_titlecase_x4 | Omar/Opal/**Wes** ; Sal/Skye | 5/8 = 0.625 |

No replacement removes the two DEV collision families. One candidate adds Wes to a collision family.

## Collision-group first disambiguating position (DEV)

Inventory-wide first-difference is 1 for every name (other names differ at token 1). The pathology metric is **within the first-token group**:

| Candidate | Mean group-conditional disambiguation | Omar/Opal | Sal/Skye | Wes |
|---|---:|---|---|---|
| v0_7 | 1.50 | 2 / 2 | 2 / 2 | 1 |
| bpe_ws_2048 | 1.50 | 2 / 2 | 2 / 2 | 1 |
| bpe_bl_4096 | 1.50 | 2 / 2 | 2 / 2 | 1 |
| bpe_bl_1536 / 2048 / prefix_space | 1.75 (worse) | 3 / 3 | 2 / 2 | 1 |
| titlecase_x4 | 1.625 (worse) | 2 / 2 | 2 / 2 | 2 |

Larger byte-level vocabs split Omar/Opal as `[sp]` + `O` + …, delaying disambiguation.

## Descriptive encodings (evaluation only; not used to train or rank)

| String | v0_7 | best new (`bpe_bl_1536`) | `bpe_ws_2048` | `bpe_bl_4096` |
|---|---|---|---|---|
| Sal | `[sp]S` `al` | same | `S` `al` | same as v0_7 |
| Salt | `[sp]S` `al` `t` | same | `S` `al` `t` | same as v0_7 |
| Skye | `[sp]S` `k` `y` `e` | same | `S` `k` `y` `e` | `[sp]S` `k` `ye` |
| Sky | `[sp]S` `k` `y` | same | `S` `k` `y` | same as v0_7 |
| Wes | `[sp]W` `es` | same | `W` `es` | same as v0_7 |
| Walt | `[sp]W` `al` `t` | same | `W` `al` `t` | same as v0_7 |
| Omar | `[sp]O` `m` `ar` | `[sp]` `O` `m` `ar` | `O` `mar` | same as v0_7 |
| Opal | `[sp]O` `p` `al` | `[sp]` `O` `p` `al` | `O` `p` `al` | same as v0_7 |

Salt remains a continuation of Sal. Sky remains a prefix of Skye. Walt remains a continuation of Wes’s first token `W`/`[sp]W`. Whitespace-first BPE makes Omar a 2-token `O`+`mar` while Opal stays `O`+`p`+`al` — still a shared first token.

## TRAIN vs DEV generalization

v0_7 TRAIN collisions are large because many TRAIN names share `[sp]` (Gia/Gus/Helen/Hope/Jan/Lois/Vic/York). Larger vocabs trained **on those TRAIN prompts** shrink TRAIN groups (e.g. 4096 and whitespace 2048: 0.750 → 0.375). DEV names never appear in the tokenizer corpus, so Sal/Skye/Omar/Opal geometry is essentially unchanged. That is in-sample TRAIN segmentation, not a general mouth redesign.

## Language compression (Phase1G TRAIN source)

v0_7: 2.201 bytes/token. All new candidates 3.45–4.00 (worse). Unknown counts on the name inventories were 0.

## Punctuation

Period remains a trailing singleton on `" " + name + "."` for v0_7 and the byte-level candidates.
