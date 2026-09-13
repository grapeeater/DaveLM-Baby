# SF15 FINAL CLASSIFICATION

**PARTIAL_FIRST_TOKEN_CE_INSUFFICIENT**

# WHAT CHANGED

The sole scientific change from sealed SF14 was first-answer-name causal CE weight `0.0 -> 0.25` during English updates. Per-record response weights were `[0.25, 1, 1, 1, 1]`, normalized by their sum. Later response/punctuation/EOS CE, first-token margin, broad KL, data, optimizer, scopes, schedule, binding, and gates were unchanged.

# PER-SEED TRAJECTORY

| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order c/e (fact/copy exact) | Language CE/PPL | D3 | Binding |
|---:|---:|---:|---:|---:|---:|---:|---|
| 87038 | 0 | 16/16/8/4 | 12/5 | 9/3 (3/0) | 3.5873/36.14 | 0.00704 | pilot0:80/80/collapse0/quartets20/rev40; pilot1:80/80/collapse0/quartets20/rev40 |
| 87038 | 100 | 16/16/8/4 | 14/9 | 9/5 (4/1) | 3.6475/38.38 | 0.01133 | pilot0:80/80/collapse0/quartets20/rev40; pilot1:80/80/collapse0/quartets20/rev40 |
| 87038 | 200 | — | — | — | — | — | NOT REACHED: frozen D3 stop at U100 |
| 87039 | 0 | 16/16/8/4 | 13/4 | 9/3 (3/0) | 3.5746/35.68 | 0.00667 | pilot0:80/80/collapse0/quartets20/rev40; pilot1:80/80/collapse0/quartets20/rev40 |
| 87039 | 100 | 16/16/8/4 | 14/10 | 9/5 (4/1) | 3.6511/38.52 | 0.01137 | pilot0:80/80/collapse0/quartets20/rev40; pilot1:80/80/collapse0/quartets20/rev40 |
| 87039 | 200 | — | — | — | — | — | NOT REACHED: frozen D3 stop at U100 |
| 87040 | 0 | 16/16/8/4 | 13/5 | 8/3 (3/0) | 3.5950/36.41 | 0.00784 | pilot0:80/80/collapse0/quartets20/rev40; pilot1:80/80/collapse0/quartets20/rev40 |
| 87040 | 100 | 16/16/8/4 | 14/8 | 9/5 (4/1) | 3.6224/37.43 | 0.00997 | pilot0:80/80/collapse0/quartets20/rev40; pilot1:80/80/collapse0/quartets20/rev40 |
| 87040 | 200 | 16/16/8/4 | 14/10 | 10/6 (4/2) | 3.6399/38.09 | 0.01455 | pilot0:80/80/collapse0/quartets20/rev40; pilot1:80/80/collapse0/quartets20/rev40 |

# U100 OH-FUCK SIGNAL

- Seed 87038: D3 `0.01133` (FAIL); TRAIN16 `16/16/8/4`; language `PASS` (`+0.0602` nat); binding `PASS`; surface exact `5->9`; order exact `3->5`.
- Seed 87039: D3 `0.01137` (FAIL); TRAIN16 `16/16/8/4`; language `PASS` (`+0.0765` nat); binding `PASS`; surface exact `4->10`; order exact `3->5`.
- Seed 87040: D3 `0.00997` (PASS); TRAIN16 `16/16/8/4`; language `PASS` (`+0.0275` nat); binding `PASS`; surface exact `5->8`; order exact `3->5`.

# FROZEN GATE RESULT

- Seed 87038 at U100: TRAIN16 `PASS`; language `PASS`; binding `PASS`; D3 `FAIL`; surface `FAIL`; order `FAIL`; complete endpoint `FAIL`.
- Seed 87039 at U100: TRAIN16 `PASS`; language `PASS`; binding `PASS`; D3 `FAIL`; surface `FAIL`; order `FAIL`; complete endpoint `FAIL`.
- Seed 87040 at U200: TRAIN16 `PASS`; language `PASS`; binding `PASS`; D3 `FAIL`; surface `FAIL`; order `FAIL`; complete endpoint `FAIL`.

- Complete endpoint passes: 0/3.
- D3 failures: 3/3.
- TRAIN16/language/binding retention failures: 0/3.
- Surface and order exact scores improved over U0 in 3/3 branches.

# SCIENTIFIC INTERPRETATION

The 0.25 restoration recovered much of the exact-generation pressure lost in SF14, but it did not preserve the frozen D3 bound reproducibly. Seeds 87038 and 87039 crossed D3 at U100; seed 87040 was narrowly green at U100 and crossed by U200. All tested original acquisition, language, and binding retention remained intact. This supports only the narrow conclusion that weight 0.25 did not resolve the observed D3-versus-generation tradeoff under SF15. It does not establish an optimal CE weight, sole causation, a scaling law, or a capacity limit.

# CHECKPOINTS

- Seed 87038 (100 updates): `C:\DaveLM-CADAVER\sf15_partial_first_token_ce_v1\runs\seed_87038_curriculum\checkpoint_100.pt` — `c327af65cf020954c91b30d012104a6d1f1de030c414fd9467f696dc9e257170`
- Seed 87039 (100 updates): `C:\DaveLM-CADAVER\sf15_partial_first_token_ce_v1\runs\seed_87039_curriculum\checkpoint_100.pt` — `351e6a0b4e499413058dbdadbf9fee70473bf17c9991b5b94d7fdd90af405970`
- Seed 87040 (200 updates): `C:\DaveLM-CADAVER\sf15_partial_first_token_ce_v1\runs\seed_87040_curriculum\checkpoint_200.pt` — `b81993a4e209b22bed85f2b3dc707193dfb28dd0e958d1204b1f57e46776b952`

# PROVENANCE AND LOCKS

- Sealed bundle receipt: `9fa6c2f591c970cc6837daefe403e78c9f5dfbdc61f5bbcd4672cd3639c6c424`
- Sealed payload manifest: `11ef8fdb73de3f1de1309bc1c653154f76af0f79be58080344dfa197ab4f718f`
- Frozen controller: `aac8f28129b2804dd39f38ae002dfba7ec82ca28440122b5b9687537731dc71c`
- Historical transfer panels: LOCKED/UNSCORED.
- FINAL and sacred material: not accessed.
- Reporting correction only: the sealed reporter omitted the separate U100 row for the U200 branch; this report reads the preserved raw artifacts and changes no metrics, gates, checkpoints, or classification.
