# T24X generation fail-mode — final

Read-only diagnostic on T24 U1000 checkpoints, using the authorized qa_dev panel only. TEST, FINAL, and sacred material remained sealed.

## Result

| Seed | Exact | First target token | First token not target | First-token-correct then diverge | Other then EOS | Wrong candidate |
|---:|---:|---:|---:|---:|---:|---:|
| 830001 | 35/128 | 81 | 47 | 46 | 26 | 21 |
| 830002 | 40/128 | 87 | 41 | 47 | 25 | 16 |
| 830003 | 37/128 | 82 | 46 | 45 | 31 | 15 |

The dominant residual mode is a correct first answer token followed by continuation divergence, with a secondary other_then_eos mode. T24 inventory unlikelihood therefore did not resolve the established native output/continuation failure.

## Integrity

No weights were modified; no optimizer or training step ran. The only repairs were mechanical import and schema-string fixes in the new T24X copy. T24 historical artifacts were untouched.

Classification: T24X_GENERATION_FAILMODE_COMPLETE
