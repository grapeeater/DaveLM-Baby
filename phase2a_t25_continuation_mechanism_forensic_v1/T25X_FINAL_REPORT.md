# T25 continuation mechanism forensic — final

Read-only, authorized qa_dev panel only. No training or optimizer update. TEST, FINAL, and sacred materials remained sealed.

## Frozen comparison

| Seed | Teacher-forced target top-1 | Free-running target top-1 | Gap |
|---:|---:|---:|---:|
| 830001 | 338/608 | 252/608 | 0.141 |
| 830002 | 353/608 | 268/608 | 0.14 |
| 830003 | 348/608 | 257/608 | 0.15 |

Teacher-forced continuation remains materially more accurate than free-running continuation across all three seeds. This supports an autoregressive exposure/continuation gap as the immediate limiting phenomenon. It does not distinguish whether the gap is caused by readout competition, EOS calibration, or broader exposure bias; the next treatment should therefore target continuation while preserving the existing relational representation machinery.

Classification: `T25_AUTOREGRESSIVE_CONTINUATION_GAP_SUPPORTED`
