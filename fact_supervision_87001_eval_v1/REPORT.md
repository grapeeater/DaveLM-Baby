Seed-87002 first matched-pair evaluation

Primary results:
{
  "Pilot1 parent": {
    "items": 192,
    "correct": 95,
    "accuracy": 0.4947916666666667,
    "families_complete": 0,
    "families_total": 24,
    "reversal_pairs_counted": 96,
    "reversal_both_correct": 5,
    "reversal_rate": 0.052083333333333336,
    "mean_margin": 0.08391453232616186,
    "greedy_exact": 0
  },
  "Factual": {
    "items": 192,
    "correct": 97,
    "accuracy": 0.5052083333333334,
    "families_complete": 0,
    "families_total": 24,
    "reversal_pairs_counted": 96,
    "reversal_both_correct": 22,
    "reversal_rate": 0.22916666666666666,
    "mean_margin": 0.003403881709800771,
    "greedy_exact": 74
  },
  "Control": {
    "items": 192,
    "correct": 97,
    "accuracy": 0.5052083333333334,
    "families_complete": 0,
    "families_total": 24,
    "reversal_pairs_counted": 96,
    "reversal_both_correct": 11,
    "reversal_rate": 0.11458333333333333,
    "mean_margin": -0.021204435219412215,
    "greedy_exact": 88
  }
}
Object/predicate strata: see STRATA.json.

Binding: all three checkpoints scored 80/80 answer exactness, 80/80 BOTH_DISTINCT, zero collapse on each frozen 80-document pool.

Acquisition classification: FAIL. Factual accuracy 48/96 object and 49/96 predicate; reversal 13/48 and 9/48; complete families 0/12; exact greedy 36/96 and 38/96. Factual-control reversal advantages 4/48 and 7/48. Factual-parent advantages 11/48 and 6/48.

TinyStories and naturalistic endpoint diagnostics were unavailable in the frozen v8 controller and were not substituted.
