# Fact-supervision (seed 87002) Stage-0 read-only measurement

## Provenance verified

- parent_sha256: `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`
- tokenizer_sha256: `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`
- items_sha256: `2dab70739e9c0ffe1ad3154076ec613a10a4964fa1138e9c7b40d8a5e55f518d`
- schedule_sha256: `7855f07c245596a937fe758b0cca923b7ce5bb0c5bc5dc2405aaedea142521b0`
- factual_500_sha256: `3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123`
- v8_shasums_verified: True
- eval_shasums_verified: True
- raw_rows: `576`

## Primary recomputation (matches corrected authoritative record)

### Pilot1 parent
{
 "items": 192,
 "correct": 95,
 "accuracy": 0.4947916666666667,
 "ties": 0,
 "families_complete": 0,
 "families_total": 24,
 "reversal_pairs": 96,
 "reversal_both_correct": 5,
 "mean_margin": 0.08391453232616186,
 "greedy_exact": 0
}
### Factual
{
 "items": 192,
 "correct": 97,
 "accuracy": 0.5052083333333334,
 "ties": 0,
 "families_complete": 0,
 "families_total": 24,
 "reversal_pairs": 96,
 "reversal_both_correct": 22,
 "mean_margin": 0.003403881709800771,
 "greedy_exact": 74
}
### Control
{
 "items": 192,
 "correct": 97,
 "accuracy": 0.5052083333333334,
 "ties": 0,
 "families_complete": 0,
 "families_total": 24,
 "reversal_pairs": 96,
 "reversal_both_correct": 11,
 "mean_margin": -0.021204435219412215,
 "greedy_exact": 88
}

## Factual-500 on factual TRAIN partition (384 items)

### Overall
{
 "items": 384,
 "correct": 193,
 "accuracy": 0.5026041666666666,
 "ties": 0,
 "families_complete": 0,
 "families_total": 48,
 "reversal_pairs": 192,
 "reversal_both_correct": 38,
 "reversal_rate": 0.19791666666666666,
 "mean_margin": -0.0014437245664945901
}
### Object stratum
{
 "items": 192,
 "correct": 92,
 "accuracy": 0.4791666666666667,
 "ties": 0,
 "families_complete": 0,
 "families_total": 24,
 "reversal_pairs": 96,
 "reversal_both_correct": 18,
 "reversal_rate": 0.1875,
 "mean_margin": -0.003754644650522702
}
### Predicate stratum
{
 "items": 192,
 "correct": 101,
 "accuracy": 0.5260416666666666,
 "ties": 0,
 "families_complete": 0,
 "families_total": 24,
 "reversal_pairs": 96,
 "reversal_both_correct": 20,
 "reversal_rate": 0.20833333333333334,
 "mean_margin": 0.0008671955175335219
}
### Margin summary
{
 "n": 384,
 "mean": -0.0014437245664945901,
 "median": 0.00701037076942157,
 "min": -1.9088075993786333,
 "max": 1.8532648077089107
}
### Asymmetry by factor
{
 "assignment": {
  "0": {
   "correct": 96,
   "total": 192
  },
  "1": {
   "correct": 97,
   "total": 192
  }
 },
 "query": {
  "0": {
   "correct": 97,
   "total": 192
  },
  "1": {
   "correct": 96,
   "total": 192
  }
 },
 "fact_order": {
  "0": {
   "correct": 94,
   "total": 192
  },
  "1": {
   "correct": 99,
   "total": 192
  }
 }
}
### Asymmetry by factor and stratum
{
 "object": {
  "assignment": {
   "0": {
    "correct": 46,
    "total": 96
   },
   "1": {
    "correct": 46,
    "total": 96
   }
  },
  "query": {
   "0": {
    "correct": 47,
    "total": 96
   },
   "1": {
    "correct": 45,
    "total": 96
   }
  },
  "fact_order": {
   "0": {
    "correct": 45,
    "total": 96
   },
   "1": {
    "correct": 47,
    "total": 96
   }
  }
 },
 "predicate": {
  "assignment": {
   "0": {
    "correct": 50,
    "total": 96
   },
   "1": {
    "correct": 51,
    "total": 96
   }
  },
  "query": {
   "0": {
    "correct": 50,
    "total": 96
   },
   "1": {
    "correct": 51,
    "total": 96
   }
  },
  "fact_order": {
   "0": {
    "correct": 49,
    "total": 96
   },
   "1": {
    "correct": 52,
    "total": 96
   }
  }
 }
}
