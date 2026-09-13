# Orthogonal shared-core + unbounded antisymmetric treatment

{
  "status": "FINAL_FROZEN_RETENTION_RESULTS",
  "documents": 320,
  "quartets": 80,
  "overall": {
    "documents": 320,
    "answer_exact": 317,
    "answer_accuracy": 0.990625,
    "both_distinct": 312,
    "exactly_one": 8,
    "slot_collapse": 0
  },
  "localization_counts": {
    "BOTH_DISTINCT": 312,
    "EXACTLY_ONE": 8
  },
  "by_localization_category": {
    "BOTH_DISTINCT": {
      "documents": 312,
      "answer_exact": 312,
      "answer_accuracy": 1.0,
      "both_distinct": 312,
      "exactly_one": 0,
      "slot_collapse": 0
    },
    "EXACTLY_ONE": {
      "documents": 8,
      "answer_exact": 5,
      "answer_accuracy": 0.625,
      "both_distinct": 0,
      "exactly_one": 8,
      "slot_collapse": 0
    }
  },
  "conditional_both_distinct": {
    "answer_correct": 312,
    "answer_total": 312,
    "answer_accuracy": 1.0,
    "selected_row_correct": 312,
    "selected_row_total": 312,
    "selected_query_row_correct": 312,
    "selected_query_row_total": 312
  },
  "per_query_slot": {
    "0": {
      "documents": 160,
      "answer_exact": 159,
      "answer_accuracy": 0.99375,
      "both_distinct": 156,
      "exactly_one": 4,
      "slot_collapse": 0
    },
    "1": {
      "documents": 160,
      "answer_exact": 158,
      "answer_accuracy": 0.9875,
      "both_distinct": 156,
      "exactly_one": 4,
      "slot_collapse": 0
    }
  },
  "per_orientation": {
    "1": {
      "documents": 160,
      "answer_exact": 158,
      "answer_accuracy": 0.9875,
      "both_distinct": 156,
      "exactly_one": 4,
      "slot_collapse": 0
    },
    "2": {
      "documents": 160,
      "answer_exact": 159,
      "answer_accuracy": 0.99375,
      "both_distinct": 156,
      "exactly_one": 4,
      "slot_collapse": 0
    }
  },
  "per_layout": {
    "p16_b12": {
      "documents": 40,
      "answer_exact": 38,
      "answer_accuracy": 0.95,
      "both_distinct": 36,
      "exactly_one": 4,
      "slot_collapse": 0
    },
    "p16_b8": {
      "documents": 40,
      "answer_exact": 40,
      "answer_accuracy": 1.0,
      "both_distinct": 40,
      "exactly_one": 0,
      "slot_collapse": 0
    },
    "p20_b8": {
      "documents": 40,
      "answer_exact": 40,
      "answer_accuracy": 1.0,
      "both_distinct": 40,
      "exactly_one": 0,
      "slot_collapse": 0
    },
    "p24_b10": {
      "documents": 40,
      "answer_exact": 40,
      "answer_accuracy": 1.0,
      "both_distinct": 40,
      "exactly_one": 0,
      "slot_collapse": 0
    },
    "p24_b8": {
      "documents": 40,
      "answer_exact": 39,
      "answer_accuracy": 0.975,
      "both_distinct": 36,
      "exactly_one": 4,
      "slot_collapse": 0
    },
    "p28_b10": {
      "documents": 40,
      "answer_exact": 40,
      "answer_accuracy": 1.0,
      "both_distinct": 40,
      "exactly_one": 0,
      "slot_collapse": 0
    },
    "p32_b12": {
      "documents": 40,
      "answer_exact": 40,
      "answer_accuracy": 1.0,
      "both_distinct": 40,
      "exactly_one": 0,
      "slot_collapse": 0
    },
    "p32_b8": {
      "documents": 40,
      "answer_exact": 40,
      "answer_accuracy": 1.0,
      "both_distinct": 40,
      "exactly_one": 0,
      "slot_collapse": 0
    }
  },
  "complete_quartets": 78,
  "quartet_patterns": {
    "4/4 BOTH_DISTINCT": 78,
    "4/4 EXACTLY_ONE": 2
  },
  "quartet_coordinate_stability": 80,
  "reversal_pairs": 160,
  "reversal_both_correct": 158,
  "reversal_both_correct_rate": 0.9875,
  "prediction_changed_under_reversal": 158,
  "residual_wrong_localization_destinations": {
    "54": 8,
    "47": 4,
    "27": 4
  },
  "checkpoint_sha256": "fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430",
  "starting_checkpoint_sha256": "cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab",
  "train_pool_sha256": "40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3",
  "retention_pool_sha256": "29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072",
  "schedule_sha256": "6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42",
  "optimizer_updates": 1000,
  "retention_optimizer_updates": 0,
  "retention_evaluations": 1,
  "model_eval": true,
  "torch_no_grad": true
}