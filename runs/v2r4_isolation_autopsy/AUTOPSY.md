# Baby v0.10 v2R4 independent autopsy (generated)

Status: diagnostic only. Frozen Gate L/C/R are unchanged.
v2R5: `absent_in_github_unresolved_pending`.

- panels sha256: `5f0d1d2c59c5d9ac07cd93460a59f542d1c65b9a10f130e61a28388b52342bb3`
- metrics sha256: `b81cb5dad5084c30c95ca6b9b913e0c814125f6b511aec06cd75637aa345e74a`
- language DEV CE: `1.2430044692009687`

## Alias-aware Gate L collapse

{
  "aliased_scored_rows": {
    "n": 992,
    "immediate_eos": 41,
    "all_same_token": 0
  },
  "unique_scored_rows": {
    "n": 800,
    "immediate_eos": 22,
    "all_same_token": 0
  },
  "note": "Gate L in the terminal report used 992 rows including novel/induction aliases of same_surface_* panels."
}

## Panel value-span census

| panel | n | first | tf_value | value_ok | free_exact |
|---|---:|---:|---:|---:|---:|
| primitive_induction | 64 | 19 | 19 | 19 | 14 |
| primitive_keyed | 64 | 64 | 64 | 64 | 62 |
| short_keyed | 64 | 54 | 54 | 54 | 54 |
| same_surface_novel | 96 | 43 | 41 | 41 | 41 |
| same_surface_induction | 96 | 6 | 6 | 6 | 5 |
| heldout_surface | 96 | 33 | 23 | 23 | 0 |
| unseen_length | 64 | 19 | 17 | 17 | 1 |
| low_prior | 64 | 16 | 15 | 15 | 0 |
| distractor | 64 | 8 | 5 | 5 | 0 |
| broken_context | 64 | 22 | 16 | 16 | 0 |
| broken_order | 64 | 1 | 0 | 0 | 0 |

## Emission source (queried vs competitor vs off-inventory)

{
  "same_surface_novel_inventory_copy": 0.96875,
  "same_surface_novel_queried": 41,
  "same_surface_novel_competitor": 52,
  "same_surface_novel_off_inventory": 3,
  "competitor_copies_are_not_rank1": true
}

## Mechanism headlines

{
  "train_novel_competitor_rest_value_tf_lock": [
    51,
    52
  ],
  "train_novel_2_pair_two_sided_p": 0.28104151505976915,
  "train_novel_2_pair_significant_0_05": false,
  "heldout_off_inventory_all_have_train_sep": true,
  "induction_eos_all_trailing_own_sep": true,
  "matched_slice_terminal_queried": 5,
  "matched_slice_preterminal_queried": 6,
  "copy_onset_update": 12000
}


### primitive_keyed

{
  "1": {
    "n": 64,
    "first_ok": 1.0,
    "tf_value": 1.0,
    "value_ok": 1.0,
    "free_exact": 0.96875,
    "chance_1_over_k": 1.0
  }
}

### short_keyed

{
  "1": {
    "n": 36,
    "first_ok": 1.0,
    "tf_value": 1.0,
    "value_ok": 1.0,
    "free_exact": 1.0,
    "chance_1_over_k": 1.0
  },
  "2": {
    "n": 28,
    "first_ok": 0.6428571428571429,
    "tf_value": 0.6428571428571429,
    "value_ok": 0.6428571428571429,
    "free_exact": 0.6428571428571429,
    "chance_1_over_k": 0.5
  }
}

### same_surface_novel

{
  "2": {
    "n": 31,
    "first_ok": 0.6129032258064516,
    "tf_value": 0.6129032258064516,
    "value_ok": 0.6129032258064516,
    "free_exact": 0.6129032258064516,
    "chance_1_over_k": 0.5
  },
  "3": {
    "n": 21,
    "first_ok": 0.3333333333333333,
    "tf_value": 0.3333333333333333,
    "value_ok": 0.3333333333333333,
    "free_exact": 0.3333333333333333,
    "chance_1_over_k": 0.3333333333333333
  },
  "4": {
    "n": 44,
    "first_ok": 0.38636363636363635,
    "tf_value": 0.3409090909090909,
    "value_ok": 0.3409090909090909,
    "free_exact": 0.3409090909090909,
    "chance_1_over_k": 0.25
  }
}

### heldout_surface

{
  "2": {
    "n": 38,
    "first_ok": 0.47368421052631576,
    "tf_value": 0.3684210526315789,
    "value_ok": 0.3684210526315789,
    "free_exact": 0.0,
    "chance_1_over_k": 0.5
  },
  "3": {
    "n": 31,
    "first_ok": 0.25806451612903225,
    "tf_value": 0.16129032258064516,
    "value_ok": 0.16129032258064516,
    "free_exact": 0.0,
    "chance_1_over_k": 0.3333333333333333
  },
  "4": {
    "n": 27,
    "first_ok": 0.25925925925925924,
    "tf_value": 0.14814814814814814,
    "value_ok": 0.14814814814814814,
    "free_exact": 0.0,
    "chance_1_over_k": 0.25
  }
}

### broken_context

{
  "2": {
    "n": 20,
    "first_ok": 0.6,
    "tf_value": 0.45,
    "value_ok": 0.45,
    "free_exact": 0.0,
    "chance_1_over_k": 0.5
  },
  "3": {
    "n": 18,
    "first_ok": 0.3333333333333333,
    "tf_value": 0.16666666666666666,
    "value_ok": 0.16666666666666666,
    "free_exact": 0.0,
    "chance_1_over_k": 0.3333333333333333
  },
  "4": {
    "n": 26,
    "first_ok": 0.15384615384615385,
    "tf_value": 0.15384615384615385,
    "value_ok": 0.15384615384615385,
    "free_exact": 0.0,
    "chance_1_over_k": 0.25
  }
}

## Hypothesis read

{
  "A_internal_identification": "supported_tf_continuation_lock_on_train_gold_span",
  "A_prime_heldout_separator": "supported_value_ok_without_exact",
  "B_query_binding": "failed_queried_copy_not_above_1_over_k_including_2_pair",
  "C_payload_copy": "supported_train_novel_inventory_copy_93_of_96",
  "D_curriculum": "supported_copy_saturates_then_selection_does_not_lift",
  "D_free_emission_of_selected_span": "supported_tf_equals_free",
  "E_separator_eos": "heldout_exact_is_separator_ood_and_induction_eos_is_trailing_sep",
  "F_surface": "heldout_inventory_copy_collapses_at_long_values",
  "probe_limit_confound": "interim_evals_n_16_only_terminal_full_panel",
  "architecture": "not_justified_as_next_claim"
}

This generated table is an audit companion. The narrative independent autopsy lives in `research/V010_V2R4_INDEPENDENT_AUTOPSY.md`.

