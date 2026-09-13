# Mechanical corrections before the final SF3 freeze

1. The first static-preflight invocation found that the copied SF1 receipt listed `BUILD.py` and `ENGINE.py`,
   which the initial copy command had omitted. Both were copied byte-for-byte from the authoritative SF1 bundle;
   its original manifest then verified completely.
2. The static masking test initially expected width `pad-1`; the pinned `pad_batch` contract correctly returns
   model-input tensors of width `pad`. The test was corrected to verify the pinned implementation's actual
   identity and next-token labels.
3. The first baseline invocation stopped before completing because `PROTOCOL.json` omitted the already-frozen
   SF2 `baseline_reproduction_M9.kl_zero_tolerance` field consumed by the pinned controller. The value `1e-9`
   was copied from authoritative `SF2_PROTOCOL.json`. The partial output was preserved as
   `preflight_baseline_failed_v1`; no optimizer was created and no update occurred.

These were uniquely mechanical completeness/test corrections. No data, objective, schedule, scope, gate,
treatment variable, or model behavior was changed.
