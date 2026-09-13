# PATH 2 — Prospective numerical-equivalence standard: analysis and classification

## Result: FAILED / NOT JUSTIFIABLE (and unnecessary once Path 3 is adopted)

## What Path 2 would need

A scientifically justified prospective equivalence criterion to replace bitwise identity for the "replay the
historical SF2 prefix, then branch" design. It must be justified independently of the observed failed SF3/SF4
outcomes (no reverse-engineering tolerances to make old runs pass).

## Analysis

1. **Exact identity is unavailable.** Bitwise reproduction of the historical prefix occurs only by stochastic
   luck (hist==p1a once in ~9 attempts). Exactness cannot be the standard.

2. **No principled tolerance can be set from first principles.** The divergence is stochastic with a heavy tail
   at the *parameter* level (up to 9e-4 in individual weights, RMS ~3-4e-6) but near-zero at every *behavioral*
   level measured (acquisition counts identical, language CE within ~4e-6, D3 within ~7e-7, per-item margin
   deltas ≤2.05e-3 at u100 and ≤6.97e-3 at u200, zero sign flips). Any tolerance chosen on these numbers is
   an empirical bound, and choosing it to admit the observed "successful" replicates while excluding the
   observed "failed" pairs would be reverse-engineering. The mission forbids this. No independent theoretical
   bound exists for how these parameter-level differences would propagate into a downstream 100-update LR
   manipulation.

3. **Parameter/optimizer distance is the wrong metric for the question.** The question in the SF3/SF4 lineage is
   whether a *fresh prefix that then receives a different LR schedule* can be compared against a control. If two
   fresh runs diverge stochastically (u100 RMS ~3-4e-6 in weights), then two arms branched from two different
   fresh prefixes carry a prefix difference of that size into the treatment window. Because the LR manipulation
   is small (step-size reduction over the final 90 English updates), the prefix difference could be the same
   order as the treatment signal at the parameter level even though it is far smaller at the behavioral level.
   A parameter-space equivalence threshold would therefore need to be *below* the observed stochastic spread —
   i.e., exactness — which is unavailable; a behavioral threshold would not certify parameter-level
   comparability of the arms.

4. **The scientific way out is not a looser gate; it is a matched design (Path 3).** If control and treatment
   arms are forked from ONE shared in-process prefix, prefix identity is exact by construction and no
   equivalence standard is needed. Any residual stochastic divergence occurs *after* the fork, identically in
   distribution for both arms, and is bounded by the measured endpoint noise floor (~7e-3 nats per item, zero
   sign flips) — two orders of magnitude below the boundary movements the LR hypothesis is about (~0.1-0.6 nats
   over the final 100 updates as measured in SF2 u100->u200). Path 2 therefore becomes unnecessary.

5. **Frozen SF3/SF4 outcomes remain stopped regardless.** No tolerance or standard proposed here would revive
   SF3 or SF4; they stay `SF3_HARD_STOP_UPDATE100_REPLAY_MISMATCH` and
   `SF4_HARD_STOP_COMMON_STATE_REPRODUCTION_MISMATCH` forever.

## Conclusion

PATH 2 FAILED as a standalone path: no independently justifiable numerical-equivalence standard exists, and the
matched-fork design of PATH 3 makes such a standard unnecessary. (The endpoint noise-floor measurement that
informs PATH 3 was taken from replica pairs and mode comparisons, i.e., platform-noise characterization, not
from SF3/SF4 treatment outcomes.)
