# Baby v0.10 capability exit gates

Status: preregistered for foundation v1. Thresholds below are fixed before the
first v0.10 optimizer update and may not be weakened after observing results.

## Stage gates

### Gate L — language health

At the selected terminal checkpoint: finite metrics, held-out language DEV CE
`<= 2.50`, train/DEV gap `<= 1.00`, and no immediate-EOS or repeated-token
collapse in more than 10% of generation prompts.

### Gate C — contextual reproduction

On a frozen, independently generated panel, each independent seed must satisfy:

| metric | threshold |
|---|---:|
| novel sequence first-token top-1 | >= 0.90 |
| novel sequence teacher-forced full-span exact | >= 0.80 |
| held-out surface/template first-token top-1 | >= 0.80 |
| held-out surface/template free-running exact | >= 0.70 |
| unseen-length first-token top-1 | >= 0.80 |
| changed positions/orderings top-1 | >= 0.75 |
| low-prior-stratum top-1 | >= 0.80 |
| low-vs-high prior accuracy gap | <= 0.10 absolute |
| distractor-resistance top-1 | >= 0.75 |
| broken-context free-running exact | <= 0.05 |
| broken-order top-1 | <= 0.20 of intact accuracy |
| distinct emitted target tokens | >= 100 |

The held-out template panel must pass independently; it cannot be averaged away
by familiar-format performance.

### Gate R — retention and mechanism

Language DEV CE must stay within `+0.50` nats of the pre-structured baseline.
For intact versus broken context, the median target logit lift must be `>= 1.0`
and the median target-vs-competitor margin must be positive. These are mechanism
proxies, not a replacement for the behavioral gates.

## Replication and selection

Foundation graduation requires the full Gate L + C + R bundle on two independent
seeds, with no seed discarded. A seed that fails is preserved as a failure and
drives the next preregistered iteration. A checkpoint is terminally selected by
the first checkpoint that passes all gates; if none passes, the run is a failure
or futility result under the frozen protocol.

## Futility criteria

Futility is declared only if, after 4000 updates on two seeds, the diverse
training objective is active, language health is finite, and held-out template
top-1 remains `< 0.30` while same-surface top-1 is `> 0.80`, or all structured
metrics remain at U0 within the preregistered bootstrap confidence interval.
Futility does not authorize changing the threshold or inspecting protected data.
