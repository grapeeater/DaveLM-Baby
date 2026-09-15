# Baby v0.10 curriculum v2R3

Protocol: `BABY_V010_FOUNDATION_V2R3`  
Status: frozen before the v2R3 diagnostic

v2R2 established a fresh language substrate (DEV CE 1.1919 at U6000) and
produced an early primitive identity signal, but catastrophic forgetting
raised DEV CE to 5.7095 after 500 structured updates. v2R3 preserves the
successful parts and makes the smallest evidence-backed correction.

## Frozen changes

- U0-5999: same fresh 6,000-update Phase1G-style language foundation.
- At U6000: freeze blocks 0-6; train only blocks 7-11, final norm, and language
  head. This is the upper-block scope directly used by T34, applied only after
  v0.10's own language training.
- U6000 onward: retain a 20% language CE mix in primitive, short, and full
  capability stages, not only in the final stage.
- Primitive/short stages continue to mask only sampled answer-span identities;
  full stage includes answer, separator, and EOS.
- Structured learning rate remains 3.75e-5; final gates and panels are
  unchanged.

The first diagnostic ends at U8,000 on fresh seed 105001. It is not a
graduation result. Success requires stable language retention together with a
measurable primitive identity signal; if supported, the same frozen protocol
can continue to the full challenge and independent replication.
