# Baby v0.10

Baby v0.10 is a fresh lineage. It is initialized from random weights and is not
a continuation of the v0.9/T34 checkpoint line.

The authoritative writable workspace supplied by the machine is
`C:\\DaveLM-v0.10`; the requested `C:\\DaveLM0.10` path was absent at start and
is recorded in the provenance notes. `C:\\DaveLM-CADAVER` is read-only source
evidence except for a separately documented, conservative archival operation.

The foundational objective is capability-first training: ordinary language
modeling, contextual reproduction, induction, and copy across held-out surface
realizations must be measured during training. Loss and familiar-example exact
match are not graduation criteria.

Start with the frozen protocol in `data_specs/V010_FOUNDATION_PROTOCOL.json`.
The design and gates are in `design/` and the implementation is in
`src/baby_v010/`.
