# HR-3 build attempt 1 mechanical stop

This unsealed construction attempt stopped during independent preflight before
checkpoint loading, model inference, optimizer creation, or any update.

The inherited `BINDING_SCHEDULE.json` stores canonical quartet **integer
indices**, exactly as the authoritative corrected-alignment trainer consumes
them. The first HR-3 adapter incorrectly treated them as `quartet_id` strings.
The failure was detected by the independent validator when it checked schedule
identity resolution. The unique specification-preserving correction is to use
the canonical indexed quartet ordering and assert the resulting document IDs
against the literal `documents` field in each schedule record.

No scientific parameter, data record, schedule ordering, seed, objective,
scope, parent, or frozen readiness material was changed. This directory is
preserved as a failed unsealed build record; the corrected preflight is built
in a fresh directory.
