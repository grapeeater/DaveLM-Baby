# HR-3 execution attempt 1 mechanical stop

The v3 trainer completed the in-memory optimizer calculation for global update
1, then failed while atomically committing `restart.pt` on Windows with
`OSError: [Errno 9] Bad file descriptor` from an `fsync` call on a read-only
file handle. No rolling restart state or permanent checkpoint exists.

Under the frozen persistence rule, an update is complete only after a valid
restart state is committed. Update 1 is therefore explicitly **uncommitted**;
it cannot supply a descendant or evaluation result and a later fresh v4 run
starts from the immutable Pilot 1 parent. `UPDATE_METRICS.jsonl` and
`DEV_TRAJECTORY.json` are retained as forensic execution evidence only.

No FINAL or sacred material was accessed. No binding or readiness evaluation
was performed after the failed update.
