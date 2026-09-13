# HR-3 v4 persistence completion

The v3 execution attempt performed update 1 in memory but failed before restart-state commit because Windows rejected `fsync` on a read-only descriptor. The attempt is preserved separately and update 1 is uncommitted by the frozen restart rule. v4 changes only the atomic write handle to `r+b`, flushes it, then fsyncs before atomic replacement. It also excludes all bytecode cache files from v4 payload. No scientific treatment component changed.
