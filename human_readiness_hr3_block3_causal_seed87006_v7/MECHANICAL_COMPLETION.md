# HR-3 v7 predecessor protocol-chain completion

The only valid committed restart is update 297 under the v4 protocol hash. v5 and v6 stopped before any new commit while correcting RNG restoration. v7 accepts exactly the finite v4/v5/v6 predecessor hashes, verifies every current payload and schedule identity, and resumes at update 298. This is mechanical provenance wiring only.
