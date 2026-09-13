# T13 focused/post-run checks

All checks passed:

- inherited `TREATMENT13_LOCALIZATION_PREFLIGHT_PASS` verified;
- train/retention pool hashes match preflight and have zero token-array overlap;
- retention layout signatures are disjoint from training signatures (8 unseen);
- both query slots and both reversal orientations are present in every signature;
- fixed `+4` row-local value offset has zero violations;
- GPU smoke passed on AMD Radeon RX 9060 XT with no optimizer step;
- training metrics contain exactly 1,000 updates and 32 supervised answer decisions per update;
- final retention audit runs under `no_grad` and records zero optimizer steps;
- final checkpoint state is finite and has T13 total parameter count 10,841,346;
- protected T12 source files were hash-checked unchanged.
