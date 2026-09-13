# HR-3 v5 RNG-resume completion

The v4 resume attempt stopped before a new optimizer update because its GPU-mapped CPU RNG state was passed directly to `torch.set_rng_state`. v5 pins the unique `.cpu()` correction and permits the valid v4 update-297 restart hash as predecessor compatibility. This preserves the committed state and resumes at update 298 without replaying updates. No scientific value changed.
