# HR-3 resume attempt 1 mechanical stop

The v4 resume process restored a valid update-297 restart state and stopped
before the next optimizer update because `torch.load(..., map_location="cuda")`
moved the saved CPU RNG byte tensor onto the GPU. `torch.set_rng_state` then
rejected the device mismatch. The unique fix is to call `.cpu()` on the saved
CPU RNG state before restoration. The valid update-297 state remains intact and
is eligible for continuation; no new update was performed in this attempt.
