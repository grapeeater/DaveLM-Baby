# HR-3 v5 resume mechanical stop

The v5 resume controller validated the update-297 state and stopped before any
new optimizer update while restoring CUDA RNG states. On this pinned runtime,
`torch.cuda.set_rng_state_all` requires CPU `ByteTensor` values after a
GPU-mapped checkpoint load. The unique fix is to move every saved CUDA RNG
state to CPU before passing the list to that API. The update-297 restart state
remains valid and is preserved for the next versioned continuation.
