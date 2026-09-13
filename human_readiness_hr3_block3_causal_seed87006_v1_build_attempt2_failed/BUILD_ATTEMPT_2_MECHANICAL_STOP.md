# HR-3 build attempt 2 mechanical stop

This unsealed construction attempt stopped during independent static preflight
before checkpoint loading, model inference, optimizer creation, or any update.

The validator correctly found that the evaluator did not textually call the
low-level `aligned_tensors` helper: it calls the shared `aligned_dev_loss`
function, which itself calls that same helper. The initial static assertion was
therefore too literal. The unique correction verifies the actual call graph:
the trainer calls `aligned_tensors`, the evaluator calls `aligned_dev_loss`,
and the shared runtime contains the corrected aligned construction.

No scientific component changed. This preserved unsealed attempt is followed
by a fresh corrected preflight directory.
