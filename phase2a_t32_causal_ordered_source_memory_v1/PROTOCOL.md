# T32 Causal Ordered Source-Memory Prefix

This is one frozen architecture test relative to the Phase1G parent. Ordered
identity mention hidden states (from block 10) are exposed as auxiliary key /
value memory to the final block's attention. The branch is answer-position
masked, uses the existing soft cosine fact-clause pointer weights, and has one
zero-initialized 640-channel gate. Normal language rows have no memory.

The parent, tokenizer, data, losses, optimizer, schedule, protected-data locks,
and 2-of-3 sequential replication rule are frozen in `T32_PROTOCOL.json`.
The inherited T24/T31 objective is retained: pointer CE, candidate margin,
full answer CE including EOS, and inventory-wide unlikelihood. No T29/T28
auxiliary loss is present.

At gate zero the ordinary transformer path is exactly the Phase1G parent. A
preflight must also show a nonzero gate gradient on a TRAIN QA row, otherwise
the design is mechanically invalid. T32 reports memory gate magnitude and
answer-position source attention, but activation is descriptive and never a
success gate by itself. T3 TEST, FINAL, sacred, and all locked historical
panels remain sealed.
