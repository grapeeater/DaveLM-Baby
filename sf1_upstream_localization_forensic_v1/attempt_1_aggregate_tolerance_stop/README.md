# Preserved first finalization stop

The full read-only sweep completed and raw results were persisted. Same-source manual logits reproduced exactly, but the cross-run aggregate comparison used an unnecessarily strict 1e-6 tolerance despite a different batching layout. Maximum aggregate difference was 3.427267074584961e-6. No hybrid result was interpreted before this stop. The finalization correction uses 1e-5 only for cross-run aggregate reproduction and performs no new model inference.
