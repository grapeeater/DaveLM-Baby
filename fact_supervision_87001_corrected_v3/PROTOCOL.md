# Fact-supervision pre-training study (seed 87001)

Two arms receive the same two-fact English formats, candidate names, response tokens, EOS targets, updates, batches, and binding rehearsal. The factual arm’s query is entailed by its rendered facts. The control arm’s facts omit the queried description, making the target underdetermined; each exact control prefix is paired equally with both practice targets.

English updates freeze blocks 0–3 and specialized localization/retrieval parameters; embeddings, blocks 4–7, final norm, and language head train. Binding updates restore the full T13 scope and existing answer-CE plus hard-min permutation-invariant localization objective. 500 updates comprise 450 English and 50 binding updates (9:1), AdamW 5e-5, weight decay .05, clip 2.0. Seeds are 87002 and matched replication 87003.

Naturalistic diagnostics use the fixed Primary renderer “<Name> <predicate> the <description> and then went home.” and fixed Confirmation renderer “After <Name> <predicate> the <description>, <Name> went home.” They are descriptive only.

No checkpoint is evaluated during this construction freeze. Sacred graduation material is excluded.
