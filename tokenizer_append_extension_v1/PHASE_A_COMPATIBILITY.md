# Phase A compatibility report

Append-only valid: **True**

- v0_7 SHA-256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b` vocab 1024 max id 1023
- embedding `[1024, 640]`, language_head `[1024, 640]` bias `[1024]`
- tied embeddings: False (must be false)
- train stream token max 1023
- in-memory add_tokens preserves unrelated encodings: True
- dummy new id 1024 (expected 1024)

Old IDs 0–1023 can be preserved. New IDs start at 1024. Old embedding/head rows can be copied bit-identically.
Do not overwrite `BABY_VNEXT_CONFIG.json` or v0_7. Historical streams remain v0_7.
v0_7 was not written. Dummy add was in-memory only.
