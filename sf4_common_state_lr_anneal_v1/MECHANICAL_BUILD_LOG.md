# Prospective mechanical build log

Before sealing or model loading, the verifier discovered the historical SF2 receipt uses a UTF-8 BOM.
Its byte hash exactly matches the detached historical hash. The new verifier now decodes that receipt with
utf-8-sig; it does not alter the historical receipt or any pinned scientific implementation. The failed first
construction attempt stopped before constructing a model, optimizer, schedule, or protocol receipt.

A second construction pass encountered Windows' default cp1252 decoding on the UTF-8 language DEV file.
All newly written orchestration text reads were made explicitly UTF-8. Historical pinned source files remain
byte-identical. No model or optimizer had been loaded; no scientific value changed.
