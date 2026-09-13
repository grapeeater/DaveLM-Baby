# Mechanical corrections before seal

The first validation invocation used a wrapper object as the mock teacher even though the pinned KL helper calls the teacher directly. The mock was changed to pass its deterministic base module. No treatment source, data, schedule, objective, gate, or runtime behavior changed. The complete preflight then passed.
