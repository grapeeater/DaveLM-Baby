# HR-3 v2 mechanical completion

HR-3 v1 stopped before checkpoint loading because its copied `treatment13_model.py` imports `treatment13_config.py`, which was absent from that bundle. v2 pins the exact authoritative configuration source and adds the pre-parent-load entrypoint check. No scientific payload, parent, dataset, literal schedule, optimizer, scope, objective, gate, or sealed-material rule changed.
