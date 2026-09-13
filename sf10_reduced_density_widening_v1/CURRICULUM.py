"""SF10 curriculum builder. Reuses the SF9 curriculum content VERBATIM (same 144 items, same
ids/prompts/tokenizations, same vocabulary) - no redesign of what Baby is taught. The ONLY change
is a deterministic 4-way balanced shard split so each English update trains 36 items (close to
SF8's historically-safe 32/update) instead of SF9's full 144/update, cycled so full curriculum
coverage and balance are preserved over the run.

Shard assignment is derived purely from existing structural identifiers (pair_id for TRAIN16,
group+object+verb for new items) - never from any observed outcome.
"""
import json, hashlib
from pathlib import Path

H = Path(__file__).resolve().parent
SF9 = Path(r"C:\DaveLM-CADAVER\sf9_surface_order_curriculum_v1")

# TRAIN16: pair_id -> shard, chosen so each shard gets exactly one g0 pair and one g1 pair
# (their fixed file order = 0..3 within each group).
TRAIN16_PAIR_SHARD = {
    "TRAIN:g0:found:small drum": 0, "TRAIN:g0:found:wooden boat": 1,
    "TRAIN:g0:carried:small drum": 2, "TRAIN:g0:carried:wooden boat": 3,
    "TRAIN:g1:found:soft scarf": 0, "TRAIN:g1:found:round plate": 1,
    "TRAIN:g1:carried:soft scarf": 2, "TRAIN:g1:carried:round plate": 3,
}
# New curriculum: (group, verb, objkey) -> shard, one shard per (object,verb) combo per group,
# matching the SF9 generator's own enumeration order (obj outer loop, verb inner loop).
COMBO_SHARD = {
    ("A", "painted", "tin_cup"): 0, ("A", "dropped", "tin_cup"): 1,
    ("A", "painted", "wool_hat"): 2, ("A", "dropped", "wool_hat"): 3,
    ("B", "bought", "green_ball"): 0, ("B", "held", "green_ball"): 1,
    ("B", "bought", "silver_bell"): 2, ("B", "held", "silver_bell"): 3,
}


def shard_for(it):
    iid = it["id"]
    if iid.startswith("TRAIN:"):
        return TRAIN16_PAIR_SHARD[it["pair_id"]]
    # SF9:surface:{group}:{verb}:{objkey}:{surface}:{name} or SF9:competing:{group}:{verb}:{objkey}:{order}:{query}:{name}
    parts = iid.split(":")
    group, verb, objkey = parts[2], parts[3], parts[4]
    return COMBO_SHARD[(group, verb, objkey)]


def main():
    items = json.loads((SF9 / "TRAIN.json").read_text(encoding="utf-8-sig"))
    assert len(items) == 144
    for it in items:
        it["shard"] = shard_for(it)

    # Verify byte-for-byte content identity with SF9 except the added 'shard' key.
    for it in items:
        stripped = {k: v for k, v in it.items() if k != "shard"}
        assert stripped in json.loads((SF9 / "TRAIN.json").read_text(encoding="utf-8-sig"))

    # Balance verification (by construction, asserted not assumed).
    from collections import Counter
    per_shard = {s: [it for it in items if it["shard"] == s] for s in range(4)}
    for s, its in per_shard.items():
        assert len(its) == 36, (s, len(its))
        names = Counter(it["candidates"][it["correct_index"]] for it in its)
        assert set(names.values()) == {9}, (s, names)
    surf_by_shard_surface = Counter((it["shard"], it["subgroup"]) for it in items if it["id"].startswith("SF9:surface:"))
    assert set(surf_by_shard_surface.values()) == {4}
    comp_by_shard_sg = Counter((it["shard"], it["subgroup"]) for it in items if it["id"].startswith("SF9:competing:"))
    assert set(comp_by_shard_sg.values()) == {4}

    # Schedule: 200 updates = 180 English + 20 binding. Binding entries reused verbatim from SF9
    # (which reused them verbatim from SF8/SF1). English updates cycle shard 0,1,2,3,... so each
    # of the 180 English updates trains exactly 36 items; full curriculum seen once every 4 updates
    # (45 full rotations total; each item exposed 45x over the run, vs SF9's 180x).
    sf9_sched = json.loads((SF9 / "SCHEDULE.json").read_text(encoding="utf-8-sig"))
    sf9_binding = {u["update"]: u for u in sf9_sched if u["kind"] == "binding"}
    pad = max(len(it["prompt_token_ids"]) for it in items) + 5
    schedule = []
    english_counter = 0
    for u in range(1, 201):
        if u % 10 == 0:
            b = sf9_binding[u]
            schedule.append({"update": u, "kind": "binding", "pad": pad, "quartets": b["quartets"], "documents": b["documents"]})
        else:
            shard = english_counter % 4
            ids = [it["id"] for it in items if it["shard"] == shard]
            assert len(ids) == 36
            schedule.append({"update": u, "kind": "english", "pad": pad, "shard": shard, "ids": ids})
            english_counter += 1
    assert english_counter == 180
    exposure = Counter()
    for u in schedule:
        if u["kind"] == "english":
            exposure.update(u["ids"])
    assert set(exposure.values()) == {45}, set(exposure.values())

    def write_json(p, v):
        p.write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    write_json(H / "TRAIN.json", items)
    write_json(H / "SCHEDULE.json", schedule)

    manifest = {}
    for p in [H / "TRAIN.json", H / "SCHEDULE.json", H / "DEV_SURFACE.json", H / "DEV_ORDER.json", H / "TRAIN16_RETENTION.json"]:
        manifest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(H / "CURRICULUM_MANIFEST.json", manifest)

    print(json.dumps({
        "items": len(items), "shards": 4, "items_per_shard": 36,
        "per_update_batch_english": 36, "sf8_per_update_batch": 32, "sf9_per_update_batch": 144,
        "english_updates": 180, "per_item_exposure_over_run": 45,
        "name_balance_per_shard": {"Alex/Owen/Mia/Nora": 9},
        "pad": pad,
    }, indent=2))


if __name__ == "__main__":
    main()
