import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
SF1 = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")
seeds = [87017, 87018, 87019]
tok_names = {314: "Alex", 536: "Owen", 925: "Mia", 512: "Nora"}

items = {}
for label in ["HELDOUT", "ALTERNATE", "COPY", "COMPETING"]:
    for it in json.loads((SF1 / f"{label}.json").read_text(encoding="utf-8-sig")):
        items[it["id"]] = it

print("### HELDOUT exact-fail items (all seeds)")
for s in seeds:
    for row in (json.loads(l) for l in (OUT / f"raw/seed_{s}_low/heldout_RAW.jsonl").read_text().splitlines()):
        if not row["exact"]:
            it = items[row["id"]]
            tgt = it["candidate_token_ids"][row["correct_index"]][0]
            gen0 = row["generated_ids"][0] if row["generated_ids"] else None
            print(f"{s} | {row['id']} | correct={row['correct']} | gen_first_matches_correct={gen0==tgt} | margin={row['margin']:.3f} | gen='{row['generated_text'][:40]}'")

def firstname_hits(seed, panel):
    rows = [json.loads(l) for l in (OUT / f"raw/seed_{seed}_low/{panel}_RAW.jsonl").read_text().splitlines()]
    by_sg = {}
    for r in rows:
        by_sg.setdefault(r["subgroup"], []).append(r)
    out = []
    for sg in sorted(by_sg):
        g = by_sg[sg]
        hit = 0
        for r in g:
            it = items[r["id"]]
            tgt = it["candidate_token_ids"][r["correct_index"]][0]
            if r["generated_ids"] and r["generated_ids"][0] == tgt:
                hit += 1
        out.append(f"{sg}={hit}/{len(g)}")
    return "  ".join(out)

print()
print("### ALTERNATE greedy first-name==correct per subgroup (all seeds)")
for s in seeds:
    print(f"seed {s}: {firstname_hits(s, 'alternate')}")
print()
print("### COMPETING greedy first-name==correct per subgroup (all seeds)")
for s in seeds:
    print(f"seed {s}: {firstname_hits(s, 'competing')}")
print()
print("### COPY greedy first-name==correct per subgroup (all seeds)")
for s in seeds:
    print(f"seed {s}: {firstname_hits(s, 'copy')}")

print()
print("### COPY: model emission when card says the non-default name (seed 87017)")
rows = [json.loads(l) for l in (OUT / "raw/seed_87017_low/copy_RAW.jsonl").read_text().splitlines()]
for r in rows:
    if r["correct_index"] == 1:
        it = items[r["id"]]
        emitted = tok_names.get(r["generated_ids"][0], str(r["generated_ids"][0]))
        print(f"{r['id']} | card={it['candidates'][1].strip()} | model_emitted={emitted} | margin={r['margin']:.3f}")

print()
print("### COMPETING o1:fact wrong items: which name is emitted? (seed 87017)")
rows = [json.loads(l) for l in (OUT / "raw/seed_87017_low/competing_RAW.jsonl").read_text().splitlines()]
for r in rows:
    if r["subgroup"] == "fact_order1:fact" and not r["correct"]:
        it = items[r["id"]]
        emitted = tok_names.get(r["generated_ids"][0], str(r["generated_ids"][0]))
        print(f"{r['id']} | expected={it['candidates'][r['correct_index']].strip()} | other={it['candidates'][1-r['correct_index']].strip()} | emitted={emitted}")
