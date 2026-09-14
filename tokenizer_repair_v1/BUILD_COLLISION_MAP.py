"""DEV-only collision map from post-T28 tokenizer artifacts. Never loads TEST."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER")
ITEMS = ROOT / "phase2a_post_t28_tokenizer_diagnostic_v1" / "ITEMS.json"
OUT = ROOT / "tokenizer_repair_v1" / "COLLISION_MAP.json"


def main() -> None:
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    by_name = defaultdict(list)
    for row in items:
        by_name[row["name"]].append(row)

    names = {}
    first_token_groups = defaultdict(set)
    for name, rows in sorted(by_name.items()):
        r0 = rows[0]
        ids = list(r0["name_token_ids"])
        first = ids[0]
        gen = Counter(x["generated_text"] for x in rows)
        modes = Counter(x["mode"] for x in rows)
        names[name] = {
            "name": name,
            "v0_7_token_ids": ids,
            "token_pieces": r0["token_pieces"],
            "answer_text": r0["answer_text"],
            "first_token": first,
            "name_token_count": r0["name_tokens"],
            "longest_shared_prefix_among_dev_names": r0["max_shared_prefix_tokens"],
            "n": len(rows),
            "exact": sum(bool(x["exact"]) for x in rows),
            "exact_rate": sum(bool(x["exact"]) for x in rows) / len(rows),
            "diverge": modes.get("first_token_correct_then_diverge", 0),
            "diverge_rate": modes.get("first_token_correct_then_diverge", 0) / len(rows),
            "first_token_correct": sum(bool(x["first_token_correct"]) for x in rows),
            "modes": dict(modes),
            "generated_text_counts": dict(gen),
        }
        first_token_groups[first].add(name)

    for rec in names.values():
        family = sorted(first_token_groups[rec["first_token"]])
        rec["competing_dev_names_sharing_first_token"] = [n for n in family if n != rec["name"]]
        rec["collision_family"] = family if len(family) > 1 else [rec["name"]]
        rec["dev_first_token_unique"] = len(family) == 1
        gold = rec["v0_7_token_ids"]
        rec["first_identity_disambiguating_position_1based"] = 1 if rec["dev_first_token_unique"] else (
            next(
                (i + 1 for i, tok in enumerate(gold) if any(
                    i >= len(names[o]["v0_7_token_ids"]) or names[o]["v0_7_token_ids"][i] != tok
                    for o in rec["competing_dev_names_sharing_first_token"]
                )),
                None,
            )
        )
        # Token-prefix of a frequent generated continuation (Salt/Sky/Walt), not just DEV-name collisions.
        cont = []
        for gtxt, c in rec["generated_text_counts"].items():
            if gtxt != rec["answer_text"] and rec["answer_text"].rstrip(".") in gtxt:
                cont.append({"generated_text": gtxt, "n": c, "relation": "gold_string_is_prefix_of_generated"})
            if rec["answer_text"].startswith(gtxt.rstrip(".")) and gtxt != rec["answer_text"]:
                cont.append({"generated_text": gtxt, "n": c, "relation": "generated_is_prefix_of_gold_string"})
        rec["string_prefix_continuations"] = cont

    shared = [n for n, r in names.items() if not r["dev_first_token_unique"]]
    unique = [n for n, r in names.items() if r["dev_first_token_unique"]]
    payload = {
        "source": str(ITEMS),
        "n_observations": len(items),
        "names": names,
        "first_token_collision_groups": {
            str(k): sorted(v) for k, v in first_token_groups.items() if len(v) > 1
        },
        "shared_first_token_names": shared,
        "unique_first_token_names": unique,
        "shared_exact_rate": sum(names[n]["exact"] for n in shared) / sum(names[n]["n"] for n in shared),
        "unique_exact_rate": sum(names[n]["exact"] for n in unique) / sum(names[n]["n"] for n in unique),
        "note": "dev_first_token_unique is uniqueness among the 8 DEV names only, not uniqueness in the 1024-piece vocab. Wes is unique in DEV but Walt shares the same first token.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
