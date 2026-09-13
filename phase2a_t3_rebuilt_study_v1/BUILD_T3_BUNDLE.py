"""Materialize the rebuilt T3 corpus, panels, schedules, and freeze hashes.

Does not load the parent checkpoint or create an optimizer.
Does not read T2-EVAL-TEST, FINAL, sacred, or locked transfer panels.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

from tokenizers import Tokenizer

BUNDLE = Path(r"C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1")
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TOK_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"

EXCLUDED = {
    # SF frozen identities
    "Sara", "Bob", "Billy", "Tony", "Fred", "Tina", "Steve", "Susan", "Remy",
    "Ralph", "Colin", "Faye", "Carl", "Wendy", "Cathy", "Tara", "Alex", "Owen",
    "Mia", "Nora",
    # T2 inspectable train/novel names (avoid panel collision without opening TEST)
    "Lily", "Ann", "Zoe", "Ivy", "Tom", "Ben", "Noah", "Sam", "Max",
    "Kai", "Pia", "Ned", "Rex", "Ted", "Ella",
}

RAW_ALLOWLIST = [
    "Ada", "Amy", "Ava", "Bea", "Cam", "Cora", "Dale", "Dan", "Deb", "Don",
    "Earl", "Eli", "Eva", "Fay", "Finn", "Fran", "Gia", "Glen", "Grace", "Gus",
    "Hal", "Helen", "Hope", "Hugo", "Ian", "Ida", "Iris", "Ivan", "Jack", "Jade",
    "Jan", "Jed", "Jen", "Jim", "Joe", "Joy", "Joyce", "Karl", "Kay", "Kent",
    "Kim", "Laura", "Lee", "Leo", "Lois", "Lou", "Luke", "Mae", "Marie", "Mark",
    "Maya", "Meg", "Mel", "Mona", "Nancy", "Nat", "Neil", "Nia", "Nina", "Olive",
    "Omar", "Opal", "Pam", "Pat", "Paul", "Pearl", "Peg", "Quinn", "Ray", "Rita",
    "Rob", "Ron", "Rose", "Ross", "Roy", "Ruth", "Sal", "Seth", "Sid", "Skye",
    "Sue", "Tess", "Tim", "Troy", "Uma", "Val", "Vera", "Vic", "Vince", "Wade",
    "Walt", "Wes", "Will", "Willa", "York", "Zane", "Zara",
]

COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["ball", "book", "box", "car", "kite", "hat"]
PHRASES = [f"{c} {o}" for c in COLORS for o in OBJECTS]
TEMPLATES_TRAIN = ["active", "passive"]
TEMPLATES_HELDOUT = ["owns", "locative"]
ALL_TEMPLATES = TEMPLATES_TRAIN + TEMPLATES_HELDOUT
BOS = 2
EOS = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def render(template: str, n0: str, n1: str, o0: str, o1: str, fact_order: int, query: str) -> str:
    if fact_order == 0:
        a, oa, b, ob = n0, o0, n1, o1
    else:
        a, oa, b, ob = n1, o1, n0, o0
    if template == "active":
        s1, s2 = f"{a} has the {oa}.", f"{b} has the {ob}."
        q = f"Who has the {query}? Answer:"
    elif template == "passive":
        s1, s2 = f"The {oa} is with {a}.", f"The {ob} is with {b}."
        q = f"Who has the {query}? Answer:"
    elif template == "owns":
        s1, s2 = f"{a} owns the {oa}.", f"{b} owns the {ob}."
        q = f"Who owns the {query}? Answer:"
    elif template == "locative":
        s1, s2 = f"The {oa} belongs to {a}.", f"The {ob} belongs to {b}."
        q = f"Who has the {query}? Answer:"
    else:
        raise ValueError(template)
    return f"{s1} {s2} {q}"


def find_name_span(prompt_ids: list[int], name: str, tok: Tokenizer) -> list[int]:
    variants = [tok.encode(name).ids, tok.encode(" " + name).ids]
    seen = []
    for v in variants:
        if v and v not in seen:
            seen.append(v)
    for v in seen:
        n = len(v)
        for i in range(len(prompt_ids) - n + 1):
            if prompt_ids[i:i + n] == v:
                return list(range(i, i + n))
    raise RuntimeError(f"name span not found for {name!r}")


def name_eligible(name: str, tok: Tokenizer) -> bool:
    if name in EXCLUDED:
        return False
    probe = render("active", name, "Zane" if name != "Zane" else "Ada",
                   "red ball", "blue book", 0, "red ball")
    ids = tok.encode(probe).ids
    if len(ids) > 250:
        return False
    try:
        find_name_span(ids, name, tok)
    except RuntimeError:
        return False
    decoded = tok.decode(tok.encode(name).ids)
    return name.lower() in decoded.lower() or name in decoded


def round_robin_pairs(names: list[str]) -> list[tuple[str, str]]:
    n = list(names)
    if len(n) % 2:
        n.append("__BYE__")
    rounds = []
    m = len(n)
    for _ in range(m - 1):
        pairs = []
        for i in range(m // 2):
            a, b = n[i], n[m - 1 - i]
            if "__BYE__" not in (a, b):
                pairs.append((a, b) if a < b else (b, a))
        rounds.append(pairs)
        n = [n[0]] + [n[-1]] + n[1:-1]
    out = []
    for r in rounds:
        out.extend(r)
    return out


def object_pairs(rng: random.Random, count: int) -> list[tuple[str, str]]:
    combos = [(PHRASES[i], PHRASES[j]) for i in range(len(PHRASES))
              for j in range(i + 1, len(PHRASES))]
    rng.shuffle(combos)
    if len(combos) < count:
        raise RuntimeError("not enough object pairs")
    return combos[:count]


def build_family(fid: str, split: str, n0: str, n1: str, o0: str, o1: str,
                 template: str, tok: Tokenizer) -> list[dict]:
    rows = []
    for assign in (0, 1):
        own0 = o0 if assign == 0 else o1
        own1 = o1 if assign == 0 else o0
        for query_i in (0, 1):
            query = o0 if query_i == 0 else o1
            if query == own0:
                correct = n0
            elif query == own1:
                correct = n1
            else:
                raise RuntimeError("query object not owned")
            for order in (0, 1):
                prompt_text = render(template, n0, n1, own0, own1, order, query)
                prompt_ids = tok.encode(prompt_text).ids
                if len(prompt_ids) > 250:
                    raise RuntimeError(f"prompt too long: {fid}")
                span0 = find_name_span(prompt_ids, n0, tok)
                span1 = find_name_span(prompt_ids, n1, tok)
                first_mentioned = n0 if min(span0) < min(span1) else n1
                last_mentioned = n1 if first_mentioned == n0 else n0
                cands = [n0, n1]
                cand_ids = [tok.encode(f" {nm}.").ids for nm in cands]
                ci = cands.index(correct)
                rid = f"{fid}:a{assign}:q{query_i}:o{order}"
                rows.append({
                    "id": rid,
                    "family_id": fid,
                    "split": split,
                    "template": template,
                    "assignment": assign,
                    "query_index": query_i,
                    "fact_order": order,
                    "names": [n0, n1],
                    "objects": [o0, o1],
                    "owned": [own0, own1],
                    "query_object": query,
                    "candidates": cands,
                    "correct_name": correct,
                    "correct_index": ci,
                    "prompt": prompt_text,
                    "prompt_token_ids": prompt_ids,
                    "prompt_len": len(prompt_ids),
                    "candidate_token_ids": cand_ids,
                    "mention_span_prompt": [span0, span1],
                    "mention_span_bos": [[i + 1 for i in span0], [i + 1 for i in span1]],
                    "decision_position_bos": len(prompt_ids),
                    "first_mentioned": first_mentioned,
                    "last_mentioned": last_mentioned,
                    "twin_id": f"{fid}:q{query_i}:o{order}",
                })
    if len(rows) != 8:
        raise RuntimeError("family size != 8")
    return rows


def name_balance(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["correct_name"]] = counts.get(r["correct_name"], 0) + 1
    return dict(sorted(counts.items()))


def audit_split(rows: list[dict], n_families: int) -> dict:
    fams = sorted({r["family_id"] for r in rows})
    assign_pairs = 0
    by = {}
    for r in rows:
        by.setdefault(r["twin_id"], {})[r["assignment"]] = r
    for d in by.values():
        if 0 in d and 1 in d:
            assign_pairs += 1
    ci = {0: 0, 1: 0}
    order = {0: 0, 1: 0}
    q = {0: 0, 1: 0}
    for r in rows:
        ci[r["correct_index"]] += 1
        order[r["fact_order"]] += 1
        q[r["query_index"]] += 1
    return {
        "n_rows": len(rows),
        "n_families": len(fams),
        "expected_families": n_families,
        "assign_flip_pairs": assign_pairs,
        "ci_balance": ci,
        "order_balance": order,
        "query_balance": q,
        "name_balance": name_balance(rows),
        "templates": sorted({r["template"] for r in rows}),
        "names": sorted({n for r in rows for n in r["names"]}),
    }


def acq16(train: list[dict]) -> list[dict]:
    fams = sorted({r["family_id"] for r in train})
    chosen = fams[:4]
    out = [r for r in train if r["family_id"] in chosen and r["fact_order"] == 0]
    if len(out) != 16:
        raise RuntimeError(f"ACQ16 size {len(out)}")
    return out


def main() -> None:
    if sha256_file(TOK_PATH) != TOK_SHA:
        raise RuntimeError("tokenizer hash mismatch")
    tok = Tokenizer.from_file(str(TOK_PATH))
    eligible = sorted(n for n in RAW_ALLOWLIST if name_eligible(n, tok))
    if len(eligible) < 32:
        raise RuntimeError(f"only {len(eligible)} eligible names")
    train_rng = random.Random(620101)
    dev_rng = random.Random(620102)
    test_rng = random.Random(620103)
    pool = list(eligible)
    train_rng.shuffle(pool)
    train_names = sorted(pool[:16])
    rest = pool[16:]
    dev_rng.shuffle(rest)
    dev_names = sorted(rest[:8])
    rest2 = rest[8:]
    test_rng.shuffle(rest2)
    test_names = sorted(rest2[:8])
    if set(train_names) & set(dev_names) or set(train_names) & set(test_names) or set(dev_names) & set(test_names):
        raise RuntimeError("name overlap")

    obj_rng = random.Random(620104)
    objs = object_pairs(obj_rng, 80)
    train_pairs = round_robin_pairs(train_names)[:48]
    dev_pairs = round_robin_pairs(dev_names)[:16]
    test_pairs = round_robin_pairs(test_names)[:16]
    if len(train_pairs) != 48 or len(dev_pairs) != 16 or len(test_pairs) != 16:
        raise RuntimeError("pair count")

    def stamp(pairs, split, prefix, templates, start_obj):
        rows = []
        for i, (a, b) in enumerate(pairs):
            tmpl = templates[i % len(templates)]
            o0, o1 = objs[start_obj + i]
            fid = f"{prefix}{i:02d}"
            rows.extend(build_family(fid, split, a, b, o0, o1, tmpl, tok))
        return rows

    train = stamp(train_pairs, "train", "TR", TEMPLATES_TRAIN, 0)
    # DEV: 8 in-template families + 8 held-out-template families
    dev_in = stamp(dev_pairs[:8], "dev", "DI", TEMPLATES_TRAIN, 48)
    dev_td = stamp(dev_pairs[8:], "dev", "DT", TEMPLATES_HELDOUT, 56)
    dev = dev_in + dev_td
    test_in = stamp(test_pairs[:8], "test", "SI", TEMPLATES_TRAIN, 64)
    test_td = stamp(test_pairs[8:], "test", "ST", TEMPLATES_HELDOUT, 72)
    test = test_in + test_td

    texts = {
        "train": {r["prompt"] for r in train},
        "dev": {r["prompt"] for r in dev},
        "test": {r["prompt"] for r in test},
    }
    if texts["train"] & texts["dev"] or texts["train"] & texts["test"] or texts["dev"] & texts["test"]:
        raise RuntimeError("prompt text overlap")

    acq = acq16(train)
    name_disjoint_ids = [r["id"] for r in dev_in]
    template_disjoint_ids = [r["id"] for r in dev_td]
    if len(name_disjoint_ids) != 64 or len(template_disjoint_ids) != 64:
        raise RuntimeError("panel size")

    data = BUNDLE / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_jsonl(data / "qa_train.jsonl", train)
    write_jsonl(data / "qa_dev.jsonl", dev)
    write_jsonl(data / "qa_test.jsonl", test)
    write_json(data / "acq16.json", acq)
    write_json(data / "panels.json", {
        "name_disjoint_ids": name_disjoint_ids,
        "template_disjoint_ids": template_disjoint_ids,
        "acq16_ids": [r["id"] for r in acq],
        "acq16_families": sorted({r["family_id"] for r in acq}),
        "dev_in_template_families": sorted({r["family_id"] for r in dev_in}),
        "dev_template_disjoint_families": sorted({r["family_id"] for r in dev_td}),
    })

    kinds = []
    for u in range(1, 501):
        kinds.append("lang" if u % 10 == 0 else "qa")
    write_json(data / "update_kinds.json", {"kinds": kinds, "qa": 450, "lang": 50})

    schedules = {}
    for seed in (620001, 620002, 620003):
        idx = list(range(len(train)))
        rng = random.Random(seed + 101)
        rng.shuffle(idx)
        order = []
        cursor = 0
        for u in range(450):
            batch = [idx[(cursor + j) % len(idx)] for j in range(32)]
            order.append(batch)
            cursor += 32
        schedules[str(seed)] = order
    write_json(data / "qa_schedules.json", schedules)

    # Shared language rehearsal: 50 updates x 64 starts, derived from EVAL train slice
    # (320 train-fit windows at indices 1280:1600 of EVAL_WINDOW_STARTS).
    lang_starts = []
    for li in range(50):
        lang_starts.append([((li * 64 + j) % 320) for j in range(64)])
    write_json(data / "rehearsal_index.json", {"starts_mod_320": lang_starts})

    audits = {
        "train": audit_split(train, 48),
        "dev": audit_split(dev, 16),
        "test": audit_split(test, 16),
        "name_disjoint": True,
        "train_names": train_names,
        "dev_names": dev_names,
        "test_names": test_names,
        "eligible_count": len(eligible),
        "prompt_overlap": False,
    }
    # TEST names/templates are recorded for freeze integrity only; runner must not load TEST rows.
    write_json(data / "audits.json", audits)
    write_json(data / "TEST_SEAL.json", {
        "status": "SEALED_UNOPENED",
        "rows": 128,
        "sha256": sha256_file(data / "qa_test.jsonl"),
        "rule": "Do not load qa_test.jsonl until terminal DEV representation AND native gates pass.",
    })

    payloads = [
        BUNDLE / "PROTOCOL.json",
        BUNDLE / "DECISION_LEDGER.md",
        BUNDLE / "BUILD_T3_BUNDLE.py",
        BUNDLE / "T3_RUNNER.py",
        data / "qa_train.jsonl",
        data / "qa_dev.jsonl",
        data / "qa_test.jsonl",
        data / "acq16.json",
        data / "panels.json",
        data / "update_kinds.json",
        data / "qa_schedules.json",
        data / "rehearsal_index.json",
        data / "audits.json",
        data / "TEST_SEAL.json",
    ]
    sums = {}
    for p in payloads:
        if p.exists():
            sums[str(p)] = sha256_file(p)
    (BUNDLE / "SHA256SUMS.txt").write_text(
        "\n".join(f"{h}  {Path(k).name}" for k, h in sums.items()) + "\n",
        encoding="utf-8",
    )
    write_json(BUNDLE / "PROVENANCE.json", {
        "study": "BABY_VNEXT_PHASE2A_T3_REBUILT_EXPLICIT_REPRESENTATION_FORMING_V1",
        "tokenizer_sha256": TOK_SHA,
        "parent_sha256": "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1",
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "training_performed": False,
        "locked_material_accessed": False,
        "test_opened": False,
        "file_sha256": {Path(k).name: v for k, v in sums.items()},
    })
    write_json(BUNDLE / "RUN_LEDGER.json", {
        "status": "BUNDLE_FROZEN_NOT_TRAINED",
        "decision": "REBUILD_AND_RUN_T3",
        "seeds": [620001, 620002, 620003],
        "runs": {},
        "test_opened": False,
    })
    print(json.dumps({
        "status": "FROZEN",
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "acq16": len(acq),
        "train_names": train_names,
        "dev_names": dev_names,
        "test_names_count": len(test_names),
        "eligible": len(eligible),
        "test_sha256": audits and sha256_file(data / "qa_test.jsonl"),
    }, indent=2))


if __name__ == "__main__":
    main()
